"""Per-turn coordinator: resolve model, assemble context, call backend, stream out.

Owns one chat turn end-to-end:
  1. resolve the friendly model name to a registry spec (fail fast if unknown,
     not pulled, or the window can't fit),
  2. load prior turns from the session store,
  3. assemble the prompt within the token budget (Level 2: recursive
     summarization first), always through the hard window guard,
  4. stream content deltas from the backend to the caller,
  5. persist the exchange, calibrate token counting, emit metrics.

It yields small event dicts (the SSE route owns the wire format):
  {"status": "loading"|"summarizing", "text": str}  — slow work before the reply
  {"status": "prefill"}   request handed to the model; no token yet
  {"delta": str}          a content token to append to the visible answer
  {"thinking": int}       cumulative reasoning-token estimate (liveness heartbeat;
                          the reasoning TEXT is never sent)
  {"stats": {...}}        terminal: ttft_s, eval_count, tok_s, thinking_tokens, ...
Failures raise (UnknownModelError, ContextOverflowError, BackendError); the route
turns them into an error event.

Persistence: the user message and the reply are written together, only once
there is reply text — a failed turn never leaves a dangling user message, and a
cancelled turn (client disconnect / Stop) keeps whatever was streamed so far.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import structlog

from config.settings import Settings
from llm.backends.base import BackendError, ChatEvent, LLMBackend
from llm.registry import ModelSpec, Registry
from metrics.instrument import TurnMetrics
from orchestrator.context_budget import Assembly, ContextBudget
from orchestrator.summarization import (
    build_summarizer,
    deserialize_state,
    make_summarize_fn,
    serialize_state,
    summary_block,
    will_call_model,
)
from recursive_summarization import from_chat_history
from session.store import SessionStore

log = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        registry: Registry,
        backend: LLMBackend,
        store: SessionStore,
        budget: ContextBudget,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.backend = backend
        self.store = store
        self.budget = budget
        # The model we last enforced single-residency for; repeated turns with the
        # same model skip the per-turn unload_others /api/ps round-trip.
        self._enforced_model: str | None = None

    # ── Level 2 ──────────────────────────────────────────────────────────

    def _plan_level2(
        self,
        spec: ModelSpec,
        history: list[dict],
        message: str,
        session_id: str,
        num_ctx: int,
        max_tokens: int,
        keep_alive: int | str | None,
        num_gpu: int | None,
    ):
        """Decide compact vs fold for this turn.

        Returns (run, finish, will_call): ``run`` does the (blocking) compaction,
        ``finish`` persists its state, ``will_call`` predicts a model call.
        """
        model = spec.model
        anchors = self.budget.message_tokens(
            {"content": self.settings.system_prompt}, model
        ) + self.budget.message_tokens({"content": message}, model)
        history_budget = num_ctx - max_tokens - self.budget.safety_margin - anchors
        summarizer = build_summarizer(
            self.settings,
            make_summarize_fn(
                self.backend, asyncio.get_running_loop(), spec, num_ctx, keep_alive, num_gpu
            ),
            history_budget=history_budget,
            num_ctx=num_ctx,
            chars_per_token=self.budget.chars_per_token(model),
        )
        turns = from_chat_history(history)
        cadence = self.settings.summary_recompact_cadence

        prior, covered, fold_count = None, 0, 0
        raw_state = self.store.get_compaction_state(session_id)
        if raw_state:
            prior, covered, fold_count = deserialize_state(raw_state)

        # Fold only once a real summary exists, the recompaction cadence hasn't
        # elapsed, and the bookkeeping is sane. Otherwise compact() from the raw
        # turns: that bootstraps, keeps short sessions in zero-call passthrough,
        # and periodically resets accumulated fold drift.
        summarized = bool(prior and prior.summary and prior.summary.text)
        can_fold = (
            summarized
            and (cadence <= 0 or fold_count < cadence)
            and 0 <= covered <= len(turns)
        )
        if can_fold:
            new_raw = turns[covered:]
            op = "fold"
            will_call = bool(new_raw) and will_call_model(summarizer, turns, prior, new_raw)

            def run():
                return summarizer.fold(prior, new_raw) if new_raw else prior

            next_fold = fold_count + 1
        else:
            op = "recompact" if summarized else "compact"
            will_call = will_call_model(summarizer, turns, None, None)

            def run():
                return summarizer.compact(turns)

            next_fold = 0

        def finish(compacted):
            self.store.set_compaction_state(
                session_id, serialize_state(compacted, len(turns), next_fold)
            )
            if compacted.summary and compacted.summary.text:
                self.store.set_summary(session_id, compacted.summary.text)
            s = compacted.stats
            log.info(
                "context_compaction",
                session_id=session_id,
                op=op,
                fold_count=next_fold,
                passthrough=s.passthrough,
                summarize_calls=s.summarize_calls,
                max_recursion_depth=s.max_recursion_depth,
                tighten_passes=s.tighten_passes,
                hard_truncations=s.hard_truncations,
                summary_tokens=compacted.summary.token_count if compacted.summary else 0,
                recent_turns=len(compacted.recent_turns),
            )
            return compacted

        return run, finish, will_call

    # ── one turn ─────────────────────────────────────────────────────────

    async def generate(
        self,
        *,
        session_id: str,
        message: str,
        model_name: str,
        options: dict | None = None,
    ) -> AsyncIterator[dict]:
        options = options or {}
        spec = self.registry.resolve(model_name)
        model = spec.model

        temperature = options.get("temperature", self.settings.default_temperature)
        max_tokens = options.get("max_tokens", self.settings.default_max_tokens)
        top_p = options.get("top_p", self.settings.default_top_p)
        top_k = options.get("top_k", self.settings.default_top_k)
        # Effective context window: user override from the LLM Config panel, else
        # the registry default. Drives the budget guard, the Level 2 summarizer and
        # Ollama's num_ctx consistently.
        num_ctx = options.get("num_ctx") or spec.default_ctx
        # Layers to offload to the GPU: None = Ollama decides; 0 = force full CPU.
        num_gpu = options.get("num_gpu")
        keep_alive = options.get("keep_alive", self.settings.ollama_keep_alive)
        think = spec.think_value(bool(options.get("think", False)))

        state = await self.backend.warm_state(model, num_ctx)
        if state == "missing":
            raise BackendError(
                f"Model '{model}' is not available in Ollama. Pull it first: ollama pull {model}"
            )

        # Single 16GB card: keep exactly one model resident. Only re-enforced when
        # the active model changes, so same-model turns skip the /api/ps round-trip.
        if self._enforced_model != model:
            try:
                await self.backend.unload_others(model)
                self._enforced_model = model
            except Exception as exc:  # noqa: BLE001 — enforcement is best-effort
                log.warning("unload_others_failed", error=str(exc))

        if state in ("cold", "reload"):
            log.info("model_cold_start", session_id=session_id, model=model,
                     num_ctx=num_ctx, state=state)
            yield {
                "status": "loading",
                "text": f"Loading {model_name} (context {num_ctx:,}) — can take up to a minute…",
            }

        history = self.store.get_turns(session_id)

        if self.settings.context_level >= 2:
            run, finish, will_call = self._plan_level2(
                spec, history, message, session_id, num_ctx, max_tokens, keep_alive, num_gpu
            )
            if will_call:
                yield {"status": "summarizing", "text": "Summarizing earlier conversation…"}
            # Synchronous library, possibly several model calls: off the event loop.
            compacted = finish(await asyncio.to_thread(run))
            assembly: Assembly = self.budget.assemble(
                system_prompt=self.settings.system_prompt,
                history=[{"role": t.role.value, "content": t.text} for t in compacted.recent_turns],
                current_message=message,
                num_ctx=num_ctx,
                max_tokens=max_tokens,
                model=model,
                summary=summary_block(compacted),
                sliding=True,  # final hard guard: recency-wins overflow is trimmed here
            )
        else:
            assembly = self.budget.assemble(
                system_prompt=self.settings.system_prompt,
                history=history,
                current_message=message,
                num_ctx=num_ctx,
                max_tokens=max_tokens,
                model=model,
            )
        if assembly.dropped_turns or assembly.summary_dropped:
            log.info("context_trimmed", session_id=session_id,
                     dropped=assembly.dropped_turns, summary_dropped=assembly.summary_dropped)
        messages = assembly.messages

        log.info(
            "turn_start",
            session_id=session_id,
            model=model,
            context_level=self.settings.context_level,
            num_ctx=num_ctx,
            context_messages=len(messages),
            history_turns=len(history),
            prompt_tokens_est=assembly.prompt_tokens_est,
            think=think,
        )

        metrics = TurnMetrics(session_id=session_id, model=model,
                              prompt_tokens_est=assembly.prompt_tokens_est)
        parts: list[str] = []
        final: ChatEvent | None = None
        thinking_chars = 0
        thinking_tokens = 0
        outcome = "error"

        # Request in flight before any token: a long prefill reads as "waiting",
        # not as a frozen turn.
        yield {"status": "prefill"}
        try:
            async for ev in self.backend.chat(
                messages,
                model,
                num_ctx=num_ctx,
                num_gpu=num_gpu,
                think=think,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                keep_alive=keep_alive,
            ):
                if ev.delta:
                    metrics.first_token()
                    parts.append(ev.delta)
                    yield {"delta": ev.delta}
                if ev.thinking_delta:
                    thinking_chars += len(ev.thinking_delta)
                    thinking_tokens = self.budget.estimate_tokens_from_chars(thinking_chars, model)
                    yield {"thinking": thinking_tokens}
                if ev.done:
                    final = ev
            outcome = "ok"
            yield {"stats": metrics.stats(final, thinking_tokens)}
        except (asyncio.CancelledError, GeneratorExit):
            outcome = "cancelled"
            raise
        finally:
            answer = "".join(parts).strip()
            if answer:
                # Synchronous SQLite write: completes even while this generator is
                # being cancelled (an awaited write would be cancelled with it).
                self.store.append_exchange(
                    session_id, model_name, message, answer,
                    user_tokens=self.budget.estimate_tokens(message, model),
                    reply_tokens=self.budget.estimate_tokens(answer, model),
                )
            if final is not None and final.prompt_eval_count:
                prompt_chars = sum(len(m.get("content", "")) for m in messages)
                self.budget.calibrate(model, prompt_chars, final.prompt_eval_count)
            metrics.emit(final, outcome=outcome, thinking_tokens=thinking_tokens)
