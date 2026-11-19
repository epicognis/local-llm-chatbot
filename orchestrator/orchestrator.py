import asyncio

import anyio
import structlog

from config.settings import settings
from llm.backends.base import BackendError, LLMBackend, StreamStatus
from llm.registry import get_model_entry
from metrics.instrument import TurnMetrics
from orchestrator.context_budget import (
    ContextBudgetAllocator,
    TokenCalibrator,
    estimate_messages,
)
from orchestrator.memory import (
    ContextProvider,
    RawPassthroughProvider,
    RollingSummaryProvider,
    SlidingWindowProvider,
)
from session.store import SessionStore

log = structlog.get_logger()

SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable AI assistant. "
    "Answer clearly and accurately. "
    "Be concise unless depth is needed."
)


class Orchestrator:
    def __init__(
        self,
        backend: LLMBackend,
        store: SessionStore,
        calibrator: TokenCalibrator | None = None,
    ):
        self.backend = backend
        self.store = store
        self.calibrator = calibrator or TokenCalibrator()

    def _get_provider(self, model: str, runtime_opts: dict | None = None) -> ContextProvider:
        level = settings.CONTEXT_LEVEL
        if level == 0:
            return RawPassthroughProvider()
        if level == 1:
            return SlidingWindowProvider(settings.SLIDING_WINDOW_TURNS)
        if level == 2:
            return RollingSummaryProvider(
                backend=self.backend,
                store=self.store,
                summarizer_model=model,
                window_turns=settings.SLIDING_WINDOW_TURNS,
                trigger_tokens=settings.SUMMARY_TRIGGER_TOKENS,
                summarize_opts=runtime_opts,
            )
        raise ValueError(f"Unsupported CONTEXT_LEVEL: {level} (only 0, 1, 2 are implemented)")

    async def chat_stream(
        self,
        session_id: str,
        message: str,
        model_name: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Assembles context and streams the model response.

        Yields events: {"type": "status"|"token", "text": str}. Failures raise
        (UnknownModelError, ValueError, BackendError) — the API layer turns them
        into error events. The user/assistant exchange is persisted only once the
        model has produced output, so a failed turn never leaves a dangling user
        message in history; a cancelled turn keeps whatever was streamed so far.
        """
        # Validate everything that can fail cheaply before touching the store.
        entry = get_model_entry(model_name)
        model = entry["model"]
        default_ctx = entry["default_ctx"]

        resolved_max_tokens = max_tokens or settings.DEFAULT_MAX_TOKENS
        resolved_temp = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE

        allocator = ContextBudgetAllocator(
            model_ctx=default_ctx,
            max_tokens=resolved_max_tokens,
            safety_margin=settings.CONTEXT_SAFETY_MARGIN,
            ratio=self.calibrator.ratio(model),
        )

        # Options every call to this model needs, including summarize().
        runtime_opts: dict = {"num_ctx": default_ctx, "keep_alive": settings.OLLAMA_KEEP_ALIVE}
        if entry.get("think") is not None:
            runtime_opts["think"] = entry["think"]  # spec §6: direct answers by default

        session = await self.store.get_session(session_id)
        turns = await self.store.get_turns(session_id)
        provider = self._get_provider(model, runtime_opts)

        state = await self.backend.warm_state(model, default_ctx)
        if state == "missing":
            raise BackendError(
                f"Model '{model}' is not available in Ollama. Pull it first: ollama pull {model}"
            )
        if state in ("cold", "reload"):
            log.info("model_cold_start", session_id=session_id, model=model,
                     num_ctx=default_ctx, state=state)
            yield {
                "type": "status",
                "text": f"Loading {model_name} into memory (ctx={default_ctx}) — "
                        f"large context windows can take up to a minute on first use…",
            }

        pending = provider.pending_work(session, turns)
        if pending:
            yield {"type": "status", "text": pending}

        context_turns, summary = await provider.prepare(session_id, session, turns)
        messages = allocator.assemble(
            system_prompt=SYSTEM_PROMPT,
            turns=context_turns,
            current_message=message,
            summary=summary,
        )

        log.info(
            "turn_start",
            session_id=session_id,
            model=model,
            context_level=settings.CONTEXT_LEVEL,
            context_messages=len(messages),
            history_turns=len(turns),
            token_ratio=round(allocator.ratio, 3),
        )

        stats: dict = {}
        opts = {
            **runtime_opts,
            "temperature": resolved_temp,
            "num_predict": resolved_max_tokens,
            "stats": stats,
        }
        metrics = TurnMetrics(session_id=session_id, model=model)
        response_parts: list[str] = []
        outcome = "error"

        try:
            async for token in self.backend.chat(messages, model, stream=True, **opts):
                if isinstance(token, StreamStatus):
                    yield {"type": "status", "text": str(token)}
                    continue
                response_parts.append(token)
                metrics.count_token()
                yield {"type": "token", "text": token}
            outcome = "ok"
        except (asyncio.CancelledError, GeneratorExit):
            outcome = "cancelled"
            raise
        finally:
            full_response = "".join(response_parts)
            if full_response:
                # Shielded: on client disconnect the surrounding scope is already
                # cancelled, and an unshielded await here would drop the reply.
                with anyio.CancelScope(shield=True):
                    await self.store.append_exchange(session_id, message, full_response)
            if stats.get("prompt_eval_count"):
                self.calibrator.observe(model, estimate_messages(messages), stats["prompt_eval_count"])
            metrics.finish(stats, outcome=outcome)
