"""Level 2 context management — recursive summarization wiring.

Bridges the chatbot to the standalone ``recursive_summarization`` library. It
builds a per-turn ``RecursiveSummarizer`` and hands it a **synchronous**
``SummarizeFn``.

The library is synchronous and may make several model calls during one
compaction, so the orchestrator runs it in a worker thread (``asyncio.to_thread``).
The summarize function bridges back onto the event loop and calls the *same*
async backend used for chat (``backend.complete``), so summaries get the same
error handling, pooled connection, and per-model settings:

  * ``num_ctx`` equal to the generation's — Ollama keys a resident runner by
    (model, num_ctx, ...), so a different value would force an unload/reload of
    the model between the summary and the answer on every turn.
  * the model's registry ``think_off`` value — ``think: false`` would cost
    GPT-OSS ~2,000 hidden reasoning tokens per summary call.
  * low temperature for stable summaries; the same ``keep_alive``.
"""

from __future__ import annotations

import asyncio
import json

from config.settings import Settings
from llm.backends.base import LLMBackend
from llm.registry import ModelSpec
from recursive_summarization import (
    CompactedContext,
    CompactionStats,
    HeuristicCounter,
    RecursiveSummarizer,
    Role,
    RunningSummary,
    SummarizerConfig,
    Turn,
)

_SUMMARY_TEMPERATURE = 0.1
_SUMMARY_MAX_TOKENS = 2048


def make_summarize_fn(
    backend: LLMBackend,
    loop: asyncio.AbstractEventLoop,
    spec: ModelSpec,
    num_ctx: int,
    keep_alive: int | str | None,
    num_gpu: int | None = None,
):
    """A sync ``prompt -> text`` callable for use from a worker thread."""
    opts = {
        "num_ctx": num_ctx,
        "think": spec.think_off,
        "temperature": _SUMMARY_TEMPERATURE,
        "max_tokens": min(_SUMMARY_MAX_TOKENS, max(256, num_ctx // 4)),
        "keep_alive": keep_alive,
    }
    if num_gpu is not None:
        opts["num_gpu"] = num_gpu  # must match generation, or the runner reloads

    def fn(prompt: str) -> str:
        future = asyncio.run_coroutine_threadsafe(
            backend.complete(prompt, spec.model, **opts), loop
        )
        return future.result()

    return fn


def build_summarizer(
    settings: Settings,
    summarize_fn,
    *,
    history_budget: int,
    num_ctx: int,
    chars_per_token: float,
) -> RecursiveSummarizer:
    """Construct a RecursiveSummarizer for one turn.

    ``history_budget`` is the room left for prior conversation once the system
    prompt, current message, reply and safety margin are reserved — compaction
    only engages once the history actually stops fitting the window.
    """
    target = settings.summary_target_budget_tokens or history_budget
    # One summarize call must fit the window alongside its own output.
    input_budget = min(settings.summary_input_budget_tokens, max(512, num_ctx - _SUMMARY_MAX_TOKENS - 512))
    config = SummarizerConfig(
        keep_recent_turns=settings.summary_keep_recent_turns,
        target_budget_tokens=max(256, target),
        summarize_input_budget=input_budget,
        chars_per_token=chars_per_token,
    )
    return RecursiveSummarizer(
        summarize_fn=summarize_fn,
        config=config,
        count_tokens=HeuristicCounter(chars_per_token),
    )


def will_call_model(
    summarizer: RecursiveSummarizer,
    turns: list[Turn],
    prior: CompactedContext | None,
    new_turns: list[Turn] | None,
) -> bool:
    """Cheap pre-check of whether compact()/fold() is about to call the model,
    so the UI can show "summarizing…" instead of unexplained silence."""
    cfg = summarizer._cfg
    count = summarizer._count
    if new_turns is not None and prior is not None:  # fold path
        working = list(prior.recent_turns) + list(new_turns)
        summary_tokens = prior.summary.token_count if prior.summary else 0
        return summary_tokens + sum(count(t.render()) for t in working) > cfg.effective_budget
    total = sum(count(t.render()) for t in turns)
    return total > cfg.effective_budget and len(turns) > cfg.keep_recent_turns


def summary_block(compacted: CompactedContext) -> str | None:
    """The running summary as the system-message text the allocator injects."""
    s = compacted.summary
    if not (s and s.text):
        return None
    return f"[Conversation summary — {s.turns_folded} earlier turns compressed]\n{s.text}"


# ---------------------------------------------------------------------------
# Compaction-state persistence — lets the incremental fold() path survive across
# stateless HTTP turns: the running summary + verbatim recency window + how many
# raw turns are already represented (covered_count) + folds since the last full
# recompaction (fold_count).
# ---------------------------------------------------------------------------


def serialize_state(compacted: CompactedContext, covered_count: int, fold_count: int) -> str:
    summary = None
    if compacted.summary:
        summary = {
            "text": compacted.summary.text,
            "turns_folded": compacted.summary.turns_folded,
            "token_count": compacted.summary.token_count,
            "recursion_depth": compacted.summary.recursion_depth,
        }
    return json.dumps(
        {
            "summary": summary,
            "recent_turns": [
                {"role": t.role.value, "text": t.text, "metadata": t.metadata}
                for t in compacted.recent_turns
            ],
            "covered_count": covered_count,
            "fold_count": fold_count,
        }
    )


def deserialize_state(raw: str) -> tuple[CompactedContext, int, int]:
    data = json.loads(raw) if isinstance(raw, str) else raw
    summary = None
    s = data.get("summary")
    if s:
        summary = RunningSummary(
            text=s["text"],
            turns_folded=s["turns_folded"],
            token_count=s["token_count"],
            recursion_depth=s["recursion_depth"],
        )
    recent = [
        Turn(role=Role(t["role"]), text=t["text"], metadata=t.get("metadata") or {})
        for t in data.get("recent_turns", [])
    ]
    compacted = CompactedContext(summary=summary, recent_turns=recent, stats=CompactionStats())
    return compacted, int(data.get("covered_count", 0)), int(data.get("fold_count", 0))
