"""Recursive summarization engine.

Implements:
  - Batch compaction (compact)
  - Recursive reduce (_reduce)
  - Tighten pass (_tighten)
  - Incremental fold (fold)

The engine is stateless between operations.  Per-call instrumentation
(summarize-call count, recursion depth) is returned in CompactionStats.
"""

from __future__ import annotations

import logging
from typing import Sequence

from .backend import SummarizeFn
from .config import SummarizerConfig
from .prompts import (
    SUMMARIZE_SYSTEM,
    build_refine_prompt,
    build_summarize_prompt,
    build_tighten_prompt,
)
from .token_counting import HeuristicCounter, TokenCounter
from .types import (
    CompactedContext,
    CompactionStats,
    RunningSummary,
    Turn,
)

logger = logging.getLogger(__name__)


class RecursiveSummarizer:
    """Recursive summarization engine.

    Thread-safe for concurrent reads of config; not designed for concurrent
    mutation of the same instance.

    Parameters
    ----------
    summarize_fn:
        The backend abstraction — takes a prompt string, returns model text.
    config:
        All tuning knobs.  See SummarizerConfig for defaults.
    count_tokens:
        Pluggable token counter.  Defaults to HeuristicCounter.
    """

    def __init__(
        self,
        summarize_fn: SummarizeFn,
        config: SummarizerConfig | None = None,
        count_tokens: TokenCounter | None = None,
    ) -> None:
        self._summarize_fn = summarize_fn
        self._cfg = config or SummarizerConfig()
        self._count = count_tokens or HeuristicCounter(self._cfg.chars_per_token)

    # ------------------------------------------------------------------
    # Batch compaction
    # ------------------------------------------------------------------

    def compact(self, turns: Sequence[Turn]) -> CompactedContext:
        """Compact a full history into a budget-fitting context.

        Four steps:
        1. Token-count the full history.  Passthrough if under budget.
        2. Partition into recency window + older tail.
        3. Compute summary target = effective_budget − recency tokens,
           floored at min_summary_tokens.
        4. Reduce the older tail, tighten to target, assemble result.

        Recency wins: if recency alone exceeds budget, the summary
        is floored and the total may exceed the nominal budget.
        """
        stats = CompactionStats()

        if not turns:
            stats.passthrough = True
            return CompactedContext(summary=None, recent_turns=[], stats=stats)

        # Step 1: total token count
        rendered = [t.render() for t in turns]
        total_tokens = sum(self._count(r) for r in rendered)

        if total_tokens <= self._cfg.effective_budget:
            stats.passthrough = True
            return CompactedContext(
                summary=None, recent_turns=list(turns), stats=stats
            )

        # Step 2: partition
        keep = min(self._cfg.keep_recent_turns, len(turns))
        recent_turns = list(turns[-keep:]) if keep > 0 else []
        older_turns = list(turns[: len(turns) - keep]) if keep < len(turns) else []

        if not older_turns:
            # All turns are in the recency window — nothing to summarize.
            stats.passthrough = True
            return CompactedContext(
                summary=None, recent_turns=recent_turns, stats=stats
            )

        # Step 3: compute summary target
        recency_tokens = sum(self._count(t.render()) for t in recent_turns)
        summary_target = max(
            self._cfg.min_summary_tokens,
            self._cfg.effective_budget - recency_tokens,
        )

        if recency_tokens > self._cfg.effective_budget:
            logger.warning(
                "Recency window (%d tokens) exceeds effective budget (%d). "
                "Recency wins; summary floored at min_summary_tokens (%d).",
                recency_tokens,
                self._cfg.effective_budget,
                self._cfg.min_summary_tokens,
            )

        # Step 4: reduce + tighten
        older_blocks = [t.render() for t in older_turns]
        raw_summary = self._reduce(older_blocks, depth=0, stats=stats)
        final_summary = self._tighten(raw_summary, summary_target, stats)

        summary = RunningSummary(
            text=final_summary,
            turns_folded=len(older_turns),
            token_count=self._count(final_summary),
            recursion_depth=stats.max_recursion_depth,
        )

        return CompactedContext(
            summary=summary, recent_turns=recent_turns, stats=stats
        )

    # ------------------------------------------------------------------
    # Incremental fold
    # ------------------------------------------------------------------

    def fold(
        self,
        prior: CompactedContext,
        new_turns: Sequence[Turn],
    ) -> CompactedContext:
        """Extend an existing compaction with new turns.

        Pins the existing running summary as a fixed prefix.  On the common
        path — the working set fits alongside the pinned summary — appends
        and returns with no model work.  Only when overflow occurs does it
        fold the oldest working turns into the running summary.
        """
        stats = CompactionStats()

        if not new_turns:
            return CompactedContext(
                summary=prior.summary,
                recent_turns=list(prior.recent_turns),
                stats=stats,
            )

        # Working set = existing recency + new turns
        working = list(prior.recent_turns) + list(new_turns)
        summary_tokens = prior.summary.token_count if prior.summary else 0
        working_tokens = sum(self._count(t.render()) for t in working)

        if summary_tokens + working_tokens <= self._cfg.effective_budget:
            # Common path: no model work needed — append and keep the whole
            # working set verbatim. (Trimming it to keep_recent_turns here would
            # silently drop the trimmed turns: they'd be in neither the recency
            # window nor the summary.) The window shrinks back to
            # keep_recent_turns on the next overflow, when the excess is folded.
            stats.passthrough = True
            return CompactedContext(
                summary=prior.summary, recent_turns=working, stats=stats
            )

        # Overflow: fold the oldest working turns into the summary.
        keep = min(self._cfg.keep_recent_turns, len(working))
        new_recent = working[-keep:] if keep > 0 else []
        to_fold = working[: len(working) - keep]

        if not to_fold:
            # Nothing to fold — recency overflow.
            return CompactedContext(
                summary=prior.summary, recent_turns=new_recent, stats=stats
            )

        fold_text = "\n\n".join(t.render() for t in to_fold)
        existing_summary_text = prior.summary.text if prior.summary else ""
        prior_folded = prior.summary.turns_folded if prior.summary else 0

        if existing_summary_text:
            prompt = build_refine_prompt(existing_summary_text, fold_text)
        else:
            prompt = build_summarize_prompt(fold_text)

        raw = self._call_summarize(prompt, stats)

        # Compute target for the updated summary
        recency_tokens = sum(self._count(t.render()) for t in new_recent)
        summary_target = max(
            self._cfg.min_summary_tokens,
            self._cfg.effective_budget - recency_tokens,
        )
        final = self._tighten(raw, summary_target, stats)

        summary = RunningSummary(
            text=final,
            turns_folded=prior_folded + len(to_fold),
            token_count=self._count(final),
            recursion_depth=stats.max_recursion_depth,
        )

        return CompactedContext(
            summary=summary, recent_turns=new_recent, stats=stats
        )

    # ------------------------------------------------------------------
    # Recursive reduce
    # ------------------------------------------------------------------

    def _reduce(
        self, blocks: list[str], depth: int, stats: CompactionStats
    ) -> str:
        """Hierarchical map-reduce summarization.

        Base case: all blocks fit the input window → one summarize call.
        Map: pack blocks into input-sized chunks, summarize each.
        Reduce: recurse on the chunk summaries.
        Guards: depth ceiling and no-progress detection.
        """
        stats.max_recursion_depth = max(stats.max_recursion_depth, depth)

        joined = "\n\n".join(blocks)
        if self._count(joined) <= self._cfg.summarize_input_budget:
            # Base case
            return self._call_summarize(build_summarize_prompt(joined), stats)

        # Map: pack into chunks
        chunks = self._pack_chunks(blocks)
        summaries = [
            self._call_summarize(build_summarize_prompt(chunk), stats)
            for chunk in chunks
        ]
        summaries = [s for s in summaries if s]  # strip empty

        # Guards
        if depth >= self._cfg.max_recursion_depth:
            logger.warning(
                "Max recursion depth %d reached; final best-effort summarize.",
                self._cfg.max_recursion_depth,
            )
            joined_summaries = "\n\n".join(summaries)
            return self._call_summarize(
                build_summarize_prompt(joined_summaries), stats
            )

        if len(summaries) >= len(blocks):
            # No progress — summaries didn't shrink the block count.
            logger.warning(
                "No progress at depth %d (%d blocks → %d summaries); "
                "terminating with best-effort summarize.",
                depth,
                len(blocks),
                len(summaries),
            )
            joined_summaries = "\n\n".join(summaries)
            return self._call_summarize(
                build_summarize_prompt(joined_summaries), stats
            )

        # Reduce: recurse
        return self._reduce(summaries, depth + 1, stats)

    # ------------------------------------------------------------------
    # Tighten pass
    # ------------------------------------------------------------------

    def _tighten(
        self, text: str, target_tokens: int, stats: CompactionStats
    ) -> str:
        """Compress summary text to fit the output budget.

        Repeats up to max_tighten_passes.  If passes exhaust without
        convergence, hard-truncates with elision marker.
        """
        current = text
        for _ in range(self._cfg.max_tighten_passes):
            if self._count(current) <= target_tokens:
                return current
            prompt = build_tighten_prompt(current, target_tokens)
            current = self._call_summarize(prompt, stats)
            stats.tighten_passes += 1
            if not current:
                return ""

        # Still over budget — hard truncate
        if self._count(current) > target_tokens:
            current = self._hard_truncate(current, target_tokens)
            stats.hard_truncations += 1

        return current

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_summarize(self, prompt: str, stats: CompactionStats) -> str:
        """Single summarize call with instrumentation."""
        stats.summarize_calls += 1
        result = self._summarize_fn(prompt)
        # Empty/null output → empty string
        return result if result else ""

    def _pack_chunks(self, blocks: list[str]) -> list[str]:
        """Greedy-pack blocks into input-window-sized chunks.

        Handles the oversized-block problem: any block larger than
        the input window is hard-split into window-sized pieces first,
        guaranteeing the base case is always reachable.
        """
        budget = self._cfg.summarize_input_budget
        # Phase 1: split oversized blocks
        split_blocks: list[str] = []
        for block in blocks:
            if self._count(block) <= budget:
                split_blocks.append(block)
            else:
                split_blocks.extend(self._hard_split(block, budget))

        # Phase 2: greedy pack
        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for block in split_blocks:
            block_tokens = self._count(block)
            if current_parts and current_tokens + block_tokens > budget:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0
            current_parts.append(block)
            current_tokens += block_tokens

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks

    def _hard_split(self, text: str, max_tokens: int) -> list[str]:
        """Split a single oversized block into pieces ≤ max_tokens.

        Uses character-level splitting based on the heuristic ratio.
        """
        # Approximate chars per chunk
        chars_per_chunk = int(max_tokens * self._cfg.chars_per_token)
        if chars_per_chunk <= 0:
            chars_per_chunk = len(text)

        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + chars_per_chunk, len(text))
            # Try to break at a newline or space boundary
            if end < len(text):
                # Look backwards for a clean break point
                for sep in ("\n\n", "\n", ". ", " "):
                    break_at = text.rfind(sep, start, end)
                    if break_at > start:
                        end = break_at + len(sep)
                        break
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            start = end

        return pieces if pieces else [text]

    def _hard_truncate(self, text: str, target_tokens: int) -> str:
        """Hard-truncate text to target_tokens with elision marker.

        Guarantees the summary respects target as a hard ceiling.
        """
        marker = self._cfg.elision_marker
        marker_tokens = self._count(marker)
        usable = max(1, target_tokens - marker_tokens)
        usable_chars = int(usable * self._cfg.chars_per_token)

        if usable_chars >= len(text):
            return text

        # Truncate at a sentence or line boundary if possible
        truncated = text[:usable_chars]
        for sep in ("\n", ". ", " "):
            break_at = truncated.rfind(sep)
            if break_at > usable_chars // 2:
                truncated = truncated[: break_at + len(sep)]
                break

        return truncated.rstrip() + marker
