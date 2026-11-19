"""Configuration surface for the recursive summarization library.

A plain dataclass so the library stays stdlib-only; the host app maps its
pydantic-settings values onto it. Every knob has a documented default.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SummarizerConfig:
    """All tuning knobs in one place.

    Attributes:
        keep_recent_turns:
            Size of the verbatim recency window.  Larger preserves more
            fidelity at the cost of budget.
        target_budget_tokens:
            Ceiling for the whole compacted history block.  Set as a fraction
            of the model's real window, leaving room for system prompt +
            retrieval + reply.
        summarize_input_budget:
            Max tokens fed to one summarize call.  Per model; well below the
            real window.  Governs when recursion triggers.
        max_recursion_depth:
            Reduce fold-up ceiling.  Rarely hit in practice; a backstop.
        max_tighten_passes:
            Compression retries before hard truncate.  1–2 is usually enough.
        min_summary_tokens:
            Floor on summary size.  Prevents degenerate one-liner summaries
            when budget is tight.
        safety_margin:
            Budget de-rating factor (0 < margin < 1).  Lower when using exact
            counting; higher when relying on the heuristic.
        chars_per_token:
            Heuristic calibration.  Fit empirically from prompt_eval_count
            per model.  ~4 for English and code.
        elision_marker:
            Appended when hard truncation is required after tighten exhaustion.
    """

    keep_recent_turns: int = 4
    target_budget_tokens: int = 4096
    summarize_input_budget: int = 6000
    max_recursion_depth: int = 4
    max_tighten_passes: int = 2
    min_summary_tokens: int = 80
    safety_margin: float = 0.90
    chars_per_token: float = 4.0
    elision_marker: str = " [... earlier context truncated]"

    @property
    def effective_budget(self) -> int:
        """Budget × safety_margin — all compaction targets this, not raw."""
        return int(self.target_budget_tokens * self.safety_margin)
