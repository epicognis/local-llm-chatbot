"""Core data types for the recursive summarization library.

Vocabulary:
- Turn: one unit of conversation (role + text + metadata bag)
- RunningSummary: compressed representation of the older tail
- CompactedContext: the assembled result (summary + verbatim recency window)
- CompactionStats: per-operation instrumentation for metrics and tests
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True)
class Turn:
    """One unit of conversation.

    The metadata bag is load-bearing for Level 3 retrieval — it holds the raw,
    unsummarized fields (e.g. tool calls, attachments) so that retrieval
    can reach the original even after the rendered text has been folded into a
    summary.
    """

    role: Role
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """Render to the text that enters summarization and token counting."""
        return f"{self.role.value}: {self.text}"


@dataclass
class RunningSummary:
    """Compressed representation of the older tail.

    Not a turn — rendered differently (as system-level context) and treated
    as a pinned prefix on the incremental fold path.
    """

    text: str
    turns_folded: int
    token_count: int
    recursion_depth: int


@dataclass
class CompactionStats:
    """Per-operation instrumentation exposed for metrics and tests."""

    summarize_calls: int = 0
    max_recursion_depth: int = 0
    tighten_passes: int = 0
    hard_truncations: int = 0
    passthrough: bool = False


@dataclass
class CompactedContext:
    """The assembled result of compaction or fold.

    summary may be None when the history fits the budget (passthrough).
    recent_turns is the verbatim recency window, in order.
    """

    summary: RunningSummary | None
    recent_turns: list[Turn]
    stats: CompactionStats

    def render_messages(self) -> list[dict[str, str]]:
        """Emit as a list of role/content entries (chat-messages shape).

        The running summary is placed as a leading system message — reads to
        the model as established context, not as something a participant said.
        """
        messages: list[dict[str, str]] = []
        if self.summary and self.summary.text:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"[Conversation summary — {self.summary.turns_folded} "
                        f"earlier turns compressed]\n{self.summary.text}"
                    ),
                }
            )
        for turn in self.recent_turns:
            messages.append({"role": turn.role.value, "content": turn.text})
        return messages

    def render_prompt_block(self) -> str:
        """Emit as a single string (single prompt-block shape).

        For prompts that take history as one text block.
        """
        parts: list[str] = []
        if self.summary and self.summary.text:
            parts.append(
                f"[Summary of {self.summary.turns_folded} earlier turns]\n"
                f"{self.summary.text}"
            )
        for turn in self.recent_turns:
            parts.append(turn.render())
        return "\n\n".join(parts)

    @property
    def total_tokens(self) -> int:
        """Approximate total tokens in the compacted context."""
        total = 0
        if self.summary:
            total += self.summary.token_count
        # Caller is responsible for counting recent turns via the same
        # token counter used during compaction; this is a convenience
        # that uses the summary's stored count only.
        return total
