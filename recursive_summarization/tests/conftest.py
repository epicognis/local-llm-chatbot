"""Mock summarizer and test fixtures.

The mock summarizer deterministically shrinks its input, making
recursion observable and tighten convergence testable.  No live model.
"""

from __future__ import annotations

import re

from recursive_summarization import (
    HeuristicCounter,
    RecursiveSummarizer,
    Role,
    SummarizerConfig,
    Turn,
)


class MockSummarizeFn:
    """Deterministic mock: returns the first ~40% of non-prompt text.

    Tracks call count for instrumentation assertions.
    """

    def __init__(self, compression_ratio: float = 0.4) -> None:
        self.call_count = 0
        self.ratio = compression_ratio

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        # Extract the text between the delimiters
        text = self._extract_content(prompt)
        if not text:
            return ""
        target_len = max(10, int(len(text) * self.ratio))
        # Truncate at a word boundary
        truncated = text[:target_len]
        last_space = truncated.rfind(" ")
        if last_space > target_len // 2:
            truncated = truncated[:last_space]
        return truncated.strip()

    @staticmethod
    def _extract_content(prompt: str) -> str:
        """Pull text from between --- delimiters in the prompt templates."""
        # Match content between any --- LABEL --- and --- END ---
        match = re.search(
            r"---\s*(?:CONVERSATION|NOTES|EXISTING SUMMARY|NEW EXCHANGES)"
            r"\s*---\s*\n(.*?)\n\s*---\s*END\s*---",
            prompt,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        # Fallback: try to find content between any --- markers
        parts = prompt.split("---")
        if len(parts) >= 3:
            return parts[-2].strip()
        return prompt


class MockExpandingSummarizeFn:
    """Pathological mock: returns text LONGER than input.

    Used to test the no-progress guard.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        text = MockSummarizeFn._extract_content(prompt)
        return text + " [expanded padding to trigger no-progress guard]"


class MockEmptySummarizeFn:
    """Mock that returns empty strings — tests empty-output handling."""

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        return ""


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def make_turns(n: int, text_len: int = 100) -> list[Turn]:
    """Generate n alternating user/assistant turns with known text length."""
    turns = []
    for i in range(n):
        role = Role.USER if i % 2 == 0 else Role.ASSISTANT
        text = f"Turn {i}: " + "x" * (text_len - len(f"Turn {i}: "))
        turns.append(Turn(role=role, text=text))
    return turns


def make_config(**overrides) -> SummarizerConfig:
    """SummarizerConfig with small defaults for fast tests."""
    defaults = dict(
        keep_recent_turns=2,
        target_budget_tokens=200,
        summarize_input_budget=300,
        max_recursion_depth=3,
        max_tighten_passes=2,
        min_summary_tokens=20,
        safety_margin=0.90,
        chars_per_token=4.0,
    )
    defaults.update(overrides)
    return SummarizerConfig(**defaults)


def make_summarizer(
    mock_fn=None, config=None, **config_overrides
) -> tuple[RecursiveSummarizer, MockSummarizeFn | object, SummarizerConfig]:
    """Build a RecursiveSummarizer with mock backend for testing."""
    cfg = config or make_config(**config_overrides)
    fn = mock_fn or MockSummarizeFn()
    summarizer = RecursiveSummarizer(
        summarize_fn=fn,
        config=cfg,
        count_tokens=HeuristicCounter(cfg.chars_per_token),
    )
    return summarizer, fn, cfg
