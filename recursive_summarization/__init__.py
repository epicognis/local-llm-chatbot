"""Recursive summarization library — context compaction for multi-turn chat.

Level 2 mechanism in the context-budget allocator.  Keeps recent turns
verbatim; recursively summarizes the older tail to fit a token budget.

Public API:
    RecursiveSummarizer  — the engine (compact, fold)
    SummarizerConfig     — all tuning knobs
    Turn, RunningSummary, CompactedContext, CompactionStats — data types
    HeuristicCounter     — default token counter
    BackendAdapter       — thin adapter for existing backends
    from_chat_history    — chat-history adapter
"""

from .adapters import from_chat_history
from .backend import BackendAdapter, SummarizeFn
from .config import SummarizerConfig
from .engine import RecursiveSummarizer
from .token_counting import HeuristicCounter, TokenCounter
from .types import (
    CompactedContext,
    CompactionStats,
    Role,
    RunningSummary,
    Turn,
)

__all__ = [
    "BackendAdapter",
    "CompactedContext",
    "CompactionStats",
    "HeuristicCounter",
    "RecursiveSummarizer",
    "Role",
    "RunningSummary",
    "SummarizeFn",
    "SummarizerConfig",
    "TokenCounter",
    "Turn",
    "from_chat_history",
]
