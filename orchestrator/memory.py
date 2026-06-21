from abc import ABC, abstractmethod
from llm.backends.base import LLMBackend
from orchestrator.context_budget import estimate_tokens
from session.store import SessionStore, Turn


class ContextProvider(ABC):
    """Prepares the (turns, summary) pair the allocator packs each turn.
    Each progressive level (§7 of the build spec) is just a different provider."""

    @abstractmethod
    async def prepare(
        self, session_id: str, session: dict, turns: list[Turn]
    ) -> tuple[list[dict], str]:
        ...


def _to_dicts(turns: list[Turn]) -> list[dict]:
    return [{"role": t.role, "content": t.content} for t in turns]


class RawPassthroughProvider(ContextProvider):
    """Level 0 — send full history every turn."""

    async def prepare(self, session_id, session, turns):
        return _to_dicts(turns), ""


class SlidingWindowProvider(ContextProvider):
    """Level 1 — system prompt + last N turns; drop oldest, no summary."""

    def __init__(self, window_turns: int):
        self.window_turns = window_turns

    async def prepare(self, session_id, session, turns):
        return _to_dicts(turns[-self.window_turns :]), ""


class RollingSummaryProvider(ContextProvider):
    """Level 2 — once total history crosses trigger_tokens, fold turns older
    than the recent window into a cached running summary; recent turns stay verbatim."""

    def __init__(
        self,
        backend: LLMBackend,
        store: SessionStore,
        summarizer_model: str,
        window_turns: int,
        trigger_tokens: int,
    ):
        self.backend = backend
        self.store = store
        self.summarizer_model = summarizer_model
        self.window_turns = window_turns
        self.trigger_tokens = trigger_tokens

    async def prepare(self, session_id, session, turns):
        recent = turns[-self.window_turns :]
        older = turns[: -self.window_turns] if len(turns) > self.window_turns else []

        summary = session.get("summary", "")
        summarized_count = session.get("summarized_count", 0)
        unsummarized = older[summarized_count:]

        total_tokens = sum(estimate_tokens(t.content) for t in turns)
        # trigger_tokens <= 0 means "summarize as soon as there's anything to fold
        # in" rather than "never" — avoids silently dropping older turns.
        if unsummarized and total_tokens >= self.trigger_tokens:
            transcript = "\n".join(f"{t.role}: {t.content}" for t in unsummarized)
            if summary:
                transcript = f"[Previous summary]\n{summary}\n\n[New turns]\n{transcript}"
            summary = await self.backend.summarize(transcript, self.summarizer_model)
            summarized_count = len(older)
            await self.store.update_summary(session_id, summary, summarized_count)

        return _to_dicts(recent), summary
