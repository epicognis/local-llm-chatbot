import asyncio

from llm.backends.base import LLMBackend

_MOCK_REPLY = (
    "This is a deterministic mock response from MockBackend — "
    "no GPU or network required. "
    "The orchestration layer, context budget, and SSE stream are all exercised."
)


def _last_user(messages: list[dict]) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


class MockBackend(LLMBackend):
    """Deterministic backend for tests. Token delay is configurable.

    Every call is recorded in `self.calls` (model, messages, opts) so tests can
    assert on what the orchestrator sent, e.g. num_ctx or the think flag.
    """

    def __init__(self, token_delay: float = 0.01):
        self.token_delay = token_delay
        self.calls: list[dict] = []
        self.summarize_calls: list[str] = []

    def _record(self, messages, model, opts) -> dict | None:
        opts = dict(opts)
        stats = opts.pop("stats", None)
        self.calls.append({"model": model, "messages": messages, "opts": opts})
        return stats

    def _chat_stream(self, messages: list[dict], model: str, **opts):
        stats = self._record(messages, model, opts)
        return self._generate(messages, stats)

    async def _generate(self, messages: list[dict], stats: dict | None):
        reply = f"[MOCK] You said: \"{_last_user(messages)[:60]}\". {_MOCK_REPLY}"
        words = reply.split(" ")
        for word in words:
            yield word + " "
            await asyncio.sleep(self.token_delay)
        if stats is not None:
            stats["eval_count"] = len(words)
            stats["prompt_eval_count"] = sum(len(m["content"]) for m in messages) // 4

    async def _chat_complete(self, messages: list[dict], model: str, **opts) -> str:
        self._record(messages, model, opts)
        return f"[MOCK] {_last_user(messages)[:80]}"

    async def summarize(self, text: str, model: str, **opts) -> str:
        self.summarize_calls.append(text)
        return f"[MOCK SUMMARY] {text[:120]}..."
