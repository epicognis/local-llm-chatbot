import asyncio
from llm.backends.base import LLMBackend

_MOCK_REPLY = (
    "This is a deterministic mock response from MockBackend — "
    "no GPU or network required. "
    "The orchestration layer, context budget, and SSE stream are all exercised."
)


class MockBackend(LLMBackend):
    """Deterministic backend for tests. Token delay is configurable."""

    def __init__(self, token_delay: float = 0.01):
        self.token_delay = token_delay

    def _chat_stream(self, messages: list[dict], model: str, **opts):
        return self._generate(messages)

    async def _generate(self, messages: list[dict]):
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        reply = f"[MOCK] You said: \"{last_user[:60]}\". {_MOCK_REPLY}"
        for word in reply.split(" "):
            yield word + " "
            await asyncio.sleep(self.token_delay)

    async def _chat_complete(self, messages: list[dict], model: str, **opts) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return f"[MOCK] {last_user[:80]}"

    async def summarize(self, text: str, model: str) -> str:
        return f"[MOCK SUMMARY] {text[:120]}..."
