from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMBackend(ABC):
    def chat(self, messages: list[dict], model: str, stream: bool = True, **opts):
        """Returns AsyncIterator[str] when stream=True, else a Coroutine returning str.

        messages = [{"role": "system"|"user"|"assistant", "content": str}, ...]
        """
        if stream:
            return self._chat_stream(messages, model, **opts)
        return self._chat_complete(messages, model, **opts)

    @abstractmethod
    def _chat_stream(self, messages: list[dict], model: str, **opts) -> AsyncIterator[str]:
        """Returns an async generator that yields token-delta strings."""

    @abstractmethod
    async def _chat_complete(self, messages: list[dict], model: str, **opts) -> str:
        """Returns the full completion as a single string."""

    @abstractmethod
    async def summarize(self, text: str, model: str) -> str:
        """Condense older conversation turns into a compact summary."""
