from abc import ABC, abstractmethod
from typing import AsyncIterator


class BackendError(RuntimeError):
    """A user-presentable failure from the model runtime (not reachable, model
    not pulled, empty reply, ...). Callers surface it as an error event rather
    than as assistant text, so it never lands in conversation history."""


class StreamStatus(str):
    """Yielded by a chat stream to report progress that is NOT answer text
    (e.g. "Thinking…" while a reasoning model works). Consumers must not append
    it to the reply."""


class LLMBackend(ABC):
    """Backend-agnostic shim (build spec §6).

    Reserved keyword options understood by every backend:
      think      -- bool or effort level ("low"...); omitted for models without reasoning
      keep_alive -- runtime hint for how long to keep the model resident
      stats      -- a dict the backend fills with end-of-turn counters
                    (prompt_eval_count, eval_count, eval_duration_s, ...)
    Everything else is forwarded as sampling options (temperature, num_ctx, ...).
    """

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

    async def summarize(self, text: str, model: str, **opts) -> str:
        """Condense older conversation turns into a compact summary."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize the following conversation compactly, "
                    "preserving key facts, decisions, names, and open questions. "
                    "Write the summary only — no preamble."
                ),
            },
            {"role": "user", "content": text},
        ]
        return await self._chat_complete(messages, model, **opts)

    async def warm_state(self, model: str, num_ctx: int | None = None) -> str:
        """"warm" (ready), "cold" (will load first), "reload" (resident with a
        different context size, will reload) or "missing" (not installed).
        Backends without this concept report "warm"."""
        return "warm"

    async def aclose(self) -> None:
        """Release pooled resources on shutdown."""
