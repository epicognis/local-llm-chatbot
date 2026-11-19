"""LLM backend shim — the abstraction every serving runtime plugs into.

Kept deliberately small so ``llama.cpp``-direct or ``vLLM`` can drop in later as a
backend change only. ``chat()`` is async-streaming because the chat UX is SSE and
must support cancellation (client disconnect -> abort upstream).

``chat()`` yields :class:`ChatEvent` objects rather than bare strings so a single
stream can carry content deltas, a reasoning heartbeat, *and* end-of-turn
counters (``prompt_eval_count`` for token-count calibration, ``eval_duration``
for an accurate decode rate). Only ``event.delta`` is ever shown as the answer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


class BackendError(RuntimeError):
    """A user-presentable failure from the model runtime (not reachable, model
    not pulled, empty reply, ...). Callers surface it as an error event rather
    than as assistant text, so it never lands in conversation history."""


@dataclass
class ChatEvent:
    """One streamed unit from a backend.

    delta              : content text to stream live to the UI (may be "").
    thinking_delta     : reasoning text produced this chunk. NEVER shown as content;
                          surfaced only as a liveness heartbeat (token count) so a
                          model that is "thinking" doesn't read as a hung turn.
    done               : final event of the turn (carries the counters below).
    prompt_eval_count  : tokens the backend counted for the prompt (calibration).
    eval_count         : tokens generated.
    eval_duration_s    : pure decode time for eval_count (exact tok/s).
    load_duration_s    : time spent loading the model for this turn.
    done_reason        : "stop" | "length" | ...
    from_thinking      : True if ``delta`` is a last-resort fallback recovered from
                          the model's buffered reasoning (no content ever arrived).
    """

    delta: str = ""
    thinking_delta: str = ""
    done: bool = False
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_duration_s: float | None = None
    load_duration_s: float | None = None
    done_reason: str | None = None
    from_thinking: bool = False


class LLMBackend(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        model: str,
        *,
        num_ctx: int | None = None,
        num_gpu: int | None = None,
        think: bool | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        keep_alive: int | str | None = None,
        **opts,
    ) -> AsyncIterator[ChatEvent]:
        """Stream a chat completion.

        messages = [{"role": "system"|"user"|"assistant", "content": str}, ...]
        Yields :class:`ChatEvent`; the concatenation of all ``delta`` values is the
        full reply. ``num_ctx`` MUST be passed through to Ollama per request.
        ``think`` is the per-model value from the registry (None = omit).
        Raises :class:`BackendError` for user-presentable failures.
        """
        raise NotImplementedError

    async def complete(self, prompt: str, model: str, **opts) -> str:
        """One-shot, non-streamed completion of a single user prompt (used by the
        Level 2 summarizer). Default: drain :meth:`chat`."""
        parts: list[str] = []
        async for ev in self.chat([{"role": "user", "content": prompt}], model, **opts):
            if ev.delta:
                parts.append(ev.delta)
        return "".join(parts).strip()

    async def warm_state(self, model: str, num_ctx: int | None = None) -> str:
        """"warm" (ready), "cold" (will load first), "reload" (resident with a
        different context size, will reload) or "missing" (not installed).
        Backends without this concept report "warm"."""
        return "warm"

    async def resident_models(self) -> list[str]:
        """Tags of the models currently resident. Default: none."""
        return []

    async def load(
        self,
        model: str,
        *,
        num_ctx: int | None = None,
        num_gpu: int | None = None,
        keep_alive: int | str | None = None,
        purge_others: bool = False,
    ) -> dict:
        """Eagerly load ``model`` so it's resident before the first turn,
        optionally unloading any other resident models first.

        ``num_ctx`` / ``num_gpu`` must match what generation will use, so the eager
        load and the first turn share one resident runner. ``num_gpu=0`` forces
        full CPU execution. Default is a no-op (mock has nothing to load).
        """
        return {"status": "noop"}

    async def unload_others(self, keep_model: str) -> list[str]:
        """Unload any resident models other than ``keep_model`` to enforce a
        single active model. Default no-op; returns the tags it unloaded."""
        return []

    async def unload_all(self) -> dict:
        """Unload every resident model to free all GPU/VRAM (and RAM) to zero.

        Returns ``{before, unloaded, still_resident}`` (lists of tags) so the caller
        can confirm VRAM was actually freed rather than assuming the request took.
        """
        return {"before": [], "unloaded": [], "still_resident": []}

    async def aclose(self) -> None:
        """Release pooled resources on shutdown."""
