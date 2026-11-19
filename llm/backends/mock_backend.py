"""Mock backend — deterministic, no GPU, no network.

Lets the full orchestration stack (budget packing, the hard window guard, session
persistence, Level 2 summarization, SSE streaming, cancellation) be tested and
demoed without Ollama. That orchestration logic is where the real bugs live, so
it must be exercisable offline.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator

from .base import ChatEvent, LLMBackend

_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / _CHARS_PER_TOKEN))


def mock_summarize(prompt: str) -> str:
    """Deterministic stand-in for a summarize call: returns ~40% of the
    conversation text carried in a recursive_summarization prompt, so Level 2's
    compaction, recursion and tighten paths are all exercisable offline."""
    body = prompt
    for label in ("--- CONVERSATION ---", "--- NOTES ---", "--- NEW EXCHANGES ---"):
        if label in prompt:
            body = prompt.split(label, 1)[1]
            break
    body = body.split("--- END ---", 1)[0].strip()
    if not body:
        return ""
    target = max(10, int(len(body) * 0.4))
    out = body[:target]
    cut = out.rfind(" ")
    if cut > target // 2:
        out = out[:cut]
    return out.strip()


class MockBackend(LLMBackend):
    """Echoes a canned, deterministic reply derived from the last user message.

    Every chat() call is recorded in ``self.calls`` (model, messages, kwargs) so
    tests can assert on what the orchestrator sent, e.g. num_ctx or think.
    """

    def __init__(self, token_delay: float = 0.0) -> None:
        self.token_delay = token_delay
        self.calls: list[dict] = []
        self.complete_calls: list[str] = []

    async def chat(
        self,
        messages: list[dict],
        model: str,
        *,
        think: bool | str | None = None,
        **kwargs,
    ) -> AsyncIterator[ChatEvent]:
        self.calls.append({"model": model, "messages": messages, "think": think, **kwargs})
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )

        # With reasoning on, emit a few thinking-only chunks first (no content)
        # so the thinking heartbeat / liveness UI is exercisable offline.
        if think is True:
            for frag in ("Let me ", "think about ", "that. "):
                yield ChatEvent(thinking_delta=frag)

        reply = (
            f"[mock:{model}] I received {len(messages)} message(s). "
            f"You said: {last_user.strip()}"
        )
        words = reply.split(" ")
        for i, word in enumerate(words):
            yield ChatEvent(delta=word if i == 0 else " " + word)
            if self.token_delay:
                await asyncio.sleep(self.token_delay)

        yield ChatEvent(
            done=True,
            # A plausible prompt_eval_count so calibration has something to chew on.
            prompt_eval_count=_estimate_tokens(" ".join(m.get("content", "") for m in messages)),
            eval_count=len(words),
            done_reason="stop",
        )

    async def complete(self, prompt: str, model: str, **opts) -> str:
        self.complete_calls.append(prompt)
        return mock_summarize(prompt)
