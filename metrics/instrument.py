"""Per-turn metrics — tokens, latency, time-to-first-token, tokens/sec.

Emits one ``turn_complete`` structlog line per turn. The standalone monitor
(monitor/llmmon/turns.py) tails it for its tok/s panel and reads ``model``,
``completion_tokens``, ``elapsed_s`` and ``tokens_per_sec`` — keep those stable.
"""

from __future__ import annotations

import time

import structlog

from llm.backends.base import ChatEvent

log = structlog.get_logger("metrics")


class TurnMetrics:
    def __init__(self, session_id: str, model: str, prompt_tokens_est: int) -> None:
        self.session_id = session_id
        self.model = model
        self.prompt_tokens_est = prompt_tokens_est
        self._start = time.monotonic()
        self._first: float | None = None

    def first_token(self) -> None:
        if self._first is None:
            self._first = time.monotonic()

    @property
    def ttft_s(self) -> float | None:
        return None if self._first is None else self._first - self._start

    def _tok_s(self, final: ChatEvent | None) -> float | None:
        """Pure decode rate: the runtime's eval_count / eval_duration when known,
        else generated tokens over the time since the first token."""
        if final and final.eval_count and final.eval_duration_s:
            return final.eval_count / final.eval_duration_s
        if final and final.eval_count and self._first is not None:
            decode = time.monotonic() - self._first
            return final.eval_count / decode if decode > 0 else None
        return None

    def stats(self, final: ChatEvent | None, thinking_tokens: int) -> dict:
        """The terminal ``stats`` event shown under the reply in the UI."""
        tps = self._tok_s(final)
        return {
            "ttft_s": round(self.ttft_s, 2) if self.ttft_s is not None else None,
            "eval_count": final.eval_count if final else None,
            "tok_s": round(tps, 1) if tps else None,
            "thinking_tokens": thinking_tokens,
            "elapsed_s": round(time.monotonic() - self._start, 2),
            "load_s": round(final.load_duration_s, 2) if final and final.load_duration_s else None,
        }

    def emit(self, final: ChatEvent | None, *, outcome: str, thinking_tokens: int = 0) -> None:
        tps = self._tok_s(final)
        log.info(
            "turn_complete",
            session_id=self.session_id,
            model=self.model,
            outcome=outcome,
            prompt_tokens_est=self.prompt_tokens_est,
            prompt_tokens=final.prompt_eval_count if final else None,
            completion_tokens=(final.eval_count if final else None) or 0,
            thinking_tokens=thinking_tokens,
            elapsed_s=round(time.monotonic() - self._start, 2),
            ttft_s=round(self.ttft_s, 2) if self.ttft_s is not None else None,
            tokens_per_sec=round(tps, 1) if tps else 0.0,
            load_s=round(final.load_duration_s, 2) if final and final.load_duration_s else None,
        )
