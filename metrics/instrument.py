import time

import structlog

log = structlog.get_logger()


class TurnMetrics:
    """Per-turn latency / throughput. Emits one `turn_complete` log line, which
    the standalone monitor (monitor/llmmon/turns.py) tails for tok/s — keep the
    field names stable."""

    def __init__(self, session_id: str, model: str):
        self.session_id = session_id
        self.model = model
        self._start = time.monotonic()
        self._first_token_at: float | None = None
        self.chunks = 0

    def count_token(self):
        if self._first_token_at is None:
            self._first_token_at = time.monotonic()
        self.chunks += 1

    def finish(self, stats: dict | None = None, **extra):
        stats = stats or {}
        elapsed = time.monotonic() - self._start
        # Prefer the runtime's real counters; fall back to counting stream chunks
        # (≈1 token per chunk for Ollama) when they're absent, e.g. on cancel.
        completion_tokens = stats.get("eval_count") or self.chunks
        eval_s = stats.get("eval_duration_s")
        if eval_s:
            tps = completion_tokens / eval_s  # pure decode rate
        elif self._first_token_at is not None:
            decode = time.monotonic() - self._first_token_at
            tps = completion_tokens / decode if decode > 0 else 0.0
        else:
            tps = 0.0
        ttft = (self._first_token_at - self._start) if self._first_token_at else None
        log.info(
            "turn_complete",
            session_id=self.session_id,
            model=self.model,
            prompt_tokens=stats.get("prompt_eval_count", 0),
            completion_tokens=completion_tokens,
            elapsed_s=round(elapsed, 2),
            ttft_s=round(ttft, 2) if ttft is not None else None,
            tokens_per_sec=round(tps, 1),
            load_s=round(stats["load_duration_s"], 2) if stats.get("load_duration_s") else None,
            **extra,
        )
