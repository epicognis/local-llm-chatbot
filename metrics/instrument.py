import time
import structlog

log = structlog.get_logger()


class TurnMetrics:
    def __init__(self, session_id: str, model: str):
        self.session_id = session_id
        self.model = model
        self._start = time.monotonic()
        self.completion_tokens = 0

    def count_token(self):
        self.completion_tokens += 1

    def finish(self, prompt_tokens: int = 0):
        elapsed = time.monotonic() - self._start
        tps = self.completion_tokens / elapsed if elapsed > 0 else 0
        log.info(
            "turn_complete",
            session_id=self.session_id,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=self.completion_tokens,
            elapsed_s=round(elapsed, 2),
            tokens_per_sec=round(tps, 1),
        )
