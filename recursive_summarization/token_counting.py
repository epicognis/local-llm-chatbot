"""Token counting abstraction.

Pluggable counter: the library depends on a callable that maps text → int.
The default is a heuristic (chars / chars_per_token).  Moving up the accuracy
ladder (calibrated heuristic, model-native tokenizer) is a one-line swap.
"""

from __future__ import annotations

from typing import Protocol


class TokenCounter(Protocol):
    """Anything that maps text to an approximate token count."""

    def __call__(self, text: str) -> int: ...


class HeuristicCounter:
    """chars / chars_per_token, floored at 1 for non-empty text.

    Combined with the safety margin this is good enough for budgeting
    decisions, which are inherently coarse.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        self._cpt = chars_per_token

    def __call__(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / self._cpt))

    def calibrate(self, text: str, actual_tokens: int) -> None:
        """Update chars_per_token from a ground-truth observation.

        Feed prompt_eval_count from real generation responses to turn
        the heuristic into an empirically-tuned estimate.
        """
        if actual_tokens > 0 and text:
            self._cpt = len(text) / actual_tokens

    @property
    def chars_per_token(self) -> float:
        return self._cpt
