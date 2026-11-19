"""Context-budget allocator — pack the model's window within a token budget.

This is the core engineering surface. Each progressive level is a packing
strategy behind the same ``assemble()`` interface, so summarizer / retrieval /
pinned-facts providers slot in without touching the API or UI.

    budget = num_ctx - max_tokens - safety_margin

Anchors (system prompt + current message) are reserved first, then an optional
running summary (Level 2), then recent turns by recency. A hard guard refuses
any assembly that would exceed ``budget`` — at every level.

  * Level 0 — raw passthrough: full history; refuses (doesn't trim) on overflow.
  * Level 1 — sliding window: system + current + as many recent turns as fit.
  * Level 2 — the orchestrator compresses older turns with recursive
    summarization, then packs (summary, recent turns) here with the Level 1
    packer — so the summarizer's "recency wins" overflow can never push the
    prompt past the window.

Token counting is a chars-per-token heuristic, calibrated per model against
the backend's reported ``prompt_eval_count``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_PER_MESSAGE_OVERHEAD_TOKENS = 4  # role markers / separators added by chat templates


class ContextOverflowError(ValueError):
    """Raised when even the anchors (or a Level-0 history) cannot fit the budget."""


@dataclass
class Assembly:
    messages: list[dict]
    prompt_tokens_est: int
    budget: int
    dropped_turns: int = 0
    summary_dropped: bool = False


class ContextBudget:
    def __init__(
        self,
        *,
        safety_margin: int = 768,
        level: int = 0,
        chars_per_token: float = 4.0,
    ) -> None:
        self.safety_margin = safety_margin
        self.level = level
        self.base_chars_per_token = chars_per_token
        self._cpt: dict[str, float] = {}

    # --- token counting -----------------------------------------------------

    def chars_per_token(self, model: str | None = None) -> float:
        return self._cpt.get(model, self.base_chars_per_token) if model else self.base_chars_per_token

    def estimate_tokens(self, text: str, model: str | None = None) -> int:
        return self.estimate_tokens_from_chars(len(text), model)

    def estimate_tokens_from_chars(self, n_chars: int, model: str | None = None) -> int:
        """Token estimate from a character count alone — also used for the
        streaming thinking heartbeat, which only tracks a running length."""
        return max(1, math.ceil(n_chars / self.chars_per_token(model)))

    def message_tokens(self, message: dict, model: str | None = None) -> int:
        return self.estimate_tokens(message.get("content", ""), model) + _PER_MESSAGE_OVERHEAD_TOKENS

    def total_tokens(self, messages: list[dict], model: str | None = None) -> int:
        return sum(self.message_tokens(m, model) for m in messages)

    def calibrate(self, model: str, prompt_chars: int, prompt_eval_count: int | None) -> float:
        """Nudge this model's chars-per-token toward the backend's real count (EMA).

        Capped at the base heuristic on purpose: calibration may only make the
        hard guard *stricter* (fewer chars per token = more tokens estimated),
        never looser, so a noisy or low reading can't cause an overflow. Ollama
        reports the full prompt even on a prompt-cache hit, so the signal is sound.
        """
        cur = self.chars_per_token(model)
        if prompt_eval_count and prompt_eval_count > 0 and prompt_chars > 0:
            observed = prompt_chars / prompt_eval_count
            cur = 0.7 * cur + 0.3 * observed
            cur = min(self.base_chars_per_token, max(1.5, cur))
            self._cpt[model] = cur
        return cur

    # --- assembly -----------------------------------------------------------

    def assemble(
        self,
        *,
        system_prompt: str,
        history: list[dict],
        current_message: str,
        num_ctx: int,
        max_tokens: int,
        model: str | None = None,
        summary: str | None = None,
        sliding: bool | None = None,
    ) -> Assembly:
        """Build the messages list for one turn, honoring the budget.

        ``history`` is prior turns ({"role","content"}), oldest first, excluding the
        current message. ``summary`` (Level 2) is injected as a system message right
        after the system prompt. ``sliding`` forces the Level 1 packer (default:
        from ``level``). Raises :class:`ContextOverflowError` if it cannot fit.
        """
        budget = num_ctx - max_tokens - self.safety_margin
        if budget <= 0:
            raise ContextOverflowError(
                f"Non-positive budget: num_ctx={num_ctx} - max_tokens={max_tokens} - "
                f"safety_margin={self.safety_margin}. Raise the context size or lower Max tokens."
            )

        system_msg = {"role": "system", "content": system_prompt}
        current_msg = {"role": "user", "content": current_message}
        anchors_tokens = self.message_tokens(system_msg, model) + self.message_tokens(current_msg, model)
        if anchors_tokens > budget:
            raise ContextOverflowError(
                f"Message too long for this context window (~{anchors_tokens} tokens needed, "
                f"{budget} available). Shorten it, raise the context size, or pick a model "
                "with a larger window."
            )
        remaining = budget - anchors_tokens

        head = [system_msg]
        summary_dropped = False
        if summary:
            summary_msg = {"role": "system", "content": summary}
            cost = self.message_tokens(summary_msg, model)
            if cost <= remaining:
                head.append(summary_msg)
                remaining -= cost
            else:
                summary_dropped = True

        if sliding is None:
            sliding = self.level >= 1

        if not sliding:
            history_tokens = self.total_tokens(history, model)
            if history_tokens > remaining:
                # Hard guard: Level 0 neither summarizes nor drops, so it refuses.
                raise ContextOverflowError(
                    f"This conversation (~{anchors_tokens + history_tokens} tokens) no longer "
                    f"fits the {num_ctx}-token context window. Start a new chat, raise the "
                    "context size, or set CONTEXT_LEVEL=2 to summarize older turns."
                )
            kept = list(history)
        else:
            kept = []
            for msg in reversed(history):  # most recent first; stop at the first misfit
                cost = self.message_tokens(msg, model)
                if cost > remaining:
                    break
                kept.append(msg)
                remaining -= cost
            kept.reverse()

        messages = [*head, *kept, current_msg]
        return Assembly(
            messages=messages,
            prompt_tokens_est=self.total_tokens(messages, model),
            budget=budget,
            dropped_turns=len(history) - len(kept),
            summary_dropped=summary_dropped,
        )
