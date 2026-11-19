import math

# Chat templates wrap every message in role markers / separators. A few tokens
# per message is typical across Ollama's templates; counting it keeps a long
# history of short turns from sneaking past the guard.
MESSAGE_OVERHEAD_TOKENS = 4


def estimate_tokens(text: str) -> int:
    """Rough heuristic: ~4 chars per token. Calibrated per model via TokenCalibrator."""
    return max(1, len(text) // 4)


def estimate_messages(messages: list[dict]) -> int:
    """Uncalibrated estimate for a whole assembled prompt (what calibration compares
    against Ollama's prompt_eval_count)."""
    return sum(estimate_tokens(m["content"]) + MESSAGE_OVERHEAD_TOKENS for m in messages)


class TokenCalibrator:
    """Per-model correction factor for the chars/4 heuristic (build spec §7).

    After each turn the orchestrator feeds in (estimated, actual prompt_eval_count);
    the ratio is an EMA of actual/estimated. It is floored at 1.0 on purpose: the
    calibration may only make the hard window guard *stricter* than the heuristic,
    never looser, so a noisy or low reading can't cause an overflow.
    """

    def __init__(self, alpha: float = 0.3, floor: float = 1.0, cap: float = 4.0):
        self.alpha = alpha
        self.floor = floor
        self.cap = cap
        self._ratios: dict[str, float] = {}

    def ratio(self, model: str) -> float:
        return self._ratios.get(model, self.floor)

    def observe(self, model: str, estimated: int, actual: int | None) -> float:
        if estimated > 0 and actual:
            sample = actual / estimated
            prev = self._ratios.get(model)
            new = sample if prev is None else prev + self.alpha * (sample - prev)
            self._ratios[model] = min(self.cap, max(self.floor, new))
        return self.ratio(model)


class ContextBudgetAllocator:
    """Packs model context within a token budget. The hard guard applies at every level."""

    def __init__(self, model_ctx: int, max_tokens: int, safety_margin: int, ratio: float = 1.0):
        self.budget = model_ctx - max_tokens - safety_margin
        self.ratio = ratio
        if self.budget <= 0:
            raise ValueError(
                f"Budget is non-positive: ctx={model_ctx}, "
                f"max_tokens={max_tokens}, margin={safety_margin}"
            )

    def cost(self, text: str) -> int:
        return math.ceil(estimate_tokens(text) * self.ratio) + MESSAGE_OVERHEAD_TOKENS

    def assemble(
        self,
        system_prompt: str,
        turns: list[dict],
        current_message: str,
        summary: str = "",
    ) -> list[dict]:
        """
        Build the messages list for the model, respecting the token budget.
        Priority (highest first): system prompt, summary, recent turns, current message.
        Oldest turns are dropped first if budget is tight.
        """
        anchor_tokens = self.cost(system_prompt) + self.cost(current_message)
        if anchor_tokens > self.budget:
            raise ValueError(
                f"Message too long for this model's context window "
                f"(~{anchor_tokens} tokens needed, {self.budget} available). "
                "Shorten the message or pick a model with a larger window."
            )

        remaining = self.budget - anchor_tokens

        summary_block = f"[Earlier conversation summary]\n{summary}" if summary else ""
        if summary_block:
            summary_tokens = self.cost(summary_block)
            if summary_tokens > remaining:
                summary_block = ""
            else:
                remaining -= summary_tokens

        # Fill with turns, most-recent first, then reverse for chronological order.
        # Stop at the first turn that doesn't fit so the kept history is contiguous.
        included: list[dict] = []
        for turn in reversed(turns):
            cost = self.cost(turn["content"])
            if cost > remaining:
                break
            included.append(turn)
            remaining -= cost
        included.reverse()

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if summary_block:
            messages.append({"role": "system", "content": summary_block})
        messages.extend(included)
        messages.append({"role": "user", "content": current_message})
        return messages
