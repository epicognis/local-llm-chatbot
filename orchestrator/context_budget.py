def estimate_tokens(text: str) -> int:
    """Rough heuristic: ~4 chars per token. Calibrate against prompt_eval_count."""
    return max(1, len(text) // 4)


class ContextBudgetAllocator:
    """Packs model context within a token budget. Start at Level 0 (full history passthrough)."""

    def __init__(self, model_ctx: int, max_tokens: int, safety_margin: int):
        self.budget = model_ctx - max_tokens - safety_margin
        if self.budget <= 0:
            raise ValueError(
                f"Budget is non-positive: ctx={model_ctx}, "
                f"max_tokens={max_tokens}, margin={safety_margin}"
            )

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
        anchor_tokens = estimate_tokens(system_prompt) + estimate_tokens(current_message)
        if anchor_tokens > self.budget:
            raise ValueError(
                f"Anchors alone exceed budget ({anchor_tokens} > {self.budget}). "
                "Reduce system_prompt or current_message."
            )

        remaining = self.budget - anchor_tokens

        # Reserve space for summary
        summary_tokens = estimate_tokens(summary) if summary else 0
        if summary and summary_tokens > remaining:
            summary = ""
            summary_tokens = 0
        remaining -= summary_tokens

        # Fill with turns, most-recent first, then reverse for chronological order
        included: list[dict] = []
        for turn in reversed(turns):
            cost = estimate_tokens(turn["content"])
            if cost > remaining:
                break
            included.append(turn)
            remaining -= cost
        included.reverse()

        # Assemble final message list
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if summary:
            messages.append(
                {"role": "system", "content": f"[Earlier conversation summary]\n{summary}"}
            )
        messages.extend(included)
        messages.append({"role": "user", "content": current_message})
        return messages
