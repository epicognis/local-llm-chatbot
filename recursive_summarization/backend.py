"""Backend abstraction seam.

The summarizer never imports a backend.  It depends only on a SummarizeFn:
a callable that takes a fully-assembled prompt string and returns the model's
text.  The library owns prompt construction; the function owns model contact.

The adapter is the ONLY place model-call specifics live — temperature,
num_ctx, think: false, max tokens — keeping all model-call hazards in one
auditable spot.
"""

from __future__ import annotations

from typing import Any, Protocol


class SummarizeFn(Protocol):
    """Takes a fully-assembled prompt string, returns model text."""

    def __call__(self, prompt: str) -> str: ...


class BackendAdapter:
    """Thin adapter wrapping an existing backend's .summarize() method.

    Applies the determinism and length settings that keep summarization
    stable and clean:
      - Low temperature (~0.1) for stable summaries across runs
      - Explicit num_ctx to defeat Ollama's silent 4096 default
      - think: false to prevent reasoning leakage into summary text
      - Generation cap appropriate to the summary target

    This is the single auditable spot for all model-call configuration.
    """

    def __init__(
        self,
        backend: Any,
        *,
        temperature: float = 0.1,
        num_ctx: int = 8192,
        max_tokens: int = 2048,
        think: bool = False,
        extra_options: dict[str, Any] | None = None,
    ) -> None:
        self._backend = backend
        self._temperature = temperature
        self._num_ctx = num_ctx
        self._max_tokens = max_tokens
        self._think = think
        self._extra = extra_options or {}

    def __call__(self, prompt: str) -> str:
        """Call the backend with all hazard mitigations applied."""
        options = {
            "temperature": self._temperature,
            "num_ctx": self._num_ctx,
            "num_predict": self._max_tokens,
            **self._extra,
        }
        if not self._think:
            options["think"] = False

        # The backend is expected to expose a .summarize(prompt, **kwargs)
        # or .generate(prompt, **kwargs) method.  Try summarize first,
        # fall back to generate.
        if hasattr(self._backend, "summarize"):
            result = self._backend.summarize(prompt, options=options)
        elif hasattr(self._backend, "generate"):
            result = self._backend.generate(prompt, options=options)
        else:
            raise TypeError(
                f"Backend {type(self._backend).__name__} has neither "
                f".summarize() nor .generate() — cannot adapt."
            )

        # Empty / null model output: treat as empty string.
        return result if result else ""
