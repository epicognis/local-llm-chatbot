"""Model registry — friendly name -> (model tag, default_ctx, reasoning controls).

Backs ``GET /models`` and resolves the friendly ``model`` field on ``POST /chat``.
Curated shortlist for a single RTX 5060 Ti 16GB card (see the build spec §3).

``default_ctx`` is a *practical* per-request window, not the model's maximum. It
is sent to Ollama as ``num_ctx`` on every request — without that, Ollama silently
caps context at 4096 regardless of the model's real window. Each value is the
largest window measured to load 100% on the GPU (weights + KV cache + compute
buffers) on this card without a speed penalty:

    GPT-OSS 20B   65536 -> 100% GPU, ~86 tok/s   (131072 spilled: 89% GPU, ~47 tok/s)
    Gemma 4 12B   32768 -> 100% GPU, ~81 tok/s   (65536 also fits but loads 3x slower)
    Qwen3 8B      32768 -> 100% GPU, ~70 tok/s
    Llama 3.1 8B  32768 -> 100% GPU, ~75 tok/s
    Qwen3 14B     16384 -> conservative (not yet measured on this card)

Raise per turn from the UI's LLM Config panel when headroom allows (watch the
monitor's GPU VRAM row). Larger (30B+) models are intentionally absent — they
overflow 16GB once KV cache is added.

Reasoning controls (spec §6 — they're per-model, never global):
  ``supports_think`` -- the model has a reasoning mode the user can turn on.
  ``think_off``      -- the value sent as Ollama's ``think`` when reasoning is
                        off. GPT-OSS can't disable reasoning: ``think: false``
                        measured ~2,000 hidden reasoning tokens (~47 s) before the
                        first visible word, while ``"low"`` measured ~10 (0.7 s).
                        None = model has no reasoning mode; the field is omitted.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str          # friendly name shown in the UI / used in the API
    model: str         # raw backend tag (e.g. Ollama "gpt-oss:20b")
    default_ctx: int   # per-request num_ctx
    backend: str = "ollama"
    supports_think: bool = False
    think_off: bool | str | None = None

    def think_value(self, requested: bool) -> bool | str | None:
        """The ``think`` value to send for this turn."""
        if requested and self.supports_think:
            return True
        return self.think_off

    def public(self) -> dict:
        """Shape returned by GET /models for the UI dropdown."""
        return {
            "name": self.name,
            "model": self.model,
            "default_ctx": self.default_ctx,
            "supports_think": self.supports_think,
        }


# Order is the UI dropdown order: Gemma (default) first, Llama above the Qwen tier.
_MODELS: list[ModelSpec] = [
    ModelSpec("Gemma 4 12B", "gemma4:12b", 32768, supports_think=True, think_off=False),
    ModelSpec("GPT-OSS 20B", "gpt-oss:20b", 65536, supports_think=True, think_off="low"),
    ModelSpec("Llama 3.1 8B", "llama3.1:8b", 32768),
    ModelSpec("Qwen3 14B", "qwen3:14b", 16384, supports_think=True, think_off=False),
    ModelSpec("Qwen3 8B", "qwen3:8b", 32768, supports_think=True, think_off=False),
]


class UnknownModelError(KeyError):
    def __str__(self) -> str:  # KeyError's default str() wraps the message in quotes
        return self.args[0]


class Registry:
    """In-memory lookup over the curated model list."""

    def __init__(self, specs: list[ModelSpec] | None = None) -> None:
        self._specs = list(specs if specs is not None else _MODELS)
        self._by_name = {s.name: s for s in self._specs}

    def list(self) -> list[ModelSpec]:
        return list(self._specs)

    def public_list(self) -> list[dict]:
        return [s.public() for s in self._specs]

    def resolve(self, name: str) -> ModelSpec:
        """Resolve a friendly name to its spec. Raises UnknownModelError if unknown."""
        try:
            return self._by_name[name]
        except KeyError:
            known = ", ".join(self._by_name)
            raise UnknownModelError(f"Unknown model '{name}'. Available: {known}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._by_name
