"""Curated model registry (build spec §3).

Each entry maps a friendly name to its Ollama tag plus per-model capabilities.
`default_ctx` is a *practical* window (capped for KV-cache headroom on a 16GB
card), sent per request as Ollama's `num_ctx`.

`think` is the per-model value sent as Ollama's `think` field to get direct
answers (spec §6: reasoning controls are per-model, not global). None = the model
has no reasoning mode, so the field is omitted. GPT-OSS can't fully disable
reasoning: `think: false` measured ~2,000 hidden reasoning tokens (~47s) before
the first visible word on this card, while `"low"` measured ~10 tokens (0.7s).
"""

MODEL_REGISTRY: list[dict] = [
    {"name": "Gemma 4 12B",  "model": "gemma4:12b",  "default_ctx": 65536,  "think": False},
    {"name": "GPT-OSS 20B",  "model": "gpt-oss:20b", "default_ctx": 131072, "think": "low"},
    {"name": "Qwen3 14B",    "model": "qwen3:14b",   "default_ctx": 40960,  "think": False},
    {"name": "Qwen3 8B",     "model": "qwen3:8b",    "default_ctx": 16384,  "think": False},
    {"name": "Llama 3.1 8B", "model": "llama3.1:8b", "default_ctx": 8192,   "think": None},
]

_by_name = {entry["name"]: entry for entry in MODEL_REGISTRY}


class UnknownModelError(KeyError):
    def __str__(self) -> str:  # KeyError's default str() wraps the message in quotes
        return self.args[0]


def get_model_entry(friendly_name: str) -> dict:
    try:
        return _by_name[friendly_name]
    except KeyError:
        raise UnknownModelError(
            f"Unknown model '{friendly_name}'. Available: {', '.join(_by_name)}"
        ) from None
