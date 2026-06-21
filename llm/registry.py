MODEL_REGISTRY: list[dict] = [
    {"name": "GPT-OSS 20B", "model": "gpt-oss:20b", "default_ctx": 131072},
    {"name": "Qwen3 14B",   "model": "qwen3:14b",   "default_ctx": 40960},
    {"name": "Qwen3 8B",    "model": "qwen3:8b",    "default_ctx": 16384},
    {"name": "Gemma 4 12B", "model": "gemma4:12b",  "default_ctx": 32768},
    {"name": "Llama 3.1 8B","model": "llama3.1:8b", "default_ctx": 8192},
]

_by_name = {entry["name"]: entry for entry in MODEL_REGISTRY}


def get_model_entry(friendly_name: str) -> dict:
    try:
        return _by_name[friendly_name]
    except KeyError:
        raise KeyError(f"Unknown model '{friendly_name}'. Available: {list(_by_name)}")
