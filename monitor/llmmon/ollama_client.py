"""Ollama daemon introspection via its HTTP API.

`/api/ps` lists the models currently resident in the daemon along with their
total size and the portion held in VRAM — that's how we attach an LLM name to
the hardware load. The endpoint does not expose PIDs, so process-level CPU/VRAM
attribution is done separately (system.find_ollama_pids + GPU compute-apps).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import requests


@dataclass
class OllamaModel:
    name: str
    size_mb: float
    vram_mb: float
    param_size: str
    quant: str
    expires_at: str

    @property
    def pct_gpu(self) -> float | None:
        """Fraction of the model resident in VRAM vs CPU RAM (0-100)."""
        if self.size_mb <= 0:
            return None
        return min(100.0, 100.0 * self.vram_mb / self.size_mb)


@dataclass
class OllamaSnapshot:
    reachable: bool
    base_url: str
    reason: str = ""
    running: list[OllamaModel] = field(default_factory=list)


def ps(base_url: str, timeout: float = 2.0) -> OllamaSnapshot:
    url = base_url.rstrip("/") + "/api/ps"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return OllamaSnapshot(False, base_url, f"{type(exc).__name__}: {exc}")
    except ValueError as exc:
        return OllamaSnapshot(False, base_url, f"bad JSON from {url}: {exc}")

    running: list[OllamaModel] = []
    for m in data.get("models", []):
        details = m.get("details") or {}
        running.append(
            OllamaModel(
                name=m.get("name") or m.get("model") or "?",
                size_mb=(m.get("size") or 0) / 1024 / 1024,
                vram_mb=(m.get("size_vram") or 0) / 1024 / 1024,
                param_size=details.get("parameter_size", "?"),
                quant=details.get("quantization_level", "?"),
                expires_at=m.get("expires_at", ""),
            )
        )
    return OllamaSnapshot(True, base_url, running=running)
