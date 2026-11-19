"""Bridge to the standalone hardware monitor in ``monitor/`` (the ``llmmon`` package).

Backs ``GET /monitor`` for the UI's real-time hardware panel. Rather than
re-implement GPU/CPU/Ollama sampling, this reuses ``llmmon.Sampler`` — the same
code path as the standalone terminal dashboard (./run_monitor.sh). ``monitor/`` is
added to ``sys.path`` at import time, and the panel degrades gracefully to
"unavailable" if ``llmmon`` or its deps (``psutil``, ``requests``) are missing.

The sampler is *stateful* — CPU% and RAPL power are deltas between successive
calls — so one instance is held per process and guarded by an asyncio lock.
``nvidia-smi`` shells out (up to a few seconds), so sampling runs in a worker
thread to keep the event loop responsive.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# <repo-root>/monitor holds the `llmmon` package.
_MONITOR_DIR = Path(__file__).resolve().parents[1] / "monitor"

_IMPORT_ERROR: str | None = None
try:
    if _MONITOR_DIR.is_dir() and str(_MONITOR_DIR) not in sys.path:
        sys.path.insert(0, str(_MONITOR_DIR))
    from llmmon.sampler import Sampler  # type: ignore
except Exception as exc:  # noqa: BLE001 — any failure means the panel shows "unavailable"
    Sampler = None  # type: ignore
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def _f(x: float | None, ndigits: int = 1) -> float | None:
    """Round a possibly-None numeric field for compact JSON."""
    return None if x is None else round(x, ndigits)


class HardwareMonitor:
    """Per-process wrapper over ``llmmon.Sampler`` producing JSON snapshots."""

    def __init__(self, ollama_url: str) -> None:
        self._lock = asyncio.Lock()
        self._sampler = Sampler(ollama_url) if Sampler is not None else None

    @property
    def available(self) -> bool:
        return self._sampler is not None

    async def snapshot(self) -> dict:
        if self._sampler is None:
            return {"available": False, "reason": _IMPORT_ERROR or "monitor unavailable"}
        # Serialize access: the sampler keeps per-tick deltas and isn't reentrant.
        async with self._lock:
            try:
                sample = await asyncio.to_thread(self._sampler.sample)
            except Exception as exc:  # noqa: BLE001
                return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        return self._to_dict(sample)

    @staticmethod
    def _to_dict(s) -> dict:
        gpu = {
            "available": s.gpu.available,
            "reason": s.gpu.reason,
            "devices": [
                {
                    "index": d.index,
                    "name": d.name,
                    "util_pct": _f(d.util_pct),
                    "mem_used_mb": _f(d.mem_used_mb, 0),
                    "mem_total_mb": _f(d.mem_total_mb, 0),
                    "power_w": _f(d.power_w),
                    "power_limit_w": _f(d.power_limit_w),
                    "temp_c": _f(d.temp_c, 0),
                }
                for d in s.gpu.devices
            ],
        }
        cpu = {
            "pct": _f(s.sys.cpu_pct),
            "mem_used_mb": _f(s.sys.mem_used_mb, 0),
            "mem_total_mb": _f(s.sys.mem_total_mb, 0),
            "mem_pct": _f(s.sys.mem_pct),
            "temp_c": _f(s.sys.cpu_temp_c, 0),
            "power_w": _f(s.sys.cpu_power_w),
        }
        ollama = {
            "reachable": s.ollama.reachable,
            "reason": s.ollama.reason,
            # vram_mb: GPU VRAM attributed to Ollama PIDs via nvidia-smi compute-apps
            # (device truth). proc_rss_mb: system RAM held by the Ollama processes.
            "vram_mb": _f(s.ollama_gpu_vram_mb, 0),
            "proc_rss_mb": _f(sum(p.rss_mb for p in s.ollama_procs), 0),
            "running": [
                {
                    "name": m.name,
                    "param_size": m.param_size,
                    "quant": m.quant,
                    "size_mb": _f(m.size_mb, 0),
                    "vram_mb": _f(m.vram_mb, 0),
                    "pct_gpu": _f(m.pct_gpu, 0),
                }
                for m in s.ollama.running
            ],
        }
        return {"available": True, "ts": s.ts, "gpu": gpu, "cpu": cpu, "ollama": ollama}
