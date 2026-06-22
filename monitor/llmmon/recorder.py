"""Append samples to a CSV for later analysis (pandas/matplotlib, your call).

One row per resident model per tick, so you get a per-model time series of
process/mem/power/temp while the hardware is exercised. If no model is resident
we still emit one row (model_name blank) to capture the hardware baseline.

Power & temperature are device-level (whole GPU / CPU package), not split per
model — a single GPU reports one power and one temperature. VRAM, by contrast,
is attributed per model (Ollama size_vram) and per process (nvidia-smi).
"""

from __future__ import annotations

import csv
import os

from .sampler import Sample

FIELDS = [
    "ts",
    "ollama_url",
    "model_name",
    "model_param",
    "model_quant",
    "model_size_mb",
    "model_vram_mb",
    "model_pct_gpu",
    "gpu_index",
    "gpu_name",
    "gpu_util_pct",
    "gpu_mem_used_mb",
    "gpu_mem_total_mb",
    "gpu_power_w",
    "gpu_power_limit_w",
    "gpu_temp_c",
    "sys_cpu_pct",
    "sys_mem_used_mb",
    "sys_mem_total_mb",
    "cpu_power_w",
    "cpu_temp_c",
    "ollama_cpu_pct",
    "ollama_rss_mb",
    "ollama_gpu_vram_mb",
]


class CsvRecorder:
    def __init__(self, path: str) -> None:
        self.path = path
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=FIELDS)
        if new:
            self._w.writeheader()
            self._fh.flush()

    def write(self, s: Sample) -> None:
        # Device-level aggregates (use the first/primary GPU when present).
        dev = s.gpu.devices[0] if s.gpu.devices else None
        ollama_cpu = sum(p.cpu_pct for p in s.ollama_procs)
        ollama_rss = sum(p.rss_mb for p in s.ollama_procs)

        base = {
            "ts": round(s.ts, 3),
            "ollama_url": s.ollama.base_url,
            "gpu_index": dev.index if dev else "",
            "gpu_name": dev.name if dev else "",
            "gpu_util_pct": dev.util_pct if dev else "",
            "gpu_mem_used_mb": _r(dev.mem_used_mb) if dev else "",
            "gpu_mem_total_mb": _r(dev.mem_total_mb) if dev else "",
            "gpu_power_w": _r(dev.power_w) if dev else "",
            "gpu_power_limit_w": _r(dev.power_limit_w) if dev else "",
            "gpu_temp_c": dev.temp_c if dev else "",
            "sys_cpu_pct": _r(s.sys.cpu_pct),
            "sys_mem_used_mb": _r(s.sys.mem_used_mb),
            "sys_mem_total_mb": _r(s.sys.mem_total_mb),
            "cpu_power_w": _r(s.sys.cpu_power_w),
            "cpu_temp_c": _r(s.sys.cpu_temp_c),
            "ollama_cpu_pct": _r(ollama_cpu),
            "ollama_rss_mb": _r(ollama_rss),
            "ollama_gpu_vram_mb": _r(s.ollama_gpu_vram_mb),
        }

        models = s.ollama.running or [None]
        for m in models:
            row = dict(base)
            if m is not None:
                row.update(
                    model_name=m.name,
                    model_param=m.param_size,
                    model_quant=m.quant,
                    model_size_mb=_r(m.size_mb),
                    model_vram_mb=_r(m.vram_mb),
                    model_pct_gpu=_r(m.pct_gpu),
                )
            self._w.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def _r(v: float | None) -> str | float:
    return "" if v is None else round(v, 2)
