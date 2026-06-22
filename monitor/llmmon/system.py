"""CPU / memory / process sampling via psutil, plus best-effort CPU power & temp.

- CPU%, RAM, and per-process CPU/RSS: psutil (cross-platform).
- CPU package power: Linux Intel/AMD RAPL (energy delta over time). Unavailable
  on Windows/macOS without vendor tooling, so it reports None there.
- CPU temperature: psutil.sensors_temperatures() (Linux). Windows psutil does
  not expose this, so it reports None.

`SystemCollector` is stateful on purpose: process cpu_percent() and RAPL power
are both deltas between successive samples, so the collector must persist across
ticks of the poll loop.
"""

from __future__ import annotations

import glob
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class ProcInfo:
    pid: int
    name: str
    cpu_pct: float          # percent of ONE core (can exceed 100 on multi-thread)
    rss_mb: float
    cmd: str
    gpu_vram_mb: float = 0.0  # filled in by the sampler from the GPU snapshot


@dataclass
class SysSnapshot:
    cpu_pct: float
    mem_used_mb: float
    mem_total_mb: float
    mem_pct: float
    cpu_temp_c: float | None
    cpu_power_w: float | None
    procs: list[ProcInfo] = field(default_factory=list)


def _cpu_temp() -> float | None:
    fn = getattr(psutil, "sensors_temperatures", None)
    if fn is None:
        return None
    try:
        temps = fn()
    except Exception:
        return None
    for key in ("coretemp", "k10temp", "zenpower", "cpu_thermal", "acpitz"):
        entries = temps.get(key)
        if not entries:
            continue
        for e in entries:
            if e.label and "package" in e.label.lower():
                return e.current
        return max(e.current for e in entries)
    return None


class _RaplReader:
    """Intel/AMD RAPL package power from /sys/class/powercap (Linux only)."""

    def __init__(self) -> None:
        self._paths = sorted(glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj"))
        self._last_uj: int | None = None
        self._last_t: float | None = None

    def watts(self) -> float | None:
        if not self._paths:
            return None
        total = 0
        for p in self._paths:
            try:
                with open(p) as fh:
                    total += int(fh.read().strip())
            except (OSError, ValueError):
                return None
        now = time.monotonic()
        prev_uj, prev_t = self._last_uj, self._last_t
        self._last_uj, self._last_t = total, now
        if prev_uj is None or prev_t is None:
            return None
        dt = now - prev_t
        d_uj = total - prev_uj
        if dt <= 0 or d_uj < 0:  # first delta or counter wraparound
            return None
        return (d_uj / 1e6) / dt  # microjoules -> joules -> watts


class SystemCollector:
    def __init__(self) -> None:
        self._rapl = _RaplReader()
        self._procs: dict[int, psutil.Process] = {}
        psutil.cpu_percent(interval=None)  # prime the system-wide counter

    def sample(self, watch_pids: set[int]) -> SysSnapshot:
        vm = psutil.virtual_memory()

        # Refresh tracked Process handles for the PIDs of interest.
        for pid in watch_pids:
            if pid not in self._procs:
                try:
                    self._procs[pid] = psutil.Process(pid)
                except psutil.Error:
                    pass
        for pid in list(self._procs):
            if pid not in watch_pids or not self._procs[pid].is_running():
                self._procs.pop(pid, None)

        procs: list[ProcInfo] = []
        for pid, proc in list(self._procs.items()):
            try:
                with proc.oneshot():
                    procs.append(
                        ProcInfo(
                            pid=pid,
                            name=proc.name(),
                            cpu_pct=proc.cpu_percent(interval=None),
                            rss_mb=proc.memory_info().rss / 1024 / 1024,
                            cmd=" ".join(proc.cmdline()[:4]),
                        )
                    )
            except psutil.Error:
                self._procs.pop(pid, None)

        return SysSnapshot(
            cpu_pct=psutil.cpu_percent(interval=None),
            mem_used_mb=vm.used / 1024 / 1024,
            mem_total_mb=vm.total / 1024 / 1024,
            mem_pct=vm.percent,
            cpu_temp_c=_cpu_temp(),
            cpu_power_w=self._rapl.watts(),
            procs=sorted(procs, key=lambda p: p.cpu_pct, reverse=True),
        )


def find_ollama_pids() -> set[int]:
    """PIDs of the Ollama daemon and its model-runner subprocesses."""
    pids: set[int] = set()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmd = " ".join(proc.info.get("cmdline") or []).lower()
            if "ollama" in name or "ollama" in cmd:
                pids.add(proc.info["pid"])
        except psutil.Error:
            continue
    return pids
