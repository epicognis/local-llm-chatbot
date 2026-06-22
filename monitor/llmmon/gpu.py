"""NVIDIA GPU sampling via the `nvidia-smi` binary.

No pip dependency — we shell out to nvidia-smi (present wherever the NVIDIA
driver is). Device-level gives power, temperature, utilization and total VRAM;
the compute-apps query gives per-PID VRAM so we can attribute GPU memory to the
Ollama runner process(es). If nvidia-smi is absent (CPU-only box, AMD, Apple)
we degrade gracefully and report it as unavailable rather than crashing.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field

_SMI = shutil.which("nvidia-smi")

_NA = {"", "N/A", "[N/A]", "[NOT SUPPORTED]", "[NOT AVAILABLE]", "[UNKNOWN ERROR]"}


def _num(s: str) -> float | None:
    """Parse a numeric nvidia-smi field, tolerating its various 'no value' tokens."""
    s = s.strip()
    if s.upper() in _NA:
        return None
    try:
        return float(s)
    except ValueError:
        return None


@dataclass
class GpuDevice:
    index: int
    name: str
    util_pct: float | None
    mem_used_mb: float | None
    mem_total_mb: float | None
    power_w: float | None
    power_limit_w: float | None
    temp_c: float | None


@dataclass
class GpuProc:
    pid: int
    used_mem_mb: float


@dataclass
class GpuSnapshot:
    available: bool
    reason: str = ""
    devices: list[GpuDevice] = field(default_factory=list)
    procs: list[GpuProc] = field(default_factory=list)

    def vram_for_pids(self, pids: set[int]) -> float:
        """Total GPU VRAM (MB) used by the given PIDs, per the compute-apps query."""
        return sum(p.used_mem_mb for p in self.procs if p.pid in pids)


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_SMI, *args], capture_output=True, text=True, timeout=5
    )


def sample_gpu() -> GpuSnapshot:
    if not _SMI:
        return GpuSnapshot(False, "nvidia-smi not found (no NVIDIA GPU/driver on this host)")

    try:
        dev = _run([
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
            "power.draw,power.limit,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
    except (subprocess.TimeoutExpired, OSError) as exc:
        return GpuSnapshot(False, f"nvidia-smi failed: {exc}")

    if dev.returncode != 0:
        return GpuSnapshot(False, f"nvidia-smi error: {dev.stderr.strip() or dev.returncode}")

    devices: list[GpuDevice] = []
    for line in dev.stdout.splitlines():
        if not line.strip():
            continue
        c = [x.strip() for x in line.split(",")]
        if len(c) < 8:
            continue
        devices.append(
            GpuDevice(
                index=int(_num(c[0]) or 0),
                name=c[1],
                util_pct=_num(c[2]),
                mem_used_mb=_num(c[3]),
                mem_total_mb=_num(c[4]),
                power_w=_num(c[5]),
                power_limit_w=_num(c[6]),
                temp_c=_num(c[7]),
            )
        )

    procs: list[GpuProc] = []
    try:
        ca = _run([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        if ca.returncode == 0:
            for line in ca.stdout.splitlines():
                if not line.strip():
                    continue
                c = [x.strip() for x in line.split(",")]
                pid = _num(c[0])
                mem = _num(c[1]) if len(c) > 1 else None
                if pid is not None and mem is not None:
                    procs.append(GpuProc(pid=int(pid), used_mem_mb=mem))
    except (subprocess.TimeoutExpired, OSError):
        pass  # per-process VRAM is a nice-to-have; device stats already captured

    return GpuSnapshot(True, devices=devices, procs=procs)
