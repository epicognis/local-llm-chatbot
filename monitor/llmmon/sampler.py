"""Tie the collectors together into a single point-in-time Sample."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import ollama_client
from .gpu import GpuSnapshot, sample_gpu
from .system import ProcInfo, SysSnapshot, SystemCollector, find_ollama_pids
from .ollama_client import OllamaSnapshot
from .turns import TurnInfo, TurnTailer


@dataclass
class Sample:
    ts: float                       # epoch seconds
    gpu: GpuSnapshot
    sys: SysSnapshot
    ollama: OllamaSnapshot
    ollama_procs: list[ProcInfo] = field(default_factory=list)
    ollama_gpu_vram_mb: float = 0.0
    recent_turns: list[TurnInfo] = field(default_factory=list)
    turns_available: bool = False
    turns_reason: str = ""


class Sampler:
    """Stateful sampler — owns the psutil/RAPL deltas. Call sample() per tick."""

    def __init__(self, ollama_url: str, server_log_path: str | None = None) -> None:
        self.ollama_url = ollama_url
        self._sys = SystemCollector()
        self._turns = TurnTailer(server_log_path) if server_log_path else None

    def sample(self) -> Sample:
        gpu = sample_gpu()
        ol = ollama_client.ps(self.ollama_url)

        pids = find_ollama_pids()
        sysnap = self._sys.sample(pids)

        ollama_procs = [p for p in sysnap.procs if p.pid in pids]
        for p in ollama_procs:
            p.gpu_vram_mb = gpu.vram_for_pids({p.pid})

        recent_turns: list[TurnInfo] = []
        turns_available = False
        turns_reason = ""
        if self._turns is not None:
            self._turns.poll()
            recent_turns = list(self._turns.recent)
            turns_available = self._turns.available
            turns_reason = self._turns.reason

        return Sample(
            ts=time.time(),
            gpu=gpu,
            sys=sysnap,
            ollama=ol,
            ollama_procs=ollama_procs,
            ollama_gpu_vram_mb=gpu.vram_for_pids(pids),
            recent_turns=recent_turns,
            turns_available=turns_available,
            turns_reason=turns_reason,
        )
