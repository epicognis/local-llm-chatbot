"""Standalone real-time monitor for the local chatbot stack.

Fully decoupled from the FastAPI server/UI — run it alongside the server
(see run_monitor.sh) in its own terminal. Shows GPU/VRAM, host CPU/RAM,
which Ollama models are currently loaded, and recent-turn tok/s (parsed
from the server's log file, since that's the only cross-process signal
of generation performance).
"""

import json
import os
import shutil
import subprocess
import time
import urllib.request
import urllib.error
from collections import deque
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

try:
    import psutil
except ImportError:
    psutil = None

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LOG_PATH = Path(os.environ.get("MONITOR_LOG_PATH", "server.log"))
REFRESH_S = 1.0

NVIDIA_SMI = shutil.which("nvidia-smi")
GPU_FIELDS = ["name", "utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw"]

recent_turns: deque[dict] = deque(maxlen=10)


def get_gpu_stats() -> list[dict]:
    if not NVIDIA_SMI:
        return []
    try:
        result = subprocess.run(
            [NVIDIA_SMI, f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(GPU_FIELDS):
            continue
        name, util, mem_used, mem_total, temp, power = parts
        try:
            gpus.append({
                "name": name,
                "util": float(util),
                "mem_used": float(mem_used),
                "mem_total": float(mem_total),
                "temp": float(temp),
                "power": None if power in ("", "N/A") else float(power),
            })
        except ValueError:
            continue
    return gpus


def get_host_stats() -> dict | None:
    if psutil is None:
        return None
    vm = psutil.virtual_memory()
    return {
        "cpu_pct": psutil.cpu_percent(interval=None),
        "ram_used_mb": (vm.total - vm.available) / (1024 ** 2),
        "ram_total_mb": vm.total / (1024 ** 2),
        "ram_pct": vm.percent,
    }


def get_loaded_models() -> list[dict]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/ps", timeout=2) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []
    return [
        {"name": m.get("name", "?"), "vram_mb": m.get("size_vram", 0) / (1024 ** 2)}
        for m in data.get("models", [])
    ]


def _open_log_tail():
    if not LOG_PATH.exists():
        return None
    f = open(LOG_PATH, "r", encoding="utf-8", errors="ignore")
    f.seek(0, os.SEEK_END)
    return f


def poll_log(f) -> None:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "turn_complete":
            recent_turns.appendleft(event)


def fmt_mb(mb: float | None) -> str:
    if mb is None:
        return "-"
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def bar(pct: float, width: int = 20) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(width * pct / 100)
    return "#" * filled + "-" * (width - filled)


def render() -> Table:
    grid = Table.grid(expand=True)
    grid.add_column()

    gpu_table = Table(title="GPU", expand=True, show_header=False, box=None)
    gpus = get_gpu_stats()
    if not gpus:
        gpu_table.add_row("No NVIDIA GPU detected (nvidia-smi not found)")
    for g in gpus:
        vram_pct = 100 * g["mem_used"] / g["mem_total"] if g["mem_total"] else 0
        power_str = f", {g['power']:.0f} W" if g["power"] is not None else ""
        gpu_table.add_row(g["name"])
        gpu_table.add_row(f"Util  {bar(g['util'])}  {g['util']:.0f}%")
        gpu_table.add_row(f"VRAM  {bar(vram_pct)}  {fmt_mb(g['mem_used'])}/{fmt_mb(g['mem_total'])}")
        gpu_table.add_row(f"Temp  {g['temp']:.0f}C{power_str}")

    host_table = Table(title="Host", expand=True, show_header=False, box=None)
    host = get_host_stats()
    if host is None:
        host_table.add_row("psutil not installed")
    else:
        host_table.add_row(f"CPU  {bar(host['cpu_pct'])}  {host['cpu_pct']:.0f}%")
        host_table.add_row(
            f"RAM  {bar(host['ram_pct'])}  {fmt_mb(host['ram_used_mb'])}/{fmt_mb(host['ram_total_mb'])}"
        )

    models_table = Table(title="Loaded models (Ollama)", expand=True, box=None)
    models_table.add_column("Model")
    models_table.add_column("VRAM", justify="right")
    loaded = get_loaded_models()
    if not loaded:
        models_table.add_row("(none loaded)", "")
    for m in loaded:
        models_table.add_row(m["name"], fmt_mb(m["vram_mb"]))

    perf_table = Table(title="Recent turns", expand=True, box=None)
    perf_table.add_column("Model")
    perf_table.add_column("Tokens", justify="right")
    perf_table.add_column("Elapsed (s)", justify="right")
    perf_table.add_column("tok/s", justify="right")
    if not recent_turns:
        perf_table.add_row("(waiting for completed turns)", "", "", "")
    for t in recent_turns:
        perf_table.add_row(
            str(t.get("model", "?")),
            str(t.get("completion_tokens", "?")),
            str(t.get("elapsed_s", "?")),
            str(t.get("tokens_per_sec", "?")),
        )

    top = Table.grid(expand=True)
    top.add_column(ratio=1)
    top.add_column(ratio=1)
    top.add_row(Panel(gpu_table), Panel(host_table))

    bottom = Table.grid(expand=True)
    bottom.add_column(ratio=1)
    bottom.add_column(ratio=1)
    bottom.add_row(Panel(models_table), Panel(perf_table))

    grid.add_row(top)
    grid.add_row(bottom)
    return grid


def main() -> None:
    console = Console()
    if not LOG_PATH.exists():
        console.print(
            f"[yellow]Note:[/] log file '{LOG_PATH}' not found yet — the tok/s panel stays empty "
            "until the server (started via run.sh) has logged a completed turn there."
        )
    log_file = _open_log_tail()

    with Live(render(), console=console, refresh_per_second=1, screen=False) as live:
        while True:
            time.sleep(REFRESH_S)
            if log_file is None:
                log_file = _open_log_tail()
            if log_file is not None:
                poll_log(log_file)
            live.update(render())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
