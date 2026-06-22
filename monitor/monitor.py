#!/usr/bin/env python3
"""llm-hw-monitor — realtime LLM + local-hardware monitor.

Samples GPU power/temperature/VRAM/utilization (nvidia-smi), CPU/memory/process
load (psutil), and the Ollama daemon (/api/ps) so you can gauge process, memory,
power, and temperature per resident model while the local hardware is exercised.
The LLM name is shown as informational alongside the hardware it is loading.

Standalone and decoupled from the chat server — it does not import anything
from the rest of this repo.

Usage:
    python monitor.py                       # live dashboard, 1s interval
    python monitor.py -i 0.5                # faster sampling
    python monitor.py --once                # print a single sample and exit
    python monitor.py --no-tui              # plain scrolling lines (logs/pipes)
    python monitor.py --csv run.csv         # also append every sample to CSV
    python monitor.py --ollama-url http://localhost:11434
    python monitor.py --server-log ../server.log    # tok/s source (default)

Prefer launching via ./run_monitor.sh, which provisions an isolated venv first.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from llmmon.sampler import Sampler

_DEFAULT_SERVER_LOG = Path(__file__).resolve().parent.parent / "server.log"


def _default_ollama_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def _default_server_log() -> str:
    return os.environ.get("SERVER_LOG_PATH", str(_DEFAULT_SERVER_LOG))


def _looks_interactive() -> bool:
    if sys.stdout.isatty():
        return True
    # Git Bash/MSYS sets MSYSTEM and routes stdout through a mintty pty that
    # often fails isatty() detection even when run directly in an interactive
    # shell. Trust that signal too rather than silently downgrading to plain
    # scrolling text on every Windows Git Bash session.
    return bool(os.environ.get("MSYSTEM"))


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="monitor.py",
        description="Realtime per-model LLM + local hardware monitor.",
    )
    p.add_argument("-i", "--interval", type=float, default=1.0,
                   help="seconds between samples (default: 1.0)")
    p.add_argument("--ollama-url", default=_default_ollama_url(),
                   help="Ollama base URL (default: $OLLAMA_BASE_URL or localhost:11434)")
    p.add_argument("--server-log", default=_default_server_log(),
                   help="path to the chat server's log file, for tok/s "
                        "(default: $SERVER_LOG_PATH or ../server.log)")
    p.add_argument("--csv", metavar="PATH", default=None,
                   help="append every sample to this CSV file")
    p.add_argument("--once", action="store_true",
                   help="take one sample, print it, and exit")
    p.add_argument("--no-tui", action="store_true",
                   help="plain text output instead of the live dashboard")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    sampler = Sampler(args.ollama_url, args.server_log)

    recorder = None
    if args.csv:
        from llmmon.recorder import CsvRecorder
        recorder = CsvRecorder(args.csv)

    from llmmon import dashboard

    # cpu_percent / RAPL are deltas: the very first sample is a baseline. For a
    # meaningful --once reading we take a throwaway sample, wait, then sample.
    try:
        if args.once:
            sampler.sample()
            time.sleep(min(args.interval, 1.0) or 1.0)
            s = sampler.sample()
            if recorder:
                recorder.write(s)
            print(dashboard.render_plain(s))
            return 0

        use_tui = not args.no_tui and _looks_interactive()

        if use_tui:
            return _run_tui(sampler, recorder, args, dashboard)
        return _run_plain(sampler, recorder, args, dashboard)
    except KeyboardInterrupt:
        return 0
    finally:
        if recorder:
            recorder.close()


def _run_plain(sampler, recorder, args, dashboard) -> int:
    while True:
        s = sampler.sample()
        if recorder:
            recorder.write(s)
        print(dashboard.render_plain(s))
        print("-" * 72, flush=True)
        time.sleep(args.interval)


def _run_tui(sampler, recorder, args, dashboard) -> int:
    from rich.live import Live
    from rich.console import Console

    # force_terminal/legacy_windows: Git Bash's mintty pty is frequently
    # misdetected (isatty() False, or routed through the Win32-console-API
    # code path that mintty doesn't honor). Force real ANSI/VT output.
    console = Console(force_terminal=True, legacy_windows=False)
    with Live(console=console, screen=True, auto_refresh=False) as live:
        while True:
            s = sampler.sample()
            if recorder:
                recorder.write(s)
            live.update(dashboard.build_renderable(s), refresh=True)
            time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
