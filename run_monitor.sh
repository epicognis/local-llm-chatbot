#!/usr/bin/env bash
#
# run_monitor.sh — provision an isolated venv, resolve dependencies, and launch
# the realtime LLM + hardware monitor. Completely standalone: it gets its own
# venv (separate from the server's .venv) so installing into it can never
# trigger the server's --reload file watcher again. Re-running is cheap — the
# venv is reused and deps only reinstall when requirements.txt changes.
#
#   ./run_monitor.sh                         # live dashboard (1s)
#   ./run_monitor.sh -i 0.5                  # faster sampling
#   ./run_monitor.sh --once                  # single snapshot, then exit
#   ./run_monitor.sh --no-tui                # plain scrolling output
#   ./run_monitor.sh --csv run.csv           # also log every sample to CSV
#   OLLAMA_BASE_URL=http://localhost:11434 ./run_monitor.sh
#   PYTHON=python3.11 ./run_monitor.sh       # pin the interpreter
#
# Any arguments are passed straight through to monitor/monitor.py (see --help).
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/monitor" && pwd)"
cd "$HERE"

VENV="$HERE/.venv"
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY="python3"

FRESH=0
if [ ! -d "$VENV" ]; then
  echo "[run_monitor] creating venv at monitor/.venv ..."
  "$PY" -m venv "$VENV"
  FRESH=1
fi

# Activate — Windows (Git Bash) uses Scripts/, POSIX uses bin/.
if [ -f "$VENV/Scripts/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/Scripts/activate"
else
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

# Install deps on first run, or whenever requirements.txt is newer than the marker.
MARKER="$VENV/.requirements.installed"
if [ "$FRESH" = "1" ] || [ requirements.txt -nt "$MARKER" ]; then
  echo "[run_monitor] installing requirements ..."
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements.txt
  touch "$MARKER"
fi

echo "[run_monitor] OLLAMA_BASE_URL = ${OLLAMA_BASE_URL:-http://localhost:11434}"
exec python monitor.py "$@"
