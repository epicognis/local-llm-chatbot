#!/usr/bin/env bash
# Standalone GPU/host/model monitor — runs independently of the chat server.
# Safe to start/stop any time; it just reads nvidia-smi, psutil, Ollama's
# /api/ps, and (for tok/s) the server's log file, if present.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_DIR=".venv"
PYTHON_BIN="$VENV_DIR/Scripts/python.exe"
[ -f "$PYTHON_BIN" ] || PYTHON_BIN="$VENV_DIR/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "Creating virtualenv in $VENV_DIR ..."
    python -m venv "$VENV_DIR"
    PYTHON_BIN="$VENV_DIR/Scripts/python.exe"
    [ -f "$PYTHON_BIN" ] || PYTHON_BIN="$VENV_DIR/bin/python"
fi

"$PYTHON_BIN" -m pip install -q -r monitor/requirements.txt

exec "$PYTHON_BIN" monitor/monitor_app.py
