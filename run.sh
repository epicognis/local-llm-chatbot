#!/usr/bin/env bash
# Activates the project venv (creating it on first run) and launches the server.
# Output is teed to server.log so the standalone monitor (run_monitor.sh) can
# tail it for per-turn tok/s, in addition to printing to this terminal.
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

"$PYTHON_BIN" -m pip install -q -r requirements.txt

"$PYTHON_BIN" -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload 2>&1 | tee server.log
