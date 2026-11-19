#!/usr/bin/env bash
# Activates the project venv (creating it on first run) and launches the server.
# Output is teed to server.log so the standalone monitor (run_monitor.sh) can
# tail it for per-turn tok/s, in addition to printing to this terminal.
#
#   ./run.sh                    # dev server with --reload on :8000
#   PORT=9000 ./run.sh
#   NO_RELOAD=1 ./run.sh        # no file watcher
#   PYTHON=python3.11 ./run.sh  # pin the interpreter used to create the venv
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_DIR=".venv"

venv_python() {
    # Windows (Git Bash) venvs use Scripts/, POSIX venvs use bin/.
    if [ -f "$VENV_DIR/Scripts/python.exe" ]; then echo "$VENV_DIR/Scripts/python.exe"
    else echo "$VENV_DIR/bin/python"; fi
}

PYTHON_BIN="$(venv_python)"
FRESH=0
if [ ! -f "$PYTHON_BIN" ]; then
    PY="${PYTHON:-python3}"
    command -v "$PY" >/dev/null 2>&1 || PY="python"
    echo "Creating virtualenv in $VENV_DIR with $PY ..."
    # Debian/Ubuntu (incl. WSL) ship python3 without ensurepip unless
    # python3-venv is installed; fall back to uv when that's the case.
    if ! "$PY" -m venv "$VENV_DIR" >/dev/null 2>&1; then
        command -v uv >/dev/null 2>&1 || {
            echo "python -m venv failed; install python3-venv or uv." >&2; exit 1; }
        rm -rf "$VENV_DIR"
        uv venv -q --python "$PY" "$VENV_DIR"
    fi
    PYTHON_BIN="$(venv_python)"
    FRESH=1
fi

# Reinstall only on first run or when requirements.txt changed.
MARKER="$VENV_DIR/.requirements.installed"
if [ "$FRESH" = "1" ] || [ requirements.txt -nt "$MARKER" ]; then
    echo "Installing requirements ..."
    if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
        "$PYTHON_BIN" -m pip install -q -r requirements.txt
    else
        uv pip install -q --python "$PYTHON_BIN" -r requirements.txt
    fi
    touch "$MARKER"
fi

RELOAD_ARGS=()
if [ -z "${NO_RELOAD:-}" ]; then
    RELOAD_ARGS=(--reload
        --reload-dir api --reload-dir config --reload-dir llm
        --reload-dir metrics --reload-dir orchestrator --reload-dir session)
fi

# Unbuffered: piped through tee, Python would otherwise block-buffer stdout and
# the monitor would see turn_complete lines only after ~8KB of log accumulates.
export PYTHONUNBUFFERED=1
"$PYTHON_BIN" -m uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}" \
    "${RELOAD_ARGS[@]}" 2>&1 | tee server.log
