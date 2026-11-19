#!/usr/bin/env bash
# Activates the project venv (creating it on first run) and launches the server.
# Output is teed to server.log so the standalone monitor (run_monitor.sh) can
# tail it for per-turn tok/s, in addition to printing to this terminal.
#
#   ./run.sh                    # dev server with --reload on :8080
#   NO_RELOAD=1 ./run.sh        # no file watcher
#   PYTHON=python3.11 ./run.sh  # pin the interpreter used to create the venv
#   LAN_IP=192.168.1.50 ./run.sh  # override the network address printed at startup
#   RUN_IN_WSL=1 ./run.sh       # (WSL only) run the server inside WSL itself
#
# Always serves on port 8080. Anything already listening there (typically an
# earlier copy of this server that didn't shut down) is stopped first.
#
# From WSL, the server is started as a *Windows* process (Windows Python in
# .venv), exactly as Git Bash would: a server inside WSL's NAT network is only
# reachable from Windows through WSL's localhost forwarding — which is flaky —
# and never from other devices on the network.
#
# Once the server answers, it prints the local URL plus the network URL (and a
# curl check) for reaching it from other devices.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Windows (Git Bash) and Linux/WSL can't share a venv — the same project folder
# is visible from both, so each gets its own.
is_wsl() { grep -qi microsoft /proc/version 2>/dev/null; }

windows_python() {
    # A real Windows python.exe (skipping the Microsoft Store stub), as a WSL path.
    local win
    win="$(cmd.exe /c where python 2>/dev/null | tr -d '\r' | grep -vi WindowsApps | head -n1)"
    [ -n "$win" ] && wslpath -u "$win"
}

WINDOWS_SERVER=0
if [ -z "${MSYSTEM:-}" ] && is_wsl && [ -z "${RUN_IN_WSL:-}" ] \
        && command -v cmd.exe >/dev/null 2>&1; then
    WINDOWS_SERVER=1
fi

if [ -n "${MSYSTEM:-}" ]; then
    VENV_DIR=".venv"          # Git Bash on Windows
    DEFAULT_PY="python"       # python3 on Windows is often the Microsoft Store stub
elif [ "$WINDOWS_SERVER" = "1" ]; then
    VENV_DIR=".venv"          # WSL launching the Windows server: same venv as Git Bash
    DEFAULT_PY="$(windows_python || true)"
    if [ -z "$DEFAULT_PY" ] && [ ! -f "$VENV_DIR/Scripts/python.exe" ]; then
        echo "No Windows Python found (needed to serve to Windows and the network)." >&2
        echo "Install Python for Windows, or run inside WSL with: RUN_IN_WSL=1 ./run.sh" >&2
        exit 1
    fi
else
    VENV_DIR=".venv-linux"    # WSL / Linux / macOS
    DEFAULT_PY="python3"
fi

venv_python() {
    # Windows (Git Bash) venvs use Scripts/, POSIX venvs use bin/.
    if [ -f "$VENV_DIR/Scripts/python.exe" ]; then echo "$VENV_DIR/Scripts/python.exe"
    else echo "$VENV_DIR/bin/python"; fi
}

PYTHON_BIN="$(venv_python)"
FRESH=0
if [ ! -f "$PYTHON_BIN" ]; then
    PY="${PYTHON:-$DEFAULT_PY}"
    echo "Creating virtualenv in $VENV_DIR with $PY ..."
    # Debian/Ubuntu (incl. WSL) ship python3 without ensurepip unless
    # python3-venv is installed; fall back to uv when that's the case.
    if ! "$PY" -m venv "$VENV_DIR" >/dev/null 2>&1; then
        [ "$WINDOWS_SERVER" = "1" ] && { echo "Creating the Windows venv failed." >&2; exit 1; }
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

PORT=8080

# ── Free the port: stop whatever is still listening on it ──────────────────
# Ports get "stuck" because uvicorn --reload's supervisor process owns the
# listening socket: killing just the worker (or closing the terminal on
# Windows) can leave the supervisor, or a child that inherited the socket,
# holding it. So we stop the whole process tree, on both sides when on WSL
# (a Windows listener on the port shadows the WSL one for the Windows browser).

windows_port_pids() {
    command -v netstat.exe >/dev/null 2>&1 || return 0
    netstat.exe -ano 2>/dev/null | tr -d '\r' \
        | awk -v p=":$PORT" '$1=="TCP" && $4=="LISTENING" && substr($2, length($2)-length(p)+1)==p {print $5}' \
        | sort -u
}

linux_port_pids() {
    [ -n "${MSYSTEM:-}" ] && return 0
    if command -v lsof >/dev/null 2>&1; then
        lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | sort -u
    elif command -v ss >/dev/null 2>&1; then
        ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u
    fi
}

kill_windows_port() {
    # Walk up from each listener through python.exe parents (venv launcher →
    # interpreter → reloader → worker), then stop that whole tree. Also stops
    # children of a dead owner PID, which can keep an inherited socket open.
    local ps1='
$port = __PORT__
function Stop-Tree($id) {
  # Only ever stops python processes, never anything else in the tree.
  Get-CimInstance Win32_Process -Filter "ParentProcessId=$id" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "python*" } | ForEach-Object { Stop-Tree $_.ProcessId }
  $p = Get-Process -Id $id -ErrorAction SilentlyContinue
  if ($p -and $p.ProcessName -like "python*") { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
}
$owners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty OwningProcess -Unique
foreach ($o in $owners) {
  $root = $o
  $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$o" -ErrorAction SilentlyContinue
  if ($proc -and $proc.Name -notlike "python*") {
    # e.g. the WSL localhost relay forwarding to a server inside WSL: never touch it.
    Write-Output "  leaving $($proc.Name) (PID $o) alone - not a Python process"
    continue
  }
  while ($proc) {
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.ParentProcessId)" -ErrorAction SilentlyContinue
    if (-not ($parent -and $parent.Name -like "python*")) { break }
    $root = $parent.ProcessId; $proc = $parent
  }
  Write-Output "  stopping Windows process tree $root (listener PID $o)"
  Stop-Tree $root
  Stop-Tree $o
}'
    powershell.exe -NoProfile -Command "${ps1//__PORT__/$PORT}" 2>/dev/null | tr -d '\r'
}

kill_linux_port() {
    local pid parent
    for pid in $(linux_port_pids); do
        # Include a python parent (the uvicorn --reload supervisor).
        parent="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"
        if [ -n "$parent" ] && ps -o comm= -p "$parent" 2>/dev/null | grep -q python; then
            echo "  stopping process $parent (supervisor of listener $pid)"
            kill -TERM "$parent" 2>/dev/null || true
        fi
        echo "  stopping process $pid"
        kill -TERM "$pid" 2>/dev/null || true
    done
}

kill_wsl_port() {
    # Under Git Bash, a server running inside WSL shows up on Windows as
    # wslrelay.exe (the localhost forwarder). Stop the real server inside each
    # *running* WSL distro (not just the default one, and without booting any
    # stopped distro) — the relay then goes away by itself.
    [ -n "${MSYSTEM:-}" ] || return 0
    command -v wsl.exe >/dev/null 2>&1 || return 0
    MSYS_NO_PATHCONV=1 tasklist.exe /FI "IMAGENAME eq wslrelay.exe" /NH 2>/dev/null \
        | grep -qi wslrelay || return 0
    local distro
    # wsl.exe prints UTF-16.
    wsl.exe -l --running -q 2>/dev/null | iconv -f UTF-16LE -t UTF-8 2>/dev/null | tr -d '\r' \
    | while IFS= read -r distro; do
        [ -n "$distro" ] || continue
        MSYS_NO_PATHCONV=1 wsl.exe -d "$distro" -e sh -c "
            pids=\$(lsof -t -iTCP:$PORT -sTCP:LISTEN 2>/dev/null \
                || ss -ltnpH 'sport = :$PORT' 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2)
            [ -n \"\$pids\" ] && echo '  stopping the server inside WSL ($distro):' \$pids && kill \$pids" \
            </dev/null 2>/dev/null | tr -d '\r' || true
    done
}

free_port() {
    if [ -z "$(windows_port_pids)$(linux_port_pids)" ]; then return 0; fi
    echo "Port $PORT is in use — stopping the old server ..."
    [ -n "$(windows_port_pids)" ] && kill_windows_port
    [ -n "$(windows_port_pids)" ] && kill_wsl_port
    [ -n "$(linux_port_pids)" ] && kill_linux_port
    for _ in $(seq 1 10); do
        [ -z "$(windows_port_pids)$(linux_port_pids)" ] && { echo "Port $PORT is free."; return 0; }
        sleep 1
    done
    # Still held: force-kill anything left on the Linux side ...
    for pid in $(linux_port_pids); do kill -KILL "$pid" 2>/dev/null || true; done
    sleep 1
    [ -z "$(windows_port_pids)$(linux_port_pids)" ] && { echo "Port $PORT is free."; return 0; }
    # ... a Windows process started with admin rights can't be stopped from here.
    echo "Couldn't free port $PORT. Still held by PID(s): $(windows_port_pids) $(linux_port_pids)" >&2
    echo "If it's a Windows process started as administrator, stop it from an admin PowerShell:" >&2
    echo "  Stop-Process -Id <PID> -Force" >&2
    exit 1
}

free_port

# ── LAN address for other devices (phones etc.) ─────────────────────────────
windows_lan_ip() {
    # The Windows host's IPv4 on the adapter that has a default gateway.
    powershell.exe -NoProfile -Command \
        '(Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -ne $null -and $_.NetAdapter.Status -eq "Up" } | Select-Object -First 1).IPv4Address.IPAddress' \
        2>/dev/null | tr -d '\r'
}

lan_ip() {
    if [ -n "${LAN_IP:-}" ]; then echo "$LAN_IP"; return; fi   # manual override
    if [ -n "${MSYSTEM:-}" ] || grep -qi microsoft /proc/version 2>/dev/null; then
        windows_lan_ip; return                                  # Git Bash or WSL
    fi
    if command -v ip >/dev/null 2>&1; then                      # Linux
        ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}'
    elif command -v ipconfig >/dev/null 2>&1; then              # macOS
        ipconfig getifaddr en0 2>/dev/null
    fi
}

wsl_nat() {
    # WSL2's default NAT mode puts the server behind a private 172.x address:
    # the Windows LAN IP doesn't forward into it (mirrored mode does).
    [ "$WINDOWS_SERVER" = "1" ] && return 1    # served by Windows, not from inside WSL
    is_wsl || return 1
    local mode
    mode="$(wslinfo --networking-mode 2>/dev/null || true)"
    [ "$mode" = "nat" ] || { [ -z "$mode" ] && ip -4 -o addr show eth0 2>/dev/null | grep -q ' 172\.'; }
}

announce() {
    # Print the URLs once the server actually answers, after uvicorn's startup lines.
    local ready=0
    for _ in $(seq 1 120); do
        if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then ready=1; break; fi
        sleep 1
    done
    [ "$ready" = 1 ] || return 0

    local ip; ip="$(lan_ip)"
    echo
    echo "  Chatbot is up"
    echo "    This machine : http://localhost:$PORT/ui"
    if [ -n "$ip" ]; then
        echo "    Your network : http://$ip:$PORT/ui"
        echo "    Check from another device:  curl http://$ip:$PORT/health"
        if wsl_nat; then
            echo "    NOTE: WSL is in NAT mode, so the network URL won't work yet. Either"
            echo "      - add 'networkingMode=mirrored' under [wsl2] in %UserProfile%\\.wslconfig,"
            echo "        then run 'wsl --shutdown' from Windows and start again, or"
            echo "      - run ./run.sh from Git Bash on Windows instead."
        fi
    else
        echo "    Your network : (couldn't detect a LAN IP; set LAN_IP=... to print it)"
    fi
    echo
}
announce &

# Unbuffered: piped through tee, Python would otherwise block-buffer stdout and
# the monitor would see turn_complete lines only after ~8KB of log accumulates.
export PYTHONUNBUFFERED=1
# --timeout-graceful-shutdown: on a code-change reload (or Ctrl+C), uvicorn
# otherwise waits forever for open connections — browser tabs polling /monitor,
# keep-alives — and the reload never completes, leaving the server unresponsive.
"$PYTHON_BIN" -m uvicorn api.main:create_app --factory --host 0.0.0.0 --port "$PORT" \
    --timeout-graceful-shutdown 3 \
    "${RELOAD_ARGS[@]}" 2>&1 | tee server.log
