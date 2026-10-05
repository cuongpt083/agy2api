#!/usr/bin/env bash
# ==============================================================================
# agy2api Daemon Manager (Headless / No-GUI Mode)
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

LOGS_DIR="$PROJECT_DIR/logs"
PID_FILE="$LOGS_DIR/daemon.pid"
LOG_FILE="$LOGS_DIR/daemon.log"
VENV_DIR="$PROJECT_DIR/.venv"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

mkdir -p "$LOGS_DIR"

check_venv() {
    if [ ! -f "$VENV_DIR/bin/python" ] || [ ! -f "$VENV_DIR/bin/uvicorn" ]; then
        echo "[!] Virtual environment not found or incomplete at $VENV_DIR. Initializing..."
        if command -v uv >/dev/null 2>&1; then
            echo "[*] Using uv to create venv and install dependencies..."
            uv venv "$VENV_DIR"
            uv pip install -r "$PROJECT_DIR/requirements.txt"
        else
            echo "[*] Using python3 to create venv..."
            python3 -m venv "$VENV_DIR"
            "$VENV_DIR/bin/pip" install --upgrade pip
            "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
        fi
        echo "[+] Dependencies installed successfully."
    fi
}

is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

start_daemon() {
    if is_running; then
        local pid
        pid="$(cat "$PID_FILE")"
        echo "[*] Daemon is already running (PID: $pid). Listening on http://$HOST:$PORT"
        return 0
    fi

    # Check if port is already occupied
    if command -v ss >/dev/null 2>&1; then
        if ss -tulpn | grep -q ":$PORT "; then
            echo "[-] Error: Port $PORT is already in use by another process."
            return 1
        fi
    fi

    check_venv

    echo "[*] Starting agy2api daemon (Headless mode, no GUI)..."

    # Use standard double-fork daemonizer to guarantee detachment from terminal/SSH
    "$VENV_DIR/bin/python" - <<EOF
import os, sys

pid = os.fork()
if pid > 0:
    sys.exit(0)

os.setsid()

pid2 = os.fork()
if pid2 > 0:
    sys.exit(0)

with open("$PID_FILE", "w") as f:
    f.write(str(os.getpid()))

env = os.environ.copy()
env["DBUS_SESSION_BUS_ADDRESS"] = os.environ.get(
    "DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus"
)
env["PATH"] = "$VENV_DIR/bin:$HOME/.local/bin:" + env.get("PATH", "")

# Remove SSH indicators so agy does not block keyring access
for k in ["SSH_CLIENT", "SSH_CONNECTION", "SSH_TTY"]:
    env.pop(k, None)

log = open("$LOG_FILE", "a")
os.dup2(log.fileno(), sys.stdout.fileno())
os.dup2(log.fileno(), sys.stderr.fileno())

devnull = open("/dev/null", "r")
os.dup2(devnull.fileno(), sys.stdin.fileno())

os.execve(
    "$VENV_DIR/bin/uvicorn",
    ["$VENV_DIR/bin/uvicorn", "app.main:app", "--host", "$HOST", "--port", "$PORT"],
    env,
)
EOF

    # Wait briefly and verify process is alive
    sleep 2
    if is_running; then
        local new_pid
        new_pid="$(cat "$PID_FILE")"
        echo "[+] agy2api daemon started successfully!"
        echo "    - PID: $new_pid"
        echo "    - Address: http://$HOST:$PORT"
        echo "    - Log file: $LOG_FILE"
        echo "    - Check status: ./daemon.sh status"
        echo "    - View logs: ./daemon.sh logs"
    else
        echo "[-] Failed to start daemon. Last logs:"
        tail -n 20 "$LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop_daemon() {
    if ! is_running; then
        echo "[*] Daemon is not running."
        rm -f "$PID_FILE"
        return 0
    fi

    local pid
    pid="$(cat "$PID_FILE")"
    echo "[*] Stopping daemon (PID: $pid)..."

    kill -15 "$pid" 2>/dev/null || true

    local count=0
    while kill -0 "$pid" 2>/dev/null && [ $count -lt 15 ]; do
        sleep 1
        count=$((count + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        echo "[!] Process did not exit gracefully, sending SIGKILL..."
        kill -9 "$pid" 2>/dev/null || true
    fi

    rm -f "$PID_FILE"
    echo "[+] Daemon stopped."
}

status_daemon() {
    if is_running; then
        local pid
        pid="$(cat "$PID_FILE")"
        echo "[+] Daemon status: RUNNING (PID: $pid)"
        echo "    - Port: $PORT"
        echo "    - Log: $LOG_FILE"
        
        # Test HTTP health check on docs
        local http_code
        http_code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/docs" 2>/dev/null || echo "000")
        if [ "$http_code" != "000" ]; then
            echo "    - API Endpoint: HEALTHY (HTTP $http_code on /docs)"
        else
            echo "    - API Endpoint: Starting up or unreachable"
        fi
    else
        echo "[-] Daemon status: STOPPED"
    fi
}

show_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        touch "$LOG_FILE"
    fi
    echo "[*] Following $LOG_FILE (Press Ctrl+C to exit)..."
    tail -f "$LOG_FILE"
}

case "${1:-}" in
    start)
        start_daemon
        ;;
    stop)
        stop_daemon
        ;;
    restart)
        stop_daemon
        sleep 1
        start_daemon
        ;;
    status)
        status_daemon
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  start   : Start the agy2api server in the background (no GUI, no npm)"
        echo "  stop    : Gracefully stop the running background daemon"
        echo "  restart : Restart the daemon"
        echo "  status  : Check daemon process and API health status"
        echo "  logs    : Follow live output logs"
        exit 1
        ;;
esac
