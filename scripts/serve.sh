#!/usr/bin/env bash
# Run API (8000) and UI (8501) from a SNAPSHOT of the project, so editing the working copy
# (or running uv sync, eval, tests) never breaks a service someone is testing.
#
#   scripts/serve.sh start    copy the working tree to ~/minichem-run/app, restart both services
#   scripts/serve.sh stop
#   scripts/serve.sh status
#
# The snapshot has its own .venv; `.env` is symlinked, never copied or read. Streamlit's file
# watcher is off, so nothing reloads until `start` is run again.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="${MINICHEM_RUN_DIR:-$HOME/minichem-run}"
APP="$RUN_ROOT/app"
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"

pid_alive() { [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; }

stop_one() {
  local pidfile="$RUN_ROOT/$1.pid"
  if pid_alive "$pidfile"; then
    kill "$(cat "$pidfile")"
    while kill -0 "$(cat "$pidfile")" 2>/dev/null; do sleep 0.2; done
  fi
  rm -f "$pidfile"
}

port_busy() { lsof -iTCP:"$1" -sTCP:LISTEN -n -P >/dev/null 2>&1; }

cmd_stop() { stop_one ui; stop_one api; echo "stopped"; }

cmd_start() {
  cmd_stop
  for port in "$API_PORT" "$UI_PORT"; do
    if port_busy "$port"; then
      echo "port $port is used by another process (lsof -iTCP:$port -sTCP:LISTEN); stop it first" >&2
      exit 1
    fi
  done
  mkdir -p "$APP" "$RUN_ROOT/logs"
  rsync -a --delete \
    --exclude .git --exclude .venv --exclude .env --exclude logs --exclude .cache \
    --exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache --exclude eval/results \
    "$SRC/" "$APP/"
  if [[ -f "$SRC/.env" ]]; then ln -sf "$SRC/.env" "$APP/.env"; fi
  (cd "$APP" && uv sync --quiet)
  # exec: the background subshell becomes the server, so $! is the server's pid and `stop` kills it
  # (without exec the pid file held the subshell and the server kept running after `stop`).
  (cd "$APP" && exec nohup .venv/bin/python -m uvicorn app.api:app --host 127.0.0.1 \
    --port "$API_PORT" >"$RUN_ROOT/logs/api.log" 2>&1) &
  echo $! >"$RUN_ROOT/api.pid"
  (cd "$APP" && exec nohup .venv/bin/python -m streamlit run ui/app.py --server.port "$UI_PORT" \
    --server.headless true --server.fileWatcherType none \
    >"$RUN_ROOT/logs/ui.log" 2>&1) &
  echo $! >"$RUN_ROOT/ui.pid"
  for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then break; fi
    sleep 1
  done
  cmd_status
}

cmd_status() {
  for name in api ui; do
    if pid_alive "$RUN_ROOT/$name.pid"; then
      echo "$name: running (pid $(cat "$RUN_ROOT/$name.pid"))"
    else
      echo "$name: stopped"
    fi
  done
  if curl -fsS "http://127.0.0.1:$API_PORT/health" 2>/dev/null; then echo; else echo "api /health: no answer"; fi
  echo "snapshot: $APP   logs: $RUN_ROOT/logs"
}

case "${1:-}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
