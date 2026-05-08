#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${CASIE_PROJECT_DIR:-$HOME/Desktop/cas-e}"
APP_DIR="$PROJECT_DIR/casie_direct"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/casie_desktop.log"

mkdir -p "$LOG_DIR"
exec >> "$LOG_FILE" 2>&1

echo "----- $(date) starting CASie desktop -----"
echo "Project: $PROJECT_DIR"

if [[ ! -f "$APP_DIR/desktop_ui.py" ]]; then
  echo "ERROR: desktop_ui.py not found at $APP_DIR"
  exit 1
fi

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export CASIE_PI_COMPAT_ENABLED="${CASIE_PI_COMPAT_ENABLED:-true}"

if command -v xset >/dev/null 2>&1; then
  for _ in $(seq 1 30); do
    if xset q >/dev/null 2>&1; then
      break
    fi
    echo "Waiting for desktop display..."
    sleep 1
  done
fi

cd "$PROJECT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
elif [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
else
  echo "WARNING: no virtual environment found; using system python3"
fi

cd "$APP_DIR"

PYTHON_BIN="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python3}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"

if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: python3 not found"
  exit 1
fi

echo "Python: $PYTHON_BIN"
exec "$PYTHON_BIN" desktop_ui.py
