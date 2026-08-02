#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

usage() {
  cat <<'EOF'
Usage: ./run.sh

Starts Local AI CAD Agent at the host and port configured in config.yaml.

Before the first run:
  sudo apt install bubblewrap libseccomp2
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
  cp .env.example .env
  cp config.example.yaml config.yaml
  # Set OPENROUTER_API_KEY in .env
EOF
}

case "${1:-}" in
  "") ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Virtual environment not found. Create it with: python3 -m venv .venv" >&2
  echo "Then install dependencies with: .venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi

if ! command -v bwrap >/dev/null 2>&1; then
  echo "bubblewrap (bwrap) is required for sandboxed CAD execution." >&2
  exit 1
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  if [[ -f "$PROJECT_DIR/.env" ]] && grep -qE '^[[:space:]]*OPENROUTER_API_KEY[[:space:]]*=' "$PROJECT_DIR/.env"; then
    echo "Using OPENROUTER_API_KEY from .env." >&2
  else
    echo "Warning: OPENROUTER_API_KEY is not configured. Copy .env.example to .env before using AI chat." >&2
  fi
fi

cd "$PROJECT_DIR"
APP_ADDRESS="$("$PYTHON_BIN" -c 'from agent.settings import load_settings; s = load_settings(); print(f"http://{s.host}:{s.port}")')"
echo "Starting Local AI CAD Agent. Open $APP_ADDRESS" >&2
exec env PYTHONUNBUFFERED=1 "$PYTHON_BIN" app.py
