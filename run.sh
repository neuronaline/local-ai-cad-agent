#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*" >&2; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*" >&2; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: ./run.sh

Starts Local AI CAD Agent at the host and port configured in config.yaml.

Before the first run:
  ./install.sh
EOF
}

case "${1:-}" in
  "") ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
esac

cd "$PROJECT_DIR"

# ── Configuration files ──
if [ ! -f .env ]; then
    warn ".env is missing. Creating from .env.example…"
    if [ -f .env.example ]; then
        cp .env.example .env
        info ".env created. Set the API key for your selected provider there."
    else
        warn ".env.example not found. Create .env with the API key for your selected provider manually."
    fi
fi

if [ ! -f config.yaml ]; then
    warn "config.yaml is missing. Creating from config.example.yaml…"
    if [ -f config.example.yaml ]; then
        cp config.example.yaml config.yaml
        info "config.yaml created with default settings."
    else
        error "config.example.yaml not found. The repository may be corrupted."
    fi
fi

# ── Virtual environment ──
if [ ! -x "$PYTHON_BIN" ]; then
    error "Virtual environment not found. Run ./install.sh first, or create it manually:
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt"
fi

# ── System dependencies ──
if ! command -v bwrap >/dev/null 2>&1; then
    error "bubblewrap (bwrap) is required for sandboxed CAD execution. Install it with:
  sudo apt install bubblewrap"
fi

if ! "$PYTHON_BIN" -c 'import ctypes; ctypes.CDLL("libseccomp.so.2")' 2>/dev/null; then
    warn "libseccomp is not available. CAD sandbox security filters will fail."
    warn "Install it with: sudo apt install libseccomp2"
fi

# ── API key ──
# Python loads .env via dotenv; this shell only checks the selected provider's key.
LLM_PROVIDER="$("$PYTHON_BIN" -c 'from agent.settings import load_settings; print(load_settings().llm_provider)')"
if [ "$LLM_PROVIDER" = "openai" ]; then
    API_KEY_NAME="OPENAI_API_KEY"
else
    API_KEY_NAME="OPENROUTER_API_KEY"
fi
if [ -f "$PROJECT_DIR/.env" ] \
    && ! grep -qE "^[[:space:]]*${API_KEY_NAME}[[:space:]]*=[[:space:]]*[^[:space:]]" "$PROJECT_DIR/.env" 2>/dev/null; then
    warn "${API_KEY_NAME} is not configured. The setup page will guide you on first launch."
fi

# ── Port check ──
HOST="$("$PYTHON_BIN" -c 'from agent.settings import load_settings; s = load_settings(); print(s.host)')"
PORT="$("$PYTHON_BIN" -c 'from agent.settings import load_settings; s = load_settings(); print(s.port)')"

if command -v ss >/dev/null 2>&1; then
    if ss -tlnH "sport = :$PORT" 2>/dev/null | grep -q ":$PORT"; then
        warn "Port $PORT is already in use. The server may fail to start."
    fi
elif command -v lsof >/dev/null 2>&1; then
    if lsof -i ":$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        warn "Port $PORT is already in use. The server may fail to start."
    fi
fi

# ── Resolve display URL ──
if [ "$HOST" = "0.0.0.0" ] || [ "$HOST" = "::" ]; then
    DISPLAY_URL="http://localhost:${PORT}"
else
    DISPLAY_URL="http://${HOST}:${PORT}"
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Local AI CAD Agent${NC}"
echo -e "${GREEN}  ${DISPLAY_URL}${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

exec env PYTHONUNBUFFERED=1 "$PYTHON_BIN" app.py
