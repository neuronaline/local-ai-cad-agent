#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*" >&2; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*" >&2; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Check Python ──
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        full_version="$("$candidate" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null || true)"
        major="${full_version%%.*}"
        minor="${full_version#*.}"
        if [ "$major" -gt 3 ] || { [ "$major" -eq 3 ] && [ "${minor:-0}" -ge 10 ]; }; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    error "Python 3.10 or newer is required but was not found. Install it with: sudo apt install python3.12"
fi

info "Using Python: $PYTHON_BIN ($full_version)"

# ── Install system packages ──
# Use apt if available; for other Debian/Ubuntu variants (or manual users) the
# script will still proceed gracefully and report missing dependencies later.
SYSTEM_PACKAGES=()
command -v bwrap     >/dev/null 2>&1 || SYSTEM_PACKAGES+=("bubblewrap")
dpkg -s libseccomp2  >/dev/null 2>&1 || SYSTEM_PACKAGES+=("libseccomp2")
command -v python3   >/dev/null 2>&1 || SYSTEM_PACKAGES+=("python3")
dpkg -s python3-venv >/dev/null 2>&1 || SYSTEM_PACKAGES+=("python3-venv")

if [ ${#SYSTEM_PACKAGES[@]} -gt 0 ]; then
    if command -v apt-get >/dev/null 2>&1; then
        warn "Installing system packages: ${SYSTEM_PACKAGES[*]}"
        sudo apt-get update -qq
        sudo apt-get install -y -qq "${SYSTEM_PACKAGES[@]}"
        info "System packages installed."
    else
        warn "apt-get not available. Please install these packages manually: ${SYSTEM_PACKAGES[*]}"
    fi
else
    info "All required system packages are installed."
fi

# ── Virtual environment ──
if [ ! -d .venv ]; then
    info "Creating Python virtual environment…"
    "$PYTHON_BIN" -m venv .venv
    info "Virtual environment created."
else
    info "Virtual environment already exists."
fi

VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
VENV_PIP="$PROJECT_DIR/.venv/bin/pip"

if [ ! -x "$VENV_PYTHON" ]; then
    error "Virtual environment Python not found at $VENV_PYTHON."
fi

# ── Install Python dependencies ──
info "Installing Python dependencies from requirements.txt…"
"$VENV_PIP" install --upgrade pip -q
"$VENV_PIP" install -r requirements.txt -q
info "Python dependencies installed."

# ── Configuration files ──
if [ ! -f .env ]; then
    cp .env.example .env
    info ".env created from .env.example. Add your OPENROUTER_API_KEY inside."
else
    info ".env already exists — left unchanged."
fi

if [ ! -f config.yaml ]; then
    cp config.example.yaml config.yaml
    info "config.yaml created from config.example.yaml."
else
    info "config.yaml already exists — left unchanged."
fi

# ── Done ──
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Local AI CAD Agent installed successfully!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "  Next steps:"
echo "  1. Set your OpenRouter API key in .env:"
echo "     OPENROUTER_API_KEY=sk-or-v1-..."
echo ""
echo "  2. (Optional) Customize the model in config.yaml"
echo ""
echo "  3. Start the application:"
echo "     ${GREEN}./run.sh${NC}"
echo ""
if [ ! -s .env ] || ! grep -qE '^[[:space:]]*OPENROUTER_API_KEY[[:space:]]*=[[:space:]]*[^[:space:]]' .env; then
    warn "OPENROUTER_API_KEY is empty in .env. The setup page will guide you on first launch."
fi
