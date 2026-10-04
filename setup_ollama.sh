#!/usr/bin/env bash
# Helper script to prepare an Ollama model with a 16k context window for Local AI CAD Agent.
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

echo -e "${CYAN}====================================================${NC}"
echo -e "${CYAN}  Local AI CAD Agent - Ollama Model Setup Helper    ${NC}"
echo -e "${CYAN}====================================================${NC}"
echo ""

# 1. Check if ollama command is installed
if ! command -v ollama >/dev/null 2>&1; then
    error "Ollama CLI is not installed. Install it via: curl -fsSL https://ollama.com/install.sh | sh"
fi

# 2. Check if Ollama server is running
if ! curl -s -f -m 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
    warn "Ollama server does not appear to be running on http://localhost:11434."
    warn "Start it in another terminal with: ollama serve"
    echo ""
    if [ -t 0 ]; then
        read -r -p "Press Enter to continue once Ollama is running, or Ctrl+C to abort..."
    else
        error "Cannot proceed non-interactively while Ollama server is not running."
    fi
fi

# 3. Model selection
DEFAULT_MODEL="qwen2.5-coder:14b"
BASE_MODEL="${1:-$DEFAULT_MODEL}"
CONTEXT_SIZE="${2:-16384}"

if [ "$CONTEXT_SIZE" -ge 1024 ] 2>/dev/null; then
    CTX_LABEL="$((CONTEXT_SIZE / 1024))k"
else
    CTX_LABEL="${CONTEXT_SIZE}"
fi
CUSTOM_MODEL="${BASE_MODEL}-${CTX_LABEL}"

info "Selected base model: ${CYAN}${BASE_MODEL}${NC}"
info "Target context size: ${CYAN}${CONTEXT_SIZE} tokens${NC}"
info "Custom model name:   ${CYAN}${CUSTOM_MODEL}${NC}"
echo ""

# 4. Pull base model
info "Pulling base model '${BASE_MODEL}' (this may take a few minutes depending on your internet connection)..."
ollama pull "${BASE_MODEL}"
info "Base model '${BASE_MODEL}' ready."

# 5. Create 16k context variant
TMP_MODELFILE="$(mktemp /tmp/Modelfile.cad.XXXXXX)"
trap 'rm -f "$TMP_MODELFILE"' EXIT

cat <<EOF > "$TMP_MODELFILE"
FROM ${BASE_MODEL}
PARAMETER num_ctx ${CONTEXT_SIZE}
PARAMETER temperature 0.2
EOF

info "Creating custom model '${CUSTOM_MODEL}' with num_ctx=${CONTEXT_SIZE}..."
ollama create "${CUSTOM_MODEL}" -f "$TMP_MODELFILE"
info "Custom model '${CUSTOM_MODEL}' created successfully!"

echo ""
echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN}  Setup Complete!                                   ${NC}"
echo -e "${GREEN}====================================================${NC}"
echo ""
echo "To use this model in Local AI CAD Agent, ensure your config.yaml has:"
echo ""
echo -e "${CYAN}llm:"
echo "  provider: ollama"
echo ""
echo "ollama:"
echo "  base_url: http://localhost:11434/v1"
echo "  model: ${CUSTOM_MODEL}"
echo -e "  timeout_seconds: 120${NC}"
echo ""
echo "Then start the agent with: ./run.sh"
