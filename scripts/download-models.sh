#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/../models"

REPO="Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
QUANT="Q8_0"

# Check dependency
if ! command -v huggingface-cli &>/dev/null; then
  echo "huggingface-cli not found. Install: pip install -U 'huggingface_hub[cli]'"
  exit 1
fi

mkdir -p "$MODEL_DIR"

# Check HF cache first (populated automatically by: llama-server -hf <URI>)
# No need to copy — llama-server reads directly from snapshots.
HF_CACHE_DIR="$HOME/.cache/huggingface/hub"
HF_MODEL_DIR="$HF_CACHE_DIR/models--Jackrong--Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
HF_CACHED=""
if [[ -f "$HF_MODEL_DIR/refs/main" ]]; then
  _commit=$(tr -d '[:space:]' < "$HF_MODEL_DIR/refs/main")
  HF_CACHED=$(find "$HF_MODEL_DIR/snapshots/${_commit}" -name "*${QUANT}*.gguf" 2>/dev/null | head -1 || true)
  unset _commit
fi
if [[ -n "$HF_CACHED" ]]; then
  echo "Model already in HuggingFace cache:"
  echo "  $HF_CACHED"
  echo "No download needed. Run: make serve"
  exit 0
fi

# Check models/ dir
EXISTING=$(find "$MODEL_DIR" -maxdepth 1 -name "*${QUANT}*.gguf" 2>/dev/null | head -1 || true)
if [[ -n "$EXISTING" ]]; then
  echo "Model already exists: $EXISTING"
  echo "To re-download, delete the file and run again."
  exit 0
fi

echo "Downloading ${QUANT} (~9.53 GB) from HuggingFace..."
echo "  Repo : $REPO"
echo "  Dest : $MODEL_DIR"
echo ""

huggingface-cli download "$REPO" \
  --include "*${QUANT}*.gguf" \
  --local-dir "$MODEL_DIR" \
  --local-dir-use-symlinks False

DOWNLOADED=$(find "$MODEL_DIR" -maxdepth 1 -name "*${QUANT}*.gguf" | head -1)
echo ""
echo "Done: $DOWNLOADED"
