#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/../models"

HF_CACHE_DIR="$HOME/.cache/huggingface/hub"
HF_URI="Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF:Q8_0"

# Scan available models from multiple sources
scan_models() {
  local -a models
  
  # Source 1: local models/ directory
  while IFS= read -r f; do
    [[ -f "$f" ]] && models+=("$f")
  done < <(find "$MODEL_DIR" -maxdepth 1 -name "*.gguf" 2>/dev/null)
  
  # Source 2: HuggingFace cache — scan all models--*/ dirs, resolve via refs/main
  for repo_dir in "$HF_CACHE_DIR"/models--*/; do
    [[ -d "$repo_dir" ]] || continue
    local refs_file="${repo_dir}refs/main"
    [[ -f "$refs_file" ]] || continue
    local commit_hash
    commit_hash=$(tr -d '[:space:]' < "$refs_file")
    [[ -n "$commit_hash" ]] || continue
    local snapshot_dir="${repo_dir}snapshots/${commit_hash}"
    [[ -d "$snapshot_dir" ]] || continue
    while IFS= read -r f; do
      local fname
      fname=$(basename "$f")
      [[ "$fname" == mmproj-* ]] && continue       # skip vision projector files
      [[ "$fname" == *-projector* ]] && continue   # skip alternate projector naming
      [[ -f "$f" ]] && models+=("$f")              # -f follows symlinks, confirms blob exists
    done < <(find "$snapshot_dir" -name "*.gguf" 2>/dev/null)
  done
  
  printf '%s\n' "${models[@]}"
}

# Display interactive model selector
select_model() {
  local -a available_models
  local choice
  
  # Collect all available models
  while IFS= read -r model; do
    [[ -n "$model" ]] && available_models+=("$model")
  done < <(scan_models | sort)
  
  # Debug: show how many models found
  if (( ${#available_models[@]} == 0 )); then
    echo "❌ No models found locally."
    echo ""
    echo "Would you like to download Qwen3.5-9B Q8_0 (~9.5 GB)?"
    read -p "Download? (y/n): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      echo "Run: make download"
      exit 0
    else
      echo "Run 'make download' to fetch the model, then try again."
      exit 1
    fi
  fi
  
  # Single model: use it directly
  if (( ${#available_models[@]} == 1 )); then
    echo "${available_models[0]}"
    return
  fi
  
  # Multiple models: show selection menu
  {
    echo "📦 Available models:"
    echo ""
    local i
    for i in "${!available_models[@]}"; do
      local model_path="${available_models[i]}"
      local size=$(stat -Lf%z "$model_path" 2>/dev/null | awk '{printf("%.1f GB", $1/1024/1024/1024)}' || echo "? GB")
      printf "  [%d] %s (%s)\n" "$((i+1))" "$model_path" "$size"
    done
    echo "  [0] Download new model (Qwen3.5-9B Q8_0, ~9.5 GB)"
    echo ""
  } >&2
  
  # Read user choice
  read -p "Select model (0-$((${#available_models[@]}))): " -r choice
  
  # Handle download option
  if [[ "$choice" == "0" ]]; then
    echo "Run: make download"
    exit 0
  fi
  
  # Validate choice
  if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#available_models[@]} )); then
    echo "❌ Invalid selection."
    exit 1
  fi
  
  # Return selected model
  echo "${available_models[$((choice-1))]}"
}

COMMON_FLAGS=(
  --n-gpu-layers 99        # load all layers onto Metal GPU
  --ctx-size 786432        # context window = ctx-size / parallel; 786432 / 3 ≈ 256K context window
  --parallel 3             # 4 slots × 16K ctx × q8_0 KV ≈ 1.9 GB; model ≈ 9.5 GB → ~11.4 GB total (fits 24 GB Metal)
  --cache-type-k q4_0      # quantised KV cache: halves KV memory, negligible quality loss
  --cache-type-v q4_0
  --flash-attn auto        # auto-detect Flash Attention support
  --embedding
  --host 127.0.0.1
  --port 50000
)

MODEL_FILE=$(select_model)
MODEL_NAME=$(basename "$MODEL_FILE" .gguf)

echo "Model   : $MODEL_NAME"
echo "Endpoint: http://127.0.0.1:50000"
echo ""
exec llama-server --model "$MODEL_FILE" --alias "$MODEL_NAME" "${COMMON_FLAGS[@]}"
