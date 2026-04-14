#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/../config/litellm.yaml"

# ── Security check ───────────────────────────────────────────────────────────
# LiteLLM 1.82.7 and 1.82.8 were compromised with credential-stealing malware.
# https://github.com/BerriAI/litellm/issues/24518
LITELLM_VERSION=$(pip show litellm 2>/dev/null | awk '/^Version:/ {print $2}' || echo "")

if [[ -z "$LITELLM_VERSION" ]]; then
  echo "LiteLLM not installed."
  echo "Run: pip install -U 'litellm[proxy]'"
  exit 1
fi

if [[ "$LITELLM_VERSION" == "1.82.7" || "$LITELLM_VERSION" == "1.82.8" ]]; then
  echo "SECURITY ALERT: LiteLLM $LITELLM_VERSION is COMPROMISED (malware)."
  echo "Remove: pip uninstall litellm"
  echo "Rotate ALL credentials on this machine."
  echo "See: https://github.com/BerriAI/litellm/issues/24518"
  exit 1
fi

# ── Pre-flight: check llama-server is running ─────────────────────────────────
if ! curl -sf http://127.0.0.1:18888/health &>/dev/null; then
  echo "llama-server is not running on port 18888."
  echo "Start it first: make serve  (or open a new terminal)"
  exit 1
fi

echo "LiteLLM $LITELLM_VERSION starting..."
echo "Anthropic-compatible endpoint: http://127.0.0.1:4000"
echo ""

exec litellm --config "$CONFIG" \
  --host 127.0.0.1 \
  --port 4000 \
  --detailed_debug
