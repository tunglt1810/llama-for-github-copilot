.PHONY: all install download serve proxy install-continue check check-proxy build run help

SHELL := /bin/bash
MODEL_DIR       := models
CONTINUE_DIR    := $(HOME)/.continue
SCRIPTS_DIR     := scripts
CONFIG_DIR      := config
BIN_DIR         := bin

# ─────────────────────────────────────────────────────────────────────────────
# Default: show help
# ─────────────────────────────────────────────────────────────────────────────
help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─────────────────────────────────────────────────────────────────────────────
# Go binary — Ollama proxy + llama-server launcher
# ─────────────────────────────────────────────────────────────────────────────
build:  ## Build the llama-proxy Go binary → bin/llama-proxy
	@mkdir -p $(BIN_DIR)
	go build -o $(BIN_DIR)/llama-proxy ./cmd/llama-proxy/

run: build  ## Build & run llama-proxy (selects model interactively, starts Ollama proxy on :11434)
	@$(BIN_DIR)/llama-proxy

# Start llama-wrapper in foreground with sensible defaults from .vscode/launch.json
start: build  ## Start llama-wrapper (foreground)
	@PORT=11434 RUN_MODE=llama-server-wrapper UPSTREAM_BASE="http://127.0.0.1:50000" LLAMA_SERVER_ARGS__HOST="127.0.0.1" LLAMA_SERVER_ARGS__PORT="50000" $(BIN_DIR)/llama-proxy

# ─────────────────────────────────────────────────────────────────────────────
# Installation
# ─────────────────────────────────────────────────────────────────────────────
install:  ## Install all dependencies (llama.cpp, huggingface-hub, litellm)
	brew install llama.cpp
	pip install -U "huggingface_hub[cli]"
	pip install -U "litellm[proxy]"
	@echo ""
	@echo "Verifying LiteLLM version (1.82.7 and 1.82.8 are compromised)..."
	@pip show litellm | grep Version
	chmod +x $(SCRIPTS_DIR)/*.sh
	@echo ""
	@echo "Installation complete. Run: make download"

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
download:  ## Download Qwen3.5-9B Q8_0 (~9.53 GB) from HuggingFace
	@bash $(SCRIPTS_DIR)/download-models.sh

# ─────────────────────────────────────────────────────────────────────────────
# Services  (run each in its own terminal tab)
# ─────────────────────────────────────────────────────────────────────────────
serve:  ## Start llama-server with Metal acceleration (port 18888) — legacy bash script
	@bash $(SCRIPTS_DIR)/start-server.sh

proxy:  ## Start LiteLLM proxy for Claude Code (port 4000)
	@bash $(SCRIPTS_DIR)/start-litellm.sh

# ─────────────────────────────────────────────────────────────────────────────
# VS Code integration
# ─────────────────────────────────────────────────────────────────────────────
install-continue:  ## Copy Continue.dev config to ~/.continue/config.yaml
	mkdir -p $(CONTINUE_DIR)
	cp $(CONFIG_DIR)/continue/config.yaml $(CONTINUE_DIR)/config.yaml
	@echo "Continue.dev config installed → $(CONTINUE_DIR)/config.yaml"
	@echo "Reload VS Code to apply."

# ─────────────────────────────────────────────────────────────────────────────
# Health checks
# ─────────────────────────────────────────────────────────────────────────────
check:  ## Check llama-server is responding (port 18888)
	@echo "=== llama-server /health ===" && \
		curl -sf http://127.0.0.1:18888/health | python3 -m json.tool && \
		echo "" && echo "=== Available models ===" && \
		curl -sf http://127.0.0.1:18888/v1/models | python3 -m json.tool

check-proxy:  ## Check LiteLLM proxy is responding (port 4000)
	@echo "=== LiteLLM /health ===" && \
		curl -sf http://127.0.0.1:4000/health | python3 -m json.tool

test-inference:  ## Send a quick test prompt to llama-server
	@curl -s http://127.0.0.1:18888/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{ \
			"model": "qwen3.5-9b-reasoning", \
			"messages": [{"role": "user", "content": "Reply with only: OK"}], \
			"max_tokens": 64, \
			"stream": false \
		}' | python3 -m json.tool
