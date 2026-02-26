#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/.neuralclaw" "$HOME/.neuralclaw/data" "$HOME/.neuralclaw/logs"

# Prefer Railway volume for WhatsApp auth/session persistence across redeploys.
# If /data is mounted and caller did not set a custom path, use /data/whatsapp.
if [[ -z "${NEURALCLAW_WHATSAPP_SESSION_DIR:-}" && -d "/data" ]]; then
  export NEURALCLAW_WHATSAPP_SESSION_DIR="/data/whatsapp"
fi
if [[ -n "${NEURALCLAW_WHATSAPP_SESSION_DIR:-}" ]]; then
  mkdir -p "${NEURALCLAW_WHATSAPP_SESSION_DIR}"
fi

AGENT_NAME="${NEURALCLAW_AGENT_NAME:-NeuralClaw}"
PROVIDER="${NEURALCLAW_PROVIDER:-openai}"
MODEL="${NEURALCLAW_MODEL:-gpt-4o}"
MESH_ENABLED_RAW="${NEURALCLAW_MESH_ENABLED:-false}"
MESH_PEERS_JSON="${NEURALCLAW_MESH_PEERS_JSON:-}"
ENABLE_DASHBOARD_RAW="${NEURALCLAW_ENABLE_DASHBOARD:-false}"
LOCAL_URL="${NEURALCLAW_LOCAL_URL:-}"
# Sanitize persona: replace double-quotes with single-quotes for TOML safety
PERSONA="${NEURALCLAW_PERSONA:-You are NeuralClaw, a helpful and intelligent AI assistant.}"
PERSONA="${PERSONA//\"/\'}"

to_bool() {
  local v="${1:-false}"
  shopt -s nocasematch
  if [[ "$v" == "1" || "$v" == "true" || "$v" == "yes" || "$v" == "on" ]]; then
    echo "true"
  else
    echo "false"
  fi
  shopt -u nocasematch
}

MESH_ENABLED="$(to_bool "$MESH_ENABLED_RAW")"
ENABLE_DASHBOARD="$(to_bool "$ENABLE_DASHBOARD_RAW")"

# Build the fallback list: only include "local" if an Ollama URL is explicitly provided.
if [[ -n "$LOCAL_URL" ]]; then
  FALLBACK_TOML='fallback = ["local"]'
else
  FALLBACK_TOML='fallback = []'
fi

if [[ -n "$MESH_PEERS_JSON" ]]; then
  # Persist for runtime adapters/debug; current upstream gateway does not auto-consume this file yet.
  printf '%s\n' "$MESH_PEERS_JSON" > "$HOME/.neuralclaw/mesh-peers.json"
fi

KNOWLEDGE_CONTENT="${NEURALCLAW_KNOWLEDGE_CONTENT:-}"
if [[ -n "$KNOWLEDGE_CONTENT" ]]; then
  printf '%s\n' "$KNOWLEDGE_CONTENT" > "$HOME/.neuralclaw/knowledge.txt"
  echo "[runtime] knowledge base written (${#KNOWLEDGE_CONTENT} bytes)"
fi

cat > "$HOME/.neuralclaw/config.toml" <<EOF
[general]
name = "${AGENT_NAME}"
persona = "${PERSONA}"
log_level = "INFO"
telemetry_stdout = true

[providers]
primary = "${PROVIDER}"
${FALLBACK_TOML}

[providers.openai]
model = "${MODEL}"
base_url = "https://api.openai.com/v1"

[providers.anthropic]
model = "${MODEL}"
base_url = "https://api.anthropic.com"

[providers.openrouter]
model = "${MODEL}"
base_url = "https://openrouter.ai/api/v1"

[providers.local]
model = "${MODEL}"
base_url = "${LOCAL_URL:-http://localhost:11434/v1}"

[memory]
db_path = "${HOME}/.neuralclaw/data/memory.db"
max_episodic_results = 10
max_semantic_results = 5
importance_threshold = 0.3

[security]
threat_threshold = 0.7
block_threshold = 0.9
allow_shell_execution = false

[features]
swarm = ${MESH_ENABLED}
dashboard = ${ENABLE_DASHBOARD}
evolution = false
reflective_reasoning = true
procedural_memory = true
semantic_memory = true

[channels.telegram]
enabled = true

[channels.discord]
enabled = true
EOF

echo "[runtime] agent=${AGENT_NAME} provider=${PROVIDER} model=${MODEL} mesh_enabled=${MESH_ENABLED} dashboard=${ENABLE_DASHBOARD}"
if [[ -f "$HOME/.neuralclaw/mesh-peers.json" ]]; then
  echo "[runtime] mesh peers file written: $HOME/.neuralclaw/mesh-peers.json"
fi

# Railway sets PORT for web-facing services; NeuralClaw gateway has its own channel listeners.
# This process is long-running.
exec python /app/mesh_gateway.py
