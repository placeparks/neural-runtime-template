#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/.neuralclaw" "$HOME/.neuralclaw/data" "$HOME/.neuralclaw/logs"

# Install NeuralClaw and channel adapters at container boot.
# Pin major/minor if you want deterministic upgrades.
pip install "neuralclaw[all-channels]" "aiohttp"

AGENT_NAME="${NEURALCLAW_AGENT_NAME:-NeuralClaw}"
PROVIDER="${NEURALCLAW_PROVIDER:-openai}"
MODEL="${NEURALCLAW_MODEL:-gpt-4o}"
MESH_ENABLED_RAW="${NEURALCLAW_MESH_ENABLED:-false}"
MESH_PEERS_JSON="${NEURALCLAW_MESH_PEERS_JSON:-}"
ENABLE_DASHBOARD_RAW="${NEURALCLAW_ENABLE_DASHBOARD:-false}"

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

if [[ -n "$MESH_PEERS_JSON" ]]; then
  # Persist for runtime adapters/debug; current upstream gateway does not auto-consume this file yet.
  printf '%s\n' "$MESH_PEERS_JSON" > "$HOME/.neuralclaw/mesh-peers.json"
fi

cat > "$HOME/.neuralclaw/config.toml" <<EOF
[general]
name = "${AGENT_NAME}"
persona = "You are NeuralClaw, a helpful and intelligent AI assistant."
log_level = "INFO"
telemetry_stdout = true

[providers]
primary = "${PROVIDER}"
fallback = ["local"]

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
base_url = "http://localhost:11434/v1"

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
