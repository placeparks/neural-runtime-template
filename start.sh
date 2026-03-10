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
PROXY_BASE_URL="${NEURALCLAW_PROXY_BASE_URL:-}"
PROXY_API_KEY="${NEURALCLAW_PROXY_API_KEY:-}"
ALLOWED_TOOLS_RAW="${NEURALCLAW_ALLOWED_TOOLS:-}"
MESH_ENABLED_RAW="${NEURALCLAW_MESH_ENABLED:-false}"
MESH_PEERS_JSON="${NEURALCLAW_MESH_PEERS_JSON:-}"
ENABLE_DASHBOARD_RAW="${NEURALCLAW_ENABLE_DASHBOARD:-false}"
ENABLE_EVOLUTION_RAW="${NEURALCLAW_ENABLE_EVOLUTION:-false}"
ENABLE_REFLECTIVE_RAW="${NEURALCLAW_REFLECTIVE_REASONING:-true}"
LOCAL_URL="${NEURALCLAW_LOCAL_URL:-}"
VOICE_ENABLED_RAW="${NEURALCLAW_VOICE_ENABLED:-false}"
VOICE_PROVIDER="${NEURALCLAW_VOICE_PROVIDER:-twilio}"
VOICE_REQUIRE_CONFIRM_RAW="${NEURALCLAW_VOICE_REQUIRE_CONFIRM:-true}"
TWILIO_ACCOUNT_SID="${TWILIO_ACCOUNT_SID:-}"
TWILIO_AUTH_TOKEN="${TWILIO_AUTH_TOKEN:-}"
TWILIO_PHONE_NUMBER="${TWILIO_PHONE_NUMBER:-}"
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

# Normalize session providers → proxy provider
SESSION_PROVIDER=""
if [[ "$PROVIDER" == "chatgpt_session" ]]; then
  SESSION_PROVIDER="chatgpt"
  PROVIDER="proxy"
elif [[ "$PROVIDER" == "claude_session" ]]; then
  SESSION_PROVIDER="claude"
  PROVIDER="proxy"
fi
# If SESSION_PROVIDER env was set explicitly (by provisioner), respect it
if [[ -n "${NEURALCLAW_SESSION_PROVIDER:-}" && -z "$SESSION_PROVIDER" ]]; then
  SESSION_PROVIDER="${NEURALCLAW_SESSION_PROVIDER}"
fi

MESH_ENABLED="$(to_bool "$MESH_ENABLED_RAW")"
ENABLE_DASHBOARD="$(to_bool "$ENABLE_DASHBOARD_RAW")"
ENABLE_EVOLUTION="$(to_bool "$ENABLE_EVOLUTION_RAW")"
ENABLE_REFLECTIVE="$(to_bool "$ENABLE_REFLECTIVE_RAW")"
VOICE_ENABLED="$(to_bool "$VOICE_ENABLED_RAW")"
VOICE_REQUIRE_CONFIRM="$(to_bool "$VOICE_REQUIRE_CONFIRM_RAW")"

# Resolve proxy model: session providers have a canonical default; otherwise use $MODEL.
if [[ "$SESSION_PROVIDER" == "chatgpt" ]]; then
  PROXY_MODEL="${MODEL:-gpt-4o}"
elif [[ "$SESSION_PROVIDER" == "claude" ]]; then
  PROXY_MODEL="${MODEL:-claude-sonnet-4-20250514}"
else
  PROXY_MODEL="${MODEL}"
fi

# Build the fallback list: only include "local" if an Ollama URL is explicitly provided.
if [[ -n "$LOCAL_URL" ]]; then
  FALLBACK_TOML='fallback = ["local"]'
else
  FALLBACK_TOML='fallback = []'
fi

POLICY_ALLOWED_TOOLS_TOML=""
if [[ -n "$ALLOWED_TOOLS_RAW" ]]; then
  IFS=',' read -r -a _TOOLS_ARRAY <<< "$ALLOWED_TOOLS_RAW"
  _TOOLS_CLEAN=()
  for _tool in "${_TOOLS_ARRAY[@]}"; do
    _trimmed="$(echo "$_tool" | xargs)"
    if [[ -n "$_trimmed" ]]; then
      _TOOLS_CLEAN+=("\"${_trimmed}\"")
    fi
  done
  if [[ ${#_TOOLS_CLEAN[@]} -gt 0 ]]; then
    POLICY_ALLOWED_TOOLS_TOML="allowed_tools = [$(IFS=,; echo "${_TOOLS_CLEAN[*]}")]"
  fi
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

[providers.g4f]
model = "${MODEL}"
base_url = ""

[providers.proxy]
model = "${PROXY_MODEL}"
base_url = "${PROXY_BASE_URL}"

[memory]
db_path = "${HOME}/.neuralclaw/data/memory.db"
max_episodic_results = 10
max_semantic_results = 5
importance_threshold = 0.3

[security]
threat_threshold = 0.7
block_threshold = 0.9
threat_verifier_model = ""
threat_borderline_low = 0.35
threat_borderline_high = 0.65
max_content_chars = 8000
max_skill_timeout_seconds = 30
allow_shell_execution = false

[policy]
max_tool_calls_per_request = 10
max_request_wall_seconds = 120.0
${POLICY_ALLOWED_TOOLS_TOML}
mutating_tools = ["write_file", "create_event", "delete_event"]
allowed_filesystem_roots = ["${HOME}/.neuralclaw"]
deny_private_networks = true
deny_shell_execution = true

[voice]
enabled = ${VOICE_ENABLED}
provider = "${VOICE_PROVIDER}"
require_confirmation = ${VOICE_REQUIRE_CONFIRM}
twilio_account_sid = "${TWILIO_ACCOUNT_SID}"
twilio_auth_token = "${TWILIO_AUTH_TOKEN}"
twilio_phone_number = "${TWILIO_PHONE_NUMBER}"

[features]
swarm = ${MESH_ENABLED}
dashboard = ${ENABLE_DASHBOARD}
evolution = ${ENABLE_EVOLUTION}
reflective_reasoning = ${ENABLE_REFLECTIVE}
procedural_memory = true
semantic_memory = true

[federation]
enabled = false

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
