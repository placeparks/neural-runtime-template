#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  "$HOME/.neuralclaw" \
  "$HOME/.neuralclaw/data" \
  "$HOME/.neuralclaw/logs" \
  "$HOME/.neuralclaw/workspace/repos" \
  "$HOME/.neuralclaw/sessions/chatgpt" \
  "$HOME/.neuralclaw/sessions/claude"

# Prefer Railway volume for WhatsApp auth/session persistence across redeploys.
if [[ -z "${NEURALCLAW_WHATSAPP_SESSION_DIR:-}" && -d "/data" ]]; then
  export NEURALCLAW_WHATSAPP_SESSION_DIR="/data/whatsapp"
fi
if [[ -n "${NEURALCLAW_WHATSAPP_SESSION_DIR:-}" ]]; then
  mkdir -p "${NEURALCLAW_WHATSAPP_SESSION_DIR}"
fi

AGENT_NAME="${NEURALCLAW_AGENT_NAME:-NeuralClaw}"
PROVIDER="${NEURALCLAW_PROVIDER:-openai}"
ORIGINAL_PROVIDER="$PROVIDER"
ALLOWED_TOOLS_RAW="${NEURALCLAW_ALLOWED_TOOLS:-}"
MESH_ENABLED_RAW="${NEURALCLAW_MESH_ENABLED:-false}"
MESH_PEERS_JSON="${NEURALCLAW_MESH_PEERS_JSON:-}"
ENABLE_DASHBOARD_RAW="${NEURALCLAW_ENABLE_DASHBOARD:-false}"
ENABLE_EVOLUTION_RAW="${NEURALCLAW_ENABLE_EVOLUTION:-false}"
ENABLE_REFLECTIVE_RAW="${NEURALCLAW_REFLECTIVE_REASONING:-true}"
LOCAL_URL="${NEURALCLAW_LOCAL_URL:-}"
PROXY_BASE_URL="${NEURALCLAW_PROXY_BASE_URL:-}"
OPENAI_BASE_URL="${NEURALCLAW_OPENAI_BASE_URL:-https://api.openai.com/v1}"
VOICE_ENABLED_RAW="${NEURALCLAW_VOICE_ENABLED:-false}"
VOICE_PROVIDER="${NEURALCLAW_VOICE_PROVIDER:-twilio}"
VOICE_REQUIRE_CONFIRM_RAW="${NEURALCLAW_VOICE_REQUIRE_CONFIRM:-true}"
TWILIO_ACCOUNT_SID="${TWILIO_ACCOUNT_SID:-}"
TWILIO_AUTH_TOKEN="${TWILIO_AUTH_TOKEN:-}"
TWILIO_PHONE_NUMBER="${TWILIO_PHONE_NUMBER:-}"
PERSONA="${NEURALCLAW_PERSONA:-You are NeuralClaw, a helpful and intelligent AI assistant.}"
PERSONA="${PERSONA//\"/\'}"

case "$PROVIDER" in
  venice)
    PROVIDER="openai"
    OPENAI_BASE_URL="${NEURALCLAW_OPENAI_BASE_URL:-https://api.venice.ai/api/v1}"
    ;;
  chatgpt_session) PROVIDER="chatgpt_token" ;;
  claude_session) PROVIDER="claude_token" ;;
esac

case "$ORIGINAL_PROVIDER" in
  venice) DEFAULT_MODEL="venice-uncensored" ;;
  openai) DEFAULT_MODEL="gpt-4o" ;;
  anthropic) DEFAULT_MODEL="claude-sonnet-4-20250514" ;;
  openrouter) DEFAULT_MODEL="anthropic/claude-sonnet-4-20250514" ;;
  local) DEFAULT_MODEL="qwen3.5:2b" ;;
  g4f) DEFAULT_MODEL="gpt-4o" ;;
  chatgpt_token|claude_token) DEFAULT_MODEL="auto" ;;
  proxy) DEFAULT_MODEL="gpt-4" ;;
  *) DEFAULT_MODEL="gpt-4o" ;;
esac
MODEL="${NEURALCLAW_MODEL:-$DEFAULT_MODEL}"

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
ENABLE_EVOLUTION="$(to_bool "$ENABLE_EVOLUTION_RAW")"
ENABLE_REFLECTIVE="$(to_bool "$ENABLE_REFLECTIVE_RAW")"
VOICE_ENABLED="$(to_bool "$VOICE_ENABLED_RAW")"
VOICE_REQUIRE_CONFIRM="$(to_bool "$VOICE_REQUIRE_CONFIRM_RAW")"

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
base_url = "${OPENAI_BASE_URL}"

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
model = "${MODEL}"
base_url = "${PROXY_BASE_URL}"

[providers.chatgpt_app]
model = "auto"
profile_dir = "${HOME}/.neuralclaw/sessions/chatgpt"
headless = false
browser_channel = ""
site_url = "https://chatgpt.com/"

[providers.claude_app]
model = "auto"
profile_dir = "${HOME}/.neuralclaw/sessions/claude"
headless = false
browser_channel = ""
site_url = "https://claude.ai/chats"

[providers.chatgpt_token]
model = "${MODEL}"
auth_method = "cookie"
profile_dir = "${HOME}/.neuralclaw/sessions/chatgpt"

[providers.claude_token]
model = "${MODEL}"
auth_method = "session_key"
profile_dir = "${HOME}/.neuralclaw/sessions/claude"

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
mutating_tools = ["write_file", "create_event", "delete_event", "clone_repo", "install_repo_deps", "remove_repo", "save_api_config"]
allowed_filesystem_roots = ["${HOME}/.neuralclaw", "${HOME}/.neuralclaw/workspace/repos"]
deny_private_networks = true
deny_shell_execution = true

[workspace]
repos_dir = "${HOME}/.neuralclaw/workspace/repos"
max_repo_size_mb = 500
allowed_git_hosts = ["github.com", "gitlab.com", "bitbucket.org"]
max_clone_timeout_seconds = 120
max_install_timeout_seconds = 300
max_exec_timeout_seconds = 300

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

python - <<'PY'
import os
import time

from neuralclaw.session.auth import AuthManager, TokenCredential


def save_from_env(env_name: str, provider: str, token_type: str) -> None:
    value = os.getenv(env_name, "").strip()
    if not value:
        return
    credential = TokenCredential(
        access_token=value,
        provider=provider,
        token_type=token_type,
        expires_at=time.time() + 86400 * 30,
    )
    AuthManager(provider).save_credential(credential)
    print(f"[runtime] imported {provider} credential from {env_name}")


save_from_env("CHATGPT_TOKEN", "chatgpt", "cookie")
save_from_env("CLAUDE_SESSION_KEY", "claude", "session_key")
PY

echo "[runtime] agent=${AGENT_NAME} provider=${PROVIDER} model=${MODEL} mesh_enabled=${MESH_ENABLED} dashboard=${ENABLE_DASHBOARD}"
if [[ -f "$HOME/.neuralclaw/mesh-peers.json" ]]; then
  echo "[runtime] mesh peers file written: $HOME/.neuralclaw/mesh-peers.json"
fi

exec python /app/mesh_gateway.py
