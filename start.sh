#!/usr/bin/env bash
set -euo pipefail

RUNTIME_HOME="$HOME/.neuralclaw"
STATE_ROOT="${NEURALCLAW_STATE_ROOT:-}"
if [[ -z "$STATE_ROOT" && -d "/data" ]]; then
  STATE_ROOT="/data/neuralclaw"
fi

if [[ -n "$STATE_ROOT" ]]; then
  DATA_ROOT="$STATE_ROOT/data"
  LOG_ROOT="$STATE_ROOT/logs"
  SESSION_ROOT="$STATE_ROOT/sessions"
  WORKSPACE_ROOT="$STATE_ROOT/workspace"
else
  DATA_ROOT="$RUNTIME_HOME/data"
  LOG_ROOT="$RUNTIME_HOME/logs"
  SESSION_ROOT="$RUNTIME_HOME/sessions"
  WORKSPACE_ROOT="$RUNTIME_HOME/workspace"
fi

mkdir -p \
  "$RUNTIME_HOME" \
  "$DATA_ROOT" \
  "$LOG_ROOT" \
  "$WORKSPACE_ROOT/repos" \
  "$SESSION_ROOT/chatgpt" \
  "$SESSION_ROOT/claude" \
  "$SESSION_ROOT/browser"

link_runtime_dir() {
  local target="$1"
  local link="$2"

  if [[ "$target" == "$link" ]]; then
    return
  fi

  mkdir -p "$target"

  if [[ -L "$link" ]]; then
    rm -f "$link"
  elif [[ -d "$link" ]]; then
    if [[ -n "$(ls -A "$link" 2>/dev/null)" ]]; then
      cp -a "$link"/. "$target"/ 2>/dev/null || true
    fi
    rm -rf "$link"
  elif [[ -e "$link" ]]; then
    rm -f "$link"
  fi

  ln -s "$target" "$link"
}

link_runtime_dir "$DATA_ROOT" "$RUNTIME_HOME/data"
link_runtime_dir "$LOG_ROOT" "$RUNTIME_HOME/logs"
link_runtime_dir "$SESSION_ROOT" "$RUNTIME_HOME/sessions"

export NEURALCLAW_CONFIG_DIR="$RUNTIME_HOME"
export NEURALCLAW_CONFIG_FILE="$RUNTIME_HOME/config.toml"
export NEURALCLAW_DATA_DIR="$DATA_ROOT"
export NEURALCLAW_LOG_DIR="$LOG_ROOT"
export NEURALCLAW_SESSION_DIR="$SESSION_ROOT"
export NEURALCLAW_CHANNEL_BINDINGS_FILE="$DATA_ROOT/channel_bindings.json"

# Prefer a persistent volume for WhatsApp session state.
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
ENABLE_EVOLUTION_RAW="${NEURALCLAW_ENABLE_EVOLUTION:-true}"
ENABLE_REFLECTIVE_RAW="${NEURALCLAW_REFLECTIVE_REASONING:-true}"
ENABLE_VECTOR_MEMORY_RAW="${NEURALCLAW_VECTOR_MEMORY_ENABLED:-true}"
ENABLE_IDENTITY_RAW="${NEURALCLAW_IDENTITY_ENABLED:-true}"
ENABLE_TRACELINE_RAW="${NEURALCLAW_TRACELINE_ENABLED:-true}"
ENABLE_STRUCTURED_OUTPUT_RAW="${NEURALCLAW_STRUCTURED_OUTPUT_ENABLED:-true}"
ENABLE_BROWSER_RAW="${NEURALCLAW_BROWSER_ENABLED:-true}"
ENABLE_DESKTOP_RAW="${NEURALCLAW_DESKTOP_ENABLED:-false}"
ENABLE_GOOGLE_WORKSPACE_RAW="${NEURALCLAW_GOOGLE_WORKSPACE_ENABLED:-false}"
ENABLE_MICROSOFT365_RAW="${NEURALCLAW_MICROSOFT365_ENABLED:-false}"
ENABLE_STREAMING_RAW="${NEURALCLAW_STREAMING_RESPONSES:-true}"
ENABLE_VISION_RAW="${NEURALCLAW_VISION_ENABLED:-true}"
LOCAL_URL="${NEURALCLAW_LOCAL_URL:-}"
PROXY_BASE_URL="${NEURALCLAW_PROXY_BASE_URL:-}"
OPENAI_BASE_URL="${NEURALCLAW_OPENAI_BASE_URL:-https://api.openai.com/v1}"
VOICE_ENABLED_RAW="${NEURALCLAW_VOICE_ENABLED:-false}"
VOICE_PROVIDER="${NEURALCLAW_VOICE_PROVIDER:-twilio}"
VOICE_REQUIRE_CONFIRM_RAW="${NEURALCLAW_VOICE_REQUIRE_CONFIRM:-true}"
TWILIO_ACCOUNT_SID="${TWILIO_ACCOUNT_SID:-}"
TWILIO_AUTH_TOKEN="${TWILIO_AUTH_TOKEN:-}"
TWILIO_PHONE_NUMBER="${TWILIO_PHONE_NUMBER:-}"
DISCORD_VOICE_ENABLED_RAW="${NEURALCLAW_DISCORD_VOICE_ENABLED:-false}"
TELEGRAM_TRUST_MODE="${NEURALCLAW_TELEGRAM_TRUST_MODE:-open}"
DISCORD_TRUST_MODE="${NEURALCLAW_DISCORD_TRUST_MODE:-open}"
SLACK_TRUST_MODE="${NEURALCLAW_SLACK_TRUST_MODE:-open}"
PERSONA="${NEURALCLAW_PERSONA:-You are NeuralClaw, a self-evolving cognitive AI agent with persistent memory and tool use capabilities.}"
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
  openai) DEFAULT_MODEL="gpt-5.4" ;;
  anthropic) DEFAULT_MODEL="claude-sonnet-4-6" ;;
  openrouter) DEFAULT_MODEL="anthropic/claude-sonnet-4-6" ;;
  local) DEFAULT_MODEL="qwen3:8b" ;;
  g4f) DEFAULT_MODEL="gpt-5.4" ;;
  chatgpt_token|claude_token) DEFAULT_MODEL="auto" ;;
  proxy) DEFAULT_MODEL="gpt-5.4" ;;
  *) DEFAULT_MODEL="gpt-5.4" ;;
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
ENABLE_VECTOR_MEMORY="$(to_bool "$ENABLE_VECTOR_MEMORY_RAW")"
ENABLE_IDENTITY="$(to_bool "$ENABLE_IDENTITY_RAW")"
ENABLE_TRACELINE="$(to_bool "$ENABLE_TRACELINE_RAW")"
ENABLE_STRUCTURED_OUTPUT="$(to_bool "$ENABLE_STRUCTURED_OUTPUT_RAW")"
ENABLE_BROWSER="$(to_bool "$ENABLE_BROWSER_RAW")"
ENABLE_DESKTOP="$(to_bool "$ENABLE_DESKTOP_RAW")"
ENABLE_GOOGLE_WORKSPACE="$(to_bool "$ENABLE_GOOGLE_WORKSPACE_RAW")"
ENABLE_MICROSOFT365="$(to_bool "$ENABLE_MICROSOFT365_RAW")"
ENABLE_STREAMING="$(to_bool "$ENABLE_STREAMING_RAW")"
ENABLE_VISION="$(to_bool "$ENABLE_VISION_RAW")"
VOICE_ENABLED="$(to_bool "$VOICE_ENABLED_RAW")"
DISCORD_VOICE_ENABLED="$(to_bool "$DISCORD_VOICE_ENABLED_RAW")"
VOICE_REQUIRE_CONFIRM="$(to_bool "$VOICE_REQUIRE_CONFIRM_RAW")"

if [[ "$DISCORD_VOICE_ENABLED" == "true" ]]; then
  ENABLE_STREAMING="true"
fi

if [[ "$ENABLE_BROWSER" == "true" ]]; then
  FALLBACK_TOML='fallback = ["openrouter", "local"]'
elif [[ -n "$LOCAL_URL" ]]; then
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

MESH_PEERS_FILE="$RUNTIME_HOME/mesh-peers.json"
if [[ -n "$MESH_PEERS_JSON" ]]; then
  printf '%s\n' "$MESH_PEERS_JSON" > "$MESH_PEERS_FILE"
fi

KNOWLEDGE_CONTENT="${NEURALCLAW_KNOWLEDGE_CONTENT:-}"
KNOWLEDGE_FILE="$RUNTIME_HOME/knowledge.txt"
if [[ -n "$KNOWLEDGE_CONTENT" ]]; then
  printf '%s\n' "$KNOWLEDGE_CONTENT" > "$KNOWLEDGE_FILE"
  echo "[runtime] knowledge base written (${#KNOWLEDGE_CONTENT} bytes)"
fi

cat > "$RUNTIME_HOME/config.toml" <<EOF
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
profile_dir = "${SESSION_ROOT}/chatgpt"
headless = false
browser_channel = ""
site_url = "https://chatgpt.com/"

[providers.claude_app]
model = "auto"
profile_dir = "${SESSION_ROOT}/claude"
headless = false
browser_channel = ""
site_url = "https://claude.ai/chats"

[providers.chatgpt_token]
model = "${MODEL}"
auth_method = "cookie"
profile_dir = "${SESSION_ROOT}/chatgpt"

[providers.claude_token]
model = "${MODEL}"
auth_method = "session_key"
profile_dir = "${SESSION_ROOT}/claude"

[memory]
db_path = "${DATA_ROOT}/memory.db"
max_episodic_results = 10
max_semantic_results = 5
importance_threshold = 0.3
vector_memory = ${ENABLE_VECTOR_MEMORY}
embedding_provider = "local"
embedding_model = "${NEURALCLAW_EMBEDDING_MODEL:-nomic-embed-text}"
embedding_dimension = ${NEURALCLAW_EMBEDDING_DIMENSION:-768}
vector_similarity_top_k = ${NEURALCLAW_VECTOR_TOP_K:-10}

[identity]
enabled = ${ENABLE_IDENTITY}
cross_channel = true
inject_in_prompt = true
notes_enabled = true

[traceline]
enabled = ${ENABLE_TRACELINE}
db_path = "${DATA_ROOT}/traces.db"
retention_days = 30
export_otlp = false
otlp_endpoint = ""
export_prometheus = false
metrics_port = 9090
include_input = true
include_output = true
max_preview_chars = 500

[audit]
enabled = true
jsonl_path = "${LOG_ROOT}/audit.jsonl"
max_memory_entries = 200
retention_days = 90
siem_export = false
include_args = true

[security]
threat_threshold = 0.7
block_threshold = 0.9
threat_verifier_model = ""
threat_borderline_low = 0.35
threat_borderline_high = 0.65
max_content_chars = 8000
max_skill_timeout_seconds = 30
allow_shell_execution = false
output_filtering = true
output_pii_detection = true
output_prompt_leak_check = true
canary_tokens = true

[policy]
max_tool_calls_per_request = 10
max_request_wall_seconds = 120.0
${POLICY_ALLOWED_TOOLS_TOML}
mutating_tools = ["write_file", "create_event", "delete_event", "clone_repo", "install_repo_deps", "remove_repo", "save_api_config"]
allowed_filesystem_roots = ["${RUNTIME_HOME}", "${WORKSPACE_ROOT}/repos"]
deny_private_networks = true
deny_shell_execution = true
parallel_tool_execution = true

[workspace]
repos_dir = "${WORKSPACE_ROOT}/repos"
max_repo_size_mb = 500
allowed_git_hosts = ["github.com", "gitlab.com", "bitbucket.org"]
max_clone_timeout_seconds = 120
max_install_timeout_seconds = 300
max_exec_timeout_seconds = 300

[desktop]
enabled = ${ENABLE_DESKTOP}
screenshot_on_action = true
action_delay_ms = 100

[tts]
enabled = ${DISCORD_VOICE_ENABLED}
provider = "openai"
voice = "${NEURALCLAW_DISCORD_TTS_VOICE:-alloy}"
speed = 1.0
output_format = "mp3"
piper_binary = "piper"
piper_model = ""
auto_speak = false
max_tts_chars = 4000
temp_dir = "${LOG_ROOT}"

[browser]
enabled = ${ENABLE_BROWSER}
headless = true
browser_type = "chromium"
viewport_width = 1280
viewport_height = 900
stealth = true
allow_js_execution = false
max_steps_per_task = 20
screenshot_on_error = true
chrome_ai_enabled = false
navigation_timeout = 30
user_data_dir = "${SESSION_ROOT}/browser"
allowed_domains = []
blocked_domains = ["localhost", "127.0.0.1", "169.254.169.254"]

[google_workspace]
enabled = ${ENABLE_GOOGLE_WORKSPACE}
scopes = [
  "https://www.googleapis.com/auth/gmail.modify",
  "https://www.googleapis.com/auth/calendar",
  "https://www.googleapis.com/auth/drive",
  "https://www.googleapis.com/auth/documents",
  "https://www.googleapis.com/auth/spreadsheets",
]
max_email_results = 10
max_drive_results = 10
default_calendar_id = "primary"
response_body_limit = 20000

[microsoft365]
enabled = ${ENABLE_MICROSOFT365}
tenant_id = "${MICROSOFT_TENANT_ID:-}"
scopes = ["Mail.ReadWrite", "Calendars.ReadWrite", "Files.ReadWrite", "Chat.ReadWrite", "ChannelMessage.Send"]
max_email_results = 10
max_file_results = 10
default_user = "me"

[voice]
enabled = ${VOICE_ENABLED}
provider = "${VOICE_PROVIDER}"
require_confirmation = ${VOICE_REQUIRE_CONFIRM}
twilio_account_sid = "${TWILIO_ACCOUNT_SID}"
twilio_auth_token = "${TWILIO_AUTH_TOKEN}"
twilio_phone_number = "${TWILIO_PHONE_NUMBER}"

[features]
vector_memory = ${ENABLE_VECTOR_MEMORY}
identity = ${ENABLE_IDENTITY}
vision = ${ENABLE_VISION}
voice = ${DISCORD_VOICE_ENABLED}
browser = ${ENABLE_BROWSER}
structured_output = ${ENABLE_STRUCTURED_OUTPUT}
streaming_responses = ${ENABLE_STREAMING}
streaming_edit_interval = 20
traceline = ${ENABLE_TRACELINE}
desktop = ${ENABLE_DESKTOP}
swarm = ${MESH_ENABLED}
dashboard = ${ENABLE_DASHBOARD}
evolution = ${ENABLE_EVOLUTION}
reflective_reasoning = ${ENABLE_REFLECTIVE}
procedural_memory = true
semantic_memory = true
a2a_federation = false

[federation]
enabled = false
a2a_enabled = false

[channels.telegram]
enabled = true
trust_mode = "${TELEGRAM_TRUST_MODE}"

[channels.discord]
enabled = true
trust_mode = "${DISCORD_TRUST_MODE}"
voice_responses = ${DISCORD_VOICE_ENABLED}
auto_disconnect_empty_vc = true
voice_channel_id = "${NEURALCLAW_DISCORD_VOICE_CHANNEL_ID:-}"

[channels.slack]
enabled = true
trust_mode = "${SLACK_TRUST_MODE}"
EOF

python - <<'PY'
import json
import os
import time

from neuralclaw.session.auth import AuthManager, TokenCredential


def save_from_env(env_name: str, provider: str, token_type: str) -> None:
    value = os.getenv(env_name, "").strip()
    if not value:
        return
    credential = None
    if value.startswith("{"):
        try:
            data = json.loads(value)
            access_token = str(data.get("access_token", "")).strip()
            if access_token:
                credential = TokenCredential(
                    access_token=access_token,
                    provider=str(data.get("provider") or provider),
                    token_type=str(data.get("token_type") or token_type),
                    expires_at=float(data.get("expires_at") or (time.time() + 86400 * 30)),
                    refresh_token=str(data.get("refresh_token") or ""),
                )
        except Exception as exc:
            print(f"[runtime] failed to parse structured credential from {env_name}: {exc}")
    if credential is None:
        credential = TokenCredential(
            access_token=value,
            provider=provider,
            token_type=token_type,
            expires_at=time.time() + 86400 * 30,
        )
    AuthManager(provider).save_credential(credential)
    print(f"[runtime] imported {provider} credential from {env_name} ({credential.token_type})")


save_from_env("CHATGPT_TOKEN", "chatgpt", "cookie")
save_from_env("CLAUDE_SESSION_KEY", "claude", "session_key")
PY

echo "[runtime] agent=${AGENT_NAME} provider=${PROVIDER} model=${MODEL} state_root=${STATE_ROOT:-$RUNTIME_HOME}"
echo "[runtime] vector_memory=${ENABLE_VECTOR_MEMORY} identity=${ENABLE_IDENTITY} browser=${ENABLE_BROWSER} traceline=${ENABLE_TRACELINE}"
if [[ -f "$MESH_PEERS_FILE" ]]; then
  echo "[runtime] mesh peers file written: $MESH_PEERS_FILE"
fi

DISCORD_TOKEN="${NEURALCLAW_DISCORD_TOKEN:-}"

if [[ "$DISCORD_VOICE_ENABLED" == "true" && -n "$DISCORD_TOKEN" ]]; then
  echo "[runtime] starting Python gateway + Discord voice worker"
  python /app/mesh_gateway.py &
  GATEWAY_PID=$!
  node /app/discord_voice_worker/index.js &
  DISCORD_WORKER_PID=$!

  cleanup() {
    kill "$GATEWAY_PID" "$DISCORD_WORKER_PID" 2>/dev/null || true
  }

  trap cleanup INT TERM
  wait -n "$GATEWAY_PID" "$DISCORD_WORKER_PID"
  STATUS=$?
  cleanup
  wait "$GATEWAY_PID" "$DISCORD_WORKER_PID" 2>/dev/null || true
  exit "$STATUS"
fi

exec python /app/mesh_gateway.py
