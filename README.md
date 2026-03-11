# neural-runtime-template

Runtime template for per-user NeuralClaw service provisioning on Railway.

## Purpose

This repo is cloned by Railway for each user deployment created by your provisioner.

Your provisioner should set these env vars per service:

- `NEURALCLAW_AGENT_NAME`
- `NEURALCLAW_PROVIDER` (`openai|anthropic|openrouter|venice|local|g4f|chatgpt_token|claude_token`)
- `NEURALCLAW_MODEL`
- Provider key (one of):
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `OPENROUTER_API_KEY`
  - `OPENAI_API_KEY` with `NEURALCLAW_PROVIDER=venice` for Venice API
  - `NEURALCLAW_OPENAI_BASE_URL` (optional override; Venice defaults to `https://api.venice.ai/api/v1`)
  - `CHATGPT_TOKEN`
  - `CLAUDE_SESSION_KEY`
- Channel tokens (as needed):
  - `NEURALCLAW_TELEGRAM_TOKEN`
  - `NEURALCLAW_DISCORD_TOKEN`
  - `NEURALCLAW_SLACK_BOT_API_KEY`
  - `NEURALCLAW_SLACK_APP_API_KEY`
  - `NEURALCLAW_WHATSAPP_API_KEY`
  - `NEURALCLAW_SIGNAL_API_KEY`
- Mesh/runtime flags:
  - `NEURALCLAW_MESH_ENABLED` (`true|false`)
  - `NEURALCLAW_MESH_PEERS_JSON` (JSON array of peer metadata)
  - `NEURALCLAW_ENABLE_DASHBOARD` (`true|false`, optional)
  - `NEURALCLAW_MESH_TIMEOUT_SECONDS` (delegation timeout, optional)
  - `NEURALCLAW_MESH_PORT` (HTTP mesh port, default `PORT` or 8100)
  - `NEURALCLAW_MESH_SHARED_SECRET` (optional shared secret for mesh requests)

## Deploy check (manual)

1. Create a Railway service from this repo.
2. Set required env vars.
3. Deploy and verify logs show gateway startup.

## Notes

- Uses exact package pin `neuralclaw==0.7.7`.
- Does not run `neuralclaw init` (interactive, not suitable for automation).
- Stores runtime config in `$HOME/.neuralclaw/config.toml`.
- Writes mesh peers to `$HOME/.neuralclaw/mesh-peers.json` when provided.
- Imports `CHATGPT_TOKEN` / `CLAUDE_SESSION_KEY` into NeuralClaw's token store on boot.
- `CHATGPT_TOKEN` may be either a raw session cookie or a JSON-serialized OAuth credential payload produced by the SaaS auth helper.
- For ChatGPT and Claude session auth, `0.7.7` supports local bootstrap with `neuralclaw session auth chatgpt --stealth` and `neuralclaw session auth claude --stealth` before pasting the resulting session credential into your SaaS deploy form.
- Uses `/app/mesh_gateway.py` to intercept `ask <agent> ...` and delegate over mesh HTTP.
- Exposes `POST /a2a/message` for remote agents to send tasks.
