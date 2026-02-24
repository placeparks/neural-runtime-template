# neural-runtime-template

Runtime template for per-user NeuralClaw service provisioning on Railway.

## Purpose

This repo is cloned by Railway for each user deployment created by your provisioner.

Your provisioner should set these env vars per service:

- `NEURALCLAW_AGENT_NAME`
- `NEURALCLAW_PROVIDER` (`openai|anthropic|openrouter|local`)
- `NEURALCLAW_MODEL`
- Provider key (one of):
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `OPENROUTER_API_KEY`
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

- Uses `pip install "neuralclaw[all-channels]"` at startup.
- Does not run `neuralclaw init` (interactive, not suitable for automation).
- Stores runtime config in `$HOME/.neuralclaw/config.toml`.
- Writes mesh peers to `$HOME/.neuralclaw/mesh-peers.json` when provided.
- Uses `/app/mesh_gateway.py` to intercept `ask <agent> ...` and delegate over mesh HTTP.
- Exposes `POST /a2a/message` for remote agents to send tasks.
