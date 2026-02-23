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

## Deploy check (manual)

1. Create a Railway service from this repo.
2. Set required env vars.
3. Deploy and verify logs show gateway startup.

## Notes

- Uses `pip install "neuralclaw[all-channels]"` at startup.
- Does not run `neuralclaw init` (interactive, not suitable for automation).
- Stores runtime config in `$HOME/.neuralclaw/config.toml`.
