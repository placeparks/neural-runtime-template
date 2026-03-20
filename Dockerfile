## Stage 1: install WhatsApp bridge deps using Bun (10-20x faster than npm)
FROM oven/bun:1-slim AS wa-builder
WORKDIR /app/wa_bridge
RUN echo '{"name":"wa-bridge","private":true}' > package.json \
    && bun add @whiskeysockets/baileys qrcode

## Stage 2: runtime image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    bash \
    nodejs \
    npm \
    ffmpeg \
    gcc \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the published NeuralClaw runtime and the extras this Railway image uses.
RUN pip install --no-cache-dir "neuralclaw[voice,vector,google,microsoft]==1.1.6" aiohttp \
    && python -m playwright install --with-deps chromium

WORKDIR /app/discord_voice_worker
COPY discord_voice_worker/package.json /app/discord_voice_worker/package.json
RUN npm install --omit=dev

# Copy pre-built node_modules from builder — no npm install needed here
COPY --from=wa-builder /app/wa_bridge/node_modules /app/wa_bridge/node_modules
COPY --from=wa-builder /app/wa_bridge/package.json /app/wa_bridge/package.json

WORKDIR /app
COPY discord_voice_worker /app/discord_voice_worker
COPY start.sh /app/start.sh
COPY mesh_gateway.py /app/mesh_gateway.py
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
