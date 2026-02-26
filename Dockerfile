## Stage 1: install WhatsApp bridge deps in a proper Node environment
FROM node:20-slim AS wa-builder
WORKDIR /app/wa_bridge
RUN npm init -y \
    && npm install --legacy-peer-deps --no-fund --no-audit --omit=optional \
       @whiskeysockets/baileys qrcode

## Stage 2: runtime image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    bash \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built node_modules from builder — no npm install needed here
COPY --from=wa-builder /app/wa_bridge/node_modules /app/wa_bridge/node_modules

WORKDIR /app
COPY start.sh /app/start.sh
COPY mesh_gateway.py /app/mesh_gateway.py
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
