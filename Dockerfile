FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    bash \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Baileys + QR generator in a fixed directory.
# The bridge .mjs file is written here at runtime so ESM imports resolve.
RUN mkdir -p /app/wa_bridge \
    && cd /app/wa_bridge \
    && npm init -y \
    && npm install @whiskeysockets/baileys qrcode

WORKDIR /app
COPY start.sh /app/start.sh
COPY mesh_gateway.py /app/mesh_gateway.py
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
