FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY start.sh /app/start.sh
COPY mesh_gateway.py /app/mesh_gateway.py
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
