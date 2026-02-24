#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from neuralclaw.config import get_api_key, load_config
from neuralclaw.gateway import NeuralClawGateway


ASK_PATTERN = re.compile(
    r"^\s*(?:ask|delegate(?:\s+to)?)\s+([a-zA-Z0-9_-]+)\s+(?:to\s+)?(.+?)\s*$",
    re.IGNORECASE,
)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


class MeshDelegateRouter:
    def __init__(self) -> None:
        self.enabled = os.getenv("NEURALCLAW_MESH_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
        self.timeout = float(os.getenv("NEURALCLAW_MESH_TIMEOUT_SECONDS", "45"))
        self.peers = self._load_peers()

    def _load_peers(self) -> list[dict[str, Any]]:
        raw = os.getenv("NEURALCLAW_MESH_PEERS_JSON", "").strip()
        if not raw:
            p = Path.home() / ".neuralclaw" / "mesh-peers.json"
            if p.exists():
                raw = p.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
        except Exception:
            return []
        return []

    def _find_peer(self, target_name: str) -> dict[str, Any] | None:
        target_key = _normalize(target_name)
        for peer in self.peers:
            permission = str(peer.get("permission", "delegate")).lower()
            if permission == "blocked":
                continue
            name = str(peer.get("agentName") or peer.get("name") or "").strip()
            if not name:
                continue
            if _normalize(name) == target_key:
                return peer
        return None

    def parse_command(self, text: str) -> tuple[str, str] | None:
        m = ASK_PATTERN.match(text)
        if not m:
            return None
        target = m.group(1).strip()
        task = m.group(2).strip()
        if not target or not task:
            return None
        return target, task

    async def delegate(self, from_agent: str, target_name: str, task: str) -> str | None:
        if not self.enabled:
            return None
        peer = self._find_peer(target_name)
        if not peer:
            return None

        endpoint = str(peer.get("endpoint") or peer.get("meshEndpoint") or "").rstrip("/")
        if not endpoint:
            return None

        # Expects peer runtime endpoint to support /a2a/message.
        payload = {
            "from": from_agent,
            "to": str(peer.get("agentName") or target_name),
            "type": "task",
            "content": task,
            "payload": {"source": "mesh"},
        }
        headers = {}
        shared = os.getenv("NEURALCLAW_MESH_SHARED_SECRET", "").strip()
        if shared:
            headers["x-mesh-secret"] = shared

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(f"{endpoint}/a2a/message", json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    content = str(data.get("content", "")).strip()
                    return content or None
        except Exception:
            return None


class MeshAwareGateway(NeuralClawGateway):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mesh_router = MeshDelegateRouter()

    async def process_message(
        self,
        content: str,
        author_id: str = "user",
        author_name: str = "User",
        channel_id: str = "cli",
        channel_type_name: str = "CLI",
    ) -> str:
        parsed = self._mesh_router.parse_command(content)
        if parsed:
            target, task = parsed
            delegated = await self._mesh_router.delegate(self._config.name, target, task)
            if delegated:
                return delegated
        return await super().process_message(
            content=content,
            author_id=author_id,
            author_name=author_name,
            channel_id=channel_id,
            channel_type_name=channel_type_name,
        )


async def _handle_a2a_message(request: web.Request) -> web.Response:
    shared = os.getenv("NEURALCLAW_MESH_SHARED_SECRET", "").strip()
    if shared:
        provided = request.headers.get("x-mesh-secret", "")
        if not provided or provided != shared:
            return web.json_response({"error": "unauthorized"}, status=401)

    data = await request.json()
    content = str(data.get("content", "")).strip()
    if not content:
        return web.json_response({"error": "missing content"}, status=400)

    app = request.app
    gw: MeshAwareGateway = app["gateway"]
    response = await gw.process_message(
        content=content,
        author_id=str(data.get("from", "peer")),
        author_name=str(data.get("from", "peer")),
        channel_id="mesh",
        channel_type_name="CLI",
    )
    return web.json_response({"content": response, "payload": {"source": "mesh"}})


async def _run_gateway() -> None:
    config = load_config()
    gw = MeshAwareGateway(config)

    for ch_config in config.channels:
        if not ch_config.enabled or not ch_config.token:
            continue

        if ch_config.name == "telegram":
            from neuralclaw.channels.telegram import TelegramAdapter

            gw.add_channel(TelegramAdapter(ch_config.token))
        elif ch_config.name == "discord":
            from neuralclaw.channels.discord_adapter import DiscordAdapter

            gw.add_channel(DiscordAdapter(ch_config.token))

    slack_bot = get_api_key("slack_bot")
    slack_app = get_api_key("slack_app")
    if slack_bot and slack_app:
        from neuralclaw.channels.slack import SlackAdapter

        gw.add_channel(SlackAdapter(slack_bot, slack_app))

    whatsapp_session = get_api_key("whatsapp")
    if whatsapp_session:
        from neuralclaw.channels.whatsapp import WhatsAppAdapter

        gw.add_channel(WhatsAppAdapter(whatsapp_session))

    signal_phone = get_api_key("signal")
    if signal_phone:
        from neuralclaw.channels.signal_adapter import SignalAdapter

        gw.add_channel(SignalAdapter(signal_phone))

    from neuralclaw.channels.web import WebChatAdapter

    gw.add_channel(WebChatAdapter())

    # Mesh HTTP server for remote delegation
    mesh_port = int(os.getenv("NEURALCLAW_MESH_PORT", os.getenv("PORT", "8100")))
    app = web.Application()
    app["gateway"] = gw
    app.add_routes([web.post("/a2a/message", _handle_a2a_message)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", mesh_port)
    await site.start()

    await gw.run_forever()


if __name__ == "__main__":
    asyncio.run(_run_gateway())
