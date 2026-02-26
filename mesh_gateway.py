#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import sys
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from neuralclaw.config import get_api_key, load_config
from neuralclaw.gateway import NeuralClawGateway

# ---------------------------------------------------------------------------
# Hotfix: ToolCall.to_dict() must serialise arguments as a JSON *string*.
# OpenAI rejects tool_calls where function.arguments is a plain dict.
# Remove this block once neuralclaw >= next release is on PyPI.
# ---------------------------------------------------------------------------
try:
    from neuralclaw.providers.router import ToolCall as _ToolCall

    def _patched_to_dict(self: _ToolCall) -> dict:  # type: ignore[override]
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }

    _ToolCall.to_dict = _patched_to_dict  # type: ignore[method-assign]
except Exception as _patch_err:  # pragma: no cover
    logging.getLogger("mesh_gateway").warning("Could not apply ToolCall patch: %s", _patch_err)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Knowledge base: if ~/.neuralclaw/knowledge.txt exists, enable file_ops
# for that directory so the agent can read it with the read_file tool.
# ---------------------------------------------------------------------------
_KNOWLEDGE_PATH = Path.home() / ".neuralclaw" / "knowledge.txt"
if _KNOWLEDGE_PATH.exists():
    try:
        from neuralclaw.skills.builtins.file_ops import set_allowed_roots
        set_allowed_roots([_KNOWLEDGE_PATH.parent])
        logging.getLogger("mesh_gateway").info(
            "[runtime] knowledge base available (%d bytes) — file_ops enabled",
            _KNOWLEDGE_PATH.stat().st_size,
        )
    except Exception as _kb_err:  # pragma: no cover
        logging.getLogger("mesh_gateway").warning("Could not enable file_ops for knowledge base: %s", _kb_err)
# ---------------------------------------------------------------------------

logger = logging.getLogger("mesh_gateway")


ASK_PATTERN = re.compile(
    r"^\s*(?:ask|delegate(?:\s+to)?)\s+(.+?)\s+to\s+(.+?)\s*$",
    re.IGNORECASE,
)

KNOWLEDGE_QUERY_PATTERN = re.compile(
    r"\b("
    r"knowledge|knowledge\s*base|docs?|document|file|stored|"
    r"nexus|memory|plan|architecture|delivery|"
    r"based\s+on|from\s+the\s+docs|what\s+do\s+you\s+know|"
    r"do\s+you\s+have"
    r")\b",
    re.IGNORECASE,
)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _maybe_call(value: Any) -> Any:
    try:
        if callable(value):
            return value()
    except Exception:
        return None
    return value


def _extract_qr_like(value: Any) -> str | None:
    """Try to normalize any QR-like value to string."""
    value = _maybe_call(value)
    if value is None:
        return None

    if isinstance(value, (bytes, bytearray)):
        try:
            text = value.decode("utf-8", errors="ignore").strip()
        except Exception:
            return None
    elif isinstance(value, str):
        text = value.strip()
    else:
        return None

    if not text:
        return None

    # Common representations:
    # - data:image/png;base64,...
    # - raw long base64-like payload
    # - textual QR payload for terminal rendering
    if text.startswith("data:image/"):
        return text
    if len(text) > 120:
        return text
    return None


def _extract_whatsapp_debug_state(adapter: Any) -> dict[str, Any]:
    if adapter is None:
        return {"enabled": False, "connected": False, "ready": False, "qr": None}

    qr_candidates = (
        "qr",
        "qr_code",
        "qr_data",
        "last_qr",
        "latest_qr",
        "current_qr",
        "_qr",
        "_last_qr",
    )
    state_candidates = ("connected", "ready", "authenticated", "logged_in", "is_connected")
    nested_candidates = ("client", "bridge", "session", "state", "_client", "_bridge")

    connected: bool | None = None
    ready: bool | None = None
    qr: str | None = None

    # Direct adapter attrs
    for name in qr_candidates:
        if hasattr(adapter, name):
            qr = _extract_qr_like(getattr(adapter, name))
            if qr:
                break

    for name in state_candidates:
        if hasattr(adapter, name):
            raw = _maybe_call(getattr(adapter, name))
            if isinstance(raw, bool):
                if name in {"ready", "authenticated"}:
                    ready = raw if ready is None else ready
                else:
                    connected = raw if connected is None else connected

    # Nested attrs
    for parent in nested_candidates:
        if qr and connected is not None and ready is not None:
            break
        if not hasattr(adapter, parent):
            continue
        node = getattr(adapter, parent)

        for name in qr_candidates:
            if qr:
                break
            if hasattr(node, name):
                qr = _extract_qr_like(getattr(node, name))
                if qr:
                    break

        for name in state_candidates:
            if hasattr(node, name):
                raw = _maybe_call(getattr(node, name))
                if isinstance(raw, bool):
                    if name in {"ready", "authenticated"}:
                        ready = raw if ready is None else ready
                    else:
                        connected = raw if connected is None else connected

    # Reasonable fallback semantics.
    if ready is None and connected is not None:
        ready = connected
    if connected is None and ready is not None:
        connected = ready
    if connected is None:
        connected = False
    if ready is None:
        ready = False

    return {
        "enabled": True,
        "connected": connected,
        "ready": ready,
        "qr": qr,
    }


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

    async def delegate(self, from_agent: str, target_name: str, task: str) -> tuple[str | None, str | None]:
        """Returns (result, error_reason). Both can be None only if not a delegation context."""
        if not self.enabled:
            logger.warning("[MESH] Delegation attempted but NEURALCLAW_MESH_ENABLED is not true")
            return None, "mesh is not enabled on this agent"

        peer = self._find_peer(target_name)
        if not peer:
            known = [str(p.get("agentName") or p.get("name") or "") for p in self.peers]
            logger.warning("[MESH] Peer '%s' not found. Known peers: %s", target_name, known or ["(none)"])
            if not self.peers:
                return None, "no mesh peers are configured — check NEURALCLAW_MESH_PEERS_JSON"
            return None, f"agent '{target_name}' is not in your mesh peer list (known: {', '.join(known)})"

        endpoint = str(peer.get("endpoint") or peer.get("meshEndpoint") or "").rstrip("/")
        if not endpoint:
            logger.warning("[MESH] Peer '%s' found but has no endpoint set", target_name)
            return None, f"agent '{target_name}' has no endpoint configured"

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

        logger.info("[MESH] Delegating to '%s' at %s — task: %s", target_name, endpoint, task[:80])
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(f"{endpoint}/a2a/message", json=payload, headers=headers) as resp:
                    if resp.status == 401:
                        logger.error("[MESH] Peer '%s' rejected request: unauthorized (wrong shared secret?)", target_name)
                        return None, f"agent '{target_name}' rejected the request (unauthorized)"
                    if resp.status != 200:
                        logger.error("[MESH] Peer '%s' returned HTTP %d", target_name, resp.status)
                        return None, f"agent '{target_name}' returned an error (HTTP {resp.status})"
                    data = await resp.json()
                    content = str(data.get("content", "")).strip()
                    if not content:
                        return None, f"agent '{target_name}' returned an empty response"
                    logger.info("[MESH] Delegation to '%s' succeeded", target_name)
                    return content, None
        except asyncio.TimeoutError:
            logger.error("[MESH] Delegation to '%s' timed out after %ss", target_name, self.timeout)
            return None, f"agent '{target_name}' did not respond within {int(self.timeout)}s"
        except Exception as exc:
            logger.error("[MESH] Delegation to '%s' failed: %s", target_name, exc)
            return None, f"could not reach agent '{target_name}': {exc}"


class MeshAwareGateway(NeuralClawGateway):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mesh_router = MeshDelegateRouter()
        self._knowledge_max_chars = int(os.getenv("NEURALCLAW_KNOWLEDGE_MAX_INJECT_CHARS", "12000"))

    def _is_knowledge_query(self, content: str) -> bool:
        return bool(KNOWLEDGE_QUERY_PATTERN.search(content))

    def _knowledge_snippet(self) -> str:
        if not _KNOWLEDGE_PATH.exists():
            return ""
        try:
            text = _KNOWLEDGE_PATH.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception as exc:
            logger.warning("[runtime] could not read knowledge file: %s", exc)
            return ""
        if not text:
            return ""
        if len(text) <= self._knowledge_max_chars:
            return text
        return (
            text[: self._knowledge_max_chars]
            + "\n\n[knowledge truncated due to size; answer from available section only]"
        )

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
            result, error = await self._mesh_router.delegate(self._config.name, target, task)
            if result:
                return result
            # Delegation was explicitly attempted but failed — return the reason.
            # Do NOT fall through to the local LLM; it would try to answer "ask X to ..."
            # as a normal query and fail (or give a nonsensical answer).
            return f"Could not delegate to '{target}': {error}"

        resolved_content = content
        if self._is_knowledge_query(content):
            snippet = self._knowledge_snippet()
            if snippet:
                logger.info("[runtime] knowledge context injected for query")
                resolved_content = (
                    "Use the knowledge context below to answer the user.\n"
                    "If the requested detail is not present, clearly say it is not in stored knowledge.\n\n"
                    f"KNOWLEDGE_CONTEXT:\n{snippet}\n\n"
                    f"USER_MESSAGE:\n{content}"
                )
        return await super().process_message(
            content=resolved_content,
            author_id=author_id,
            author_name=author_name,
            channel_id=channel_id,
            channel_type_name=channel_type_name,
        )


async def _handle_health(request: web.Request) -> web.Response:
    gw: MeshAwareGateway = request.app["gateway"]
    mesh = gw._mesh_router
    return web.json_response({
        "status": "ok",
        "mesh_enabled": mesh.enabled,
        "peers": len(mesh.peers),
    })


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


# ---------------------------------------------------------------------------
# QRTrackingWhatsAppAdapter — pure-Python WhatsApp adapter via Evolution API.
#
# No Node.js, no npm, no Chromium. Uses aiohttp (already installed) to talk
# to an Evolution API instance over HTTP.
#
# Required env vars in the runtime container:
#   NEURALCLAW_WHATSAPP_BRIDGE_URL      — Evolution API base URL
#   NEURALCLAW_WHATSAPP_BRIDGE_API_KEY  — Evolution API key
#   NEURALCLAW_WHATSAPP_API_KEY         — instance name (set by provisioner)
#
# Optional:
#   NEURALCLAW_PUBLIC_URL               — runtime's public URL for webhook
#                                         registration (e.g. https://xxx.railway.app)
# ---------------------------------------------------------------------------

try:
    from neuralclaw.channels.protocol import ChannelAdapter as _ChannelAdapterBase
except Exception:
    class _ChannelAdapterBase:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self._handlers: list[Any] = []

        def add_handler(self, handler: Any) -> None:
            self._handlers.append(handler)

        async def _dispatch(self, msg: Any) -> None:
            for h in self._handlers:
                try:
                    await h(msg)
                except Exception as exc:
                    logging.getLogger("mesh_gateway").warning("[WhatsApp] handler error: %s", exc)

        async def start(self) -> None: ...
        async def stop(self) -> None: ...


class QRTrackingWhatsAppAdapter(_ChannelAdapterBase):
    """
    WhatsApp adapter that talks to an Evolution API instance over HTTP.
    Exposes .qr / .connected / .ready for the /channels/whatsapp endpoint.
    """

    name = "whatsapp"

    def __init__(self, instance_name: str) -> None:
        super().__init__()
        self._instance_name = instance_name
        self._evo_url = (
            os.getenv("NEURALCLAW_WHATSAPP_BRIDGE_URL") or
            os.getenv("EVOLUTION_API_URL") or ""
        ).rstrip("/")
        self._evo_key = (
            os.getenv("NEURALCLAW_WHATSAPP_BRIDGE_API_KEY") or
            os.getenv("EVOLUTION_API_KEY") or
            os.getenv("AUTHENTICATION_API_KEY") or ""
        )
        self._poll_task: asyncio.Task[None] | None = None
        self.qr: str | None = None
        self.connected: bool = False
        self.ready: bool = False

    @property
    def _configured(self) -> bool:
        return bool(self._evo_url and self._evo_key)

    async def start(self) -> None:
        if not self._configured:
            logger.warning(
                "[WhatsApp] NEURALCLAW_WHATSAPP_BRIDGE_URL / NEURALCLAW_WHATSAPP_BRIDGE_API_KEY "
                "not set — WhatsApp channel inactive"
            )
            return
        await self._ensure_instance()
        await self._register_webhook()
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("[WhatsApp] Evolution API adapter started for instance '%s'", self._instance_name)

    async def stop(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

    async def send(self, channel_id: str, content: str, **kwargs: Any) -> None:
        if not self._configured:
            return
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{self._evo_url}/message/sendText/{self._instance_name}",
                    headers={"apikey": self._evo_key, "Content-Type": "application/json"},
                    json={"number": channel_id, "text": content},
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status not in (200, 201):
                        logger.warning("[WhatsApp] sendText failed: HTTP %d", resp.status)
        except Exception as exc:
            logger.warning("[WhatsApp] send error: %s", exc)

    async def _ensure_instance(self) -> None:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    f"{self._evo_url}/instance/connect/{self._instance_name}",
                    headers={"apikey": self._evo_key},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        return  # already exists
                async with s.post(
                    f"{self._evo_url}/instance/create",
                    headers={"apikey": self._evo_key, "Content-Type": "application/json"},
                    json={"instanceName": self._instance_name, "qrcode": True, "integration": "WHATSAPP-BAILEYS"},
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status not in (200, 201):
                        text = await resp.text()
                        logger.warning("[WhatsApp] instance create failed: HTTP %d — %s", resp.status, text[:200])
        except Exception as exc:
            logger.warning("[WhatsApp] ensure instance error: %s", exc)

    async def _register_webhook(self) -> None:
        public_url = (os.getenv("NEURALCLAW_PUBLIC_URL") or "").rstrip("/")
        if not public_url:
            return
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{self._evo_url}/webhook/set/{self._instance_name}",
                    headers={"apikey": self._evo_key, "Content-Type": "application/json"},
                    json={
                        "url": f"{public_url}/channels/whatsapp/webhook",
                        "webhook_by_events": False,
                        "webhook_base64": False,
                        "events": ["MESSAGES_UPSERT"],
                    },
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status in (200, 201):
                        logger.info("[WhatsApp] Webhook registered at %s", public_url)
        except Exception as exc:
            logger.warning("[WhatsApp] webhook registration error: %s", exc)

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self._refresh_state()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("[WhatsApp] poll error: %s", exc)
            await asyncio.sleep(5)

    async def _refresh_state(self) -> None:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{self._evo_url}/instance/connectionState/{self._instance_name}",
                headers={"apikey": self._evo_key},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    state = (data.get("instance") or {}).get("state", "")
                    if state == "open":
                        self.connected = True
                        self.ready = True
                        self.qr = None
                        return

            self.connected = False
            self.ready = False
            async with s.get(
                f"{self._evo_url}/instance/connect/{self._instance_name}",
                headers={"apikey": self._evo_key},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.qr = self._pick_qr(data)

    @staticmethod
    def _pick_qr(data: Any) -> str | None:
        if not isinstance(data, dict):
            return None
        qrcode = data.get("qrcode") if isinstance(data.get("qrcode"), dict) else {}
        for val in (
            data.get("base64"), data.get("qr"), data.get("code"),
            qrcode.get("base64"), qrcode.get("qr"), qrcode.get("code"),
        ):
            if not isinstance(val, str) or not val.strip():
                continue
            v = val.strip()
            if v.startswith("data:image/"):
                return v
            if len(v) > 120:
                return f"data:image/png;base64,{v}" if re.match(r"^[A-Za-z0-9+/=]+$", v) else v
        return None

    async def handle_webhook(self, data: dict[str, Any]) -> None:
        """Called by the /channels/whatsapp/webhook HTTP handler."""
        event_data = data.get("data") or {}
        messages = event_data if isinstance(event_data, list) else [event_data]
        for msg in messages:
            key = msg.get("key") or {}
            if key.get("fromMe"):
                continue
            message = msg.get("message") or {}
            body = (
                message.get("conversation") or
                (message.get("extendedTextMessage") or {}).get("text") or
                ""
            )
            if not body:
                continue
            try:
                from neuralclaw.channels.protocol import ChannelMessage
                channel_msg = ChannelMessage(
                    content=body,
                    author_id=key.get("remoteJid", "unknown"),
                    author_name=msg.get("pushName", "Unknown"),
                    channel_id=key.get("remoteJid", ""),
                    metadata={"platform": "whatsapp"},
                )
                await self._dispatch(channel_msg)
            except Exception as exc:
                logger.warning("[WhatsApp] webhook dispatch error: %s", exc)


async def _handle_whatsapp_status(request: web.Request) -> web.Response:
    adapter = request.app.get("whatsapp_adapter")
    state = _extract_whatsapp_debug_state(adapter)
    return web.json_response(state)


async def _handle_whatsapp_webhook(request: web.Request) -> web.Response:
    adapter = request.app.get("whatsapp_adapter")
    if isinstance(adapter, QRTrackingWhatsAppAdapter):
        try:
            data = await request.json()
            await adapter.handle_webhook(data)
        except Exception as exc:
            logger.warning("[WhatsApp] webhook error: %s", exc)
    return web.json_response({"ok": True})


async def _run_gateway() -> None:
    # Small startup delay so the previous container's Telegram polling
    # has time to stop before this instance starts polling.
    # Prevents 409 Conflict during Railway rolling deploys.
    startup_delay = float(os.getenv("NEURALCLAW_STARTUP_DELAY", "8"))
    if startup_delay > 0:
        logger.info("[runtime] startup delay %.0fs (set NEURALCLAW_STARTUP_DELAY=0 to skip)", startup_delay)
        await asyncio.sleep(startup_delay)

    config = load_config()

    # Inject knowledge base hint into persona so the LLM knows to use read_file
    if _KNOWLEDGE_PATH.exists():
        knowledge_hint = (
            f"\n\nYou have access to a knowledge base stored at '{_KNOWLEDGE_PATH}'. "
            "When a user asks about stored information, company details, or anything "
            "that might be in the knowledge base, use the read_file tool to read it first."
        )
        config.persona = (config.persona or "") + knowledge_hint

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
    whatsapp_adapter: Any = None
    if whatsapp_session:
        whatsapp_adapter = QRTrackingWhatsAppAdapter(whatsapp_session)
        gw.add_channel(whatsapp_adapter)

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
    app["whatsapp_adapter"] = whatsapp_adapter
    app.add_routes([
        web.get("/health", _handle_health),
        web.get("/channels/whatsapp", _handle_whatsapp_status),
        web.post("/channels/whatsapp/webhook", _handle_whatsapp_webhook),
        web.post("/a2a/message", _handle_a2a_message),
    ])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", mesh_port)
    await site.start()

    loop = asyncio.get_running_loop()

    async def _shutdown(sig_name: str) -> None:
        logger.info("[runtime] received %s — shutting down gracefully", sig_name)
        await gw.stop()
        await runner.cleanup()
        loop.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(_shutdown(s.name)))
        except NotImplementedError:
            # Windows does not support add_signal_handler
            pass

    logger.info("[runtime] gateway started on port %d (mesh_enabled=%s, peers=%d)",
                mesh_port, gw._mesh_router.enabled, len(gw._mesh_router.peers))
    await gw.run_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    asyncio.run(_run_gateway())
