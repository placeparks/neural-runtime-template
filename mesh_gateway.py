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
        return {"enabled": False, "connected": False, "ready": False, "qr": None, "reason": "adapter_missing"}

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
    reason_candidates = ("reason", "last_error", "error", "_last_error")

    connected: bool | None = None
    ready: bool | None = None
    qr: str | None = None
    reason: str | None = None

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
    for name in reason_candidates:
        if hasattr(adapter, name):
            raw = _maybe_call(getattr(adapter, name))
            if isinstance(raw, str) and raw.strip():
                reason = raw.strip()
                break

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
        if not reason:
            for name in reason_candidates:
                if hasattr(node, name):
                    raw = _maybe_call(getattr(node, name))
                    if isinstance(raw, str) and raw.strip():
                        reason = raw.strip()
                        break

    # Reasonable fallback semantics.
    if ready is None and connected is not None:
        ready = connected
    if connected is None and ready is not None:
        connected = ready
    if connected is None:
        connected = False
    if ready is None:
        ready = False
    if not reason and not ready and not qr:
        reason = "waiting_for_qr"

    return {
        "enabled": True,
        "connected": connected,
        "ready": ready,
        "qr": qr,
        "reason": reason,
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
# QRTrackingWhatsAppAdapter — direct Baileys bridge (OpenClaw pattern).
#
# Runs a Node.js Baileys process as a subprocess. The bridge script is
# written to /app/wa_bridge/bridge.mjs at startup; node_modules are
# pre-installed there by the Docker build stage.
#
# No Evolution API, no Chromium — just Node.js + Baileys, same as OpenClaw.
# ---------------------------------------------------------------------------

_WA_BRIDGE_DIR = Path("/app/wa_bridge")

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
    Direct Baileys WhatsApp adapter - same pattern as OpenClaw.
    Spawns a Node.js subprocess running @whiskeysockets/baileys.
    Exposes .qr / .connected / .ready for the /channels/whatsapp endpoint.
    """

    name = "whatsapp"

    def __init__(self, session_name: str = "default") -> None:
        super().__init__()
        self._session_name = session_name
        self._session_dir = str(_WA_BRIDGE_DIR / f"session-{session_name}")
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self.qr: str | None = None
        self.connected: bool = False
        self.ready: bool = False
        self.reason: str | None = "initializing"

    async def start(self) -> None:
        _WA_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
        bridge_file = _WA_BRIDGE_DIR / "bridge.mjs"
        bridge_file.write_text(self._bridge_script(), encoding="utf-8")

        self.reason = "starting_baileys_bridge"
        self._process = await asyncio.create_subprocess_exec(
            "node", str(bridge_file),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_messages())
        self._stderr_task = asyncio.create_task(self._read_stderr())
        logger.info("[WhatsApp] Baileys bridge started - session=%s", self._session_name)

    async def stop(self) -> None:
        if self._process:
            self._process.terminate()
            await self._process.wait()
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass

    async def send(self, channel_id: str, content: str, **kwargs: Any) -> None:
        if self._process and self._process.stdin:
            payload = json.dumps({"type": "send", "to": channel_id, "content": content})
            self._process.stdin.write(f"{payload}\n".encode())
            await self._process.stdin.drain()

    async def _read_messages(self) -> None:
        if not self._process or not self._process.stdout:
            return
        while True:
            try:
                line = await self._process.stdout.readline()
                if not line:
                    break
                data = json.loads(line.decode().strip())
                t = data.get("type")
                if t == "qr":
                    self.qr = data.get("data")
                    self.connected = False
                    self.ready = False
                    self.reason = "scan_qr_in_dashboard"
                    logger.info("[WhatsApp] QR received for session=%s", self._session_name)
                elif t == "ready":
                    self.qr = None
                    self.connected = True
                    self.ready = True
                    self.reason = None
                    logger.info("[WhatsApp] session=%s connected", self._session_name)
                elif t == "state":
                    state = str(data.get("value", "")).strip()
                    if state:
                        self.reason = f"state:{state}"
                elif t == "message":
                    try:
                        from neuralclaw.channels.protocol import ChannelMessage
                        msg = ChannelMessage(
                            content=data.get("content", ""),
                            author_id=data.get("from", "unknown"),
                            author_name=data.get("name", "Unknown"),
                            channel_id=data.get("chat_id", ""),
                            metadata={"platform": "whatsapp"},
                        )
                        await self._dispatch(msg)
                    except Exception as exc:
                        logger.warning("[WhatsApp] dispatch error: %s", exc)
            except json.JSONDecodeError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.reason = f"read_error:{exc}"
                logger.warning("[WhatsApp] read error: %s", exc)
                await asyncio.sleep(1)
        if self._process and self._process.returncode not in (None, 0):
            self.connected = False
            self.ready = False
            self.reason = f"bridge_exited:{self._process.returncode}"

    async def _read_stderr(self) -> None:
        if not self._process or not self._process.stderr:
            return
        while True:
            try:
                line = await self._process.stderr.readline()
                if not line:
                    break
                text = line.decode(errors="ignore").strip()
                if not text:
                    continue
                self.reason = text[:500]
                logger.warning("[WhatsApp][bridge] %s", text)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.reason = f"stderr_error:{exc}"
                logger.warning("[WhatsApp] stderr read error: %s", exc)
                await asyncio.sleep(1)

    def _bridge_script(self) -> str:
        session_dir_js = json.dumps(self._session_dir)
        return f"""
import {{ makeWASocket, useMultiFileAuthState, DisconnectReason }} from '@whiskeysockets/baileys';
import QRCode from 'qrcode';
import {{ createInterface }} from 'readline';

const SESSION_DIR = {session_dir_js};

async function main() {{
    const {{ state, saveCreds }} = await useMultiFileAuthState(SESSION_DIR);
    process.stdout.write(JSON.stringify({{ type: 'state', value: 'connecting' }}) + '\\n');
    const sock = makeWASocket({{ auth: state, printQRInTerminal: false }});

    sock.ev.on('creds.update', saveCreds);

    sock.ev.on('connection.update', async ({{ connection, qr, lastDisconnect }}) => {{
        if (qr) {{
            try {{
                const dataUrl = await QRCode.toDataURL(qr);
                process.stdout.write(JSON.stringify({{ type: 'qr', data: dataUrl }}) + '\\n');
            }} catch {{
                process.stdout.write(JSON.stringify({{ type: 'qr', data: qr }}) + '\\n');
            }}
        }}
        if (connection === 'open') {{
            process.stdout.write(JSON.stringify({{ type: 'ready' }}) + '\\n');
        }}
        if (connection === 'close') {{
            const code = lastDisconnect?.error?.output?.statusCode;
            process.stderr.write(`connection_close code=${{code}}\\n`);
            if (code !== DisconnectReason.loggedOut) setTimeout(main, 3000);
        }}
    }});

    sock.ev.on('messages.upsert', async ({{ messages }}) => {{
        for (const msg of messages) {{
            if (!msg.message || msg.key.fromMe) continue;
            const body = msg.message.conversation
                ?? msg.message.extendedTextMessage?.text
                ?? '';
            if (!body) continue;
            process.stdout.write(JSON.stringify({{
                type: 'message', content: body,
                from: msg.key.remoteJid, name: msg.pushName || 'Unknown',
                chat_id: msg.key.remoteJid,
            }}) + '\\n');
        }}
    }});

    const rl = createInterface({{ input: process.stdin }});
    rl.on('line', async line => {{
        try {{
            const cmd = JSON.parse(line);
            if (cmd.type === 'send') await sock.sendMessage(cmd.to, {{ text: cmd.content }});
        }} catch {{}}
    }});
}}

main().catch(e => process.stderr.write(`bridge_fatal ${{String(e)}}\\n`));
"""



async def _handle_whatsapp_status(request: web.Request) -> web.Response:
    adapter = request.app.get("whatsapp_adapter")
    state = _extract_whatsapp_debug_state(adapter)
    return web.json_response(state)




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
