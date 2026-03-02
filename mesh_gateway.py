#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import signal
import sys
import time
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
        self._voice_manager: VoiceCallManager | None = None
        self._last_request_context: dict[str, str] | None = None

    async def initialize(self) -> None:
        await super().initialize()
        if self._voice_manager:
            self._voice_manager.register_tools()

    def _get_channel_type(self, msg: Any) -> str:
        """Override to honor metadata.source for Slack/Telegram/Discord."""
        if msg.raw:
            raw_module = type(msg.raw).__module__
            if "telegram" in raw_module:
                return "TELEGRAM"
            if "discord" in raw_module:
                return "DISCORD"
            if "slack" in raw_module:
                return "SLACK"

        meta = msg.metadata or {}
        source = str(meta.get("source", "")).lower()
        if "telegram" in source:
            return "TELEGRAM"
        if "discord" in source:
            return "DISCORD"
        if "slack" in source:
            return "SLACK"
        if "whatsapp" in source:
            return "WHATSAPP"
        if "signal" in source:
            return "SIGNAL"
        if "web" in source:
            return "CLI"
        return "CLI"

    def _get_source_adapter(self, msg: Any) -> str | None:
        """Override to honor metadata.source for Slack/Telegram/Discord."""
        if msg.raw:
            raw_module = type(msg.raw).__module__
            if "telegram" in raw_module:
                return "telegram"
            if "discord" in raw_module:
                return "discord"
            if "slack" in raw_module:
                return "slack"

        meta = msg.metadata or {}
        source = str(meta.get("source", "")).lower()
        if "telegram" in source:
            return "telegram"
        if "discord" in source:
            return "discord"
        if "slack" in source:
            return "slack"
        if "whatsapp" in source:
            return "whatsapp"
        if "signal" in source:
            return "signal"
        if "web" in source:
            return "web"
        return None

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
        if not channel_id.startswith("voice:"):
            source_map = {
                "TELEGRAM": "telegram",
                "DISCORD": "discord",
                "SLACK": "slack",
                "WHATSAPP": "whatsapp",
                "SIGNAL": "signal",
                "WEB": "web",
            }
            source_channel = source_map.get(channel_type_name.upper())
            if source_channel:
                self._last_request_context = {
                    "source_channel": source_channel,
                    "channel_id": channel_id,
                    "author_id": author_id,
                    "author_name": author_name,
                }

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


async def _handle_voice_start(request: web.Request) -> web.Response:
    manager: VoiceCallManager = request.app["voice_manager"]
    return await manager.handle_start(request)


async def _handle_voice_continue(request: web.Request) -> web.Response:
    manager: VoiceCallManager = request.app["voice_manager"]
    return await manager.handle_continue(request)


async def _handle_voice_status(request: web.Request) -> web.Response:
    manager: VoiceCallManager = request.app["voice_manager"]
    return await manager.handle_status(request)


class VoiceCallManager:
    def __init__(self, gateway: "MeshAwareGateway") -> None:
        self._gateway = gateway
        self._enabled = os.getenv("NEURALCLAW_VOICE_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
        self._provider = os.getenv("NEURALCLAW_VOICE_PROVIDER", "twilio").strip().lower()
        self._require_confirmation = os.getenv("NEURALCLAW_VOICE_REQUIRE_CONFIRM", "true").lower() in {"1", "true", "yes", "on"}
        self._twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
        self._twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
        self._twilio_number = os.getenv("TWILIO_PHONE_NUMBER", "").strip()
        self._voice_name = os.getenv("TWILIO_VOICE", "Polly.Joanna").strip() or "Polly.Joanna"
        self._sessions: dict[str, dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled and self._provider == "twilio"

    def register_tools(self) -> None:
        if not self.enabled:
            return
        self._gateway._skills.register_tool(
            name="place_call",
            description=(
                "Place a live outbound phone call through Twilio, speak to the callee, "
                "listen to their spoken replies, and continue the conversation."
            ),
            function=self.place_call,
            parameters={
                "phone_number": {
                    "type": "string",
                    "description": "Destination phone number in E.164 format, e.g. +15551234567",
                },
                "purpose": {
                    "type": "string",
                    "description": "What the call is about and what you should handle on the user's behalf",
                },
            },
        )
        logger.info("[Voice] Twilio voice tool registered")

    def _public_base_url(self) -> str | None:
        explicit = os.getenv("NEURALCLAW_PUBLIC_BASE_URL", "").strip().rstrip("/")
        if explicit:
            return explicit
        railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "").strip()
        if railway_domain:
            return f"https://{railway_domain}".rstrip("/")
        railway_static = os.getenv("RAILWAY_STATIC_URL", "").strip()
        if railway_static:
            return f"https://{railway_static}".rstrip("/")
        return None

    def _xml_response(self, body: str) -> web.Response:
        return web.Response(
            text=f'<?xml version="1.0" encoding="UTF-8"?><Response>{body}</Response>',
            content_type="text/xml",
        )

    def _gather_twiml(self, prompt: str, action_url: str) -> web.Response:
        safe_prompt = html.escape(prompt, quote=False)
        safe_action = html.escape(action_url, quote=True)
        safe_voice = html.escape(self._voice_name, quote=True)
        return self._xml_response(
            f'<Gather input="speech" action="{safe_action}" method="POST" speechTimeout="auto" timeout="6">'
            f'<Say voice="{safe_voice}">{safe_prompt}</Say>'
            f"</Gather>"
            f'<Redirect method="POST">{safe_action}</Redirect>'
        )

    def _normalize_destination(self, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            return raw
        if raw.startswith("+"):
            return raw
        digits = "".join(ch for ch in raw if ch.isdigit())
        return f"+{digits}" if digits else raw

    async def place_call(self, phone_number: str, purpose: str) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "error": "voice agent is not enabled"}
        if not self._twilio_sid or not self._twilio_token or not self._twilio_number:
            return {"ok": False, "error": "twilio credentials are not configured"}

        base_url = self._public_base_url()
        if not base_url:
            return {"ok": False, "error": "public base URL is not available; set NEURALCLAW_PUBLIC_BASE_URL or enable Railway public domain"}

        destination = self._normalize_destination(phone_number)
        objective = purpose.strip()
        if not destination or not objective:
            return {"ok": False, "error": "phone_number and purpose are required"}

        session_id = f"call-{int(time.time() * 1000)}"
        origin = getattr(self._gateway, "_last_request_context", None) or {}
        self._sessions[session_id] = {
            "purpose": objective,
            "turns": 0,
            "retries": 0,
            "created_at": time.time(),
            "origin_channel": str(origin.get("source_channel", "")).strip(),
            "origin_channel_id": str(origin.get("channel_id", "")).strip(),
        }

        payload = {
            "To": destination,
            "From": self._twilio_number,
            "Url": f"{base_url}/voice/twilio/start?session_id={session_id}",
            "Method": "POST",
            "StatusCallback": f"{base_url}/voice/twilio/status?session_id={session_id}",
            "StatusCallbackMethod": "POST",
        }

        try:
            async with aiohttp.ClientSession(auth=aiohttp.BasicAuth(self._twilio_sid, self._twilio_token)) as session:
                async with session.post(
                    f"https://api.twilio.com/2010-04-01/Accounts/{self._twilio_sid}/Calls.json",
                    data=payload,
                ) as resp:
                    data = await resp.json(content_type=None)
                    if resp.status not in {200, 201}:
                        return {"ok": False, "error": data.get("message", f"twilio call create failed ({resp.status})")}
        except Exception as exc:
            return {"ok": False, "error": f"twilio request failed: {exc}"}

        call_sid = str(data.get("sid", "")).strip()
        if call_sid:
            self._sessions[session_id]["call_sid"] = call_sid

        logger.info("[Voice] outbound call started session=%s call_sid=%s to=%s", session_id, call_sid or "unknown", destination)
        return {
            "ok": True,
            "session_id": session_id,
            "call_sid": call_sid,
            "status": data.get("status", "queued"),
            "to": destination,
            "message": (
                f"Outbound AI call started to {destination}."
                + (" Confirmation mode is configured for audit only." if self._require_confirmation else "")
            ),
        }

    def _resolve_session(self, session_id: str | None) -> dict[str, Any] | None:
        if not session_id:
            return None
        return self._sessions.get(session_id)

    async def _relay_update(self, state: dict[str, Any], text: str) -> None:
        source_channel = str(state.get("origin_channel", "")).strip()
        channel_id = str(state.get("origin_channel_id", "")).strip()
        if not source_channel or not channel_id:
            return

        adapter = self._gateway._channels.get(source_channel)
        if not adapter:
            return

        try:
            await adapter.send(channel_id, text)
        except Exception as exc:
            logger.warning("[Voice] relay send failed: %s", exc)

    async def handle_start(self, request: web.Request) -> web.Response:
        if not self.enabled:
            return self._xml_response("<Say>Voice agent is not enabled.</Say><Hangup/>")

        session_id = request.query.get("session_id")
        state = self._resolve_session(session_id)
        if not state:
            return self._xml_response("<Say>Call session was not found.</Say><Hangup/>")

        if request.method == "POST":
            try:
                form = await request.post()
                call_sid = str(form.get("CallSid", "")).strip()
                if call_sid:
                    state["call_sid"] = call_sid
            except Exception:
                pass

        action_url = f"https://{request.host}/voice/twilio/continue?session_id={session_id}"
        prompt = str(state["purpose"]).strip() or "Hello."
        return self._gather_twiml(prompt, action_url)

    async def handle_continue(self, request: web.Request) -> web.Response:
        if not self.enabled:
            return self._xml_response("<Say>Voice agent is not enabled.</Say><Hangup/>")

        session_id = request.query.get("session_id")
        state = self._resolve_session(session_id)
        if not state:
            return self._xml_response("<Say>Call session was not found.</Say><Hangup/>")

        # Twilio may send GET (after an http→https redirect converts POST→GET)
        # or POST directly. Read params from whichever is present.
        if request.method == "POST":
            params = await request.post()
        else:
            params = request.rel_url.query
        speech = str(params.get("SpeechResult", "")).strip()
        call_sid = str(params.get("CallSid", "")).strip()
        caller = str(params.get("From", "callee")).strip() or "callee"
        if call_sid:
            state["call_sid"] = call_sid

        action_url = f"https://{request.host}/voice/twilio/continue?session_id={session_id}"

        if not speech:
            retries = int(state.get("retries", 0)) + 1
            state["retries"] = retries
            if retries >= 2:
                self._sessions.pop(session_id or "", None)
                return self._xml_response("<Say>I could not hear a response. I will end the call now. Goodbye.</Say><Hangup/>")
            return self._gather_twiml("I did not catch that. Please say that again.", action_url)

        await self._relay_update(state, f"[Voice] Callee: {speech}")

        state["retries"] = 0
        turn_count = int(state.get("turns", 0))
        if turn_count == 0:
            content = (
                "You are on a live phone call. "
                "Speak naturally and conversationally — use short, clear sentences as a real person would in a phone conversation. "
                "Never use bullet points, lists, markdown, or formatting of any kind. "
                "Keep each response to two or three sentences at most. Sound warm, calm, and human. "
                f"Your task for this call: {state['purpose']}. "
                f"The person you called just said: {speech}"
            )
        else:
            content = speech

        try:
            response = await self._gateway.process_message(
                content=content,
                author_id=caller,
                author_name=caller,
                channel_id=f"voice:{call_sid or session_id}",
                channel_type_name="CLI",
            )
        except Exception as exc:
            logger.error("[Voice] call response failed: %s", exc)
            self._sessions.pop(session_id or "", None)
            return self._xml_response("<Say>I hit an internal error and need to end the call. Goodbye.</Say><Hangup/>")

        state["turns"] = turn_count + 1
        trimmed = (response or "").strip()
        if not trimmed:
            self._sessions.pop(session_id or "", None)
            return self._xml_response("<Say>I do not have anything further to add. Goodbye.</Say><Hangup/>")

        await self._relay_update(state, f"[Voice] AI: {trimmed}")

        if any(term in trimmed.lower() for term in ("goodbye", "bye for now", "end the call", "hang up")):
            self._sessions.pop(session_id or "", None)
            safe_voice = html.escape(self._voice_name, quote=True)
            return self._xml_response(f'<Say voice="{safe_voice}">{html.escape(trimmed, quote=False)}</Say><Hangup/>')

        return self._gather_twiml(trimmed, action_url)

    async def handle_status(self, request: web.Request) -> web.Response:
        session_id = request.query.get("session_id")
        if session_id:
            form = await request.post()
            call_state = str(form.get("CallStatus", "")).strip().lower()
            if call_state in {"completed", "busy", "failed", "no-answer", "canceled"}:
                self._sessions.pop(session_id, None)
        return web.Response(status=204)


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
_WA_SESSION_BASE = Path(
    os.getenv(
        "NEURALCLAW_WHATSAPP_SESSION_DIR",
        ("/data/whatsapp" if Path("/data").exists() else str(Path.home() / ".neuralclaw" / "data" / "whatsapp")),
    )
)

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
        self._session_dir = str(_WA_SESSION_BASE / f"session-{session_name}")
        self._reply_mode = os.getenv("NEURALCLAW_WHATSAPP_REPLY_MODE", "direct_only").strip().lower()
        raw_allowed = os.getenv("NEURALCLAW_WHATSAPP_ALLOWED_IDS", "").strip()
        self._allowed_ids = {item.strip() for item in raw_allowed.split(",") if item.strip()}
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._restart_task: asyncio.Task[None] | None = None
        self._stopping: bool = False
        self.qr: str | None = None
        self.connected: bool = False
        self.ready: bool = False
        self.reason: str | None = "initializing"
        self._recent_outbound: dict[str, float] = {}

    async def start(self) -> None:
        self._stopping = False
        _WA_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
        Path(self._session_dir).mkdir(parents=True, exist_ok=True)
        bridge_file = _WA_BRIDGE_DIR / "bridge.mjs"
        bridge_file.write_text(self._bridge_script(), encoding="utf-8")
        await self._spawn_bridge(bridge_file)

    async def stop(self) -> None:
        self._stopping = True
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
        if self._restart_task:
            self._restart_task.cancel()
            try:
                await self._restart_task
            except asyncio.CancelledError:
                pass

    async def _spawn_bridge(self, bridge_file: Path) -> None:
        self.reason = "starting_baileys_bridge"
        self._process = await asyncio.create_subprocess_exec(
            "node", str(bridge_file),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1024 * 1024,
        )
        self._reader_task = asyncio.create_task(self._read_messages())
        self._stderr_task = asyncio.create_task(self._read_stderr())
        logger.info("[WhatsApp] Baileys bridge started - session=%s", self._session_name)

    async def _restart_after_delay(self, delay_seconds: float = 3.0) -> None:
        if self._stopping:
            return
        await asyncio.sleep(delay_seconds)
        if self._stopping:
            return
        try:
            bridge_file = _WA_BRIDGE_DIR / "bridge.mjs"
            await self._spawn_bridge(bridge_file)
        except Exception as exc:
            self.reason = f"restart_failed:{exc}"
            logger.warning("[WhatsApp] bridge restart failed: %s", exc)

    async def send(self, channel_id: str, content: str, **kwargs: Any) -> None:
        if self._process and self._process.stdin:
            key = f"{channel_id}::{content}"
            self._recent_outbound[key] = time.time()
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
                        # Avoid reply loops when testing self-chat:
                        # ignore the echo of messages sent by the bot itself.
                        from_me = bool(data.get("from_me", False))
                        chat_id = data.get("chat_id", "")
                        content = data.get("content", "")
                        sender_id = str(data.get("from", "") or "").strip()
                        # Optional strict allowlist: if set, ignore everyone else.
                        if self._allowed_ids and sender_id not in self._allowed_ids:
                            continue
                        # By default, do not auto-reply in WhatsApp groups.
                        # Set NEURALCLAW_WHATSAPP_REPLY_MODE=all to reply in groups too.
                        if self._reply_mode != "all":
                            if not chat_id or chat_id.endswith("@g.us"):
                                continue
                        key = f"{chat_id}::{content}"
                        if from_me and key in self._recent_outbound:
                            # Drop only recent bot echoes.
                            ts = self._recent_outbound.get(key, 0.0)
                            if time.time() - ts < 30:
                                self._recent_outbound.pop(key, None)
                                continue

                        from neuralclaw.channels.protocol import ChannelMessage
                        msg = ChannelMessage(
                            content=content,
                            author_id=data.get("from", "unknown"),
                            author_name=data.get("name", "Unknown"),
                            channel_id=chat_id,
                            metadata={"platform": "whatsapp", "source": "whatsapp", "channel": "whatsapp"},
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
            if not self._stopping and not self._restart_task:
                self._restart_task = asyncio.create_task(self._restart_after_delay())
                self._restart_task.add_done_callback(lambda _: setattr(self, "_restart_task", None))

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
import {{ makeWASocket, useMultiFileAuthState, DisconnectReason, fetchLatestBaileysVersion, Browsers }} from '@whiskeysockets/baileys';
import QRCode from 'qrcode';
import {{ createInterface }} from 'readline';

const SESSION_DIR = {session_dir_js};

async function main() {{
    const rl = createInterface({{ input: process.stdin }});
    const wait = (ms) => new Promise(res => setTimeout(res, ms));

    while (true) {{
        const {{ state, saveCreds }} = await useMultiFileAuthState(SESSION_DIR);
        const {{ version }} = await fetchLatestBaileysVersion();
        process.stdout.write(JSON.stringify({{ type: 'state', value: 'connecting' }}) + '\\n');
        process.stdout.write(JSON.stringify({{ type: 'state', value: `version:${{version.join('.')}}` }}) + '\\n');

        const sock = makeWASocket({{
            auth: state,
            version,
            browser: Browsers.ubuntu('Chrome'),
            printQRInTerminal: false,
            markOnlineOnConnect: false,
            syncFullHistory: false,
        }});

        const closed = await new Promise((resolve) => {{
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
                    if (code === DisconnectReason.loggedOut) {{
                        process.stdout.write(JSON.stringify({{ type: 'state', value: 'logged_out' }}) + '\\n');
                        resolve({{ restart: false }});
                        return;
                    }}
                    process.stdout.write(JSON.stringify({{ type: 'state', value: `closed:${{code}}` }}) + '\\n');
                    resolve({{ restart: true }});
                }}
            }});

            sock.ev.on('messages.upsert', async ({{ messages }}) => {{
                for (const msg of messages) {{
                    if (!msg.message) continue;
                    const body = msg.message.conversation
                        ?? msg.message.extendedTextMessage?.text
                        ?? '';
                    if (!body) continue;
                    process.stdout.write(JSON.stringify({{
                        type: 'message', content: body,
                        from_me: Boolean(msg.key.fromMe),
                        from: msg.key.remoteJid, name: msg.pushName || 'Unknown',
                        chat_id: msg.key.remoteJid,
                    }}) + '\\n');
                }}
            }});

            rl.on('line', async line => {{
                try {{
                    const cmd = JSON.parse(line);
                    if (cmd.type === 'send') await sock.sendMessage(cmd.to, {{ text: cmd.content }});
                }} catch {{}}
            }});
        }});

        if (!closed?.restart) {{
            break;
        }}
        await wait(3000);
    }}
}}

main().catch(e => process.stderr.write(`bridge_fatal ${{String(e)}}\\n`));
"""



async def _handle_whatsapp_status(request: web.Request) -> web.Response:
    adapter = request.app.get("whatsapp_adapter")
    state = _extract_whatsapp_debug_state(adapter)
    return web.json_response(state)


class LocalSlackAdapter(_ChannelAdapterBase):
    """
    Slack adapter pinned in runtime template so we can enforce metadata.source
    regardless of the currently installed neuralclaw package version.
    """

    name = "slack"

    def __init__(self, bot_token: str, app_token: str) -> None:
        super().__init__()
        self._bot_token = bot_token
        self._app_token = app_token
        self._app: Any = None
        self._handler: Any = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        try:
            from slack_bolt.async_app import AsyncApp
            from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
            from neuralclaw.channels.protocol import ChannelMessage
        except ImportError as exc:
            raise RuntimeError("slack-bolt not installed. Run: pip install slack-bolt") from exc

        self._app = AsyncApp(token=self._bot_token)
        adapter = self

        @self._app.event("message")
        async def handle_message(event: dict, say: Any) -> None:
            if event.get("bot_id") or event.get("subtype"):
                return
            logger.info(
                "[Slack] inbound message event channel=%s user=%s subtype=%s",
                event.get("channel"),
                event.get("user"),
                event.get("subtype"),
            )

            msg = ChannelMessage(
                content=event.get("text", ""),
                author_id=event.get("user", "unknown"),
                author_name=event.get("user", "Unknown"),
                channel_id=event.get("channel", ""),
                raw=event,
                metadata={
                    "platform": "slack",
                    "source": "slack",
                    "channel": "slack",
                    "thread_ts": event.get("thread_ts"),
                },
            )
            try:
                await adapter._dispatch(msg)
            except Exception:
                logger.exception("[Slack] dispatch failure for message event")

        @self._app.event("app_mention")
        async def handle_mention(event: dict, say: Any) -> None:
            logger.info(
                "[Slack] inbound app_mention event channel=%s user=%s",
                event.get("channel"),
                event.get("user"),
            )
            text = event.get("text", "")
            if "<@" in text:
                text = text.split(">", 1)[-1].strip()

            msg = ChannelMessage(
                content=text,
                author_id=event.get("user", "unknown"),
                author_name=event.get("user", "Unknown"),
                channel_id=event.get("channel", ""),
                raw=event,
                metadata={
                    "platform": "slack",
                    "source": "slack",
                    "channel": "slack",
                    "is_mention": True,
                    "thread_ts": event.get("thread_ts", event.get("ts")),
                },
            )
            try:
                await adapter._dispatch(msg)
            except Exception:
                logger.exception("[Slack] dispatch failure for app_mention event")

        self._handler = AsyncSocketModeHandler(self._app, self._app_token)
        self._task = asyncio.create_task(self._handler.start_async())
        logger.info("[Slack] Bot started in Socket Mode (local adapter)")

    async def stop(self) -> None:
        if self._handler:
            await self._handler.close_async()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def send(self, channel_id: str, content: str, **kwargs: Any) -> None:
        if self._app:
            thread_ts = kwargs.get("thread_ts")
            await self._app.client.chat_postMessage(
                channel=channel_id,
                text=content,
                thread_ts=thread_ts,
            )




async def _run_gateway() -> None:
    # Small startup delay so the previous container's Telegram polling
    # has time to stop before this instance starts polling.
    # Prevents 409 Conflict during Railway rolling deploys.
    startup_delay = float(os.getenv("NEURALCLAW_STARTUP_DELAY", "8"))
    if startup_delay > 0:
        logger.info("[runtime] startup delay %.0fs (set NEURALCLAW_STARTUP_DELAY=0 to skip)", startup_delay)
        await asyncio.sleep(startup_delay)

    config = load_config()
    if os.getenv("NEURALCLAW_VOICE_ENABLED", "false").lower() in {"1", "true", "yes", "on"}:
        if "place_call" not in config.policy.allowed_tools:
            config.policy.allowed_tools.append("place_call")

    # Inject knowledge base hint into persona so the LLM knows to use read_file
    if _KNOWLEDGE_PATH.exists():
        knowledge_hint = (
            f"\n\nYou have access to a knowledge base stored at '{_KNOWLEDGE_PATH}'. "
            "When a user asks about stored information, company details, or anything "
            "that might be in the knowledge base, use the read_file tool to read it first."
        )
        config.persona = (config.persona or "") + knowledge_hint

    gw = MeshAwareGateway(config)
    voice_manager = VoiceCallManager(gw)
    gw._voice_manager = voice_manager

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
        gw.add_channel(LocalSlackAdapter(slack_bot, slack_app))

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
    app["voice_manager"] = voice_manager
    app.add_routes([
        web.get("/health", _handle_health),
        web.get("/channels/whatsapp", _handle_whatsapp_status),
        web.post("/a2a/message", _handle_a2a_message),
        web.get("/voice/twilio/start", _handle_voice_start),
        web.post("/voice/twilio/start", _handle_voice_start),
        web.get("/voice/twilio/continue", _handle_voice_continue),
        web.post("/voice/twilio/continue", _handle_voice_continue),
        web.post("/voice/twilio/status", _handle_voice_status),
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
