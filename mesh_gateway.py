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

from neuralclaw.bus.neural_bus import EventType
from neuralclaw.cortex.action.policy import RequestContext
from neuralclaw.cortex.reasoning.deliberate import ConfidenceEnvelope, DeliberativeReasoner
from neuralclaw.config import get_api_key, load_config
from neuralclaw.gateway import NeuralClawGateway

# ---------------------------------------------------------------------------
# Emoji stripper — TTS engines read emoji as "hand wave sign" etc.
# ---------------------------------------------------------------------------
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002702-\U000027B0"   # dingbats
    "\U000024C2-\U0001F251"   # enclosed characters
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\ufe0f"                   # variation selector
    "]+",
    flags=re.UNICODE,
)


def _strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text).strip()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


_SEARCH_QUERY_RE = re.compile(
    r"\b(search|find|look up|google|browse|review|reviews|latest|news|reddit|trustpilot)\b",
    re.IGNORECASE,
)


def _looks_like_web_search_query(text: str) -> bool:
    return bool(_SEARCH_QUERY_RE.search(text or ""))


async def _openai_web_search(query: str) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    provider = os.getenv("NEURALCLAW_PROVIDER", "").strip().lower()
    base_url = os.getenv("NEURALCLAW_OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    model = os.getenv("NEURALCLAW_MODEL", "gpt-4o").strip() or "gpt-4o"

    if not api_key:
        return {"error": "OPENAI_API_KEY is not configured for provider-backed search fallback."}
    if provider != "openai":
        return {"error": f"Provider-backed search fallback is only enabled for OpenAI; current provider is '{provider or 'unknown'}'."}
    if base_url != "https://api.openai.com/v1":
        return {"error": f"Provider-backed search fallback requires the default OpenAI base URL; current base URL is '{base_url}'."}

    payload = {
        "model": model,
        "tools": [{"type": "web_search"}],
        "tool_choice": "auto",
        "include": ["web_search_call.action.sources"],
        "input": query,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(
                "https://api.openai.com/v1/responses",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                raw_text = await resp.text()
                if resp.status != 200:
                    return {"error": f"OpenAI web search failed ({resp.status}): {raw_text[:300]}"}
        data = json.loads(raw_text)
    except Exception as exc:
        return {"error": f"OpenAI web search request failed: {exc}"}

    output_text = str(data.get("output_text") or "").strip()
    results: list[dict[str, str]] = []
    sources: list[dict[str, str]] = []

    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") not in ("output_text", "text"):
                continue
            text_value = str(content.get("text") or "").strip()
            if text_value and not output_text:
                output_text = text_value
            for ann in content.get("annotations", []):
                if ann.get("type") != "url_citation":
                    continue
                url = str(ann.get("url") or "").strip()
                title = str(ann.get("title") or url or "Source").strip()
                if url:
                    sources.append({"title": title, "url": url})

    for source in sources[:5]:
        results.append({
            "title": source["title"],
            "snippet": output_text[:500] if output_text else source["title"],
            "url": source["url"],
        })

    if not output_text and not results:
        return {"error": "OpenAI web search returned no usable output."}

    return {
        "query": query,
        "provider": "openai_web_search",
        "summary": output_text,
        "results": results,
        "sources": sources,
    }


async def _patched_web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    from neuralclaw.cortex.action.network import validate_url
    from neuralclaw.skills.builtins import web_search as _web_search_mod

    url_check = validate_url(_web_search_mod._DUCKDUCKGO_API)
    if not url_check.allowed:
        return {"error": f"URL blocked by SSRF policy: {url_check.reason}"}

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain;q=0.9,*/*;q=0.8",
    }
    params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(
                _web_search_mod._DUCKDUCKGO_API,
                params=params,
                timeout=aiohttp.ClientTimeout(total=12),
            ) as resp:
                raw_text = await resp.text()
                if resp.status not in (200, 202):
                    return {"error": f"Search failed with status {resp.status}: {raw_text[:200]}"}

        try:
            data = json.loads(raw_text or "{}")
        except json.JSONDecodeError:
            return {"error": f"Search returned non-JSON response (status {_safe_status(resp)})"}

        results: list[dict[str, str]] = []
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", ""),
                "snippet": data.get("Abstract", ""),
                "url": data.get("AbstractURL", ""),
            })

        for topic in data.get("RelatedTopics", [])[: max_results * 2]:
            if isinstance(topic, dict) and "Text" in topic:
                topic_url = topic.get("FirstURL", "")
                if topic_url:
                    topic_url_check = validate_url(topic_url)
                    if not topic_url_check.allowed:
                        topic_url = "[blocked]"
                results.append({
                    "title": topic.get("Text", "")[:100],
                    "snippet": topic.get("Text", ""),
                    "url": topic_url,
                })
            elif isinstance(topic, dict) and isinstance(topic.get("Topics"), list):
                for nested in topic["Topics"]:
                    if isinstance(nested, dict) and "Text" in nested:
                        topic_url = nested.get("FirstURL", "")
                        if topic_url:
                            topic_url_check = validate_url(topic_url)
                            if not topic_url_check.allowed:
                                topic_url = "[blocked]"
                        results.append({
                            "title": nested.get("Text", "")[:100],
                            "snippet": nested.get("Text", ""),
                            "url": topic_url,
                        })

        if results:
            return {
                "query": query,
                "results": results[:max_results],
                "status": "ok" if resp.status == 200 else "accepted_partial",
            }

        if resp.status == 202:
            openai_fallback = await _openai_web_search(query)
            if not openai_fallback.get("error"):
                return openai_fallback
            return {
                "query": query,
                "results": [],
                "warning": "Search provider returned 202 Accepted with no usable results. Try a broader query.",
                "fallback_error": openai_fallback.get("error"),
            }

        openai_fallback = await _openai_web_search(query)
        if not openai_fallback.get("error"):
            return openai_fallback
        return {
            "message": f"No results found for '{query}'",
            "results": [],
            "fallback_error": openai_fallback.get("error"),
        }
    except Exception as exc:
        openai_fallback = await _openai_web_search(query)
        if not openai_fallback.get("error"):
            return openai_fallback
        return {"error": f"Search failed: {exc}", "fallback_error": openai_fallback.get("error")}


def _safe_status(resp: Any) -> Any:
    try:
        return resp.status
    except Exception:
        return "unknown"


from neuralclaw.skills.builtins import web_search as _web_search_mod
_web_search_mod.web_search = _patched_web_search


_ORIGINAL_DELIBERATE_REASON = DeliberativeReasoner.reason


async def _patched_deliberative_reason(
    self: DeliberativeReasoner,
    signal: Any,
    memory_ctx: Any,
    tools: list[Any] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> ConfidenceEnvelope:
    if not _looks_like_web_search_query(getattr(signal, "content", "")):
        return await _ORIGINAL_DELIBERATE_REASON(self, signal, memory_ctx, tools, conversation_history)

    web_tool = next((t for t in (tools or []) if getattr(t, "name", "") == "web_search"), None)
    if not web_tool:
        return ConfidenceEnvelope(
            response="Web search was requested, but this agent does not have the web_search tool enabled.",
            confidence=0.0,
            source="error",
            uncertainty_factors=["web_search_unavailable"],
        )

    request_ctx = RequestContext(request_id=getattr(signal, "id", "forced-web-search"))

    class _ForcedToolCall:
        def __init__(self, signal_id: str, query: str) -> None:
            self.id = f"forced-web-search-{signal_id}"
            self.name = "web_search"
            self.arguments = {"query": query, "max_results": 5}

    forced_search = await self._execute_tool_call(
        _ForcedToolCall(getattr(signal, "id", "forced"), getattr(signal, "content", "")),
        tools or [],
        request_ctx,
    )
    if isinstance(forced_search, dict) and forced_search.get("error"):
        return ConfidenceEnvelope(
            response=f"Web search failed: {forced_search['error']}",
            confidence=0.0,
            source="error",
            uncertainty_factors=["web_search_failed"],
        )

    await self._bus.publish(
        EventType.REASONING_STARTED,
        {"signal_id": signal.id, "path": "deliberative", "tools_available": len(tools or [])},
        source="reasoning.deliberate",
    )

    messages = self._build_messages(signal, memory_ctx, conversation_history)
    messages.append({
        "role": "system",
        "content": (
            "Web search was executed for this request. Base your answer on these results. "
            "If the results are sparse or inconclusive, say that clearly instead of pretending to browse.\n\n"
            f"WEB_SEARCH_RESULTS:\n{json.dumps(forced_search, ensure_ascii=False)}"
        ),
    })

    tool_defs = [t.to_openai_format() for t in (tools or [])] if tools else None
    tool_calls_made = 1
    iterations = 0

    while iterations < self.MAX_ITERATIONS:
        iterations += 1
        try:
            response = await self._provider.complete(messages=messages, tools=tool_defs)
        except Exception as e:
            return ConfidenceEnvelope(
                response=f"I encountered an error: {str(e)}",
                confidence=0.0,
                source="error",
                uncertainty_factors=["provider_error"],
            )

        if response.tool_calls and tools:
            for tc in response.tool_calls:
                tool_calls_made += 1
                result = await self._execute_tool_call(tc, tools, request_ctx)
                messages.append({"role": "assistant", "content": None, "tool_calls": [tc.to_dict()]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
            continue

        content = response.content or "I'm not sure how to respond to that."
        confidence = self._estimate_confidence(content, memory_ctx, tool_calls_made)
        source = "tool_verified" if tool_calls_made > 0 else "llm"
        envelope = ConfidenceEnvelope(
            response=content,
            confidence=confidence,
            source=source,
            tool_calls_made=tool_calls_made,
            uncertainty_factors=self._detect_uncertainty(content),
        )

        await self._bus.publish(
            EventType.REASONING_COMPLETE,
            {
                "signal_id": signal.id,
                "confidence": envelope.confidence,
                "source": envelope.source,
                "tool_calls": tool_calls_made,
                "iterations": iterations,
            },
            source="reasoning.deliberate",
        )
        return envelope

    return ConfidenceEnvelope(
        response="I spent too many iterations trying to answer. Let me try a simpler approach â€” could you rephrase?",
        confidence=0.1,
        source="max_iterations",
        uncertainty_factors=["max_iterations_reached"],
        tool_calls_made=tool_calls_made,
    )


DeliberativeReasoner.reason = _patched_deliberative_reason


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

    def _build_reply_kwargs(self, msg: Any) -> dict[str, Any]:
        kwargs = super()._build_reply_kwargs(msg)
        meta = getattr(msg, "metadata", {}) or {}
        if meta.get("platform") == "slack" and meta.get("slack_voice_reply"):
            kwargs["slack_voice_reply"] = True
            filename = str(meta.get("slack_audio_filename") or "").strip()
            if filename:
                kwargs["slack_audio_title"] = f"Voice reply to {filename}"
        return kwargs

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
        message_metadata: dict[str, Any] | None = None,
        raw_message: Any = None,
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
            message_metadata=message_metadata,
            raw_message=raw_message,
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


async def _handle_voice_ws(request: web.Request) -> web.WebSocketResponse:
    """Twilio Media Streams WebSocket — bridges to OpenAI Realtime API."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    manager: VoiceCallManager = request.app["voice_manager"]
    session_id = request.query.get("session_id")
    await manager.handle_stream(ws, session_id)
    return ws


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
        # OpenAI Realtime — preferred key is NEURALCLAW_VOICE_OPENAI_KEY,
        # falls back to OPENAI_API_KEY if the agent is already using OpenAI.
        self._realtime_key = (
            os.getenv("NEURALCLAW_VOICE_OPENAI_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
        self._realtime_model = (
            os.getenv("NEURALCLAW_VOICE_REALTIME_MODEL", "gpt-4o-realtime-preview").strip()
            or "gpt-4o-realtime-preview"
        )
        self._realtime_voice = (
            os.getenv("NEURALCLAW_VOICE_REALTIME_VOICE", "coral").strip() or "coral"
        )
        self._sessions: dict[str, dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled and self._provider == "twilio"

    @property
    def realtime_enabled(self) -> bool:
        """True when an OpenAI key is available for the Realtime API."""
        return self.enabled and bool(self._realtime_key)

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
            "history": [],
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

        # --- OpenAI Realtime path (natural voice, <1s latency) ---
        if self.realtime_enabled:
            # Twilio strips query parameters from WebSocket URLs, so pass
            # session_id via <Parameter> (arrives in start.customParameters).
            # We also keep it in the URL as a fallback.
            ws_url = f"wss://{request.host}/voice/twilio/ws?session_id={session_id}"
            safe_url = html.escape(ws_url, quote=True)
            safe_sid = html.escape(session_id or "", quote=True)
            logger.info("[Voice][Realtime] routing to WebSocket stream session=%s", session_id)
            return self._xml_response(
                f'<Connect>'
                f'<Stream url="{safe_url}">'
                f'<Parameter name="session_id" value="{safe_sid}"/>'
                f'</Stream>'
                f'</Connect>'
            )

        # --- Fallback: Twilio Gather + LLM + Polly TTS ---
        action_url = f"https://{request.host}/voice/twilio/continue?session_id={session_id}"
        channel_id = f"voice:{state.get('call_sid', session_id)}"
        try:
            opening = await self._gateway.process_message(
                content=(
                    "You are starting an outbound phone call right now. "
                    "Say a brief, natural greeting and immediately state the purpose of your call. "
                    "Speak as a real person — one or two sentences, no emoji, no lists, no markdown. "
                    f"Call purpose: {state['purpose']}"
                ),
                author_id="system",
                author_name="system",
                channel_id=channel_id,
                channel_type_name="CLI",
            )
            prompt = _strip_emoji((opening or "").strip()) or "Hello, is this a good time to talk?"
        except Exception as exc:
            logger.error("[Voice] opening generation failed: %s", exc)
            prompt = "Hello, is this a good time to talk?"
        state.setdefault("history", []).append({"role": "ai", "text": prompt})
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

        # Build conversation history for context on every turn
        history: list[dict] = state.setdefault("history", [])
        history_lines = []
        for entry in history[-8:]:  # keep last 8 exchanges to avoid token bloat
            role = "You" if entry["role"] == "ai" else "Callee"
            history_lines.append(f"{role}: {entry['text']}")
        history_text = "\n".join(history_lines)

        content = (
            "You are on a live phone call. Respond naturally and conversationally. "
            "Use short, clear sentences — 1 to 3 max per reply. "
            "No emoji, no markdown, no bullet points. Sound warm and human. "
            "Do not introduce yourself again if you have already done so.\n"
            f"Call objective: {state['purpose']}\n"
            + (f"Conversation so far:\n{history_text}\n" if history_text else "")
            + f"Callee just said: {speech}"
        )

        history.append({"role": "callee", "text": speech})

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
        trimmed = _strip_emoji((response or "").strip())
        if not trimmed:
            self._sessions.pop(session_id or "", None)
            return self._xml_response("<Say>I do not have anything further to add. Goodbye.</Say><Hangup/>")

        history.append({"role": "ai", "text": trimmed})
        await self._relay_update(state, f"[Voice] AI: {trimmed}")

        if any(term in trimmed.lower() for term in ("goodbye", "bye for now", "end the call", "hang up")):
            self._sessions.pop(session_id or "", None)
            safe_voice = html.escape(self._voice_name, quote=True)
            return self._xml_response(f'<Say voice="{safe_voice}">{html.escape(trimmed, quote=False)}</Say><Hangup/>')

        return self._gather_twiml(trimmed, action_url)

    # ------------------------------------------------------------------
    # OpenAI Realtime API bridge
    # ------------------------------------------------------------------

    def _build_call_instructions(self, state: dict[str, Any]) -> str:
        custom_persona = os.getenv("NEURALCLAW_VOICE_PERSONA", "").strip()
        base = custom_persona or (
            "You are a friendly, professional phone caller making an outbound call on behalf of the user."
        )
        return (
            f"{base}\n\n"
            f"Your task for this call: {state['purpose']}\n\n"
            "Speak naturally and conversationally. Keep responses concise — 1 to 3 sentences. "
            "Be warm and human. Do not use markdown, bullet points, or emoji. "
            "When the task is complete or the conversation reaches a natural end, "
            "thank the person warmly and say goodbye."
        )

    async def handle_stream(self, ws: web.WebSocketResponse, url_session_id: str | None) -> None:
        """Bridge a Twilio Media Stream WebSocket to the OpenAI Realtime API."""
        # --- Phase 1: consume the Twilio handshake events ---
        # Twilio sends "connected" then "start" before any media.
        # The "start" event carries callSid, streamSid, and customParameters
        # (where we put session_id, since Twilio strips URL query params).
        session_id = url_session_id
        stream_sid: str | None = None
        call_sid_from_start: str | None = None
        try:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                ev = data.get("event")
                if ev == "connected":
                    continue  # protocol greeting — wait for "start"
                if ev == "start":
                    start = data.get("start", {})
                    stream_sid = data.get("streamSid") or start.get("streamSid")
                    call_sid_from_start = start.get("callSid", "")
                    # URL query param takes priority; fall back to customParameters
                    if not session_id:
                        session_id = start.get("customParameters", {}).get("session_id", "")
                    break
                logger.warning("[Voice][Realtime] unexpected pre-start event: %s", ev)
                break
        except Exception as exc:
            logger.error("[Voice][Realtime] error reading start event: %s", exc)
            return

        if not session_id:
            logger.warning("[Voice][Realtime] no session_id in URL or start customParameters")
            await ws.close()
            return

        state = self._resolve_session(session_id)
        if not state:
            logger.warning("[Voice][Realtime] session not found: %s", session_id)
            await ws.close()
            return

        if call_sid_from_start and not state.get("call_sid"):
            state["call_sid"] = call_sid_from_start

        instructions = self._build_call_instructions(state)
        oai_url = f"wss://api.openai.com/v1/realtime?model={self._realtime_model}"
        oai_headers = {
            "Authorization": f"Bearer {self._realtime_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        # Shared context between the two concurrent bridge coroutines
        ctx: dict[str, Any] = {
            "stream_sid": stream_sid,
            "call_sid": call_sid_from_start or state.get("call_sid", session_id),
            "transcript": [],   # list of (role, text) in chronological order
        }
        done = asyncio.Event()

        try:
            async with aiohttp.ClientSession() as http_sess:
                async with http_sess.ws_connect(oai_url, headers=oai_headers) as oai_ws:
                    await oai_ws.send_json({
                        "type": "session.update",
                        "session": {
                            "turn_detection": {"type": "server_vad"},
                            "input_audio_format": "g711_ulaw",
                            "output_audio_format": "g711_ulaw",
                            "voice": self._realtime_voice,
                            "instructions": instructions,
                            "modalities": ["text", "audio"],
                            "temperature": 0.8,
                            "input_audio_transcription": {"model": "whisper-1"},
                        },
                    })
                    logger.info(
                        "[Voice][Realtime] session configured model=%s voice=%s session=%s",
                        self._realtime_model, self._realtime_voice, session_id,
                    )

                    t1 = asyncio.create_task(
                        self._bridge_twilio_to_openai(ws, oai_ws, ctx, done)
                    )
                    t2 = asyncio.create_task(
                        self._bridge_openai_to_twilio(oai_ws, ws, ctx, done, state)
                    )
                    await asyncio.gather(t1, t2, return_exceptions=True)

        except Exception as exc:
            logger.error("[Voice][Realtime] stream error session=%s: %s", session_id, exc)
        finally:
            transcript = ctx.get("transcript", [])
            if transcript:
                lines = ["[{}] {}".format("AI" if r == "ai" else "Callee", t) for r, t in transcript]
                await self._relay_update(state, "Call ended.\n" + "\n".join(lines))
            else:
                await self._relay_update(state, "Call ended (no transcript captured).")
            self._sessions.pop(session_id or "", None)

    async def _bridge_twilio_to_openai(
        self,
        twilio_ws: web.WebSocketResponse,
        oai_ws: Any,
        ctx: dict[str, Any],
        done: asyncio.Event,
    ) -> None:
        """Read Twilio Media Stream audio and forward it to OpenAI Realtime.
        The start event has already been consumed by handle_stream."""
        try:
            async for msg in twilio_ws:
                if done.is_set():
                    break
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    event = data.get("event")
                    if event == "media":
                        payload = data.get("media", {}).get("payload", "")
                        if payload and not oai_ws.closed:
                            await oai_ws.send_json({
                                "type": "input_audio_buffer.append",
                                "audio": payload,
                            })
                    elif event == "stop":
                        logger.info("[Voice][Realtime] stream stopped")
                        done.set()
                        break
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                    done.set()
                    break
        except Exception as exc:
            logger.error("[Voice][Realtime] twilio→openai error: %s", exc)
        finally:
            done.set()
            if not oai_ws.closed:
                await oai_ws.close()

    async def _bridge_openai_to_twilio(
        self,
        oai_ws: Any,
        twilio_ws: web.WebSocketResponse,
        ctx: dict[str, Any],
        done: asyncio.Event,
        state: dict[str, Any],
    ) -> None:
        """Read OpenAI Realtime events and forward audio back to Twilio."""
        ai_buf: list[str] = []   # accumulate current AI utterance
        try:
            async for msg in oai_ws:
                if done.is_set():
                    break
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                        break
                    continue

                event = json.loads(msg.data)
                etype = event.get("type", "")

                if etype == "response.audio.delta":
                    chunk = event.get("delta", "")
                    if chunk and ctx.get("stream_sid") and not twilio_ws.closed:
                        await twilio_ws.send_json({
                            "event": "media",
                            "streamSid": ctx["stream_sid"],
                            "media": {"payload": chunk},
                        })

                elif etype == "response.audio_transcript.delta":
                    ai_buf.append(event.get("delta", ""))

                elif etype == "response.audio_transcript.done":
                    text = "".join(ai_buf).strip()
                    ai_buf.clear()
                    if text:
                        ctx["transcript"].append(("ai", text))
                        await self._relay_update(state, f"[Voice] AI: {text}")

                elif etype == "conversation.item.input_audio_transcription.completed":
                    text = event.get("transcript", "").strip()
                    if text:
                        ctx["transcript"].append(("callee", text))
                        await self._relay_update(state, f"[Voice] Callee: {text}")

                elif etype == "input_audio_buffer.speech_started":
                    # Callee interrupted — clear buffered audio so agent stops speaking
                    if ctx.get("stream_sid") and not twilio_ws.closed:
                        await twilio_ws.send_json({
                            "event": "clear",
                            "streamSid": ctx["stream_sid"],
                        })

                elif etype == "error":
                    logger.error("[Voice][Realtime] OpenAI error: %s", event.get("error"))

        except Exception as exc:
            logger.error("[Voice][Realtime] openai→twilio error: %s", exc)
        finally:
            done.set()

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
        self._voice_enabled = _env_flag("NEURALCLAW_SLACK_VOICE_ENABLED", False)
        self._voice_reply_text = _env_flag("NEURALCLAW_SLACK_VOICE_REPLY_WITH_TEXT", True)
        self._voice_openai_key = (
            os.getenv("NEURALCLAW_VOICE_OPENAI_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
        self._transcribe_model = os.getenv("NEURALCLAW_SLACK_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe").strip() or "gpt-4o-mini-transcribe"
        self._tts_model = os.getenv("NEURALCLAW_SLACK_TTS_MODEL", "gpt-4o-mini-tts").strip() or "gpt-4o-mini-tts"
        self._tts_voice = os.getenv("NEURALCLAW_SLACK_TTS_VOICE", "alloy").strip() or "alloy"
        self._tts_instructions = os.getenv(
            "NEURALCLAW_SLACK_TTS_INSTRUCTIONS",
            "Speak naturally, warmly, and conversationally. Use expressive but professional delivery.",
        ).strip()
        self._max_audio_bytes = int(os.getenv("NEURALCLAW_SLACK_MAX_AUDIO_BYTES", str(15 * 1024 * 1024)))

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
            if event.get("bot_id"):
                return
            if self._voice_enabled and await adapter._maybe_handle_audio_message(event):
                return
            if event.get("subtype"):
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
            if kwargs.get("slack_voice_reply") and self._voice_enabled:
                try:
                    audio_bytes, filename = await self._synthesize_tts_audio(content)
                    upload_kwargs: dict[str, Any] = {
                        "channel": channel_id,
                        "thread_ts": thread_ts,
                        "filename": filename,
                        "file": audio_bytes,
                        "title": kwargs.get("slack_audio_title", "NeuralClaw voice reply"),
                    }
                    if self._voice_reply_text:
                        upload_kwargs["initial_comment"] = content
                    await self._app.client.files_upload_v2(**upload_kwargs)
                    return
                except Exception:
                    logger.exception("[Slack] voice reply synthesis/upload failed - falling back to text")

            await self._app.client.chat_postMessage(
                channel=channel_id,
                text=content,
                thread_ts=thread_ts,
            )

    async def _maybe_handle_audio_message(self, event: dict[str, Any]) -> bool:
        from neuralclaw.channels.protocol import ChannelMessage

        audio_file = self._extract_audio_file(event)
        if not audio_file:
            return False
        if not self._voice_openai_key:
            logger.warning("[Slack] voice event ignored - no OpenAI key configured")
            return False

        channel_id = str(event.get("channel") or "").strip()
        thread_ts = event.get("thread_ts", event.get("ts"))
        user_id = str(event.get("user") or "unknown").strip() or "unknown"
        user_name = str(event.get("username") or event.get("user_profile", {}).get("display_name") or "Unknown").strip() or "Unknown"

        try:
            file_bytes = await self._download_slack_file(audio_file)
            transcript = await self._transcribe_audio(audio_file, file_bytes)
            if not transcript:
                raise RuntimeError("Transcription returned empty text.")
        except Exception as exc:
            logger.warning("[Slack] voice message processing failed: %s", exc)
            if self._app and channel_id:
                try:
                    await self._app.client.chat_postMessage(
                        channel=channel_id,
                        text=f"I couldn't process that audio message: {exc}",
                        thread_ts=thread_ts,
                    )
                except Exception:
                    logger.exception("[Slack] failed to post audio-processing error")
            return True

        logger.info("[Slack] inbound audio message channel=%s user=%s transcript_len=%d", channel_id, user_id, len(transcript))
        msg = ChannelMessage(
            content=transcript,
            author_id=user_id,
            author_name=user_name,
            channel_id=channel_id,
            raw=event,
            metadata={
                "platform": "slack",
                "source": "slack",
                "channel": "slack",
                "thread_ts": thread_ts,
                "slack_voice_reply": True,
                "slack_audio_file_id": audio_file.get("id"),
                "slack_audio_filename": audio_file.get("name"),
            },
        )
        try:
            await self._dispatch(msg)
        except Exception:
            logger.exception("[Slack] dispatch failure for audio message event")
        return True

    def _extract_audio_file(self, event: dict[str, Any]) -> dict[str, Any] | None:
        files = event.get("files")
        if not isinstance(files, list):
            return None
        for file_obj in files:
            if not isinstance(file_obj, dict):
                continue
            mimetype = str(file_obj.get("mimetype") or "").lower()
            filetype = str(file_obj.get("filetype") or "").lower()
            if mimetype.startswith("audio/") or filetype in {"mp3", "m4a", "wav", "ogg", "opus", "mpeg", "mp4"}:
                return file_obj
        return None

    async def _download_slack_file(self, file_obj: dict[str, Any]) -> bytes:
        url = str(file_obj.get("url_private_download") or file_obj.get("url_private") or "").strip()
        if not url:
            raise RuntimeError("Slack audio file has no downloadable URL.")

        headers = {"Authorization": f"Bearer {self._bot_token}"}
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=45)) as resp:
                body = await resp.read()
                if resp.status != 200:
                    raise RuntimeError(f"Slack file download failed ({resp.status}): {body[:200]!r}")
        if not body:
            raise RuntimeError("Slack audio download returned empty content.")
        if len(body) > self._max_audio_bytes:
            raise RuntimeError(
                f"Slack audio file exceeds limit ({len(body)} bytes > {self._max_audio_bytes} bytes)."
            )
        return body

    async def _transcribe_audio(self, file_obj: dict[str, Any], file_bytes: bytes) -> str:
        filename = str(file_obj.get("name") or "slack-audio").strip() or "slack-audio"
        mimetype = str(file_obj.get("mimetype") or "application/octet-stream").strip() or "application/octet-stream"
        form = aiohttp.FormData()
        form.add_field("model", self._transcribe_model)
        form.add_field("file", file_bytes, filename=filename, content_type=mimetype)
        headers = {"Authorization": f"Bearer {self._voice_openai_key}"}
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(
                "https://api.openai.com/v1/audio/transcriptions",
                data=form,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                raw_text = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"OpenAI transcription failed ({resp.status}): {raw_text[:300]}")
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Transcription returned invalid JSON: {exc}") from exc
        transcript = str(data.get("text") or "").strip()
        if not transcript:
            raise RuntimeError("OpenAI transcription returned empty text.")
        return transcript

    async def _synthesize_tts_audio(self, text: str) -> tuple[bytes, str]:
        clean_text = _strip_emoji(text).strip()
        if not clean_text:
            raise RuntimeError("Cannot synthesize empty reply.")
        if len(clean_text) > 4000:
            clean_text = clean_text[:4000].rsplit(" ", 1)[0].strip() or clean_text[:4000]

        headers = {
            "Authorization": f"Bearer {self._voice_openai_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._tts_model,
            "voice": self._tts_voice,
            "input": clean_text,
            "response_format": "mp3",
            "instructions": self._tts_instructions,
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(
                "https://api.openai.com/v1/audio/speech",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                audio_bytes = await resp.read()
                if resp.status != 200:
                    preview = audio_bytes[:300].decode("utf-8", errors="replace")
                    raise RuntimeError(f"OpenAI TTS failed ({resp.status}): {preview}")
        if not audio_bytes:
            raise RuntimeError("OpenAI TTS returned empty audio.")
        return audio_bytes, f"neuralclaw-reply-{int(time.time())}.mp3"




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

        # Inject voice call behaviour into the system prompt so it overrides
        # the base persona when the LLM is on a phone call.
        custom_voice_persona = os.getenv("NEURALCLAW_VOICE_PERSONA", "").strip()
        if custom_voice_persona:
            voice_hint = (
                f"\n\nFor outbound phone calls (messages that contain 'Call objective:'): "
                f"{custom_voice_persona} "
                "Respond in 1-3 short sentences. No emoji or markdown."
            )
        else:
            voice_hint = (
                "\n\nFor outbound phone calls (messages that contain 'Call objective:'): "
                "act as a focused, professional caller completing the stated task. "
                "Do not use your usual greeting or ask 'How can I help?'. "
                "Speak naturally in 1-3 short sentences. No emoji or markdown."
            )
        config.persona = (config.persona or "") + voice_hint

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
        web.get("/voice/twilio/ws", _handle_voice_ws),
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
