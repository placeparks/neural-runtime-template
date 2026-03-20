#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime as dt
import html
import io
import json
import logging
import os
import re
import signal
import shutil
import sys
import tempfile
import time
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp
from aiohttp import web

from neuralclaw.bus.neural_bus import EventType
from neuralclaw.cortex.action.policy import RequestContext
from neuralclaw.cortex.reasoning.deliberate import ConfidenceEnvelope, DeliberativeReasoner
from neuralclaw.config import get_api_key, load_config
from neuralclaw.gateway import NeuralClawGateway

# ---------------------------------------------------------------------------
# Emoji stripper â€” TTS engines read emoji as "hand wave sign" etc.
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

_ONBOARDING_SKIP_RE = re.compile(
    r"^\s*(join|leave|stop|pause|search|find|look up|google|open|click|type|call|message|send|show|take)\b",
    re.IGNORECASE,
)
_EXPLICIT_NAME_RE = re.compile(
    r"\b(?:my name is|call me|you can call me|i am|i'm|im)\s+([A-Za-z][A-Za-z' -]{1,40})\b",
    re.IGNORECASE,
)
_SHORT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z' -]{1,32}$")
_PREFERENCE_RE = re.compile(
    r"\b(?:i prefer|please call me|from now on call me|i like it when)\s+(.+)",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(
    r"\b(?:i am a|i'm a|im a|i work on|i'm working on|im working on|i run|i build|i'm building|im building)\s+(.+)",
    re.IGNORECASE,
)
_ROLE_HINT_WORDS = {
    "developer", "engineer", "founder", "student", "designer", "manager", "freelancer",
    "marketer", "writer", "consultant", "full", "stack", "frontend", "backend",
    "software", "product", "qa", "tester", "devops", "architect"
}
_SCREENSHOT_QUERY_RE = re.compile(r"\b(screenshot|screen shot|take a shot of|capture my screen|see my screen|share my screen)\b", re.IGNORECASE)
_REMINDER_RE = re.compile(
    r"^(?:please\s+)?remind\s+me(?:\s+to)?\s+(?P<task>.+?)\s+(?:at|on|in|after|tomorrow|today|every)\s+(?P<when>.+)$",
    re.IGNORECASE,
)
_REMINDER_AFTER_RE = re.compile(
    r"^(?:please\s+)?remind\s+me\s+(?:in|after)\s+(?P<when>.+?)\s+(?:to|that)\s+(?P<task>.+)$",
    re.IGNORECASE,
)


def _looks_like_web_search_query(text: str) -> bool:
    return bool(_SEARCH_QUERY_RE.search(text or ""))


def _looks_like_screenshot_request(text: str) -> bool:
    return bool(_SCREENSHOT_QUERY_RE.search(text or ""))


def _extract_schedule_request(text: str) -> dict[str, str] | None:
    content = (text or "").strip()
    if not content:
        return None
    content = re.sub(
        r"^(?:please\s+)?(?:i\s+want\s+you\s+to|can\s+you|could\s+you|would\s+you)\s+",
        "",
        content,
        flags=re.IGNORECASE,
    ).strip()
    match = _REMINDER_AFTER_RE.match(content)
    if match:
        task = match.group("task").strip(" .!")
        when = f"in {match.group('when').strip()}"
        if task and when:
            return {"task": task, "schedule_text": when}
    match = _REMINDER_RE.match(content)
    if match:
        task = match.group("task").strip(" .!")
        keyword_match = re.search(r"\b(at|on|in|after|tomorrow|today|every)\b", content, re.IGNORECASE)
        when = ""
        if keyword_match:
            when = content[keyword_match.start():].strip()
        else:
            when = match.group("when").strip()
        if task and when:
            return {"task": task, "schedule_text": when}
    return None


def _compact_text(value: str, limit: int = 1200) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _compact_history_messages(history: list[dict[str, Any]] | None, max_messages: int = 8, max_chars: int = 700) -> list[dict[str, Any]]:
    if not history:
        return []
    trimmed = history[-max_messages:]
    compact: list[dict[str, Any]] = []
    for item in trimmed:
        role = str(item.get("role") or "user")
        content = item.get("content")
        if isinstance(content, str):
            compact.append({"role": role, "content": _compact_text(content, max_chars)})
        else:
            compact.append({"role": role, "content": content})
    return compact


def _score_tool(tool: Any, content: str) -> int:
    name = str(getattr(tool, "name", "") or "").lower()
    desc = str(getattr(tool, "description", "") or "").lower()
    text = (content or "").lower()
    score = 0
    for token in re.findall(r"[a-z0-9_]+", text):
        if len(token) < 3:
            continue
        if token in name:
            score += 4
        if token in desc:
            score += 1
    if "remind" in text and name in {"create_schedule", "list_schedules"}:
        score += 50
    if _looks_like_screenshot_request(text) and name == "companion_take_screenshot":
        score += 50
    if _looks_like_web_search_query(text) and name in {"web_search", "fetch_url", "companion_search_web"}:
        score += 40
    if any(word in text for word in ["open", "launch", "app", "browser", "laptop", "computer"]) and name.startswith("companion_"):
        score += 20
    return score


def _select_relevant_tools(tools: list[Any] | None, content: str) -> list[Any] | None:
    if not tools:
        return None
    scored = sorted(tools, key=lambda t: (_score_tool(t, content), str(getattr(t, "name", ""))), reverse=True)
    chosen = [tool for tool in scored if _score_tool(tool, content) > 0][:18]
    if not chosen:
        preferred = {"web_search", "create_schedule", "list_schedules"}
        chosen = [tool for tool in tools if str(getattr(tool, "name", "")) in preferred][:8]
    return chosen or tools[:12]


def _summarize_tool_result_for_model(result: Any) -> str:
    if isinstance(result, dict):
        compact = dict(result)
        for key in list(compact.keys()):
            lowered = key.lower()
            if lowered in {"screenshot_b64", "base64", "image_b64", "image_base64", "binary", "bytes"}:
                value = compact.pop(key)
                compact[f"{key}_omitted"] = f"omitted {len(str(value))} chars of binary/base64 data"
        if "results" in compact and isinstance(compact["results"], list):
            compact["results"] = compact["results"][:3]
        text = json.dumps(compact, ensure_ascii=False)
        return _compact_text(text, 1800)
    if isinstance(result, list):
        return _compact_text(json.dumps(result[:3], ensure_ascii=False), 1200)
    return _compact_text(str(result), 1200)


def _is_probable_role_phrase(value: str) -> bool:
    words = [part for part in re.findall(r"[a-z]+", value.lower()) if part]
    return bool(words) and any(word in _ROLE_HINT_WORDS for word in words)


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
    extra_system_sections: list[str] | None = None,
) -> ConfidenceEnvelope:
    if not self._provider:
        return ConfidenceEnvelope(
            response="I'm not configured with an LLM provider yet.",
            confidence=0.0,
            source="error",
            uncertainty_factors=["no_provider_configured"],
        )

    content_text = str(getattr(signal, "content", "") or "")
    selected_tools = _select_relevant_tools(tools, content_text) or (tools or [])
    request_ctx = RequestContext(request_id=getattr(signal, "id", "runtime"))

    class _ForcedToolCall:
        def __init__(self, signal_id: str, name: str, arguments: dict[str, Any]) -> None:
            self.id = f"forced-{name}-{signal_id}"
            self.name = name
            self.arguments = arguments

    schedule_payload = _extract_schedule_request(content_text)
    schedule_tool = next((t for t in selected_tools if getattr(t, "name", "") == "create_schedule"), None)
    if schedule_payload and schedule_tool:
        forced_schedule = await self._execute_tool_call(
            _ForcedToolCall(getattr(signal, "id", "forced"), "create_schedule", schedule_payload),
            selected_tools,
            request_ctx,
        )
        if isinstance(forced_schedule, dict) and forced_schedule.get("error"):
            return ConfidenceEnvelope(
                response=f"I couldn't schedule that reminder: {forced_schedule['error']}",
                confidence=0.0,
                source="error",
                uncertainty_factors=["schedule_failed"],
            )
        message = str((forced_schedule or {}).get("message") or "Scheduled that reminder.").strip()
        return ConfidenceEnvelope(
            response=message,
            confidence=0.94,
            source="tool_verified",
            tool_calls_made=1,
        )

    screenshot_tool = next((t for t in selected_tools if getattr(t, "name", "") == "companion_take_screenshot"), None)
    if _looks_like_screenshot_request(content_text) and screenshot_tool:
        forced_capture = await self._execute_tool_call(
            _ForcedToolCall(getattr(signal, "id", "forced"), "companion_take_screenshot", {"monitor": 0}),
            selected_tools,
            request_ctx,
        )
        if isinstance(forced_capture, dict) and forced_capture.get("error"):
            return ConfidenceEnvelope(
                response=f"I couldn't capture a screenshot right now: {forced_capture['error']}",
                confidence=0.0,
                source="error",
                uncertainty_factors=["screenshot_failed"],
            )
        message = str((forced_capture or {}).get("message") or "I've captured your screen and shared it here.").strip()
        return ConfidenceEnvelope(
            response=message,
            confidence=0.96,
            source="tool_verified",
            tool_calls_made=1,
        )

    forced_search = None
    if _looks_like_web_search_query(content_text):
        web_tool = next((t for t in selected_tools if getattr(t, "name", "") == "web_search"), None)
        if web_tool:
            forced_search = await self._execute_tool_call(
                _ForcedToolCall(getattr(signal, "id", "forced"), "web_search", {"query": content_text, "max_results": 5}),
                selected_tools,
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
        {"signal_id": signal.id, "path": "deliberative", "tools_available": len(selected_tools)},
        source="reasoning.deliberate",
    )

    compact_history = _compact_history_messages(conversation_history)
    try:
        messages = self._build_messages(
            signal,
            memory_ctx,
            compact_history,
            extra_system_sections=extra_system_sections,
        )
    except TypeError:
        messages = self._build_messages(signal, memory_ctx, compact_history)
        if extra_system_sections:
            messages.insert(1, {"role": "system", "content": "\n\n".join(extra_system_sections)})

    if forced_search is not None:
        messages.append({
            "role": "system",
            "content": (
                "Web search was executed for this request. Base your answer on these results. "
                "If the results are sparse or inconclusive, say that clearly instead of pretending to browse.\n\n"
                f"WEB_SEARCH_RESULTS:\n{_summarize_tool_result_for_model(forced_search)}"
            ),
        })

    tool_defs = [t.to_openai_format() for t in selected_tools] if selected_tools else None
    tool_calls_made = 1 if forced_search is not None else 0
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

        if response.tool_calls and selected_tools:
            for tc in response.tool_calls:
                tool_calls_made += 1
                result = await self._execute_tool_call(tc, selected_tools, request_ctx)
                messages.append({"role": "assistant", "content": None, "tool_calls": [tc.to_dict()]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": _summarize_tool_result_for_model(result)})
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
        response="I spent too many iterations trying to answer. Let me try a simpler approach - could you rephrase?",
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
            "[runtime] knowledge base available (%d bytes) â€” file_ops enabled",
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
                return None, "no mesh peers are configured â€” check NEURALCLAW_MESH_PEERS_JSON"
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

        logger.info("[MESH] Delegating to '%s' at %s â€” task: %s", target_name, endpoint, task[:80])
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
        self._companion_manager: CompanionRelayManager | None = None
        self._cron_manager: CronJobManager | None = None
        self._last_request_context: dict[str, str] | None = None
        self._control_base_url = os.getenv("NEURALCLAW_CONTROL_BASE_URL", "").strip().rstrip("/")
        self._provisioner_secret = os.getenv("NEURALCLAW_PROVISIONER_SECRET", "").strip()

    async def _peek_identity_model(self, platform: str, platform_user_id: str) -> Any | None:
        store = getattr(self, "_identity", None)
        db = getattr(store, "_db", None)
        if not store or not db:
            return None
        try:
            rows = await db.execute_fetchall(
                """
                SELECT m.user_id, m.display_name, m.platform_aliases_json,
                       m.communication_style_json, m.active_projects_json,
                       m.expertise_domains_json, m.language, m.timezone,
                       m.preferences_json, m.last_seen, m.first_seen,
                       m.session_count, m.message_count, m.notes
                FROM user_aliases a
                JOIN user_models m ON m.user_id = a.user_id
                WHERE a.platform = ? AND a.platform_user_id = ?
                """,
                (platform, platform_user_id),
            )
            if not rows:
                return None
            return store._row_to_model(rows[0])
        except Exception as exc:
            logger.warning("[runtime] could not peek identity model: %s", exc)
            return None

    def _last_assistant_message(self, channel_id: str) -> str:
        history = getattr(self, "_history", {}).get(channel_id, [])
        for item in reversed(history):
            if item.get("role") == "assistant":
                return str(item.get("content") or "")
        return ""

    def _extract_preferred_name(self, text: str, channel_id: str) -> str | None:
        content = (text or "").strip()
        if not content:
            return None

        match = _EXPLICIT_NAME_RE.search(content)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip(" .,!?:;\"'")
            if value and not _is_probable_role_phrase(value):
                return value.title()

        last_assistant = self._last_assistant_message(channel_id).lower()
        short = re.sub(r"^[^A-Za-z]+|[^A-Za-z' -]+$", "", content).strip()
        if (
            any(phrase in last_assistant for phrase in ("what should i call you", "your name", "what's your name"))
            and _SHORT_NAME_RE.fullmatch(short)
            and len(short.split()) <= 3
            and not _is_probable_role_phrase(short)
        ):
            return re.sub(r"\s+", " ", short).strip().title()
        return None

    def _should_prompt_for_intro(self, model: Any | None, content: str, channel_id: str) -> bool:
        text = (content or "").strip()
        if not text:
            return False
        if _ONBOARDING_SKIP_RE.match(text):
            return False
        if self._extract_preferred_name(text, channel_id):
            return False
        if not model:
            return True
        if str(getattr(model, "notes", "") or "").strip():
            return False
        if int(getattr(model, "message_count", 0) or 0) > 2:
            return False
        return True

    def _extract_profile_payload(
        self,
        *,
        content: str,
        channel_id: str,
        author_name: str,
        model: Any | None,
    ) -> dict[str, Any] | None:
        text = (content or "").strip()
        if not text:
            return None

        preferred_name = self._extract_preferred_name(text, channel_id)
        preference_match = _PREFERENCE_RE.search(text)
        summary_match = _SUMMARY_RE.search(text)
        preference = (
            preference_match.group(1).strip(" .,!?:;\"'") if preference_match else ""
        )
        summary = summary_match.group(1).strip(" .,!?:;\"'") if summary_match else ""

        if not any([preferred_name, preference, summary]) and model and int(getattr(model, "message_count", 0) or 0) > 2:
            return None

        aliases: list[str] = []
        current_name = str(getattr(model, "display_name", "") or "").strip()
        clean_author = (author_name or "").strip()
        if preferred_name and current_name and preferred_name.lower() != current_name.lower():
            aliases.append(current_name)
        if clean_author and preferred_name and clean_author.lower() != preferred_name.lower():
            aliases.append(clean_author)

        notes_parts: list[str] = []
        if summary:
            notes_parts.append(f"Shared context: {summary}")
        if preference:
            notes_parts.append(f"Preference: {preference}")

        return {
            "preferred_name": preferred_name or "",
            "aliases": aliases,
            "summary": summary or None,
            "preferences": preference or None,
            "notes": "\n".join(notes_parts).strip() or None,
        }

    async def _update_local_identity_memory(self, user_id: str, payload: dict[str, Any]) -> None:
        if not getattr(self, "_identity", None) or not user_id:
            return
        model = await self._identity.get(user_id)
        if not model:
            return

        updates: dict[str, Any] = {}
        preferred_name = str(payload.get("preferred_name") or "").strip()
        if preferred_name and preferred_name.lower() != model.display_name.lower():
            updates["display_name"] = preferred_name

        preference = str(payload.get("preferences") or "").strip()
        if preference:
            existing_rules = []
            if isinstance(model.preferences, dict):
                existing_rules = list(model.preferences.get("custom_rules", []) or [])
            if preference not in existing_rules:
                updates["preferences"] = {"custom_rules": [*existing_rules, preference]}

        note_lines = [str(payload.get("notes") or "").strip()]
        merged_note_lines = [line for line in note_lines if line]
        if merged_note_lines:
            existing_notes = str(model.notes or "").strip()
            next_notes = existing_notes
            for line in merged_note_lines:
                if line.lower() not in existing_notes.lower():
                    next_notes = f"{next_notes}\n{line}".strip() if next_notes else line
            if next_notes != existing_notes:
                updates["notes"] = next_notes

        if updates:
            await self._identity.update(user_id, updates)

    async def _sync_people_memory(
        self,
        *,
        agent_id: str,
        platform: str,
        platform_user_id: str,
        author_name: str,
        payload: dict[str, Any],
        model: Any | None,
    ) -> None:
        if not (self._control_base_url and self._provisioner_secret and agent_id):
            return

        preferred_name = str(payload.get("preferred_name") or "").strip()
        if not any([preferred_name, payload.get("preferences"), payload.get("summary"), model]):
            return

        first_seen = None
        last_seen = None
        if model:
            if getattr(model, "first_seen", 0):
                first_seen = dt.datetime.fromtimestamp(float(model.first_seen), tz=dt.timezone.utc).isoformat()
            if getattr(model, "last_seen", 0):
                last_seen = dt.datetime.fromtimestamp(float(model.last_seen), tz=dt.timezone.utc).isoformat()

        url = f"{self._control_base_url}/api/internal/agents/{agent_id}/people/ingest"
        body = {
            "platform": platform,
            "platformUserId": platform_user_id,
            "displayName": author_name,
            "preferredName": preferred_name or None,
            "aliases": payload.get("aliases") or [],
            "summary": payload.get("summary"),
            "preferences": payload.get("preferences"),
            "notes": payload.get("notes"),
            "firstSeenAt": first_seen,
            "lastSeenAt": last_seen or dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        headers = {
            "Content-Type": "application/json",
            "x-provisioner-secret": self._provisioner_secret,
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                async with session.post(url, json=body, headers=headers) as resp:
                    if resp.status >= 400:
                        preview = (await resp.text())[:300]
                        logger.warning(
                            "[runtime] people ingest failed (%s): %s",
                            resp.status,
                            preview,
                        )
        except Exception as exc:
            logger.warning("[runtime] people ingest request failed: %s", exc)

    async def _load_saved_person_record(
        self,
        *,
        agent_id: str,
        platform: str,
        platform_user_id: str,
    ) -> dict[str, Any] | None:
        if not (self._control_base_url and self._provisioner_secret and agent_id and platform and platform_user_id):
            return None
        url = (
            f"{self._control_base_url}/api/internal/agents/{agent_id}/people"
            f"?platform={platform}&platformUserId={platform_user_id}"
        )
        headers = {"x-provisioner-secret": self._provisioner_secret}
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status >= 400:
                        return None
                    data = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning("[runtime] people lookup failed: %s", exc)
            return None
        person = data.get("person") if isinstance(data, dict) else None
        return person if isinstance(person, dict) else None

    def _person_prompt_block(self, person: dict[str, Any], author_name: str) -> str:
        aliases = [str(item).strip() for item in (person.get("aliases") or []) if str(item).strip()]
        canonical = str(person.get("canonical_name") or "").strip()
        display_name = canonical
        profile_lines: list[str] = []
        if canonical and _is_probable_role_phrase(canonical):
            profile_lines.append(f"Known role/context: {canonical}")
            display_name = aliases[0] if aliases else (author_name or canonical)
        elif canonical:
            display_name = canonical

        if aliases:
            profile_lines.append(f"Known aliases: {', '.join(aliases[:4])}")
        if person.get("relationship"):
            profile_lines.append(f"Relationship: {person['relationship']}")
        if person.get("summary"):
            profile_lines.append(f"Summary: {person['summary']}")
        if person.get("preferences"):
            profile_lines.append(f"Preferences: {person['preferences']}")
        if person.get("notes"):
            profile_lines.append(f"Notes: {_compact_text(str(person['notes']), 500)}")

        identity_map = person.get("channel_identities") or {}
        if isinstance(identity_map, dict) and identity_map:
            parts = [f"{key}={value}" for key, value in list(identity_map.items())[:4] if str(value).strip()]
            if parts:
                profile_lines.append(f"Known channels: {', '.join(parts)}")

        lead = f"KNOWN PERSON: The current user is {display_name}."
        if profile_lines:
            lead += "\n" + "\n".join(profile_lines)
        lead += "\nUse this information confidently when the user asks what you know about them. Do not claim you have no memory if this block is present."
        return lead


    async def initialize(self) -> None:
        await super().initialize()
        if self._voice_manager:
            self._voice_manager.register_tools()
        if self._companion_manager:
            self._companion_manager.register_tools()
        if self._cron_manager:
            self._cron_manager.register_tools()

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
        if meta.get("platform") == "discord" and meta.get("discord_voice_reply"):
            kwargs["discord_voice_reply"] = True
            guild_name = str(meta.get("discord_guild_name") or "").strip()
            if guild_name:
                kwargs["discord_voice_title"] = f"Voice reply in {guild_name}"
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
        media: list[dict[str, Any]] | None = None,
        message_metadata: dict[str, Any] | None = None,
        raw_message: Any = None,
    ) -> str:
        is_cron_run = bool(channel_id.startswith("cron:") or ((message_metadata or {}).get("source") == "cron"))
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
            # Delegation was explicitly attempted but failed â€” return the reason.
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
        platform = channel_type_name.lower()
        agent_id = os.getenv("NEURALCLAW_AGENT_ID", "").strip()
        saved_person = await self._load_saved_person_record(
            agent_id=agent_id,
            platform=platform,
            platform_user_id=author_id,
        )
        if saved_person:
            resolved_content = (
                f"{self._person_prompt_block(saved_person, author_name)}\n\n"
                f"USER_MESSAGE:\n{resolved_content}"
            )
        if not is_cron_run:
            schedule_payload = _extract_schedule_request(content)
            if schedule_payload and self._cron_manager and self._cron_manager.enabled:
                scheduled = await self._cron_manager.create_schedule(
                    task=schedule_payload["task"],
                    schedule_text=schedule_payload["schedule_text"],
                )
                if isinstance(scheduled, dict) and scheduled.get("ok"):
                    return str(scheduled.get("message") or "Scheduled that reminder.").strip()
                if isinstance(scheduled, dict) and scheduled.get("error"):
                    return f"I couldn't schedule that reminder: {scheduled['error']}"
            if _looks_like_screenshot_request(content) and self._companion_manager and self._companion_manager.enabled:
                captured = await self._companion_manager.take_screenshot(monitor=0)
                if isinstance(captured, dict) and captured.get("ok"):
                    return str(captured.get("message") or "I've captured your screen and shared it here.").strip()
                if isinstance(captured, dict) and captured.get("error"):
                    return f"I couldn't capture a screenshot right now: {captured['error']}"
        existing_model = await self._peek_identity_model(platform, author_id)
        onboarding_suffix = ""
        if not is_cron_run and self._should_prompt_for_intro(existing_model, content, channel_id):
            onboarding_suffix = (
                "\n\nYou may be meeting this person for the first time. "
                "Before giving a long answer, ask one short, warm onboarding question to learn "
                "what they prefer to be called, unless they already gave their name in this message "
                "or the request is urgent. Keep it natural and low-pressure. "
                "After you learn a preferred name or durable preference, remember it and use it later. "
                "Do not say that you lack permanent memory when memory is enabled."
            )

        original_persona = self._config.persona or ""
        if onboarding_suffix:
            self._config.persona = original_persona + onboarding_suffix

        try:
            response = await super().process_message(
                content=resolved_content,
                author_id=author_id,
                author_name=author_name,
                channel_id=channel_id,
                channel_type_name=channel_type_name,
                media=media,
                message_metadata=message_metadata,
                raw_message=raw_message,
            )
        finally:
            self._config.persona = original_persona

        if is_cron_run:
            return response

        current_model = await self._peek_identity_model(platform, author_id)
        payload = self._extract_profile_payload(
            content=content,
            channel_id=channel_id,
            author_name=author_name,
            model=current_model,
        )
        if current_model and payload:
            await self._update_local_identity_memory(current_model.user_id, payload)
            refreshed_model = await self._peek_identity_model(platform, author_id)
            await self._sync_people_memory(
                agent_id=os.getenv("NEURALCLAW_AGENT_ID", "").strip(),
                platform=platform,
                platform_user_id=author_id,
                author_name=author_name,
                payload=payload,
                model=refreshed_model or current_model,
            )
        elif current_model and int(getattr(current_model, "message_count", 0) or 0) <= 2:
            await self._sync_people_memory(
                agent_id=os.getenv("NEURALCLAW_AGENT_ID", "").strip(),
                platform=platform,
                platform_user_id=author_id,
                author_name=author_name,
                payload={"preferred_name": "", "aliases": [], "summary": None, "preferences": None, "notes": None},
                model=current_model,
            )

        return response


async def _handle_health(request: web.Request) -> web.Response:
    gw: MeshAwareGateway = request.app["gateway"]
    mesh = gw._mesh_router
    return web.json_response({
        "status": "ok",
        "mesh_enabled": mesh.enabled,
        "peers": len(mesh.peers),
    })


def _control_request_allowed(request: web.Request, gw: MeshAwareGateway) -> bool:
    expected = (getattr(gw, "_provisioner_secret", "") or "").strip()
    if not expected:
        return True
    provided = str(request.headers.get("x-provisioner-secret", "")).strip()
    return bool(provided and provided == expected)


async def _handle_stats(request: web.Request) -> web.Response:
    gw: MeshAwareGateway = request.app["gateway"]
    if not _control_request_allowed(request, gw):
        return web.json_response({"error": "unauthorized"}, status=401)

    stats = gw._get_dashboard_stats()
    stats["uptime"] = f"{int(time.time() - gw._started_at)}s" if hasattr(gw, "_started_at") else "unknown"
    stats["skills"] = getattr(gw._skills, "count", 0)
    stats["tool_count"] = getattr(gw._skills, "tool_count", 0)
    stats["channels"] = len(getattr(gw, "_channels", {}) or {})
    stats["channel_names"] = list((getattr(gw, "_channels", {}) or {}).keys())
    stats["companion_enabled"] = bool(getattr(gw, "_companion_manager", None) and gw._companion_manager.enabled)
    stats["cron_enabled"] = bool(getattr(gw, "_cron_manager", None) and gw._cron_manager.enabled)
    stats["traceline_enabled"] = bool(getattr(gw, "_traceline", None))
    stats["audit_enabled"] = bool(getattr(gw, "_audit", None))

    if getattr(gw, "_traceline", None):
        try:
            trace_metrics = await gw._traceline.get_metrics()
            stats.update(trace_metrics)
            if "error_rate" in trace_metrics:
                stats["success_rate"] = round(1.0 - float(trace_metrics.get("error_rate", 0.0) or 0.0), 3)
            if "total_traces" in trace_metrics:
                stats["interactions"] = int(trace_metrics.get("total_traces", 0) or 0)
        except Exception as exc:
            stats["traceline_error"] = str(exc)

    if getattr(gw, "_audit", None):
        try:
            audit_stats = await gw._audit.stats()
            stats["audit_total_records"] = int(audit_stats.get("total_records", 0) or 0)
            stats["audit_denied_records"] = int(audit_stats.get("denied_records", 0) or 0)
            stats["audit_denial_rate"] = float(audit_stats.get("denial_rate", 0.0) or 0.0)
            stats["audit_top_tools"] = audit_stats.get("top_tools", [])
        except Exception as exc:
            stats["audit_error"] = str(exc)

    return web.json_response(stats)


async def _handle_traces(request: web.Request) -> web.Response:
    gw: MeshAwareGateway = request.app["gateway"]
    if not _control_request_allowed(request, gw):
        return web.json_response({"error": "unauthorized"}, status=401)

    limit = max(1, min(int(request.query.get("limit", "20")), 100))
    if not getattr(gw, "_traceline", None):
        return web.json_response([])

    try:
        traces = await gw._traceline.query_traces(limit=limit)
        return web.json_response([asdict(trace) for trace in traces])
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


async def _handle_audit(request: web.Request) -> web.Response:
    gw: MeshAwareGateway = request.app["gateway"]
    if not _control_request_allowed(request, gw):
        return web.json_response({"error": "unauthorized"}, status=401)

    limit = max(1, min(int(request.query.get("limit", "20")), 100))
    if not getattr(gw, "_audit", None):
        return web.json_response({"records": [], "stats": {}})

    try:
        records = [record.to_dict() for record in gw._audit.get_recent(limit)]
        stats = await gw._audit.stats()
        return web.json_response({"records": records, "stats": stats})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


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
    message_metadata = data.get("message_metadata")
    if not isinstance(message_metadata, dict):
        message_metadata = None
    channel_id = str(data.get("channel_id", "mesh") or "mesh")
    channel_type_name = str(data.get("channel_type_name", "CLI") or "CLI")
    response = await gw.process_message(
        content=content,
        author_id=str(data.get("from", "peer")),
        author_name=str(data.get("author_name", data.get("from", "peer"))),
        channel_id=channel_id,
        channel_type_name=channel_type_name,
        message_metadata=message_metadata,
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
    """Twilio Media Streams WebSocket â€” bridges to OpenAI Realtime API."""
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
        # OpenAI Realtime â€” preferred key is NEURALCLAW_VOICE_OPENAI_KEY,
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
                    "Speak as a real person â€” one or two sentences, no emoji, no lists, no markdown. "
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

        # Twilio may send GET (after an httpâ†’https redirect converts POSTâ†’GET)
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
            "Use short, clear sentences â€” 1 to 3 max per reply. "
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
            "Speak naturally and conversationally. Keep responses concise â€” 1 to 3 sentences. "
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
                    continue  # protocol greeting â€” wait for "start"
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
            logger.error("[Voice][Realtime] twilioâ†’openai error: %s", exc)
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
                    # Callee interrupted â€” clear buffered audio so agent stops speaking
                    if ctx.get("stream_sid") and not twilio_ws.closed:
                        await twilio_ws.send_json({
                            "event": "clear",
                            "streamSid": ctx["stream_sid"],
                        })

                elif etype == "error":
                    logger.error("[Voice][Realtime] OpenAI error: %s", event.get("error"))

        except Exception as exc:
            logger.error("[Voice][Realtime] openaiâ†’twilio error: %s", exc)
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


class CompanionRelayManager:
    def __init__(self, gateway: "MeshAwareGateway") -> None:
        self._gateway = gateway
        self._relay_url = os.getenv("NEURALCLAW_COMPANION_RELAY_URL", "").strip().rstrip("/")
        self._shared_secret = os.getenv("NEURALCLAW_COMPANION_RELAY_SHARED_SECRET", "").strip()
        self._agent_id = os.getenv("NEURALCLAW_AGENT_ID", "").strip()
        self._timeout = max(5, int(os.getenv("NEURALCLAW_COMPANION_TASK_TIMEOUT", "45")))

    @property
    def enabled(self) -> bool:
        return bool(self._relay_url and self._shared_secret and self._agent_id)

    def register_tools(self) -> None:
        if not self.enabled:
            return
        self._gateway._skills.register_tool(
            name="companion_open_url",
            description="Open a URL in the paired user's real local browser on their computer.",
            function=self.open_url,
            parameters={
                "url": {"type": "string", "description": "The URL to open on the paired computer."},
            },
        )
        self._gateway._skills.register_tool(
            name="companion_search_web",
            description="Open a web search in the paired user's local browser.",
            function=self.search_web,
            parameters={
                "query": {"type": "string", "description": "The search query to run on the paired computer."},
                "provider": {
                    "type": "string",
                    "description": "Search provider, typically 'google' or 'duckduckgo'.",
                },
            },
        )
        self._gateway._skills.register_tool(
            name="companion_launch_app",
            description="Launch an app or command on the paired user's computer.",
            function=self.launch_app,
            parameters={
                "command": {"type": "string", "description": "Executable or command to launch."},
                "args": {
                    "type": "array",
                    "description": "Optional command arguments.",
                    "items": {"type": "string"},
                },
            },
        )
        self._gateway._skills.register_tool(
            name="companion_open_path",
            description="Open a file or folder path on the paired user's computer.",
            function=self.open_path,
            parameters={
                "path": {"type": "string", "description": "Local path on the paired computer."},
            },
        )
        self._gateway._skills.register_tool(
            name="companion_reveal_path",
            description="Reveal a file or folder in the paired user's file explorer.",
            function=self.reveal_path,
            parameters={
                "path": {"type": "string", "description": "Local path on the paired computer."},
            },
        )
        self._gateway._skills.register_tool(
            name="companion_notify",
            description="Show a local desktop notification on the paired user's computer.",
            function=self.notify,
            parameters={
                "title": {"type": "string", "description": "Notification title."},
                "body": {"type": "string", "description": "Notification message body."},
            },
        )
        self._gateway._skills.register_tool(
            name="companion_take_screenshot",
            description=(
                "Capture a screenshot from the paired user's real computer and return it as an image. "
                "Use this when the user asks you to see their screen, inspect something on their laptop, "
                "or send a screenshot back."
            ),
            function=self.take_screenshot,
            parameters={
                "monitor": {
                    "type": "integer",
                    "description": "Optional zero-based display index. Defaults to the primary display.",
                },
            },
        )
        logger.info("[Companion] Relay tools registered")

    async def _execute(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "error": "companion relay is not configured"}

        headers = {
            "Content-Type": "application/json",
            "x-companion-secret": self._shared_secret,
        }
        body = {
            "agentId": self._agent_id,
            "action": action,
            "payload": payload,
            "timeoutMs": self._timeout * 1000,
        }

        try:
            logger.info("[Companion] dispatch action=%s agent_id=%s", action, self._agent_id[:8] if self._agent_id else "unknown")
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout + 5)) as session:
                async with session.post(
                    f"{self._relay_url}/api/execute",
                    json=body,
                    headers=headers,
                ) as resp:
                    data = await resp.json(content_type=None)
                    if resp.status != 200:
                        logger.warning(
                            "[Companion] action=%s failed status=%s error=%s",
                            action,
                            resp.status,
                            data.get("error"),
                        )
                        return {"ok": False, "error": str(data.get("error") or f"companion relay returned HTTP {resp.status}")}
                    result = data.get("result")
                    logger.info("[Companion] action=%s completed ok=%s", action, bool(isinstance(result, dict) and result.get("ok", True)))
                    if isinstance(result, dict):
                        return result
                    return {"ok": True, "result": result}
        except Exception as exc:
            logger.warning("[Companion] action=%s request failed: %s", action, exc)
            return {"ok": False, "error": f"companion relay request failed: {exc}"}

    async def open_url(self, url: str) -> dict[str, Any]:
        return await self._execute("browser.open_url", {"url": url})

    async def search_web(self, query: str, provider: str = "google") -> dict[str, Any]:
        return await self._execute("browser.search_query", {"query": query, "provider": provider})

    async def launch_app(self, command: str, args: list[str] | None = None) -> dict[str, Any]:
        return await self._execute("app.launch", {"command": command, "args": args or []})

    async def open_path(self, path: str) -> dict[str, Any]:
        return await self._execute("file.open_path", {"path": path})

    async def reveal_path(self, path: str) -> dict[str, Any]:
        return await self._execute("file.reveal_path", {"path": path})

    async def notify(self, title: str, body: str) -> dict[str, Any]:
        return await self._execute("system.notify", {"title": title, "body": body})

    async def take_screenshot(self, monitor: int = 0) -> dict[str, Any]:
        result = await self._execute("screen.capture", {"monitor": monitor})
        if not isinstance(result, dict) or not result.get("ok"):
            return result

        screenshot_b64 = str(result.get("screenshot_b64") or result.get("image_b64") or "").strip()
        request_ctx = dict(getattr(self._gateway, "_last_request_context", None) or {})
        source_channel = str(request_ctx.get("source_channel") or "").strip()
        channel_id = str(request_ctx.get("channel_id") or "").strip()
        if screenshot_b64 and source_channel and channel_id:
            try:
                raw = screenshot_b64.split(",", 1)[1] if screenshot_b64.startswith("data:image/") else screenshot_b64
                photo_bytes = base64.b64decode(raw)
                adapter = self._gateway._channels.get(source_channel)
                if adapter is not None:
                    if source_channel == "telegram" and getattr(adapter, "_app", None) and getattr(adapter._app, "bot", None):
                        buf = io.BytesIO(photo_bytes)
                        buf.name = "neuralclaw-screenshot.png"
                        try:
                            await adapter._app.bot.send_photo(
                                chat_id=int(channel_id),
                                photo=buf,
                                caption="Here is the screenshot from your computer.",
                            )
                        except Exception:
                            buf.seek(0)
                            await adapter._app.bot.send_document(
                                chat_id=int(channel_id),
                                document=buf,
                                caption="Here is the screenshot from your computer.",
                            )
                    elif hasattr(adapter, "send_photo"):
                        await adapter.send_photo(channel_id, photo_bytes, caption="Here is the screenshot from your computer.")
                    result.pop("screenshot_b64", None)
                    result.pop("image_b64", None)
                    result["message"] = "I've captured your screen and shared the screenshot here."
                    result["image_sent"] = True
                    return result
            except Exception as exc:
                logger.warning("[Companion] screenshot delivery failed: %s", exc)

        if screenshot_b64:
            result["message"] = "I've captured the screenshot, but couldn't send the image back automatically in this channel."
        return result


class CronJobManager:
    _WEEKDAY_MAP = {
        "sun": "0",
        "sunday": "0",
        "mon": "1",
        "monday": "1",
        "tue": "2",
        "tues": "2",
        "tuesday": "2",
        "wed": "3",
        "weds": "3",
        "wednesday": "3",
        "thu": "4",
        "thur": "4",
        "thurs": "4",
        "thursday": "4",
        "fri": "5",
        "friday": "5",
        "sat": "6",
        "saturday": "6",
    }

    def __init__(self, gateway: "MeshAwareGateway") -> None:
        self._gateway = gateway
        self._control_base_url = os.getenv("NEURALCLAW_CONTROL_BASE_URL", "").strip().rstrip("/")
        self._shared_secret = os.getenv("NEURALCLAW_PROVISIONER_SECRET", "").strip()
        self._agent_id = os.getenv("NEURALCLAW_AGENT_ID", "").strip()
        self._poll_seconds = max(15, int(os.getenv("NEURALCLAW_CRON_POLL_SECONDS", "30")))
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    @property
    def enabled(self) -> bool:
        return bool(self._control_base_url and self._agent_id)

    def register_tools(self) -> None:
        if not self.enabled:
            return

        self._gateway._skills.register_tool(
            name="create_schedule",
            description=(
                "Create a reminder or scheduled job for the current user. "
                "Use this when the user asks to be reminded later or wants something to happen at a specific time."
            ),
            function=self.create_schedule,
            parameters={
                "name": {
                    "type": "string",
                    "description": "Short title for the schedule, e.g. 'Call John reminder'.",
                },
                "task": {
                    "type": "string",
                    "description": (
                        "What should happen when the schedule runs. "
                        "For reminders, phrase it as the message the user should receive."
                    ),
                },
                "schedule_text": {
                    "type": "string",
                    "description": (
                        "Human schedule phrase such as 'at 8pm', 'tomorrow at 9', "
                        "'in 30 minutes', 'every day at 8', or 'every Monday at 9am'."
                    ),
                },
                "cron_expression": {
                    "type": "string",
                    "description": "Optional raw cron expression if you already know it, e.g. '0 9 * * *'.",
                },
                "run_once_at": {
                    "type": "string",
                    "description": "Optional one-time ISO timestamp if you already know the exact time.",
                },
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone such as 'Asia/Karachi'. Defaults to UTC if omitted.",
                },
                "delete_after_run": {
                    "type": "boolean",
                    "description": "If true, disable the job after a successful one-time run.",
                },
            },
        )
        self._gateway._skills.register_tool(
            name="list_schedules",
            description="List the current agent's scheduled jobs and recent statuses.",
            function=self.list_schedules,
            parameters={
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of schedules to return.",
                },
            },
        )
        logger.info("[Cron] scheduling tools registered")

    async def start(self) -> None:
        if not self.enabled or self._task:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("[Cron] Scheduler started for agent %s", self._agent_id[:8] if self._agent_id else "unknown")

    async def stop(self) -> None:
        if not self._task:
            return
        self._stop_event.set()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._shared_secret:
            headers["x-provisioner-secret"] = self._shared_secret
        return headers

    @staticmethod
    def _default_timezone() -> str:
        return os.getenv("NEURALCLAW_DEFAULT_TIMEZONE", "UTC").strip() or "UTC"

    @staticmethod
    def _clean_schedule_text(value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip().lower())

    @staticmethod
    def _parse_clock(value: str) -> tuple[int, int]:
        match = re.fullmatch(r"(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", value.strip().lower())
        if not match:
            raise ValueError("Could not understand the time. Use forms like '8pm', '08:30', or '9:15 am'.")
        hour = int(match.group(1))
        minute = int(match.group(2) or "0")
        ampm = match.group(3)
        if minute > 59:
            raise ValueError("Minute must be between 0 and 59.")
        if ampm:
            if hour < 1 or hour > 12:
                raise ValueError("12-hour times must use an hour from 1 to 12.")
            if ampm == "pm" and hour != 12:
                hour += 12
            if ampm == "am" and hour == 12:
                hour = 0
        elif hour > 23:
            raise ValueError("Hour must be between 0 and 23.")
        return hour, minute

    def _validate_timezone(self, timezone_name: str) -> ZoneInfo:
        try:
            return ZoneInfo(timezone_name)
        except Exception as exc:
            raise ValueError(f"Unknown timezone '{timezone_name}'. Use an IANA timezone like Asia/Karachi.") from exc

    def _validate_cron_expression(self, expression: str) -> str:
        normalized = re.sub(r"\s+", " ", expression.strip())
        parts = normalized.split(" ")
        if len(parts) != 5:
            raise ValueError("Cron expressions must have 5 fields.")
        self._parse_field(parts[0], 0, 59)
        self._parse_field(parts[1], 0, 23)
        self._parse_field(parts[2], 1, 31)
        self._parse_field(parts[3], 1, 12)
        self._parse_field(parts[4].replace("7", "0"), 0, 6)
        return normalized

    def _coerce_run_once_at(self, run_once_at: str, timezone_name: str) -> str:
        tz = self._validate_timezone(timezone_name)
        raw = run_once_at.strip()
        if not raw:
            raise ValueError("run_once_at is required.")
        try:
            parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("run_once_at must be a valid ISO datetime.") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tz)
        return parsed.astimezone(dt.timezone.utc).replace(second=0, microsecond=0).isoformat()

    def _parse_schedule_text(self, schedule_text: str, timezone_name: str) -> dict[str, Any]:
        text = self._clean_schedule_text(schedule_text)
        if not text:
            raise ValueError("A schedule is required.")

        tz = self._validate_timezone(timezone_name)
        now_local = dt.datetime.now(tz)

        match = re.fullmatch(r"in (\d+) minute[s]?", text)
        if match:
            when = now_local + dt.timedelta(minutes=int(match.group(1)))
            when = when.replace(second=0, microsecond=0)
            return {"run_once_at": when.astimezone(dt.timezone.utc).isoformat(), "delete_after_run": True}

        match = re.fullmatch(r"in (\d+) hour[s]?", text)
        if match:
            when = now_local + dt.timedelta(hours=int(match.group(1)))
            when = when.replace(second=0, microsecond=0)
            return {"run_once_at": when.astimezone(dt.timezone.utc).isoformat(), "delete_after_run": True}

        match = re.fullmatch(r"(?:every day|daily)(?: at)? (.+)", text)
        if match:
            hour, minute = self._parse_clock(match.group(1))
            return {"cron_expression": f"{minute} {hour} * * *", "delete_after_run": False}

        match = re.fullmatch(r"every (\d{1,2}) minute[s]?", text)
        if match:
            step = max(1, min(59, int(match.group(1))))
            return {"cron_expression": f"*/{step} * * * *", "delete_after_run": False}

        match = re.fullmatch(r"every (\d{1,2}) hour[s]?", text)
        if match:
            step = max(1, min(23, int(match.group(1))))
            return {"cron_expression": f"0 */{step} * * *", "delete_after_run": False}

        match = re.fullmatch(r"every weekday(?:s)?(?: at)? (.+)", text)
        if match:
            hour, minute = self._parse_clock(match.group(1))
            return {"cron_expression": f"{minute} {hour} * * 1-5", "delete_after_run": False}

        match = re.fullmatch(r"every weekend(?:s)?(?: at)? (.+)", text)
        if match:
            hour, minute = self._parse_clock(match.group(1))
            return {"cron_expression": f"{minute} {hour} * * 0,6", "delete_after_run": False}

        match = re.fullmatch(r"every ([a-z]+)(?: at)? (.+)", text)
        if match:
            weekday = self._WEEKDAY_MAP.get(match.group(1))
            if weekday is not None:
                hour, minute = self._parse_clock(match.group(2))
                return {"cron_expression": f"{minute} {hour} * * {weekday}", "delete_after_run": False}

        match = re.fullmatch(r"(?:on )?(\d{4}-\d{2}-\d{2})(?: at)? (.+)", text)
        if match:
            hour, minute = self._parse_clock(match.group(2))
            date_part = dt.date.fromisoformat(match.group(1))
            when = dt.datetime.combine(date_part, dt.time(hour=hour, minute=minute), tzinfo=tz)
            return {"run_once_at": when.astimezone(dt.timezone.utc).isoformat(), "delete_after_run": True}

        match = re.fullmatch(r"(today|tomorrow)(?: at)? (.+)", text)
        if match:
            day_label = match.group(1)
            hour, minute = self._parse_clock(match.group(2))
            day_offset = 0 if day_label == "today" else 1
            target_date = (now_local + dt.timedelta(days=day_offset)).date()
            when = dt.datetime.combine(target_date, dt.time(hour=hour, minute=minute), tzinfo=tz)
            if day_label == "today" and when <= now_local:
                raise ValueError("That time has already passed today. Say 'tomorrow' or choose a later time.")
            return {"run_once_at": when.astimezone(dt.timezone.utc).isoformat(), "delete_after_run": True}

        try:
            hour, minute = self._parse_clock(text)
        except ValueError as exc:
            raise ValueError(
                "I couldn't understand that schedule. Try 'at 8pm', 'tomorrow at 9', "
                "'in 30 minutes', 'every day at 8', or a cron expression."
            ) from exc

        when = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if when <= now_local:
            when += dt.timedelta(days=1)
        return {"run_once_at": when.astimezone(dt.timezone.utc).isoformat(), "delete_after_run": True}

    @staticmethod
    def _schedule_name(name: str, task: str) -> str:
        cleaned = (name or "").strip()
        if cleaned:
            return cleaned[:120]
        fallback = (task or "").strip()
        return (fallback[:117] + "...") if len(fallback) > 120 else (fallback or "Scheduled task")

    @staticmethod
    def _build_task_prompt(task: str) -> str:
        cleaned = (task or "").strip()
        if not cleaned:
            raise ValueError("task is required.")
        return (
            "This is a scheduled task that is firing now. "
            "Carry it out and respond with the exact reminder or result the user should receive in this chat. "
            "Be concise unless the task clearly requires more detail.\n\n"
            f"SCHEDULED_TASK:\n{cleaned}"
        )

    async def create_schedule(
        self,
        name: str = "",
        task: str = "",
        schedule_text: str = "",
        cron_expression: str = "",
        run_once_at: str = "",
        timezone: str = "",
        delete_after_run: bool | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "error": "scheduler is not enabled"}

        timezone_name = (timezone or self._default_timezone()).strip() or "UTC"
        self._validate_timezone(timezone_name)

        normalized_cron = cron_expression.strip()
        normalized_run_once = run_once_at.strip()
        parsed: dict[str, Any] = {}
        if normalized_cron and normalized_run_once:
            return {"ok": False, "error": "Provide either cron_expression or run_once_at, not both."}
        if normalized_cron:
            parsed["cron_expression"] = self._validate_cron_expression(normalized_cron)
        elif normalized_run_once:
            parsed["run_once_at"] = self._coerce_run_once_at(normalized_run_once, timezone_name)
            parsed["delete_after_run"] = True
        else:
            parsed = self._parse_schedule_text(schedule_text, timezone_name)

        one_time = bool(parsed.get("run_once_at"))
        delete_flag = bool(delete_after_run) if delete_after_run is not None else bool(parsed.get("delete_after_run") or one_time)
        request_ctx = dict(getattr(self._gateway, "_last_request_context", None) or {})
        payload = {
            "name": self._schedule_name(name, task),
            "prompt": self._build_task_prompt(task),
            "cronExpression": parsed.get("cron_expression", ""),
            "runOnceAt": parsed.get("run_once_at", ""),
            "timezone": timezone_name,
            "deleteAfterRun": delete_flag,
            "enabled": True,
            "deliveryChannel": request_ctx.get("source_channel") or None,
            "deliveryChannelId": request_ctx.get("channel_id") or None,
            "deliveryAuthorId": request_ctx.get("author_id") or None,
            "deliveryAuthorName": request_ctx.get("author_name") or None,
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                async with session.post(
                    f"{self._control_base_url}/api/internal/agents/{self._agent_id}/cron",
                    headers=self._headers(),
                    json=payload,
                ) as resp:
                    data = await resp.json(content_type=None)
                    if resp.status != 200:
                        return {"ok": False, "error": data.get("error", f"schedule create failed ({resp.status})")}
        except Exception as exc:
            return {"ok": False, "error": f"schedule create failed: {exc}"}

        job = data.get("job") if isinstance(data, dict) else None
        schedule_value = payload.get("runOnceAt") or payload.get("cronExpression")
        if isinstance(job, dict):
            schedule_value = job.get("run_once_at") or job.get("cron_expression") or schedule_value
        return {
            "ok": True,
            "job": job,
            "message": f"Scheduled '{payload['name']}' for {schedule_value} ({timezone_name}).",
        }

    async def list_schedules(self, limit: int = 10) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "error": "scheduler is not enabled"}

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                async with session.get(
                    f"{self._control_base_url}/api/internal/agents/{self._agent_id}/cron",
                    headers=self._headers(),
                ) as resp:
                    data = await resp.json(content_type=None)
                    if resp.status != 200:
                        return {"ok": False, "error": data.get("error", f"schedule list failed ({resp.status})")}
        except Exception as exc:
            return {"ok": False, "error": f"schedule list failed: {exc}"}

        jobs = data.get("jobs") if isinstance(data, dict) else []
        if not isinstance(jobs, list):
            jobs = []
        jobs = jobs[: max(1, min(int(limit or 10), 50))]
        return {"ok": True, "count": len(jobs), "jobs": jobs}

    @staticmethod
    def _parse_field(field: str, min_value: int, max_value: int) -> tuple[set[int], bool]:
        if field == "*":
            return set(range(min_value, max_value + 1)), True

        values: set[int] = set()
        for chunk in field.split(","):
            chunk = chunk.strip()
            if not chunk:
                raise ValueError("empty cron field")

            step = 1
            base = chunk
            if "/" in chunk:
                base, step_raw = chunk.split("/", 1)
                step = int(step_raw)
                if step <= 0:
                    raise ValueError("cron step must be positive")

            if base == "*":
                start, end = min_value, max_value
            elif "-" in base:
                start_raw, end_raw = base.split("-", 1)
                start, end = int(start_raw), int(end_raw)
            else:
                start = end = int(base)

            if start < min_value or end > max_value or start > end:
                raise ValueError("cron value out of range")

            for value in range(start, end + 1, step):
                values.add(value)

        return values, False

    def _matches(self, expression: str, when_local: dt.datetime) -> bool:
        minute, hour, day, month, weekday = expression.split()
        minute_values, _ = self._parse_field(minute, 0, 59)
        hour_values, _ = self._parse_field(hour, 0, 23)
        day_values, day_any = self._parse_field(day, 1, 31)
        month_values, _ = self._parse_field(month, 1, 12)
        weekday_values, weekday_any = self._parse_field(weekday.replace("7", "0"), 0, 6)

        cron_weekday = (when_local.weekday() + 1) % 7
        if when_local.minute not in minute_values or when_local.hour not in hour_values or when_local.month not in month_values:
            return False

        day_match = when_local.day in day_values
        weekday_match = cron_weekday in weekday_values
        if day_any and weekday_any:
            return True
        if day_any:
            return weekday_match
        if weekday_any:
            return day_match
        return day_match or weekday_match

    async def _fetch_jobs(self) -> list[dict[str, Any]]:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(
                    f"{self._control_base_url}/api/internal/agents/{self._agent_id}/cron",
                    headers=self._headers(),
                ) as resp:
                    data = await resp.json(content_type=None)
                    if resp.status != 200:
                        logger.warning("[Cron] fetch failed status=%s error=%s", resp.status, data.get("error"))
                        return []
                    jobs = data.get("jobs")
                    return jobs if isinstance(jobs, list) else []
        except Exception as exc:
            logger.warning("[Cron] fetch failed: %s", exc)
            return []

    async def _claim_job(self, job_id: str, scheduled_for: str) -> bool:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.post(
                    f"{self._control_base_url}/api/internal/agents/{self._agent_id}/cron/{job_id}/claim",
                    headers=self._headers(),
                    json={"scheduledFor": scheduled_for},
                ) as resp:
                    data = await resp.json(content_type=None)
                    return bool(resp.status == 200 and data.get("claimed"))
        except Exception as exc:
            logger.warning("[Cron] claim failed for %s: %s", job_id, exc)
            return False

    async def _complete_job(self, job_id: str, *, status: str, result_preview: str | None = None, error: str | None = None) -> None:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                await session.post(
                    f"{self._control_base_url}/api/internal/agents/{self._agent_id}/cron/{job_id}/result",
                    headers=self._headers(),
                    json={
                        "status": status,
                        "resultPreview": (result_preview or "")[:800] or None,
                        "error": (error or "")[:800] or None,
                        "finishedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                )
        except Exception as exc:
            logger.warning("[Cron] result sync failed for %s: %s", job_id, exc)

    async def _deliver_result(self, job: dict[str, Any], response: str) -> None:
        delivery_channel = str(job.get("delivery_channel") or "").strip()
        delivery_channel_id = str(job.get("delivery_channel_id") or "").strip()
        if not delivery_channel or not delivery_channel_id:
            return
        adapter = self._gateway._channels.get(delivery_channel)
        if not adapter:
            raise RuntimeError(f"Delivery channel '{delivery_channel}' is not connected.")
        await adapter.send(delivery_channel_id, response)

    async def _execute_job(self, job: dict[str, Any], scheduled_for: str) -> None:
        job_id = str(job.get("id") or "").strip()
        prompt = str(job.get("prompt") or "").strip()
        name = str(job.get("name") or "Scheduled job").strip()
        if not job_id or not prompt:
            return

        if not await self._claim_job(job_id, scheduled_for):
            return

        logger.info("[Cron] running job=%s name=%s scheduled_for=%s", job_id[:8], name, scheduled_for)
        try:
            response = await self._gateway.process_message(
                content=prompt,
                author_id="cron",
                author_name="Scheduler",
                channel_id=f"cron:{job_id}",
                channel_type_name="CLI",
                message_metadata={
                    "source": "cron",
                    "cron_job_id": job_id,
                    "cron_job_name": name,
                    "scheduled_for": scheduled_for,
                    "is_private": True,
                },
            )
            if response:
                await self._deliver_result(job, response)
            await self._complete_job(job_id, status="completed", result_preview=response)
            logger.info("[Cron] completed job=%s", job_id[:8])
        except Exception as exc:
            logger.exception("[Cron] job=%s failed", job_id[:8])
            await self._complete_job(job_id, status="failed", error=str(exc))

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            jobs = await self._fetch_jobs()
            now_utc = dt.datetime.now(dt.timezone.utc)
            for job in jobs:
                try:
                    timezone_name = str(job.get("timezone") or "UTC").strip() or "UTC"
                    tz = self._validate_timezone(timezone_name)
                    local_now = now_utc.astimezone(tz).replace(second=0, microsecond=0)
                    expression = str(job.get("cron_expression") or "").strip()
                    run_once_at = str(job.get("run_once_at") or "").strip()
                    scheduled_for = ""

                    if expression:
                        if not self._matches(expression, local_now):
                            continue
                        scheduled_for = local_now.astimezone(dt.timezone.utc).isoformat()
                    elif run_once_at:
                        scheduled_at = self._coerce_run_once_at(run_once_at, timezone_name)
                        scheduled_dt = dt.datetime.fromisoformat(scheduled_at)
                        if scheduled_dt > now_utc:
                            continue
                        scheduled_for = scheduled_at
                    else:
                        continue

                    if str(job.get("last_scheduled_for") or "") == scheduled_for:
                        continue
                    await self._execute_job(job, scheduled_for)
                except Exception as exc:
                    logger.warning("[Cron] scheduler loop error for job=%s: %s", job.get("id"), exc)

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._poll_seconds)
            except asyncio.TimeoutError:
                pass


# ---------------------------------------------------------------------------
# QRTrackingWhatsAppAdapter â€” direct Baileys bridge (OpenClaw pattern).
#
# Runs a Node.js Baileys process as a subprocess. The bridge script is
# written to /app/wa_bridge/bridge.mjs at startup; node_modules are
# pre-installed there by the Docker build stage.
#
# No Evolution API, no Chromium â€” just Node.js + Baileys, same as OpenClaw.
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


class LocalDiscordAdapter(_ChannelAdapterBase):
    name = "discord"

    def __init__(self, token: str) -> None:
        super().__init__()
        self._token = token
        self._client: Any = None
        self._task: asyncio.Task[None] | None = None
        self._voice_enabled = _env_flag("NEURALCLAW_DISCORD_VOICE_ENABLED", False)
        self._voice_reply_text = _env_flag("NEURALCLAW_DISCORD_VOICE_REPLY_WITH_TEXT", True)
        self._voice_openai_key = (
            os.getenv("NEURALCLAW_VOICE_OPENAI_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
        self._transcribe_model = (
            os.getenv("NEURALCLAW_DISCORD_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe").strip()
            or "gpt-4o-mini-transcribe"
        )
        self._tts_model = os.getenv("NEURALCLAW_DISCORD_TTS_MODEL", "gpt-4o-mini-tts").strip() or "gpt-4o-mini-tts"
        self._tts_voice = os.getenv("NEURALCLAW_DISCORD_TTS_VOICE", "alloy").strip() or "alloy"
        self._tts_instructions = (
            os.getenv(
                "NEURALCLAW_DISCORD_TTS_INSTRUCTIONS",
                "Speak naturally, warmly, and conversationally. Sound human and responsive.",
            ).strip()
            or "Speak naturally and conversationally."
        )
        self._segment_silence_ms = max(300, int(os.getenv("NEURALCLAW_DISCORD_VOICE_SILENCE_MS", "900")))
        self._min_segment_ms = max(150, int(os.getenv("NEURALCLAW_DISCORD_VOICE_MIN_MS", "450")))
        self._max_segment_seconds = max(3, int(os.getenv("NEURALCLAW_DISCORD_VOICE_MAX_SECONDS", "15")))
        self._voice_buffers: dict[tuple[int, int], bytearray] = {}
        self._voice_users: dict[tuple[int, int], Any] = {}
        self._voice_flush_handles: dict[tuple[int, int], asyncio.Handle] = {}
        self._voice_sessions: dict[int, dict[str, Any]] = {}
        self._playback_tasks: dict[int, asyncio.Task[None]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        try:
            import discord
        except ImportError as exc:
            raise RuntimeError("discord.py is not installed in the runtime image.") from exc

        self._loop = asyncio.get_running_loop()
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        intents.members = True
        self._client = discord.Client(intents=intents)

        adapter = self

        @self._client.event
        async def on_ready() -> None:
            logger.info("[Discord] Logged in as %s", self._client.user)

        @self._client.event
        async def on_voice_state_update(member: Any, before: Any, after: Any) -> None:
            if not self._client or not self._client.user or member.id != self._client.user.id:
                return
            if before.channel and not after.channel:
                await adapter._teardown_voice_session(before.channel.guild.id)

        @self._client.event
        async def on_message(message: Any) -> None:
            if message.author == self._client.user:
                return

            is_dm = isinstance(message.channel, discord.DMChannel)
            is_mentioned = self._client.user in message.mentions if self._client.user else False
            if not is_dm and not is_mentioned:
                return

            content = message.content or ""
            if is_mentioned and self._client.user:
                content = (
                    content.replace(f"<@{self._client.user.id}>", "")
                    .replace(f"<@!{self._client.user.id}>", "")
                    .strip()
                )

            if adapter._voice_enabled and message.guild:
                lowered = content.strip().lower()
                if adapter._is_join_voice_command(lowered):
                    await adapter._handle_join_voice_command(message)
                    return
                if adapter._is_leave_voice_command(lowered):
                    await adapter._handle_leave_voice_command(message)
                    return

            from neuralclaw.channels.protocol import ChannelMessage
            media: list[dict[str, Any]] = []
            for attachment in list(getattr(message, "attachments", []) or []):
                content_type = str(getattr(attachment, "content_type", "") or "").lower()
                filename = str(getattr(attachment, "filename", "") or "").lower()
                if not (content_type.startswith("image/") or filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))):
                    continue
                try:
                    blob = await attachment.read()
                    if not blob:
                        continue
                    media.append({
                        "type": "image",
                        "mime": content_type or "image/png",
                        "base64": base64.b64encode(bytes(blob)).decode("ascii"),
                    })
                except Exception as exc:
                    logger.warning("[Discord] failed to read attachment: %s", exc)

            msg = ChannelMessage(
                content=content,
                author_id=str(message.author.id),
                author_name=message.author.display_name,
                channel_id=str(message.channel.id),
                raw=message,
                media=media,
                metadata={
                    "platform": "discord",
                    "source": "discord",
                    "is_dm": is_dm,
                    "guild": message.guild.name if message.guild else None,
                },
            )
            await adapter._dispatch(msg)

        self._task = asyncio.create_task(self._client.start(self._token))

    async def stop(self) -> None:
        for guild_id in list(self._voice_sessions):
            await self._teardown_voice_session(guild_id)
        for handle in self._voice_flush_handles.values():
            handle.cancel()
        self._voice_flush_handles.clear()
        if self._client:
            await self._client.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def send(self, channel_id: str, content: str, **kwargs: Any) -> None:
        if not self._client:
            return
        if channel_id.startswith("discord-voice:") and kwargs.get("discord_voice_reply") and self._voice_enabled:
            guild_id = int(channel_id.split(":", 1)[1])
            await self._send_voice_reply(guild_id, content, **kwargs)
            return

        channel = self._client.get_channel(int(channel_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(channel_id))
        if channel:
            await channel.send(content)

    async def send_photo(self, channel_id: str, photo_bytes: bytes, caption: str = "") -> None:
        if not self._client:
            return
        try:
            import discord
        except ImportError:
            return
        channel = self._client.get_channel(int(channel_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(channel_id))
        if channel:
            file = discord.File(io.BytesIO(photo_bytes), filename="neuralclaw-image.png")
            await channel.send(content=caption or None, file=file)

    def _is_join_voice_command(self, content: str) -> bool:
        return content in {
            "join voice",
            "join voice channel",
            "join my voice channel",
            "join vc",
            "come to voice",
            "come to my voice channel",
            "join me in voice",
        }

    def _is_leave_voice_command(self, content: str) -> bool:
        return content in {
            "leave voice",
            "leave voice channel",
            "leave vc",
            "disconnect",
            "disconnect voice",
        }

    async def _handle_join_voice_command(self, message: Any) -> None:
        if not self._voice_openai_key:
            await message.channel.send("Discord voice is enabled, but no OpenAI key is configured for speech.")
            return

        voice_state = getattr(message.author, "voice", None)
        voice_channel = getattr(voice_state, "channel", None)
        if not voice_channel:
            await message.channel.send("Join a voice channel first, then ask me to join.")
            return

        try:
            await self._join_voice_channel(voice_channel, message.channel)
        except Exception as exc:
            logger.exception("[Discord] failed to join voice channel")
            await message.channel.send(f"I couldn't join that voice channel: {exc}")
            return

        await message.channel.send(f"Joined voice channel `{voice_channel.name}`. Speak normally and I'll answer in voice.")

    async def _handle_leave_voice_command(self, message: Any) -> None:
        guild = message.guild
        if not guild:
            await message.channel.send("That command only works in a server text channel.")
            return
        if guild.id not in self._voice_sessions:
            await message.channel.send("I'm not in a voice channel right now.")
            return
        await self._teardown_voice_session(guild.id)
        await message.channel.send("Left the voice channel.")

    async def _join_voice_channel(self, voice_channel: Any, control_channel: Any) -> None:
        try:
            from discord.ext import voice_recv
        except ImportError as exc:
            raise RuntimeError("discord-ext-voice-recv is not installed in the runtime image.") from exc

        guild_id = voice_channel.guild.id
        existing = self._voice_sessions.get(guild_id)
        if existing:
            voice_client = existing["voice_client"]
            if voice_client.channel.id != voice_channel.id:
                await voice_client.move_to(voice_channel)
            existing["control_channel_id"] = control_channel.id
            existing["voice_channel_id"] = voice_channel.id
            return

        voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
        sink = _create_discord_voice_sink(voice_recv, self, guild_id)
        voice_client.listen(sink)
        self._voice_sessions[guild_id] = {
            "voice_client": voice_client,
            "sink": sink,
            "control_channel_id": control_channel.id,
            "voice_channel_id": voice_channel.id,
            "guild_name": voice_channel.guild.name,
            "queue": asyncio.Queue(),
        }
        logger.info("[Discord] voice session started guild=%s channel=%s", guild_id, voice_channel.id)

    async def _teardown_voice_session(self, guild_id: int) -> None:
        session = self._voice_sessions.pop(guild_id, None)
        if not session:
            return
        task = self._playback_tasks.pop(guild_id, None)
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        voice_client = session.get("voice_client")
        try:
            if voice_client:
                if hasattr(voice_client, "stop_listening"):
                    voice_client.stop_listening()
                if voice_client.is_connected():
                    await voice_client.disconnect(force=True)
        except Exception:
            logger.exception("[Discord] error tearing down voice session guild=%s", guild_id)
        for key in [k for k in self._voice_buffers if k[0] == guild_id]:
            self._voice_buffers.pop(key, None)
            self._voice_users.pop(key, None)
            handle = self._voice_flush_handles.pop(key, None)
            if handle:
                handle.cancel()
        logger.info("[Discord] voice session stopped guild=%s", guild_id)

    def _schedule_voice_frame(self, guild_id: int, user: Any, pcm: bytes) -> None:
        if not self._loop:
            return
        self._loop.call_soon_threadsafe(self._handle_voice_frame, guild_id, user, pcm)

    def _handle_voice_frame(self, guild_id: int, user: Any, pcm: bytes) -> None:
        if getattr(user, "bot", False):
            return
        key = (guild_id, int(user.id))
        buf = self._voice_buffers.setdefault(key, bytearray())
        buf.extend(pcm)
        self._voice_users[key] = user

        handle = self._voice_flush_handles.pop(key, None)
        if handle:
            handle.cancel()

        max_bytes = self._pcm_bytes_for_seconds(self._max_segment_seconds)
        if len(buf) >= max_bytes:
            asyncio.create_task(self._flush_voice_segment(guild_id, int(user.id)))
            return

        delay = self._segment_silence_ms / 1000
        self._voice_flush_handles[key] = asyncio.get_running_loop().call_later(
            delay,
            lambda: asyncio.create_task(self._flush_voice_segment(guild_id, int(user.id))),
        )

    async def _flush_voice_segment(self, guild_id: int, user_id: int) -> None:
        key = (guild_id, user_id)
        handle = self._voice_flush_handles.pop(key, None)
        if handle:
            handle.cancel()

        pcm_bytes = bytes(self._voice_buffers.pop(key, b""))
        user = self._voice_users.pop(key, None)
        if not pcm_bytes or not user:
            return
        if len(pcm_bytes) < self._pcm_bytes_for_ms(self._min_segment_ms):
            return

        try:
            wav_bytes = self._build_wav_from_pcm(pcm_bytes)
            transcript = await self._transcribe_wav_bytes(wav_bytes)
            if not transcript:
                return
        except Exception as exc:
            logger.warning("[Discord] voice segment processing failed guild=%s user=%s: %s", guild_id, user_id, exc)
            await self._send_text_to_control_channel(guild_id, f"I couldn't process that voice segment: {exc}")
            return

        logger.info("[Discord] inbound voice segment guild=%s user=%s transcript_len=%d", guild_id, user_id, len(transcript))

        from neuralclaw.channels.protocol import ChannelMessage

        session = self._voice_sessions.get(guild_id) or {}
        msg = ChannelMessage(
            content=transcript,
            author_id=str(user_id),
            author_name=getattr(user, "display_name", getattr(user, "name", f"user-{user_id}")),
            channel_id=f"discord-voice:{guild_id}",
            raw=user,
            metadata={
                "platform": "discord",
                "source": "discord",
                "discord_voice_reply": True,
                "discord_guild_id": guild_id,
                "discord_guild_name": session.get("guild_name"),
                "discord_voice_channel_id": session.get("voice_channel_id"),
                "discord_control_channel_id": session.get("control_channel_id"),
            },
        )
        await self._dispatch(msg)

    def _build_wav_from_pcm(self, pcm_bytes: bytes) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(48000)
            wav_file.writeframes(pcm_bytes)
        return output.getvalue()

    async def _transcribe_wav_bytes(self, wav_bytes: bytes) -> str:
        form = aiohttp.FormData()
        form.add_field("model", self._transcribe_model)
        form.add_field("file", wav_bytes, filename="discord-voice.wav", content_type="audio/wav")
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
        data = json.loads(raw_text)
        transcript = str(data.get("text") or "").strip()
        if not transcript:
            raise RuntimeError("OpenAI transcription returned empty text.")
        return transcript

    async def _send_voice_reply(self, guild_id: int, content: str, **kwargs: Any) -> None:
        session = self._voice_sessions.get(guild_id)
        if not session:
            await self._send_text_to_control_channel(guild_id, content)
            return
        audio_bytes, extension = await self._synthesize_tts_audio(content)
        await session["queue"].put((audio_bytes, extension, content))
        task = self._playback_tasks.get(guild_id)
        if not task or task.done():
            self._playback_tasks[guild_id] = asyncio.create_task(self._consume_voice_queue(guild_id))

    async def _consume_voice_queue(self, guild_id: int) -> None:
        session = self._voice_sessions.get(guild_id)
        if not session:
            return
        queue: asyncio.Queue = session["queue"]
        voice_client = session["voice_client"]
        while True:
            try:
                audio_bytes, extension, text = await queue.get()
            except asyncio.CancelledError:
                break

            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
                    tmp.write(audio_bytes)
                    temp_path = tmp.name

                loop = asyncio.get_running_loop()
                finished = loop.create_future()

                def _after_play(err: Exception | None) -> None:
                    if err and not finished.done():
                        loop.call_soon_threadsafe(finished.set_exception, err)
                    elif not finished.done():
                        loop.call_soon_threadsafe(finished.set_result, None)

                import discord

                source = discord.FFmpegPCMAudio(temp_path)
                voice_client.play(source, after=_after_play)
                await finished
                if self._voice_reply_text:
                    await self._send_text_to_control_channel(guild_id, text)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("[Discord] voice playback failed guild=%s: %s", guild_id, exc)
                await self._send_text_to_control_channel(guild_id, f"Voice reply failed, falling back to text: {text}")
            finally:
                if temp_path:
                    with contextlib.suppress(Exception):
                        os.unlink(temp_path)
                queue.task_done()

            if queue.empty():
                break

        self._playback_tasks.pop(guild_id, None)

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
            "instructions": self._tts_instructions,
            "format": "mp3",
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(
                "https://api.openai.com/v1/audio/speech",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                audio_bytes = await resp.read()
                if resp.status != 200:
                    detail = audio_bytes.decode("utf-8", errors="replace")[:300]
                    raise RuntimeError(f"OpenAI speech failed ({resp.status}): {detail}")
        if not audio_bytes:
            raise RuntimeError("OpenAI speech returned empty audio.")
        return audio_bytes, ".mp3"

    async def _send_text_to_control_channel(self, guild_id: int, content: str) -> None:
        if not self._client or not content:
            return
        session = self._voice_sessions.get(guild_id)
        channel_id = session.get("control_channel_id") if session else None
        if not channel_id:
            return
        channel = self._client.get_channel(int(channel_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(channel_id))
        if channel:
            await channel.send(content)

    def _pcm_bytes_for_ms(self, duration_ms: int) -> int:
        return int((48000 * 2 * 2) * (duration_ms / 1000))

    def _pcm_bytes_for_seconds(self, duration_seconds: int) -> int:
        return int(48000 * 2 * 2 * duration_seconds)


def _create_discord_voice_sink(voice_recv_mod: Any, adapter: LocalDiscordAdapter, guild_id: int) -> Any:
    class _DiscordVoiceReceiveSink(voice_recv_mod.AudioSink):
        def __init__(self) -> None:
            super().__init__()

        def wants_opus(self) -> bool:
            return False

        def write(self, user: Any, data: Any) -> None:
            pcm = getattr(data, "pcm", None)
            if not pcm or not user:
                return
            adapter._schedule_voice_frame(guild_id, user, pcm)

        def cleanup(self) -> None:
            return

    return _DiscordVoiceReceiveSink()


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
                content_type = str(resp.headers.get("Content-Type") or "").strip().lower()
                if resp.status != 200:
                    raise RuntimeError(f"Slack file download failed ({resp.status}): {body[:200]!r}")
        if not body:
            raise RuntimeError("Slack audio download returned empty content.")
        if len(body) > self._max_audio_bytes:
            raise RuntimeError(
                f"Slack audio file exceeds limit ({len(body)} bytes > {self._max_audio_bytes} bytes)."
            )
        if content_type.startswith("text/") or content_type.startswith("application/json"):
            preview = body[:200].decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Slack file download returned non-audio content-type '{content_type or 'unknown'}': {preview}"
            )
        if body.startswith(b"<!DOCTYPE html") or body.startswith(b"<html") or body.startswith(b"{"):
            preview = body[:200].decode("utf-8", errors="replace")
            raise RuntimeError(f"Slack file download returned unexpected payload: {preview}")
        logger.info(
            "[Slack] downloaded audio file id=%s name=%s mimetype=%s bytes=%d",
            file_obj.get("id"),
            file_obj.get("name"),
            content_type or file_obj.get("mimetype"),
            len(body),
        )
        return body

    async def _transcribe_audio(self, file_obj: dict[str, Any], file_bytes: bytes) -> str:
        prepared_bytes, filename, mimetype = await self._prepare_audio_for_transcription(file_obj, file_bytes)
        form = aiohttp.FormData()
        form.add_field("model", self._transcribe_model)
        form.add_field("file", prepared_bytes, filename=filename, content_type=mimetype)
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

    async def _prepare_audio_for_transcription(
        self,
        file_obj: dict[str, Any],
        file_bytes: bytes,
    ) -> tuple[bytes, str, str]:
        filename = str(file_obj.get("name") or "slack-audio").strip() or "slack-audio"
        mimetype = str(file_obj.get("mimetype") or "application/octet-stream").strip().lower() or "application/octet-stream"
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                f"ffmpeg is required to normalize Slack audio format '{mimetype or 'unknown'}' before transcription."
            )

        logger.info("[Slack] normalizing audio before transcription filename=%s mimetype=%s size=%d", filename, mimetype, len(file_bytes))
        return await self._transcode_audio_to_wav(file_bytes, filename)

    async def _transcode_audio_to_wav(self, file_bytes: bytes, filename: str) -> tuple[bytes, str, str]:
        source_suffix = Path(filename).suffix or ".input"
        with tempfile.TemporaryDirectory(prefix="slack-audio-") as tmpdir:
            source_path = Path(tmpdir) / f"source{source_suffix}"
            output_path = Path(tmpdir) / "normalized.wav"
            source_path.write_bytes(file_bytes)

            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-y",
                "-i",
                str(source_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(output_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0 or not output_path.exists():
                stderr_text = stderr.decode("utf-8", errors="replace")
                preview = stderr_text[-600:] if len(stderr_text) > 600 else stderr_text
                raise RuntimeError(f"Audio conversion failed: {preview}")

            return output_path.read_bytes(), "slack-audio.wav", "audio/wav"

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

    companion_tools = [
        "companion_open_url",
        "companion_search_web",
        "companion_launch_app",
        "companion_open_path",
        "companion_reveal_path",
        "companion_notify",
        "companion_take_screenshot",
    ]
    companion_enabled = bool(
        os.getenv("NEURALCLAW_COMPANION_RELAY_URL", "").strip()
        and os.getenv("NEURALCLAW_COMPANION_RELAY_SHARED_SECRET", "").strip()
        and os.getenv("NEURALCLAW_AGENT_ID", "").strip()
    )
    if companion_enabled:
        for tool_name in companion_tools:
            if tool_name not in config.policy.allowed_tools:
                config.policy.allowed_tools.append(tool_name)
        companion_hint = (
            "\n\nYou may have companion_* tools connected to the user's PAIRED COMPUTER. "
            "Use companion_* tools when the user asks to open a real browser on their laptop, "
            "launch local apps, reveal files, capture a real screenshot from their device, "
            "or do anything on their actual computer. "
            "Use browser_* tools for the hosted cloud browser running inside Railway. "
            "If a companion action fails, explain that the paired computer is offline or unavailable."
        )
        config.persona = (config.persona or "") + companion_hint

    onboarding_hint = (
        "\n\nWhen speaking with a new person you do not know well yet, learn their preferred name in a friendly, natural way and remember it. "
        "If they share a stable preference or personal context worth remembering for future chats, keep that in memory and use it later."
    )
    config.persona = (config.persona or "") + onboarding_hint

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
    gw._companion_manager = CompanionRelayManager(gw)
    gw._cron_manager = CronJobManager(gw)

    for ch_config in config.channels:
        if not ch_config.enabled or not ch_config.token:
            continue

        if ch_config.name == "telegram":
            from neuralclaw.channels.telegram import TelegramAdapter

            gw.add_channel(TelegramAdapter(ch_config.token))
        elif ch_config.name == "discord":
            if _env_flag("NEURALCLAW_DISCORD_VOICE_ENABLED", False):
                logger.info("[Discord] Python adapter disabled; Node voice worker will own Discord transport")
            else:
                gw.add_channel(LocalDiscordAdapter(ch_config.token))

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
        web.get("/api/stats", _handle_stats),
        web.get("/api/traces", _handle_traces),
        web.get("/api/audit", _handle_audit),
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
    if gw._cron_manager:
        await gw._cron_manager.start()

    loop = asyncio.get_running_loop()

    async def _shutdown(sig_name: str) -> None:
        logger.info("[runtime] received %s â€” shutting down gracefully", sig_name)
        if gw._cron_manager:
            await gw._cron_manager.stop()
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

