"""Translate OpenAI Chat Completions (OMP) to Gemini generateContent REST and back.

Uses the public Gemini API JSON shape (camelCase). Does not invoke or wrap `agy`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional

from app.core.openai_sse import format_sse, openai_chunk

_EFFORT_TO_LEVEL = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}

_DISPLAY_NAME = re.compile(
    r"^Gemini (.+) \((High|Medium|Low)\)$",
    re.IGNORECASE,
)
_SLUG_EFFORT = re.compile(
    r"^(gemini-.+)-(high|medium|low)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ResolvedModel:
    api_model: str
    thinking_level: Optional[str] = None


@dataclass(frozen=True)
class GeminiCall:
    model: str
    body: dict[str, Any]


class ThoughtSignatureCache:
    """In-process store so OMP can drop extra_content and we still replay signatures."""

    def __init__(self) -> None:
        self._by_id: dict[str, str] = {}
        self._by_id_name: dict[tuple[str, str], str] = {}

    def put(self, tool_call_id: str, signature: str, name: Optional[str] = None) -> None:
        if not tool_call_id or not signature:
            return
        self._by_id[tool_call_id] = signature
        if name:
            self._by_id_name[(tool_call_id, name)] = signature

    def get(self, tool_call_id: str, name: Optional[str] = None) -> Optional[str]:
        if name:
            found = self._by_id_name.get((tool_call_id, name))
            if found:
                return found
        return self._by_id.get(tool_call_id)


SIGNATURE_CACHE = ThoughtSignatureCache()


def resolve_gemini_model(model_id: str) -> ResolvedModel:
    raw = (model_id or "").strip()
    if not raw:
        raise ValueError("model is required for the Gemini API tool-calling path")

    display = _DISPLAY_NAME.fullmatch(raw)
    if display:
        slug = "gemini-" + re.sub(r"\s+", "-", display.group(1).lower().strip())
        return ResolvedModel(api_model=slug, thinking_level=display.group(2).lower())

    lower = raw.lower()
    if not lower.startswith("gemini"):
        raise ValueError(
            f"Model '{model_id}' is not available on the Gemini API tool-calling path. "
            "Use a gemini-* model id. Claude/GPT models on Antigravity cannot expose "
            "OpenAI tools to Oh-My-Pi."
        )

    effort = _SLUG_EFFORT.fullmatch(raw)
    if effort:
        return ResolvedModel(
            api_model=effort.group(1).lower(),
            thinking_level=effort.group(2).lower(),
        )
    return ResolvedModel(api_model=lower, thinking_level=None)


def build_gemini_request(
    req: dict[str, Any],
    signatures: Optional[ThoughtSignatureCache] = None,
) -> GeminiCall:
    signatures = signatures or ThoughtSignatureCache()
    resolved = resolve_gemini_model(req["model"])
    effort = _EFFORT_TO_LEVEL.get((req.get("reasoning_effort") or "").lower())
    thinking_level = effort or resolved.thinking_level

    system_text, contents = _openai_messages_to_contents(req.get("messages") or [], signatures)

    body: dict[str, Any] = {"contents": contents}
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}

    gemini_tools = _openai_tools_to_gemini(req.get("tools"))
    if gemini_tools:
        body["tools"] = gemini_tools
        tool_config = _openai_tool_choice_to_config(req.get("tool_choice"))
        if tool_config:
            body["toolConfig"] = {"functionCallingConfig": tool_config}

    generation: dict[str, Any] = {}
    if req.get("temperature") is not None:
        generation["temperature"] = req["temperature"]
    max_tokens = req.get("max_tokens")
    if max_tokens is None:
        max_tokens = req.get("max_completion_tokens")
    if max_tokens is not None:
        generation["maxOutputTokens"] = int(max_tokens)
    if thinking_level:
        generation["thinkingConfig"] = {"thinkingLevel": thinking_level.upper()}
    if generation:
        body["generationConfig"] = generation

    return GeminiCall(model=resolved.api_model, body=body)


def gemini_candidate_to_openai_choice(
    candidate: dict[str, Any],
    signatures: Optional[ThoughtSignatureCache] = None,
) -> dict[str, Any]:
    signatures = signatures or ThoughtSignatureCache()
    parts = ((candidate.get("content") or {}).get("parts")) or []
    text_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for idx, part in enumerate(parts):
        if part.get("thought"):
            continue
        fc = part.get("functionCall")
        if fc:
            tool_calls.append(_function_call_to_tool_call(fc, part, idx, signatures))
            continue
        text = part.get("text")
        if text:
            text_chunks.append(text)

    message: dict[str, Any] = {"role": "assistant"}
    if tool_calls:
        message["content"] = None
        message["tool_calls"] = tool_calls
        finish = "tool_calls"
    else:
        message["content"] = "".join(text_chunks)
        finish = "stop"
    return {"index": 0, "message": message, "finish_reason": finish}


class GeminiSseEncoder:
    """Incremental Gemini stream → OpenAI SSE frames, including a terminal [DONE]."""

    def __init__(
        self,
        chat_id: str,
        created: int,
        model: str,
        signatures: Optional[ThoughtSignatureCache] = None,
    ) -> None:
        self.chat_id = chat_id
        self.created = created
        self.model = model
        self.signatures = signatures or ThoughtSignatureCache()
        self.role_sent = False
        self.saw_tool_calls = False
        self.tool_index = 0
        self.usage = None
        self.finish_seen = False

    def feed(self, event: dict[str, Any]) -> list[bytes]:
        frames: list[bytes] = []
        if event.get("usageMetadata"):
            self.usage = _usage_from_gemini(event["usageMetadata"])
        candidates = event.get("candidates") or []
        if not candidates:
            return frames
        cand = candidates[0]
        parts = ((cand.get("content") or {}).get("parts")) or []
        for part in parts:
            if part.get("thought"):
                continue
            fc = part.get("functionCall")
            if fc:
                self.saw_tool_calls = True
                delta: dict[str, Any] = {
                    "tool_calls": [
                        _function_call_to_tool_call(
                            fc, part, self.tool_index, self.signatures
                        )
                    ]
                }
                delta["tool_calls"][0]["index"] = self.tool_index
                if not self.role_sent:
                    delta["role"] = "assistant"
                    self.role_sent = True
                frames.append(format_sse(openai_chunk(self.chat_id, self.created, self.model, delta)))
                self.tool_index += 1
                continue
            text = part.get("text")
            if text:
                delta = {"content": text}
                if not self.role_sent:
                    delta["role"] = "assistant"
                    self.role_sent = True
                frames.append(format_sse(openai_chunk(self.chat_id, self.created, self.model, delta)))
        if cand.get("finishReason"):
            self.finish_seen = True
            frames.append(self._finish_frame())
        return frames

    def close(self) -> list[bytes]:
        frames: list[bytes] = []
        if not self.finish_seen:
            frames.append(self._finish_frame())
        frames.append(format_sse("[DONE]"))
        return frames

    def _finish_frame(self) -> bytes:
        finish = "tool_calls" if self.saw_tool_calls else "stop"
        return format_sse(
            openai_chunk(
                self.chat_id,
                self.created,
                self.model,
                {},
                finish_reason=finish,
                usage=self.usage,
            )
        )


def gemini_stream_to_sse(
    events: Iterable[dict[str, Any]],
    chat_id: str,
    created: int,
    model: str,
    signatures: Optional[ThoughtSignatureCache] = None,
) -> Iterator[bytes]:
    encoder = GeminiSseEncoder(chat_id, created, model, signatures)
    for event in events:
        yield from encoder.feed(event)
    yield from encoder.close()


def _openai_tools_to_gemini(tools: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
    if not tools:
        return None
    declarations = []
    for tool in tools:
        if (tool or {}).get("type") not in (None, "function"):
            continue
        fn = (tool or {}).get("function") or tool
        name = fn.get("name")
        if not name:
            continue
        decl: dict[str, Any] = {"name": name}
        if fn.get("description"):
            decl["description"] = fn["description"]
        if fn.get("parameters") is not None:
            decl["parameters"] = fn["parameters"]
        declarations.append(decl)
    if not declarations:
        return None
    return [{"functionDeclarations": declarations}]


def _openai_tool_choice_to_config(tool_choice: Any) -> Optional[dict[str, Any]]:
    if tool_choice is None or tool_choice == "auto":
        return {"mode": "AUTO"}
    if tool_choice == "none":
        return {"mode": "NONE"}
    if tool_choice == "required":
        return {"mode": "ANY"}
    if isinstance(tool_choice, dict):
        name = ((tool_choice.get("function") or {}).get("name")) or tool_choice.get("name")
        cfg: dict[str, Any] = {"mode": "ANY"}
        if name:
            cfg["allowedFunctionNames"] = [name]
        return cfg
    return {"mode": "AUTO"}


def _openai_messages_to_contents(
    messages: list[dict[str, Any]],
    signatures: ThoughtSignatureCache,
) -> tuple[Optional[str], list[dict[str, Any]]]:
    system_bits: list[str] = []
    contents: list[dict[str, Any]] = []
    id_to_name: dict[str, str] = {}
    pending_tool_parts: list[dict[str, Any]] = []

    def flush_tool_parts() -> None:
        nonlocal pending_tool_parts
        if pending_tool_parts:
            contents.append({"role": "user", "parts": pending_tool_parts})
            pending_tool_parts = []

    for msg in messages:
        role = (msg.get("role") or "").lower()
        if role in ("system", "developer"):
            flush_tool_parts()
            text = _plain_text(msg.get("content"))
            if text:
                system_bits.append(text)
            continue

        if role == "tool":
            tool_call_id = msg.get("tool_call_id") or ""
            name = msg.get("name") or id_to_name.get(tool_call_id) or "unknown"
            pending_tool_parts.append(
                {
                    "functionResponse": {
                        "name": name,
                        "id": tool_call_id,
                        "response": _tool_response_object(msg.get("content")),
                    }
                }
            )
            continue

        flush_tool_parts()

        if role == "assistant":
            parts: list[dict[str, Any]] = []
            text_parts = _content_parts(msg.get("content"))
            parts.extend(text_parts)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                tc_id = tc.get("id") or ""
                name = fn.get("name") or ""
                if tc_id and name:
                    id_to_name[tc_id] = name
                sig = _signature_from_tool_call(tc, signatures)
                fc_part: dict[str, Any] = {
                    "functionCall": {
                        "name": name,
                        "args": _parse_arguments(fn.get("arguments")),
                        "id": tc_id,
                    }
                }
                if sig:
                    fc_part["thoughtSignature"] = sig
                parts.append(fc_part)
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue

        # user (and anything else treated as user)
        parts = _content_parts(msg.get("content"))
        if parts:
            contents.append({"role": "user", "parts": parts})

    flush_tool_parts()
    system_text = "\n".join(system_bits) if system_bits else None
    return system_text, contents


def _signature_from_tool_call(tc: dict[str, Any], cache: ThoughtSignatureCache) -> Optional[str]:
    tc_id = tc.get("id") or ""
    name = (tc.get("function") or {}).get("name")
    extra = tc.get("extra_content") or {}
    google = extra.get("google") or {}
    sig = google.get("thought_signature")
    if sig:
        cache.put(tc_id, sig, name=name)
        return sig
    return cache.get(tc_id, name=name)


def _function_call_to_tool_call(
    fc: dict[str, Any],
    part: dict[str, Any],
    idx: int,
    signatures: ThoughtSignatureCache,
) -> dict[str, Any]:
    name = fc.get("name") or "unknown"
    fc_id = fc.get("id") or f"call_{name}_{idx}"
    args = fc.get("args") if fc.get("args") is not None else {}
    sig = part.get("thoughtSignature") or part.get("thought_signature")
    if sig:
        signatures.put(fc_id, sig, name=name)
    tool_call: dict[str, Any] = {
        "id": fc_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":")),
        },
    }
    if sig:
        tool_call["extra_content"] = {"google": {"thought_signature": sig}}
    return tool_call


def _plain_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    bits = []
    if isinstance(content, list):
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                bits.append(p.get("text") or "")
            elif isinstance(p, str):
                bits.append(p)
    return "\n".join(bit for bit in bits if bit)


def _content_parts(content: Any) -> list[dict[str, Any]]:
    if content is None or content == "":
        return []
    if isinstance(content, str):
        return [{"text": content}]
    parts: list[dict[str, Any]] = []
    if isinstance(content, list):
        for p in content:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                text = p.get("text") or ""
                if text:
                    parts.append({"text": text})
            elif p.get("type") == "image_url":
                url = ((p.get("image_url") or {}).get("url")) or ""
                inline = _data_uri_to_inline(url)
                if inline:
                    parts.append(inline)
                elif url:
                    parts.append({"text": f"[Image URL: {url}]"})
    return parts


def _data_uri_to_inline(url: str) -> Optional[dict[str, Any]]:
    if not url.startswith("data:"):
        return None
    header, _, b64 = url.partition(",")
    mime = "image/png"
    match = re.match(r"^data:([^;]+)", header)
    if match:
        mime = match.group(1)
    if not b64:
        return None
    return {"inlineData": {"mimeType": mime, "data": b64}}


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {"value": arguments}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"output": arguments}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


def _tool_response_object(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = _plain_text(content) if not isinstance(content, str) else content
    if text is None:
        text = ""
    return {"output": text}


def usage_from_gemini(usage: dict[str, Any]) -> dict[str, int]:
    prompt = int(usage.get("promptTokenCount") or 0)
    completion = int(usage.get("candidatesTokenCount") or 0)
    total = int(usage.get("totalTokenCount") or (prompt + completion))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _usage_from_gemini(usage: dict[str, Any]) -> dict[str, int]:
    return usage_from_gemini(usage)
