"""OpenAI tools → agy --json-schema emulation.

agy has no --tools flag. When the client sends OpenAI `tools`, we constrain
print-mode output with a discriminated union and parse the FIRST valid JSON
object. agy otherwise retries the virtual tool several times and concatenates
objects into `response`.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional

TOOLS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["message", "tool_call"]},
        "content": {"type": "string"},
        "name": {"type": "string"},
        "arguments": {"type": "object"},
    },
    "required": ["kind"],
}

_PREAMBLE = """You are answering through a strict JSON schema. You have CLIENT tools listed below.
You cannot execute those client tools yourself. Do not run shell commands, write files, or use
built-in agent tools to fake them.

If you need a client tool, emit ONE JSON object with kind="tool_call", name, and arguments, then STOP.
Do not retry. Do not emit a second object. Do not wait for a result in this turn.

If you can answer without a client tool, emit ONE JSON object with kind="message" and content.

Available client tools (OpenAI format):
"""


def first_json_object(text: str) -> Optional[dict]:
    """Return the first complete JSON object in `text`, even if more objects follow."""
    if not text:
        return None
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        ch = text[idx]
        if ch != "{":
            idx += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(obj, dict):
            return obj
        idx = max(end, idx + 1)
    return None


def interpret_schema_object(obj: dict | None) -> dict[str, Any]:
    """Normalize a parsed schema object to {kind, content?, name?, arguments?}."""
    if not isinstance(obj, dict):
        return {"kind": "message", "content": ""}
    kind = obj.get("kind")
    name = obj.get("name") or (obj.get("function") or {}).get("name")
    arguments = obj.get("arguments")
    if arguments is None:
        arguments = (obj.get("function") or {}).get("arguments")
    if kind == "tool_call" or (kind is None and name):
        if not name:
            return {"kind": "message", "content": json.dumps(obj, ensure_ascii=False)}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        if not isinstance(arguments, dict):
            arguments = {}
        return {"kind": "tool_call", "name": str(name), "arguments": arguments}
    content = obj.get("content")
    if content is None:
        content = obj.get("response") or obj.get("text") or ""
    return {"kind": "message", "content": str(content)}


def parse_emulated_output(text: str, structured_output: dict | None = None) -> dict[str, Any]:
    obj = structured_output if isinstance(structured_output, dict) else first_json_object(text)
    return interpret_schema_object(obj)


def to_openai_tool_calls(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    args = parsed.get("arguments") or {}
    if isinstance(args, dict):
        args_s = json.dumps(args, ensure_ascii=False)
    else:
        args_s = str(args)
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": parsed["name"], "arguments": args_s},
        }
    ]


def format_tools_preamble(tools: list[dict[str, Any]] | None, tool_choice: Any = None) -> str:
    if not tools:
        return ""
    lines = [_PREAMBLE, json.dumps(tools, ensure_ascii=False, indent=2)]
    if tool_choice == "none":
        lines.append('\ntool_choice is "none": do not call any client tool. Emit kind="message".')
    elif tool_choice in ("required", "any"):
        lines.append('\ntool_choice is "required": you MUST emit kind="tool_call".')
    elif isinstance(tool_choice, dict):
        forced = ((tool_choice.get("function") or {}).get("name")
                  or (tool_choice.get("function") or {}).get("Name"))
        if forced:
            lines.append(f'\nYou MUST call the client tool named "{forced}".')
    return "\n".join(lines) + "\n\n"


def format_history_message(msg) -> str:
    role = (getattr(msg, "role", None) or "user").capitalize()
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        bits = []
        for tc in tool_calls:
            fn = (tc.get("function") if isinstance(tc, dict) else None) or {}
            cid = (tc.get("id") if isinstance(tc, dict) else None) or ""
            bits.append(f"{fn.get('name')}({fn.get('arguments')}) [id={cid}]")
        return f"Assistant called {'; '.join(bits)}"
    content = getattr(msg, "content", None)
    if (getattr(msg, "role", "") or "").lower() == "tool":
        ident = getattr(msg, "tool_call_id", None) or getattr(msg, "name", None) or ""
        text = content if isinstance(content, str) else _flatten_content_parts(content)
        return f"Tool result [{ident}]: {text or ''}"
    if isinstance(content, str):
        return f"{role}: {content}"
    if isinstance(content, list):
        return f"{role}: {_flatten_content_parts(content)}"
    return f"{role}: "


def _flatten_content_parts(parts) -> str:
    if not isinstance(parts, list):
        return "" if parts is None else str(parts)
    texts = []
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text":
            texts.append(p.get("text") or "")
    return " ".join(texts)
