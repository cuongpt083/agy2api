import json
from typing import Any, Optional


def next_text_delta(text_delta: str, sent: str) -> tuple[str, str]:
    """Return (piece_to_emit, updated_sent) for cumulative or fragment deltas."""
    if not text_delta:
        return "", sent
    if text_delta.startswith(sent):
        piece = text_delta[len(sent):]
        return piece, sent + piece
    if sent.startswith(text_delta):
        return "", sent
    return text_delta, sent + text_delta


def extract_agent_text_delta(event: dict, sent: str) -> tuple[str, str]:
    if event.get("event") != "step_update":
        return "", sent
    step = event.get("step_update") or {}
    if step.get("step_type") != "agent_response":
        return "", sent
    return next_text_delta(step.get("text_delta") or "", sent)


def usage_from_agy(usage: Optional[dict]) -> dict:
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = int(usage.get("input_tokens") or 0)
    completion = int(usage.get("output_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    cache_read = int(usage.get("cache_read_tokens") or 0)
    thinking = int(usage.get("thinking_tokens") or 0)

    res: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cache_read_tokens": cache_read,
        "prompt_tokens_details": {
            "cached_tokens": cache_read,
        },
    }
    if thinking:
        res["completion_tokens_details"] = {
            "reasoning_tokens": thinking,
        }
    return res


def openai_chunk(
    chat_id: str,
    created: int,
    model: str,
    delta: dict,
    finish_reason: Optional[str] = None,
    usage: Optional[dict] = None,
) -> dict:
    chunk: dict[str, Any] = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def format_sse(payload: Any) -> bytes:
    if payload == "[DONE]":
        return b"data: [DONE]\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def events_to_sse_bytes(events: list[dict], chat_id: str, created: int, model: str):
    """Pure mapping used by the route and unit tests."""
    sent = ""
    role_sent = False
    result = None

    for event in events:
        piece, sent = extract_agent_text_delta(event, sent)
        if piece:
            delta = {"content": piece}
            if not role_sent:
                delta["role"] = "assistant"
                role_sent = True
            yield format_sse(openai_chunk(chat_id, created, model, delta))
        if event.get("event") == "result":
            result = event.get("result") or {}

    if result is not None:
        final_text = result.get("response") or result.get("text") or ""
        piece, sent = next_text_delta(final_text, sent)
        if piece:
            delta = {"content": piece}
            if not role_sent:
                delta["role"] = "assistant"
                role_sent = True
            yield format_sse(openai_chunk(chat_id, created, model, delta))

        status = (result.get("status") or "SUCCESS").upper()
        finish = "stop" if status == "SUCCESS" else "stop"
        yield format_sse(
            openai_chunk(
                chat_id,
                created,
                model,
                {},
                finish_reason=finish,
                usage=usage_from_agy(result.get("usage")),
            )
        )

    yield format_sse("[DONE]")
