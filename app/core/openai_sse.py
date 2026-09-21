import json
from dataclasses import dataclass
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


def extract_reasoning_delta(event: dict, sent: str) -> tuple[str, str]:
    if event.get("event") == "thinking_delta":
        return next_text_delta(event.get("thinking_delta") or "", sent)
    if event.get("event") == "result":
        return next_text_delta((event.get("result") or {}).get("reasoning_content") or "", sent)
    return "", sent


@dataclass
class SseStreamState:
    sent: str = ""
    sent_reasoning: str = ""
    role_sent: bool = False


def sse_frames_for_event(
    event: dict,
    state: SseStreamState,
    chat_id: str,
    created: int,
    model: str,
) -> list[bytes]:
    """Map one agy stream event to OpenAI SSE frames. Reasoning is emitted before content."""
    frames: list[bytes] = []
    rpiece, state.sent_reasoning = extract_reasoning_delta(event, state.sent_reasoning)
    if rpiece:
        delta: dict[str, Any] = {"reasoning_content": rpiece}
        if not state.role_sent:
            delta["role"] = "assistant"
            state.role_sent = True
        frames.append(format_sse(openai_chunk(chat_id, created, model, delta)))
    piece, state.sent = extract_agent_text_delta(event, state.sent)
    if piece:
        delta = {"content": piece}
        if not state.role_sent:
            delta["role"] = "assistant"
            state.role_sent = True
        frames.append(format_sse(openai_chunk(chat_id, created, model, delta)))
    if event.get("event") != "result":
        return frames
    result = event.get("result") or {}
    final_text = result.get("response") or result.get("text") or ""
    piece, state.sent = next_text_delta(final_text, state.sent)
    if piece:
        delta = {"content": piece}
        if not state.role_sent:
            delta["role"] = "assistant"
            state.role_sent = True
        frames.append(format_sse(openai_chunk(chat_id, created, model, delta)))
    frames.append(
        format_sse(
            openai_chunk(
                chat_id,
                created,
                model,
                {},
                finish_reason="stop",
                usage=usage_from_agy(result.get("usage")),
            )
        )
    )
    return frames


def usage_from_agy(usage: Optional[dict]) -> dict:
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = int(usage.get("input_tokens") or 0)
    completion = int(usage.get("output_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    res = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
    if usage.get("thinking_tokens") is not None:
        res["completion_tokens_details"] = {
            "reasoning_tokens": int(usage.get("thinking_tokens") or 0)
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
    state = SseStreamState()
    for event in events:
        yield from sse_frames_for_event(event, state, chat_id, created, model)
    yield format_sse("[DONE]")
