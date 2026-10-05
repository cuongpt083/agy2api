import os
import json
import asyncio
import time
import uuid
import io
from typing import Optional
from fastapi import Request, APIRouter, Depends, BackgroundTasks, UploadFile, File, Form, Header
from fastapi.responses import StreamingResponse, Response, JSONResponse
from app.api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Usage,
    ModelList,
    Model,
    SpeechRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageObject,
    ToolCall,
)
from app.core.security import get_api_key
from app.core.agy_runner import run_agy_prompt, stream_agy_prompt
from app.core.openai_sse import (
    extract_agent_text_delta,
    events_to_sse_bytes_with_tools,
    format_sse,
    next_text_delta,
    openai_chunk,
    sse_keepalive,
    sse_role_open,
    usage_from_agy,
)
from app.core.tool_emulation import (
    TOOLS_JSON_SCHEMA,
    format_history_message,
    format_tools_preamble,
    parse_emulated_output,
    to_openai_tool_calls,
)
import logging
from app.core.logging_setup import trace_id_var
import re
from app.core.file_handler import TempFileManager
from app.core.capcut_api import AsyncCapCutWrapper
from app.core.model_manager import get_available_models
from app.core.image_generation import (
    MAX_REFERENCE_IMAGES,
    ImageWorkspace,
    agy_text_from_response,
    build_image_prompt,
    clamp_n,
    collect_generated_images,
    encode_image_objects,
    snapshot_image_sizes,
    stable_output_images,
)
from app.core.image_limiter import (
    POLL_INTERVAL_S,
    ImageSlotAcquisition,
    get_image_queue_timeout,
    get_image_waiting_count,
    get_semaphore,
)
from app.core.metrics import (
    record_chat_completion,
    record_image_generation,
    record_speech_request,
    record_tokens,
)
from app.core.capture import global_capture_manager

logger = logging.getLogger(__name__)

router = APIRouter()
capcut_wrapper = AsyncCapCutWrapper()

@router.get("/models", response_model=ModelList, summary="List Models", description="Returns a list of available AI models.")
async def list_models(api_key: str = Depends(get_api_key)):
    models = await get_available_models()
    return ModelList(data=models)

def _ext_from_data_uri(url: str) -> str:
    ext = ".png"
    match = re.match(r"^data:([^/]+)/([^;,]+)", url)
    if not match:
        return ext
    mime_sub = match.group(2).lower()
    if mime_sub in ["jpeg", "jpg"]:
        return ".jpg"
    if mime_sub == "pdf":
        return ".pdf"
    if mime_sub == "msword":
        return ".doc"
    if "wordprocessingml" in mime_sub:
        return ".docx"
    if mime_sub == "plain":
        return ".txt"
    if mime_sub == "csv":
        return ".csv"
    if mime_sub in ["png", "gif", "webp"]:
        return f".{mime_sub}"
    return f".{mime_sub}"


# agy truncates a stdin user message at roughly 190KB ("<truncated N bytes>"), and the tail is
# what gets cut: the latest user request. OpenClaw-style clients send ~130KB system prompts plus
# ~120KB of tool schemas, so the prompt must be fitted under this budget before it reaches agy.
_MAX_PROMPT_BYTES = int(os.environ.get("AGY_MAX_PROMPT_BYTES", "170000"))
# Per-message caps applied while fitting (head+tail kept, middle elided).
_TAIL_MSG_CAP = 24_000
_OLD_MSG_CAP = 4_000
_MIN_SYSTEM_BYTES = 8_000


def _nbytes(text: str) -> int:
    return len(text.encode("utf-8"))


def _clip_middle(text: str, max_bytes: int) -> str:
    if _nbytes(text) <= max_bytes:
        return text
    keep = max(max_bytes - 80, 200) // 2
    raw = text.encode("utf-8")
    head = raw[:keep].decode("utf-8", errors="ignore")
    tail = raw[-keep:].decode("utf-8", errors="ignore")
    return f"{head}\n[... {len(raw) - 2 * keep} bytes omitted by agy2api ...]\n{tail}"


def _render_message(msg, file_mgr: TempFileManager, files_to_attach: list[str]) -> Optional[str]:
    if msg.tool_calls or (msg.role or "").lower() == "tool":
        return format_history_message(msg)
    content_text = ""
    if isinstance(msg.content, str):
        content_text = msg.content
    elif isinstance(msg.content, list):
        text_parts = []
        for p in msg.content:
            if p.get("type") == "text":
                text_parts.append(p.get("text", ""))
            elif p.get("type") == "image_url":
                url = p.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    ext = _ext_from_data_uri(url)
                    try:
                        fpath = file_mgr.add_base64_file(url, ext=ext)
                        files_to_attach.append(fpath)
                        text_parts.append(f"[Attached Image: {fpath}]")
                    except Exception as e:
                        text_parts.append(f"[Failed to attach image: {e}]")
                else:
                    text_parts.append(f"[Image URL: {url}]")
        content_text = " ".join(text_parts)
    if content_text:
        return f"{msg.role.capitalize()}: {content_text}"
    return None


def build_chat_prompt(req: ChatCompletionRequest, file_mgr: TempFileManager) -> tuple[str, list[str]]:
    files_to_attach: list[str] = []
    system_lines: list[str] = []
    convo: list[str] = []
    last_user_idx = -1
    for msg in req.messages:
        line = _render_message(msg, file_mgr, files_to_attach)
        if line is None:
            continue
        if (msg.role or "").lower() in ("system", "developer") and not convo:
            system_lines.append(line)
            continue
        if (msg.role or "").lower() == "user":
            last_user_idx = len(convo)
        convo.append(line)

    preamble = format_tools_preamble(req.tools, req.tool_choice).rstrip() if req.tools else ""
    system = "\n".join(system_lines)
    closing = "Assistant: "

    def assemble(pre: str, sys_text: str, history: list[str]) -> str:
        parts = [x for x in (pre, sys_text) if x] + history + [closing]
        return "\n".join(parts)

    full = assemble(preamble, system, convo)
    if _nbytes(full) <= _MAX_PROMPT_BYTES:
        return full, files_to_attach

    # 1. Slim tool schemas (drop per-field docs, clip long descriptions).
    if req.tools:
        preamble = format_tools_preamble(req.tools, req.tool_choice, slim=True).rstrip()

    # 2. Latest user turn and everything after it is kept (each message capped).
    split = last_user_idx if last_user_idx >= 0 else max(len(convo) - 1, 0)
    tail = [_clip_middle(x, _TAIL_MSG_CAP) for x in convo[split:]]
    older = convo[:split]

    budget = _MAX_PROMPT_BYTES - _nbytes(preamble) - sum(_nbytes(x) + 1 for x in tail) - 200
    # 3. System prompt gets what is left, but never so much that no history fits.
    sys_budget = max(min(_nbytes(system), budget - min(budget // 4, 20_000)), _MIN_SYSTEM_BYTES)
    system = _clip_middle(system, sys_budget) if system else ""
    budget -= _nbytes(system)

    # 4. Older history newest-first, each clipped, until the budget runs out.
    kept: list[str] = []
    for line in reversed(older):
        line = _clip_middle(line, _OLD_MSG_CAP)
        cost = _nbytes(line) + 1
        if cost > budget:
            break
        kept.append(line)
        budget -= cost
    kept.reverse()
    dropped = len(older) - len(kept)
    if dropped:
        kept.insert(0, f"[... {dropped} earlier conversation messages omitted by agy2api to fit the prompt limit ...]")

    fitted = assemble(preamble, system, kept + tail)
    logger.info(
        "Prompt fitted to budget: %d -> %d bytes (limit %d, tools_slim=%s, older_dropped=%d)",
        _nbytes(full), _nbytes(fitted), _MAX_PROMPT_BYTES, bool(req.tools), dropped,
    )
    return fitted, files_to_attach


def _assistant_text(agy_response) -> str:
    if isinstance(agy_response, dict):
        return agy_response.get("text") or agy_response.get("content") or agy_response.get("response") or str(agy_response)
    return str(agy_response)


# Set AGY_TRACE_REQUESTS=1 to dump each incoming chat request (headers + body) to
# logs/trace/requests.jsonl, and log a one-line summary + pool key decision.
TRACE_REQUESTS = os.environ.get("AGY_TRACE_REQUESTS", "").lower() in ("1", "true", "yes")
_TRACE_FILE = os.environ.get("AGY_TRACE_FILE", os.path.join("logs", "trace", "requests.jsonl"))
_SENSITIVE_HEADERS = {"authorization", "x-api-key", "cookie"}


def _trace_request(request: Request, req: ChatCompletionRequest) -> None:
    try:
        headers = {
            k: ("<redacted>" if k.lower() in _SENSITIVE_HEADERS else v)
            for k, v in request.headers.items()
        }
        body = req.model_dump()
        tools = req.tools or []
        tool_names = [((t.get("function") or {}).get("name") or t.get("name")) for t in tools]
        emulate = bool(req.tools) and req.tool_choice != "none"
        logger.info(
            "TRACE model=%r stream=%s n_tools=%d tool_choice=%r emulate_tools=%s -> pool_key=%s||%s "
            "n_messages=%d roles=%s ua=%r",
            req.model, req.stream, len(tools), req.tool_choice, emulate,
            req.model, "tools" if emulate else "plain",
            len(req.messages), [m.role for m in req.messages][:12], headers.get("user-agent"),
        )
        os.makedirs(os.path.dirname(_TRACE_FILE), exist_ok=True)
        with open(_TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.time(), "trace_id": trace_id_var.get(None),
                "client": request.client.host if request.client else None,
                "headers": headers, "tool_names": tool_names, "body": body,
            }, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.warning("trace dump failed: %s", e)


# aicoworker/OpenClaw's first-token watchdog (attempt.ts FIRST_TOKEN_STALL_TIMEOUT_MS=30s) is only
# disarmed by a content-bearing event (text/thinking/toolcall delta). Role-only and empty-delta
# keepalive chunks are ignored, so a tools turn that buffers >30s gets aborted and failed over to
# the fallback model. Emit one non-empty reasoning_content delta up front (shown as "thinking",
# never as assistant text) for tool-emulated streams. Set AGY_TOOLS_EARLY_THINKING=0 to disable.
EARLY_THINKING = os.environ.get("AGY_TOOLS_EARLY_THINKING", "1").lower() not in ("0", "false", "no")
_EARLY_THINKING_TEXT = "Working…"

_KEEPALIVE_SECONDS = 15.0
_QUEUE_END = object()


async def _pump_agy_events(agen, queue: asyncio.Queue) -> None:
    """Push agy events onto a queue so the SSE loop can emit keepalives without cancelling the generator."""
    try:
        async for event in agen:
            await queue.put(event)
    except Exception as exc:
        await queue.put(exc)
    finally:
        try:
            await agen.aclose()
        except Exception:
            pass
        await queue.put(_QUEUE_END)


async def _sse_chat_stream(
    prompt: str,
    model: str,
    chat_id: str,
    created: int,
    file_mgr: TempFileManager,
    emulate_tools: bool = False,
    req_payload: dict = None,
    source_agent: str = None,
):
    sent = ""
    role_sent = False
    events: list[dict] = []
    t0 = time.time()
    first_token_time = None
    captured_frames: list[bytes] = []
    conv_id = None
    pump_task = None
    try:
        # Immediate data chunk: OpenClaw/openai-completions times out waiting for first event
        # while agy plans or reads files. Role-only delta is valid and not shown as content.
        role_frame = sse_role_open(chat_id, created, model)
        captured_frames.append(role_frame)
        role_sent = True
        first_token_time = 0.0
        yield role_frame
        if emulate_tools and EARLY_THINKING:
            think_frame = format_sse(
                openai_chunk(chat_id, created, model, {"reasoning_content": _EARLY_THINKING_TEXT})
            )
            captured_frames.append(think_frame)
            yield think_frame

        agen = stream_agy_prompt(
            prompt=prompt,
            model=model,
            json_schema=TOOLS_JSON_SCHEMA if emulate_tools else None,
            stop_after_first_schema_object=emulate_tools,
        )
        queue: asyncio.Queue = asyncio.Queue()
        pump_task = asyncio.create_task(_pump_agy_events(agen, queue))

        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                yield sse_keepalive(chat_id, created, model)
                continue

            if item is _QUEUE_END:
                break
            if isinstance(item, Exception):
                raise item
            event = item

            if event.get("event") == "init":
                conv_id = event.get("conversation_id") or (
                    (event.get("init") or {}).get("conversation_id")
                )
            elif event.get("event") == "result":
                res = event.get("result") or {}
                if res.get("conversation_id"):
                    conv_id = res.get("conversation_id")

            if emulate_tools:
                events.append(event)
                continue
            piece, sent = extract_agent_text_delta(event, sent)
            if piece:
                now = time.time()
                if first_token_time in (None, 0.0):
                    first_token_time = now - t0
                delta = {"content": piece}
                frame = format_sse(openai_chunk(chat_id, created, model, delta))
                captured_frames.append(frame)
                yield frame
            if event.get("event") != "result":
                continue
            result = event.get("result") or {}
            final_text = result.get("response") or result.get("text") or ""
            piece, sent = next_text_delta(final_text, sent)
            if piece:
                now = time.time()
                if first_token_time in (None, 0.0):
                    first_token_time = now - t0
                delta = {"content": piece}
                frame = format_sse(openai_chunk(chat_id, created, model, delta))
                captured_frames.append(frame)
                yield frame
            usage_dict = usage_from_agy(result.get("usage"))
            record_tokens(
                model,
                usage_dict.get("prompt_tokens"),
                usage_dict.get("completion_tokens"),
                usage_dict.get("total_tokens"),
                usage_dict.get("cache_read_tokens"),
            )
            stop_frame = format_sse(
                openai_chunk(
                    chat_id,
                    created,
                    model,
                    {},
                    finish_reason="stop",
                    usage=usage_dict,
                )
            )
            captured_frames.append(stop_frame)
            yield stop_frame
        duration = time.time() - t0
        record_chat_completion(
            model,
            stream=True,
            status="success",
            duration=duration,
            first_token_duration=first_token_time,
        )
        if emulate_tools:
            res_evt = next((e.get("result") for e in events if e.get("event") == "result"), None)
            if res_evt and res_evt.get("usage"):
                u = usage_from_agy(res_evt.get("usage"))
                record_tokens(
                    model,
                    u.get("prompt_tokens"),
                    u.get("completion_tokens"),
                    u.get("total_tokens"),
                    u.get("cache_read_tokens"),
                )
            for frame in events_to_sse_bytes_with_tools(events, chat_id, created, model):
                captured_frames.append(frame)
                yield frame
        else:
            yield format_sse("[DONE]")
    except Exception as e:
        duration = time.time() - t0
        record_chat_completion(
            model,
            stream=True,
            status="error",
            duration=duration,
            first_token_duration=first_token_time,
        )
        logger.error("AGY stream failed: %s", e, exc_info=True)
        err = {
            "error": {
                "message": str(e),
                "type": "server_error",
            }
        }
        yield format_sse(err)
        yield format_sse("[DONE]")
    finally:
        if pump_task is not None and not pump_task.done():
            pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):
                pass
        file_mgr.cleanup()
        if req_payload and captured_frames:
            try:
                global_capture_manager.enqueue_turn(
                    request_payload=req_payload,
                    response_data=captured_frames,
                    source_agent=source_agent,
                    latency_ms=int((time.time() - t0) * 1000),
                    stream=True,
                    conversation_id=conv_id,
                )
            except Exception as capture_err:
                logger.warning("Failed to enqueue streaming capture: %s", capture_err)


@router.post("/chat/completions", summary="Chat Completions", description="Creates a model response for the given chat conversation. Supports multimodal inputs via base64 data URIs. Set stream=true for OpenAI-compatible SSE. Pass OpenAI `tools` to emulate function calling via agy --json-schema.")
async def chat_completions(
    request: Request,
    req: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key),
    x_source_agent: Optional[str] = Header(default=None, alias="X-Source-Agent"),
):
    logger.info(f"Processing chat completions for model: {req.model} stream={req.stream} tools={bool(req.tools)}")
    if TRACE_REQUESTS:
        _trace_request(request, req)
    file_mgr = TempFileManager()
    final_prompt, files_to_attach = build_chat_prompt(req, file_mgr)
    emulate_tools = bool(req.tools) and req.tool_choice != "none"
    schema = TOOLS_JSON_SCHEMA if emulate_tools else None
    req_dict = req.model_dump()

    if req.stream:
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        return StreamingResponse(
            _sse_chat_stream(
                final_prompt,
                req.model,
                chat_id,
                created,
                file_mgr,
                emulate_tools=emulate_tools,
                req_payload=req_dict,
                source_agent=x_source_agent,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    background_tasks.add_task(file_mgr.cleanup)
    t0 = time.time()
    try:
        agy_response = await run_agy_prompt(
            prompt=final_prompt,
            model=req.model,
            files=files_to_attach,
            json_schema=schema,
        )
        duration = time.time() - t0
        usage_data = Usage()
        if isinstance(agy_response, dict) and agy_response.get("usage"):
            mapped = usage_from_agy(agy_response.get("usage"))
            usage_data = Usage(
                prompt_tokens=mapped.get("prompt_tokens", 0),
                completion_tokens=mapped.get("completion_tokens", 0),
                total_tokens=mapped.get("total_tokens", 0),
                cache_read_tokens=mapped.get("cache_read_tokens", 0),
                prompt_tokens_details=mapped.get("prompt_tokens_details"),
                completion_tokens_details=mapped.get("completion_tokens_details"),
            )
            record_tokens(
                req.model,
                usage_data.prompt_tokens,
                usage_data.completion_tokens,
                usage_data.total_tokens,
                usage_data.cache_read_tokens,
            )
        record_chat_completion(req.model, stream=False, status="success", duration=duration)

        non_stream_conv_id = agy_response.get("conversation_id") if isinstance(agy_response, dict) else None
        if emulate_tools and isinstance(agy_response, dict):
            raw = agy_response.get("response") or agy_response.get("text") or agy_response.get("content") or ""
            parsed = parse_emulated_output(str(raw), agy_response.get("structured_output"))
            if parsed.get("kind") == "tool_call":
                tcs = [ToolCall(**tc) for tc in to_openai_tool_calls(parsed)]
                resp = ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    created=int(time.time()),
                    model=req.model,
                    choices=[
                        Choice(
                            message=ChoiceMessage(content=None, tool_calls=tcs),
                            finish_reason="tool_calls",
                        )
                    ],
                    usage=usage_data,
                )
                global_capture_manager.enqueue_turn(
                    request_payload=req_dict,
                    response_data=resp.model_dump(),
                    source_agent=x_source_agent,
                    latency_ms=int(duration * 1000),
                    stream=False,
                    conversation_id=non_stream_conv_id,
                )
                return resp
            assistant_text = parsed.get("content") or ""
        else:
            assistant_text = _assistant_text(agy_response)

        resp = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=req.model,
            choices=[Choice(message=ChoiceMessage(content=assistant_text))],
            usage=usage_data,
        )
        global_capture_manager.enqueue_turn(
            request_payload=req_dict,
            response_data=resp.model_dump(),
            source_agent=x_source_agent,
            latency_ms=int(duration * 1000),
            stream=False,
            conversation_id=non_stream_conv_id,
        )
        return resp
    except Exception:
        record_chat_completion(req.model, stream=False, status="error", duration=time.time() - t0)
        raise
    except Exception:
        record_chat_completion(req.model, stream=False, status="error", duration=time.time() - t0)
        raise

def _prepare_image_job(req: ImageGenerationRequest) -> tuple[ImageWorkspace, str, int]:
    n = clamp_n(req.n)
    workspace = ImageWorkspace()
    ref_paths: list[str] = []
    for i, url in enumerate((req.reference_images or [])[:MAX_REFERENCE_IMAGES]):
        if not url:
            continue
        try:
            ref_paths.append(workspace.add_reference(url, i))
        except Exception as e:
            logger.warning("Skipping invalid reference image %s: %s", i, e)
    prompt = build_image_prompt(
        prompt=req.prompt,
        out_dir=workspace.out_dir,
        n=n,
        size=req.size,
        ref_paths=ref_paths,
    )
    return workspace, prompt, n


def _image_error_payload(message: str) -> dict:
    return {
        "type": "error",
        "error": {"message": message, "type": "image_generation_error"},
    }


async def _sse_image_generation(
    req: ImageGenerationRequest,
    workspace: ImageWorkspace,
    prompt: str,
    n: int,
):
    yield format_sse({"type": "status", "stage": "started"})

    timeout_s = get_image_queue_timeout()
    loop = asyncio.get_running_loop()
    sem = get_semaphore()
    deadline = loop.time() + timeout_s

    # Acquire semaphore cleanly using a single long-lived task polled with asyncio.wait
    # to avoid cancelling and leaking semaphore permits in Python 3.10+
    acquired = False
    acquire_future = asyncio.ensure_future(sem.acquire())
    try:
        while not acquired:
            time_left = deadline - loop.time()
            if time_left <= 0:
                acquire_future.cancel()
                if acquire_future.done() and not acquire_future.cancelled() and not acquire_future.exception():
                    sem.release()
                yield format_sse(_image_error_payload("Server is busy. Image generation request timed out waiting for a free slot."))
                yield format_sse({"type": "done"})
                yield format_sse("[DONE]")
                workspace.cleanup()
                return

            poll_duration = min(time_left, POLL_INTERVAL_S)
            done, _ = await asyncio.wait({acquire_future}, timeout=poll_duration)
            if acquire_future in done:
                if acquire_future.cancelled() or acquire_future.exception():
                    raise acquire_future.exception() or asyncio.CancelledError()
                acquired = True
            else:
                waiting = get_image_waiting_count()
                yield format_sse({"type": "status", "stage": "queued", "position": waiting})
    except (asyncio.CancelledError, GeneratorExit):
        if not acquire_future.done():
            acquire_future.cancel()
        elif not acquire_future.cancelled() and not acquire_future.exception():
            sem.release()
        workspace.cleanup()
        raise

    yield format_sse({"type": "status", "stage": "generating"})
    prev_sizes: dict[str, int] = {}
    last_text = ""
    emitted = False
    agen = None
    try:
        agen = stream_agy_prompt(prompt=prompt, model=req.model, extra_dirs=[workspace.root])
        while True:
            event = None
            try:
                event = await asyncio.wait_for(agen.__anext__(), timeout=0.2)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                pass
            if isinstance(event, dict):
                result = event.get("result")
                if isinstance(result, dict):
                    last_text = agy_text_from_response(result) or last_text
            ready = stable_output_images(workspace.out_dir, n, prev_sizes)
            prev_sizes = snapshot_image_sizes(workspace.out_dir)
            if ready:
                encoded = encode_image_objects(ready, req.response_format or "url")
                yield format_sse(
                    {"type": "image", "created": int(time.time()), "data": encoded}
                )
                emitted = True
                break
        if not emitted:
            paths = collect_generated_images(workspace.out_dir, last_text, n, allowed_root=workspace.root)
            if not paths:
                yield format_sse(_image_error_payload("AGY did not produce an image file."))
            else:
                encoded = encode_image_objects(paths, req.response_format or "url")
                yield format_sse(
                    {"type": "image", "created": int(time.time()), "data": encoded}
                )
        record_image_generation("success")
        yield format_sse({"type": "done"})
        yield format_sse("[DONE]")
    except Exception as e:
        record_image_generation("error")
        logger.error("AGY image stream failed: %s", e, exc_info=True)
        yield format_sse(_image_error_payload(str(e)))
        yield format_sse({"type": "done"})
        yield format_sse("[DONE]")
    finally:
        if agen is not None:
            try:
                await agen.aclose()
            except Exception:
                pass
        if acquired:
            sem.release()
        workspace.cleanup()


@router.post("/images/generations", response_model=ImageGenerationResponse, summary="Image Generations", description="Creates an image given a prompt using AGY. Returns image bytes as a data URI or b64_json. Set stream=true for SSE status/image events.")
async def generate_image(req: ImageGenerationRequest, background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
    logger.info("Generating image. Prompt: %s... stream=%s", req.prompt[:50], req.stream)
    workspace, prompt, n = _prepare_image_job(req)

    if req.stream:
        return StreamingResponse(
            _sse_image_generation(req, workspace, prompt, n),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    background_tasks.add_task(workspace.cleanup)
    try:
        async with ImageSlotAcquisition():
            agy_response = await run_agy_prompt(
                prompt=prompt,
                model=req.model,
                output_format="json",
                extra_dirs=[workspace.root],
            )
    except asyncio.TimeoutError:
        record_image_generation("error")
        logger.warning("Image generation queue timeout exceeded.")
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": "10"},
            content={
                "error": {
                    "message": "Server is busy. Image generation request timed out waiting for a free slot.",
                    "type": "rate_limit_error",
                }
            },
        )
    except Exception as e:
        record_image_generation("error")
        logger.error("AGY image generation failed: %s", e, exc_info=True)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(e), "type": "image_generation_error"}},
        )

    paths = collect_generated_images(
        workspace.out_dir,
        agy_text_from_response(agy_response),
        n,
        allowed_root=workspace.root,
    )
    if not paths:
        record_image_generation("error")
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "AGY did not produce an image file.",
                    "type": "image_generation_error",
                }
            },
        )

    record_image_generation("success")
    encoded = encode_image_objects(paths, req.response_format or "url")
    return ImageGenerationResponse(
        created=int(time.time()),
        data=[ImageObject(**item) for item in encoded],
    )

import subprocess

@router.get("/logs", summary="Get System Logs", description="Read the latest system logs of the AGY Wrapper service.")
async def get_logs(lines: int = 100, api_key: str = Depends(get_api_key)):
    try:
        result = subprocess.run(
            ["journalctl", "--user", "-u", "agy-wrapper.service", "-n", str(lines), "--no-pager"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return {"logs": result.stdout}
        else:
            return {"logs": f"Failed to read logs (code {result.returncode}): {result.stderr}"}
    except Exception as e:
        return {"logs": f"Error reading logs: {str(e)}"}

@router.post(
    "/audio/speech", 
    summary="Text to Speech (Audio Generations)", 
    description="Tạo tệp âm thanh từ văn bản dựa trên chuẩn OpenAI Audio API (engine CapCut).",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Binary stream của file MP3 (audio/mpeg)",
            "content": {"audio/mpeg": {}}
        }
    }
)
async def audio_speech(req: SpeechRequest, api_key: str = Depends(get_api_key)):
    logger.info(f"Generating speech (voice={req.voice}, speed={req.speed}). Text: {req.input[:50]}...")
    try:
        audio_bytes = await capcut_wrapper.generate_speech(
            text=req.input,
            voice=req.voice,
            speed=req.speed
        )
        record_speech_request("success")
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")
    except Exception as e:
        record_speech_request("error")
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get(
    "/audio/voices", 
    summary="List Voices", 
    description="Lấy danh sách tất cả các giọng đọc (voices) khả dụng từ engine CapCut."
)
async def audio_voices(api_key: str = Depends(get_api_key)):
    try:
        voices = capcut_wrapper.get_voices()
        return JSONResponse(content={"voices": voices})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post(
    "/audio/transcriptions", 
    summary="Speech to Text (Audio Transcriptions)", 
    description="Chuyển đổi file âm thanh thành văn bản hoặc phụ đề thời gian chuẩn."
)
async def audio_transcriptions(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Tệp âm thanh cần upload (mp3, mp4, wav, v.v...)"),
    model: str = Form("whisper-1", description="ID của mô hình (vd: whisper-1)"),
    language: str = Form(None, description="Mã ngôn ngữ (vd: en-US, vi-VN). Bỏ trống để tự nhận diện."),
    response_format: str = Form("json", description="Định dạng trả về (json, text, srt, vtt)"),
    api_key: str = Depends(get_api_key)
):
    logger.info(f"Transcribing audio file: {file.filename}, language: {language}, format: {response_format}")
    try:
        file_mgr = TempFileManager()
        background_tasks.add_task(file_mgr.cleanup)
        
        import os
        ext = os.path.splitext(file.filename)[1] if file.filename else ".mp3"
        temp_path = os.path.join(file_mgr.temp_dir.name, f"upload{ext}")
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        transcription = await capcut_wrapper.transcribe_audio(
            file_path=temp_path,
            response_format=response_format,
            language=language
        )
        
        if response_format in ["json", "verbose_json"]:
            import json
            return JSONResponse(content=json.loads(transcription))
        else:
            return Response(content=transcription, media_type="text/plain")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
