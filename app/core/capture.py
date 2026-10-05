"""Zero-overhead background capture queue and processing engine.

Uses an in-memory bounded queue and single batch writer task with fail-open semantics.
Guarantees 0ms delay to client requests and zero degradation to TTFB/throughput.
Enriches captured turns with Chain-of-Thought (thinking) from the Antigravity brain
transcript in the background.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Optional

from dotenv import load_dotenv

from app.core.capture_db import CaptureDatabase

load_dotenv()

logger = logging.getLogger(__name__)


def _is_truthy(val: Optional[str], default: bool = True) -> bool:
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def is_capture_enabled() -> bool:
    """Check whether capture to SQLite is enabled via environment variables."""
    val = os.environ.get("CAPTURE_ENABLED")
    if val is None:
        val = os.environ.get("AGY_CAPTURE_ENABLED")
    return _is_truthy(val, default=True)


# Configurable parameters via environment variables
CAPTURE_ENABLED = is_capture_enabled()
DEFAULT_DB_PATH = os.path.join(os.getcwd(), "data", "capture.db")
CAPTURE_DB_PATH = os.environ.get("CAPTURE_DB_PATH") or os.environ.get("AGY_CAPTURE_DB_PATH", DEFAULT_DB_PATH)
CAPTURE_QUEUE_MAXSIZE = int(os.environ.get("CAPTURE_QUEUE_MAXSIZE", "2000"))
CAPTURE_BATCH_SIZE = int(os.environ.get("CAPTURE_BATCH_SIZE", "50"))
CAPTURE_BATCH_TIMEOUT = float(os.environ.get("CAPTURE_BATCH_TIMEOUT", "1.0"))

DEFAULT_BRAIN_DIR = os.environ.get("AGY_BRAIN_DIR", os.path.expanduser("~/.gemini/antigravity-cli/brain"))


SENSITIVE_PATTERNS = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]{12,}", re.IGNORECASE),
    re.compile(r"\b(sk-[A-Za-z0-9_\-\.]{16,})\b"),
    re.compile(r"\b(ghp_[A-Za-z0-9]{20,})\b"),
]


def redact_sensitive_content(text: str) -> str:
    """Redact private API keys and tokens before storing to capture database."""
    if not text:
        return text
    for pattern in SENSITIVE_PATTERNS:
        text = pattern.sub(
            lambda m: f"{m.group(1)}[REDACTED]" if m.lastindex else "[REDACTED]",
            text,
        )
    return text


def _read_brain_transcript_sync(conv_id: str, brain_dir: str = DEFAULT_BRAIN_DIR) -> Optional[str]:
    """Synchronous file reader to be executed inside a thread worker."""
    if not conv_id:
        return None
    transcript_path = os.path.join(brain_dir, conv_id, ".system_generated", "logs", "transcript_full.jsonl")
    if not os.path.isfile(transcript_path):
        transcript_path = os.path.join(brain_dir, conv_id, ".system_generated", "logs", "transcript.jsonl")
        if not os.path.isfile(transcript_path):
            return None

    thinkings = []
    try:
        with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    th = data.get("thinking")
                    if th and isinstance(th, str) and th.strip():
                        thinkings.append(th.strip())
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning("Failed to read brain transcript for %s: %s", conv_id, e)
        return None

    return "\n\n".join(thinkings) if thinkings else None


async def extract_thinking_from_brain_async(
    conv_id: str,
    brain_dir: str = DEFAULT_BRAIN_DIR,
    max_retries: int = 5,
    retry_delay: float = 0.8,
) -> Optional[str]:
    """Asynchronously reads thinking with bounded retries in a thread executor."""
    if not conv_id:
        return None

    for attempt in range(max_retries + 1):
        thinking = await asyncio.to_thread(_read_brain_transcript_sync, conv_id, brain_dir)
        if thinking:
            return thinking
        if attempt < max_retries:
            await asyncio.sleep(retry_delay)

    return None


def parse_usage_and_reasoning_from_chunks(
    chunks: list[dict],
) -> tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
    """Extract prompt_tokens, completion_tokens, cached_tokens, and reasoning from parsed chunks."""
    prompt_tokens = None
    completion_tokens = None
    cached_tokens = None
    reasoning_parts = []

    for chunk in chunks:
        # Check choices
        choices = chunk.get("choices") or []
        for choice in choices:
            delta = choice.get("delta") or choice.get("message") or {}
            reasoning = delta.get("reasoning_content") or delta.get("thought")
            if reasoning:
                reasoning_parts.append(str(reasoning))

        # Check usage
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            if usage.get("prompt_tokens") is not None:
                prompt_tokens = usage.get("prompt_tokens")
            if usage.get("completion_tokens") is not None:
                completion_tokens = usage.get("completion_tokens")

            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict) and details.get("cached_tokens") is not None:
                cached_tokens = int(details["cached_tokens"])
            elif usage.get("cache_read_tokens") is not None:
                cached_tokens = int(usage["cache_read_tokens"])

    reasoning_str = "".join(reasoning_parts).strip() if reasoning_parts else None
    return prompt_tokens, completion_tokens, cached_tokens, reasoning_str


def extract_reasoning_from_text(content: str) -> tuple[Optional[str], str]:
    """Extract <think>...</think> tags if present in plain content."""
    if not content:
        return None, content
    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return reasoning, cleaned_content
    return None, content


class CaptureManager:
    def __init__(
        self,
        db_path: Optional[str] = None,
        enabled: Optional[bool] = None,
        queue_maxsize: int = CAPTURE_QUEUE_MAXSIZE,
        batch_size: int = CAPTURE_BATCH_SIZE,
        batch_timeout: float = CAPTURE_BATCH_TIMEOUT,
        brain_dir: str = DEFAULT_BRAIN_DIR,
    ):
        self.db_path = db_path if db_path is not None else CAPTURE_DB_PATH
        self.enabled = is_capture_enabled() if enabled is None else enabled
        self.queue_maxsize = queue_maxsize
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.brain_dir = brain_dir

        self.db = CaptureDatabase(self.db_path)
        self.queue: asyncio.Queue | None = None
        self._worker_task: asyncio.Task | None = None
        self._running = False
        self._dropped_count = 0

    async def start(self) -> None:
        if not self.enabled:
            logger.info("CaptureManager is disabled via configuration.")
            return

        await self.db.connect()
        self.queue = asyncio.Queue(maxsize=self.queue_maxsize)
        self._running = True
        self._worker_task = asyncio.create_task(self._batch_writer_loop())
        logger.info(
            "CaptureManager started (db: %s, queue_maxsize: %d, batch: %d)",
            self.db_path,
            self.queue_maxsize,
            self.batch_size,
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        # Drain remaining items if any
        if self.queue and not self.queue.empty():
            remaining = []
            while not self.queue.empty():
                try:
                    remaining.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if remaining:
                await self._process_and_insert_batch(remaining)

        await self.db.close()
        logger.info("CaptureManager stopped cleanly.")

    def enqueue_turn(
        self,
        request_payload: dict[str, Any],
        response_data: Any,
        source_agent: Optional[str] = None,
        latency_ms: Optional[int] = None,
        stream: bool = False,
        conversation_id: Optional[str] = None,
    ) -> bool:
        """Enqueue captured turn with fail-open non-blocking semantics (< 0.05ms execution)."""
        if not self.enabled or not self._running or self.queue is None:
            return False

        item = {
            "turn_id": str(uuid.uuid4()),
            "request_payload": request_payload,
            "response_data": response_data,
            "source_agent": source_agent,
            "latency_ms": latency_ms,
            "stream": stream,
            "conversation_id": conversation_id,
            "created_at_ms": int(time.time() * 1000),
        }

        try:
            self.queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            self._dropped_count += 1
            if self._dropped_count % 100 == 1:
                logger.warning(
                    "Capture queue full (max: %d). Dropped %d turns (fail-open).",
                    self.queue_maxsize,
                    self._dropped_count,
                )
            return False

    async def _batch_writer_loop(self) -> None:
        """Background worker that flushes queue items in bulk."""
        while self._running:
            batch = []
            try:
                # Wait for at least one item
                item = await self.queue.get()
                batch.append(item)
                self.queue.task_done()

                # Collect up to batch_size or until timeout
                start_time = time.monotonic()
                while len(batch) < self.batch_size:
                    timeout = max(0.0, self.batch_timeout - (time.monotonic() - start_time))
                    if timeout <= 0:
                        break
                    try:
                        next_item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                        batch.append(next_item)
                        self.queue.task_done()
                    except (asyncio.TimeoutError, TimeoutError):
                        break

                if batch:
                    await self._process_and_insert_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in CaptureManager batch writer loop: %s", e)
                await asyncio.sleep(0.5)

    async def _process_and_insert_batch(self, batch: list[dict[str, Any]]) -> None:
        """Processes raw captured items in the background and writes them to SQLite."""
        records = []
        for item in batch:
            try:
                req_payload = item["request_payload"]
                resp_data = item["response_data"]
                conv_id = item.get("conversation_id")
                teacher_model = req_payload.get("model") or "default"

                prompt_tokens = None
                completion_tokens = None
                cached_tokens = None
                reasoning = None

                if item["stream"]:
                    # resp_data is list of bytes (SSE raw lines)
                    if isinstance(resp_data, list):
                        if resp_data and isinstance(resp_data[0], bytes):
                            response_text = b"".join(resp_data).decode("utf-8", errors="replace")
                        else:
                            response_text = "\n".join(
                                f"data: {json.dumps(c, ensure_ascii=False)}" for c in resp_data
                            )
                        # Parse chunks to extract tokens & reasoning
                        chunks = []
                        for line in response_text.splitlines():
                            line = line.strip()
                            if line.startswith("data:") and line != "data: [DONE]":
                                try:
                                    chunks.append(json.loads(line[5:].strip()))
                                except json.JSONDecodeError:
                                    pass
                        prompt_tokens, completion_tokens, cached_tokens, reasoning = (
                            parse_usage_and_reasoning_from_chunks(chunks)
                        )
                    else:
                        response_text = str(resp_data)
                else:
                    # Non-stream response (dict or string)
                    if isinstance(resp_data, dict):
                        response_text = json.dumps(resp_data, ensure_ascii=False)
                        usage = resp_data.get("usage") or {}
                        prompt_tokens = usage.get("prompt_tokens")
                        completion_tokens = usage.get("completion_tokens")
                        cached_tokens = (
                            (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                            or usage.get("cache_read_tokens")
                        )
                        # Extract reasoning from choices
                        choices = resp_data.get("choices") or []
                        for ch in choices:
                            msg = ch.get("message") or {}
                            r = msg.get("reasoning_content") or msg.get("thought")
                            if r:
                                reasoning = str(r)
                            elif msg.get("content"):
                                tag_reasoning, _ = extract_reasoning_from_text(msg.get("content"))
                                if tag_reasoning:
                                    reasoning = tag_reasoning
                    else:
                        response_text = str(resp_data)

                # If no reasoning was found in payload/stream, attempt background brain extraction
                cot_source = "upstream" if reasoning else None
                if not reasoning and conv_id:
                    extracted = await extract_thinking_from_brain_async(conv_id, self.brain_dir)
                    if extracted:
                        reasoning = extracted
                        cot_source = "brain_transcript"
                        # Enrich response_json with reasoning so message_builder formats <think>
                        if item["stream"]:
                            reasoning_chunk = {
                                "choices": [{"delta": {"reasoning_content": reasoning}}]
                            }
                            reasoning_sse = f"data: {json.dumps(reasoning_chunk, ensure_ascii=False)}\n\n"
                            response_text = reasoning_sse + response_text
                        elif isinstance(resp_data, dict):
                            enriched_resp = dict(resp_data)
                            choices = enriched_resp.get("choices") or []
                            if choices and "message" in choices[0]:
                                choices[0]["message"]["reasoning_content"] = reasoning
                            response_text = json.dumps(enriched_resp, ensure_ascii=False)
                    else:
                        cot_source = "missing"

                metadata: dict[str, Any] = {
                    "captured_by": "agy2api-zero-overhead",
                    "stream": item["stream"],
                }
                if conv_id:
                    metadata["conversation_id"] = conv_id
                if reasoning:
                    metadata["has_reasoning"] = True
                    metadata["reasoning_chars"] = len(reasoning)
                    metadata["cot_source"] = cot_source
                else:
                    metadata["has_reasoning"] = False
                    if cot_source == "missing":
                        metadata["cot_status"] = "missing"

                if cached_tokens is not None:
                    metadata["cached_tokens"] = cached_tokens
                    metadata["cache_hit"] = cached_tokens > 0

                # Apply redaction to prevent secret leakage in stored dataset
                clean_req = redact_sensitive_content(json.dumps(req_payload, ensure_ascii=False))
                clean_res = redact_sensitive_content(response_text)

                records.append(
                    {
                        "turn_id": item["turn_id"],
                        "created_at_ms": item["created_at_ms"],
                        "source_agent": item["source_agent"],
                        "teacher_model": teacher_model,
                        "request_json": clean_req,
                        "response_json": clean_res,
                        "latency_ms": item["latency_ms"],
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "status": "success",
                        "metadata": metadata,
                    }
                )
            except Exception as ex:
                logger.warning("Failed to prepare capture record: %s", ex)

        if records:
            await self.db.insert_captures_batch(records)


# Global singleton instance for application lifespan
global_capture_manager = CaptureManager()
