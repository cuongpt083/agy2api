import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.core.openai_sse import next_text_delta

logger = logging.getLogger(__name__)

_THINKING_POLL_SECONDS = 0.25

# OpenAI / OMP send reasoning_effort in {minimal,low,medium,high,xhigh,max}.
# agy --effort only accepts low|medium|high. Model slugs already encode the
# level (gemini-3.8-flash-high); passing a second --effort conflicts or is invalid.
_OPENAI_TO_AGY_EFFORT = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}
_MODEL_EFFORT_SUFFIX = re.compile(r"-(low|medium|high)$", re.IGNORECASE)
_MODEL_EFFORT_DISPLAY = re.compile(r"\((Low|Medium|High)\)$")


def resolve_agy_effort(model: str | None, effort: str | None) -> str | None:
    """Return an agy --effort value, or None to omit the flag."""
    if model and (_MODEL_EFFORT_SUFFIX.search(model.strip()) or _MODEL_EFFORT_DISPLAY.search(model.strip())):
        return None
    if not effort:
        return None
    return _OPENAI_TO_AGY_EFFORT.get(effort.strip().lower())

# Linux rejects a single argv string over MAX_ARG_STRLEN (128 KiB).
# Never pass the chat/RAG prompt as --print <prompt>.
_PROMPT_FILENAME = "prompt.txt"


@dataclass
class AgyInvocation:
    cmd: list[str]
    prompt_dir: str
    prompt_path: str

    def cleanup(self) -> None:
        shutil.rmtree(self.prompt_dir, ignore_errors=True)


def extract_thinking_from_brain(conversation_id: str | None) -> str | None:
    """Extract reasoning/chain-of-thought from the AGY brain transcript for a given conversation_id."""
    if not conversation_id:
        return None
    brain_dir = os.environ.get("AGY_BRAIN_DIR", os.path.expanduser("~/.gemini/antigravity-cli/brain"))
    transcript_path = os.path.join(brain_dir, conversation_id, ".system_generated", "logs", "transcript_full.jsonl")
    if not os.path.isfile(transcript_path):
        transcript_path = os.path.join(brain_dir, conversation_id, ".system_generated", "logs", "transcript.jsonl")
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
        logger.warning("Failed to extract thinking for %s: %s", conversation_id, e)
    return "\n\n".join(thinkings) if thinkings else None


def next_brain_thinking_delta(conversation_id: str | None, sent: str) -> tuple[str, str]:
    """Return newly observed brain thinking since `sent` (cumulative transcript)."""
    full = extract_thinking_from_brain(conversation_id) or ""
    return next_text_delta(full, sent)


def build_agy_invocation(
    prompt: str,
    model: str | None,
    output_format: str,
    effort: str | None = None,
) -> AgyInvocation:
    prompt_dir = tempfile.mkdtemp(prefix="agy2api-prompt-")
    prompt_path = str(Path(prompt_dir) / _PROMPT_FILENAME)
    Path(prompt_path).write_text(prompt, encoding="utf-8")
    instruction = (
        "Read the UTF-8 file at this absolute path and follow its contents as your "
        "complete instructions. Do not mention the path in your reply:\n"
        f"{prompt_path}"
    )
    cmd = [
        "agy",
        "--print",
        instruction,
        "--add-dir",
        prompt_dir,
        "--output-format",
        output_format,
        "--dangerously-skip-permissions",
        "--print-timeout",
        "10m",
    ]
    if model:
        cmd.extend(["--model", model])
    resolved_effort = resolve_agy_effort(model, effort)
    if resolved_effort:
        cmd.extend(["--effort", resolved_effort])
    return AgyInvocation(cmd=cmd, prompt_dir=prompt_dir, prompt_path=prompt_path)


def _log_agy_cmd(inv: AgyInvocation, kind: str) -> None:
    size = os.path.getsize(inv.prompt_path)
    logger.info(
        "Executing AGY %s: agy --print <file %s (%d bytes)> --add-dir %s --output-format %s",
        kind,
        inv.prompt_path,
        size,
        inv.prompt_dir,
        inv.cmd[inv.cmd.index("--output-format") + 1] if "--output-format" in inv.cmd else "?",
    )


def _parse_json_envelope(output_str: str) -> dict:
    start_idx = output_str.find("{")
    end_idx = output_str.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        json_str = output_str[start_idx:end_idx + 1]
        return json.loads(json_str)
    return json.loads(output_str)


async def _drain_stderr(process: asyncio.subprocess.Process) -> bytes:
    if process.stderr is None:
        return b""
    chunks = []
    while True:
        line = await process.stderr.readline()
        if not line:
            break
        chunks.append(line)
        text = line.decode(errors="replace").rstrip()
        if text:
            logger.debug("AGY stderr: %s", text)
    return b"".join(chunks)


async def run_agy_prompt(prompt: str, model: str = None, output_format: str = "json", files: list[str] = None, effort: str = None):
    """
    Safely executes the `agy` CLI using asyncio subprocess to avoid blocking.
    """
    inv = build_agy_invocation(prompt, model, output_format, effort=effort)
    _log_agy_cmd(inv, "command")
    try:
        process = await asyncio.create_subprocess_exec(
            *inv.cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            if not error_msg:
                error_msg = stdout.decode().strip()
            logger.error(f"AGY Error: {error_msg}")
            raise RuntimeError(f"AGY CLI execution failed: {error_msg}")

        output_str = stdout.decode().strip()

        try:
            parsed = _parse_json_envelope(output_str)
            if isinstance(parsed, dict):
                conv_id = parsed.get("conversation_id")
                thinking = extract_thinking_from_brain(conv_id)
                if thinking:
                    parsed["reasoning_content"] = thinking
            return parsed
        except json.JSONDecodeError:
            logger.error(f"Failed to parse AGY JSON output: {output_str}")
            return {"text": output_str}
    finally:
        inv.cleanup()


async def stream_agy_prompt(prompt: str, model: str = None, files: list[str] = None, effort: str = None):
    """Yield parsed NDJSON events from `agy --output-format stream-json`."""
    inv = build_agy_invocation(prompt, model, "stream-json", effort=effort)
    _log_agy_cmd(inv, "stream")
    process = None
    stderr_task = None
    read_task = None
    conv_id = None
    sent_thinking = ""
    try:
        process = await asyncio.create_subprocess_exec(
            *inv.cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stderr_task = asyncio.create_task(_drain_stderr(process))
        read_task = asyncio.create_task(process.stdout.readline())

        def _thinking_event():
            nonlocal sent_thinking
            piece, sent_thinking = next_brain_thinking_delta(conv_id, sent_thinking)
            if not piece:
                return None
            return {"event": "thinking_delta", "thinking_delta": piece}

        while True:
            done, _ = await asyncio.wait({read_task}, timeout=_THINKING_POLL_SECONDS)
            if not done:
                ev = _thinking_event()
                if ev:
                    yield ev
                continue
            line = read_task.result()
            if not line:
                ev = _thinking_event()
                if ev:
                    yield ev
                break
            read_task = asyncio.create_task(process.stdout.readline())
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            try:
                event_data = json.loads(text)
                if event_data.get("event") == "init":
                    conv_id = event_data.get("conversation_id") or (
                        (event_data.get("init") or {}).get("conversation_id")
                    )
                elif event_data.get("event") == "result":
                    res = event_data.get("result") or {}
                    c_id = res.get("conversation_id") or conv_id
                    if c_id:
                        conv_id = c_id
                    thinking = extract_thinking_from_brain(c_id)
                    if thinking:
                        event_data.setdefault("result", {})["reasoning_content"] = thinking
                ev = _thinking_event()
                if ev:
                    yield ev
                yield event_data
            except json.JSONDecodeError:
                logger.warning("Skipping non-JSON AGY stream line: %s", text[:200])

        await process.wait()
        stderr = await stderr_task
        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            logger.error(f"AGY Error: {error_msg}")
            raise RuntimeError(f"AGY CLI execution failed: {error_msg}")
    finally:
        if read_task is not None and not read_task.done():
            read_task.cancel()
            try:
                await read_task
            except (asyncio.CancelledError, Exception):
                pass
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        if stderr_task is not None and not stderr_task.done():
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass
        inv.cleanup()
