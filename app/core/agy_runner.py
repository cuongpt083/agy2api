import asyncio
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

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


def build_agy_invocation(
    prompt: str,
    model: str | None,
    output_format: str,
) -> AgyInvocation:
    prompt_dir = tempfile.mkdtemp(prefix="agy2api-prompt-")
    prompt_path = str(Path(prompt_dir) / _PROMPT_FILENAME)
    Path(prompt_path).write_text(prompt, encoding="utf-8")
    instruction = (
        "Read the UTF-8 file prompt.txt in the current directory and follow its contents as your "
        "complete instructions. Do not mention the file name in your reply."
    )
    cmd = [
        "agy",
        "--print",
        instruction,
        "--output-format",
        output_format,
        "--dangerously-skip-permissions",
        "--print-timeout",
        "10m",
    ]
    if model:
        cmd.extend(["--model", model])
    return AgyInvocation(cmd=cmd, prompt_dir=prompt_dir, prompt_path=prompt_path)


def _log_agy_cmd(inv: AgyInvocation, kind: str) -> None:
    size = os.path.getsize(inv.prompt_path)
    logger.info(
        "Executing AGY %s in %s: agy --print <prompt.txt (%d bytes)> --output-format %s",
        kind,
        inv.prompt_dir,
        size,
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


async def run_agy_prompt(prompt: str, model: str = None, output_format: str = "json", files: list[str] = None):
    """
    Safely executes the `agy` CLI using asyncio subprocess to avoid blocking.
    """
    inv = build_agy_invocation(prompt, model, output_format)
    _log_agy_cmd(inv, "command")
    try:
        process = await asyncio.create_subprocess_exec(
            *inv.cmd,
            cwd=inv.prompt_dir,
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
            return _parse_json_envelope(output_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse AGY JSON output: {output_str}")
            return {"text": output_str}
    finally:
        inv.cleanup()


async def stream_agy_prompt(prompt: str, model: str = None, files: list[str] = None):
    """Yield parsed NDJSON events from `agy --output-format stream-json`."""
    inv = build_agy_invocation(prompt, model, "stream-json")
    _log_agy_cmd(inv, "stream")
    process = None
    stderr_task = None
    try:
        process = await asyncio.create_subprocess_exec(
            *inv.cmd,
            cwd=inv.prompt_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stderr_task = asyncio.create_task(_drain_stderr(process))

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError:
                logger.warning("Skipping non-JSON AGY stream line: %s", text[:200])

        await process.wait()
        stderr = await stderr_task
        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            logger.error(f"AGY Error: {error_msg}")
            raise RuntimeError(f"AGY CLI execution failed: {error_msg}")
    finally:
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
