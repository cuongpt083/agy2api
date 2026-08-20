import asyncio
import json
import logging

logger = logging.getLogger(__name__)


def _agy_cmd(prompt: str, model: str | None, output_format: str) -> list[str]:
    cmd = [
        "agy",
        "--print", prompt,
        "--output-format", output_format,
        "--dangerously-skip-permissions",
        "--print-timeout", "10m",
    ]
    if model:
        cmd.extend(["--model", model])
    return cmd


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
    cmd = _agy_cmd(prompt, model, output_format)
    safe_cmd_log = " ".join(cmd).replace("\n", " ")
    logger.info(f"Executing AGY command: {safe_cmd_log}")

    process = await asyncio.create_subprocess_exec(
        *cmd,
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


async def stream_agy_prompt(prompt: str, model: str = None, files: list[str] = None):
    """Yield parsed NDJSON events from `agy --output-format stream-json`."""
    cmd = _agy_cmd(prompt, model, "stream-json")
    safe_cmd_log = " ".join(cmd).replace("\n", " ")
    logger.info(f"Executing AGY stream command: {safe_cmd_log}")

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stderr_task = asyncio.create_task(_drain_stderr(process))

    try:
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
        if process.returncode is None:
            process.kill()
            await process.wait()
        if not stderr_task.done():
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass
