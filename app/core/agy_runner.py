import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.core.metrics import record_runner_execution
from app.core.process_pool import global_pool, WarmWorker

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
    extra_dirs: list[str] | None = None,
    json_schema: dict | str | None = None,
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
        "--add-dir",
        prompt_dir,
        "--output-format",
        output_format,
        "--dangerously-skip-permissions",
        "--print-timeout",
        "10m",
    ]
    if json_schema is not None:
        schema_path = str(Path(prompt_dir) / "schema.json")
        if isinstance(json_schema, str):
            Path(schema_path).write_text(json_schema, encoding="utf-8")
        else:
            Path(schema_path).write_text(json.dumps(json_schema), encoding="utf-8")
        cmd.extend(["--json-schema", schema_path])
    for extra in extra_dirs or []:
        cmd.extend(["--add-dir", extra])
    if model:
        cmd.extend(["--model", model])
    return AgyInvocation(cmd=cmd, prompt_dir=prompt_dir, prompt_path=prompt_path)


def _agy_env() -> dict[str, str]:
    env = os.environ.copy()
    env["AGY_IS_API_CALL"] = "1"
    return env


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


async def stream_agy_prompt_pooled(
    worker: WarmWorker,
    prompt: str,
    model: str = None,
    stop_after_first_schema_object: bool = False,
):
    """Execute a single turn on a pre-warmed agy process using stdin stream-json."""
    from app.core.tool_emulation import first_json_object

    t0 = time.time()
    proc = worker.process
    stderr_task = asyncio.create_task(_drain_stderr(proc))

    try:
        # Send user prompt as JSON event over stdin
        user_event = {"event": "user", "message": {"content": prompt}}
        payload = json.dumps(user_event, ensure_ascii=False) + "\n"
        proc.stdin.write(payload.encode("utf-8"))
        await proc.stdin.drain()

        accumulated = ""
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                continue

            yield event

            if stop_after_first_schema_object:
                step = event.get("step_update") or {}
                if event.get("event") == "step_update" and step.get("step_type") == "agent_response":
                    accumulated += step.get("text_delta") or ""
                    if first_json_object(accumulated):
                        break

            if event.get("event") == "result":
                break

        record_runner_execution(model or "default", "stream-json-warm", "success", time.time() - t0)
    except Exception:
        record_runner_execution(model or "default", "stream-json-warm", "error", time.time() - t0)
        raise
    finally:
        # Single-use warm worker: always terminate after turn to guarantee 100% context isolation
        await worker.close()
        if stderr_task and not stderr_task.done():
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass


async def run_agy_prompt(
    prompt: str,
    model: str = None,
    output_format: str = "json",
    files: list[str] = None,
    extra_dirs: list[str] | None = None,
    json_schema: dict | str | None = None,
):
    """
    Safely executes the `agy` CLI using warm pool if available, otherwise cold spawn.
    """
    # If standard text completion with no special extra_dirs or json_schema, try warm pool
    if not extra_dirs and not json_schema:
        worker = await global_pool.acquire(model)
        if worker:
            logger.info("Using warm worker for run_agy_prompt (pid: %d)", worker.process.pid)
            final_result = None
            async for event in stream_agy_prompt_pooled(worker, prompt, model):
                if event.get("event") == "result":
                    final_result = event.get("result")
            if final_result is not None:
                return final_result

    # Fallback to cold spawn
    inv = build_agy_invocation(
        prompt, model, output_format, extra_dirs=extra_dirs, json_schema=json_schema
    )
    _log_agy_cmd(inv, "command")
    t0 = time.time()
    try:
        process = await asyncio.create_subprocess_exec(
            *inv.cmd,
            cwd=inv.prompt_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_agy_env(),
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            if not error_msg:
                error_msg = stdout.decode().strip()
            logger.error(f"AGY Error: {error_msg}")
            record_runner_execution(model or "default", output_format, "error", time.time() - t0)
            raise RuntimeError(f"AGY CLI execution failed: {error_msg}")

        output_str = stdout.decode().strip()
        record_runner_execution(model or "default", output_format, "success", time.time() - t0)

        try:
            return _parse_json_envelope(output_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse AGY JSON output: {output_str}")
            return {"text": output_str}
    except Exception:
        record_runner_execution(model or "default", output_format, "error", time.time() - t0)
        raise
    finally:
        inv.cleanup()


async def stream_agy_prompt(
    prompt: str,
    model: str = None,
    files: list[str] = None,
    extra_dirs: list[str] | None = None,
    json_schema: dict | str | None = None,
    stop_after_first_schema_object: bool = False,
):
    """Yield parsed NDJSON events from warm pool or cold spawn."""
    if not extra_dirs and not json_schema:
        worker = await global_pool.acquire(model)
        if worker:
            logger.info("Using warm worker for stream_agy_prompt (pid: %d)", worker.process.pid)
            async for event in stream_agy_prompt_pooled(
                worker, prompt, model, stop_after_first_schema_object=stop_after_first_schema_object
            ):
                yield event
            return

    # Fallback to cold spawn
    inv = build_agy_invocation(
        prompt, model, "stream-json", extra_dirs=extra_dirs, json_schema=json_schema
    )
    _log_agy_cmd(inv, "stream")
    process = None
    stderr_task = None
    stopped_early = False
    t0 = time.time()
    try:
        process = await asyncio.create_subprocess_exec(
            *inv.cmd,
            cwd=inv.prompt_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_agy_env(),
        )
        stderr_task = asyncio.create_task(_drain_stderr(process))

        accumulated = ""
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("Skipping non-JSON AGY stream line: %s", text[:200])
                continue
            yield event
            if stop_after_first_schema_object:
                step = event.get("step_update") or {}
                if event.get("event") == "step_update" and step.get("step_type") == "agent_response":
                    accumulated += step.get("text_delta") or ""
                    from app.core.tool_emulation import first_json_object
                    if first_json_object(accumulated):
                        stopped_early = True
                        process.kill()
                        break

        await process.wait()
        stderr = await stderr_task
        if stopped_early:
            record_runner_execution(model or "default", "stream-json", "success", time.time() - t0)
            return
        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            logger.error(f"AGY Error: {error_msg}")
            record_runner_execution(model or "default", "stream-json", "error", time.time() - t0)
            raise RuntimeError(f"AGY CLI execution failed: {error_msg}")
        record_runner_execution(model or "default", "stream-json", "success", time.time() - t0)
    except Exception:
        record_runner_execution(model or "default", "stream-json", "error", time.time() - t0)
        raise
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
