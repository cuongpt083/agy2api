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
from app.core.sandbox import wrap_cmd

logger = logging.getLogger(__name__)

# Never put the chat/RAG prompt on argv (Linux MAX_ARG_STRLEN is 128 KiB).
# Feed it as one NDJSON user event on stdin instead of a prompt.txt + view_file turn.
_STDOUT_LIMIT = 16 * 1024 * 1024
_PRINT_TIMEOUT = "10m"
# Per-turn wall clock (readline-level). Warm spawn intentionally has NO --print-timeout
# so idle pooled workers are not killed by the CLI clock.
_TURN_TIMEOUT_S = float(os.environ.get("AGY_TURN_TIMEOUT_SECONDS", "600"))


# Tools-emulation turns must only emit a JSON object. If agy's own agent starts running built-in
# tools (run_command/view_file/...) it loops for 60-100s+ and the client times out; abort early.
_MAX_BUILTIN_TOOL_STEPS = int(os.environ.get("AGY_MAX_BUILTIN_TOOL_STEPS", "3"))


class AgyBuiltinToolLoopError(RuntimeError):
    """Raised when agy runs too many built-in tool steps in a schema-only (tools emulation) turn."""


class AgyTimeoutError(RuntimeError):
    """Raised when agy emits no stdout until the turn deadline."""



@dataclass
class AgyInvocation:
    cmd: list[str]
    prompt_dir: str
    prompt: str
    prompt_path: str | None = None  # kept for tests; unused (prompt is stdin)

    def cleanup(self) -> None:
        shutil.rmtree(self.prompt_dir, ignore_errors=True)


def build_agy_invocation(
    prompt: str,
    model: str | None,
    output_format: str,
    extra_dirs: list[str] | None = None,
    json_schema: dict | str | None = None,
) -> AgyInvocation:
    """Build a stream-json stdin invocation. `output_format` is ignored; stdout is always NDJSON."""
    prompt_dir = tempfile.mkdtemp(prefix="agy2api-prompt-")
    cmd = [
        "agy",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--dangerously-skip-permissions",
        "--add-dir",
        prompt_dir,
        "--print-timeout",
        _PRINT_TIMEOUT,
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
    return AgyInvocation(cmd=cmd, prompt_dir=prompt_dir, prompt=prompt, prompt_path=None)


# Dump raw agy NDJSON events (truncated to 4KB each) when request tracing is on.
_TRACE_EVENTS = os.environ.get("AGY_TRACE_REQUESTS", "").lower() in ("1", "true", "yes")


def _agy_env() -> dict[str, str]:
    env = os.environ.copy()
    env["AGY_IS_API_CALL"] = "1"
    return env


def _log_agy_cmd(inv: AgyInvocation, kind: str) -> None:
    logger.info(
        "Executing AGY %s in %s: agy --input-format stream-json <stdin %d chars> --output-format stream-json",
        kind,
        inv.prompt_dir,
        len(inv.prompt),
    )


def encode_user_event(prompt: str) -> bytes:
    """Official headless stdin shape: one NDJSON user event per turn."""
    return (json.dumps({"event": "user", "message": {"content": prompt}}, ensure_ascii=False) + "\n").encode("utf-8")


def _raise_if_agy_result_error(result: dict | None) -> None:
    """Turn an ERROR result event into a RuntimeError (auth, quota, etc.)."""
    if not isinstance(result, dict):
        return
    status = (result.get("status") or "").upper()
    if status in ("", "SUCCESS"):
        return
    err = result.get("error") or result.get("response") or status
    raise RuntimeError(f"AGY CLI execution failed: {err}")


async def _wait_for_init(
    process: asyncio.subprocess.Process,
    timeout: float = 60.0,
) -> dict:
    """Read the first NDJSON init event (warm pool already did this at spawn)."""
    try:
        line = await asyncio.wait_for(process.stdout.readline(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"agy did not emit init within {timeout:.0f}s (pid={process.pid})"
        ) from exc
    if not line:
        raise RuntimeError(
            f"agy exited before init (pid={process.pid}); "
            "often authentication required — run interactive `agy` login on the host"
        )
    text_line = line.decode(errors="replace").strip()
    try:
        event = json.loads(text_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"agy init line is not JSON: {text_line[:200]}") from exc
    if event.get("event") != "init":
        logger.warning("Expected init event, got %s", event.get("event"))
    return event


async def _write_user_event_and_close(process: asyncio.subprocess.Process, prompt: str) -> None:
    if process.stdin is None:
        raise RuntimeError("agy process has no stdin pipe")
    try:
        process.stdin.write(encode_user_event(prompt))
        await process.stdin.drain()
        process.stdin.close()
        try:
            await process.stdin.wait_closed()
        except (AttributeError, ConnectionResetError, BrokenPipeError):
            pass
    except (BrokenPipeError, ConnectionResetError) as exc:
        # agy exited before reading stdin (auth failure, crash). Caller should attach stderr.
        raise RuntimeError(
            f"AGY CLI closed stdin before accepting the prompt (pid={process.pid}); "
            "often authentication required — run interactive `agy` login on the host"
        ) from exc


async def _drain_stderr(process: asyncio.subprocess.Process, collect: bool = True) -> bytes:
    if process.stderr is None:
        return b""
    chunks = []
    while True:
        line = await process.stderr.readline()
        if not line:
            break
        if collect:
            chunks.append(line)
        text_line = line.decode(errors="replace").rstrip()
        if text_line:
            logger.debug("AGY stderr: %s", text_line)
    return b"".join(chunks) if collect else b""


async def _iter_stdout_events(
    process: asyncio.subprocess.Process,
    stop_after_first_schema_object: bool = False,
    deadline: float | None = None,
    seen_events: list | None = None,
):
    from app.core.tool_emulation import first_json_object

    accumulated = ""
    builtin_tool_steps = 0
    loop = asyncio.get_running_loop()
    while True:
        if deadline is not None:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise AgyTimeoutError(
                    f"agy turn timed out after {_TURN_TIMEOUT_S:.0f}s; "
                    f"events_seen={seen_events or []} pid={process.pid}"
                )
            try:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise AgyTimeoutError(
                    f"agy turn timed out after {_TURN_TIMEOUT_S:.0f}s; "
                    f"events_seen={seen_events or []} pid={process.pid}"
                ) from exc
        else:
            line = await process.stdout.readline()
        if not line:
            break
        text_line = line.decode(errors="replace").strip()
        if not text_line:
            continue
        try:
            event = json.loads(text_line)
        except json.JSONDecodeError:
            logger.warning("Skipping non-JSON AGY stream line: %s", text_line[:200])
            continue
        if seen_events is not None:
            seen_events.append(event.get("event") or "?")
        logger.info(
            "AGY event=%s step_type=%s pid=%s",
            event.get("event"),
            (event.get("step_update") or {}).get("step_type"),
            process.pid,
        )
        if _TRACE_EVENTS:
            logger.info("AGY raw pid=%s %s", process.pid, text_line[:4000])
        yield event
        if stop_after_first_schema_object:
            step = event.get("step_update") or {}
            if step.get("step_type") == "tool" and step.get("state") == "ACTIVE":
                builtin_tool_steps += 1
                if _MAX_BUILTIN_TOOL_STEPS > 0 and builtin_tool_steps > _MAX_BUILTIN_TOOL_STEPS:
                    raise AgyBuiltinToolLoopError(
                        f"agy ran {builtin_tool_steps} built-in tool steps (last={step.get('tool_name')}) "
                        f"in a schema-only turn; aborted pid={process.pid}"
                    )
            if event.get("event") == "step_update" and step.get("step_type") == "agent_response":
                accumulated += step.get("text_delta") or ""
                if first_json_object(accumulated):
                    return
        if event.get("event") == "result":
            return


async def stream_agy_prompt_pooled(
    worker: WarmWorker,
    prompt: str,
    model: str = None,
    stop_after_first_schema_object: bool = False,
):
    """Execute a single turn on a pre-warmed agy process using stdin stream-json."""
    t0 = time.time()
    proc = worker.process
    # stderr is already drained continuously by the pool from spawn time.
    seen_events: list[str] = []
    deadline = asyncio.get_running_loop().time() + _TURN_TIMEOUT_S
    logger.info(
        "Warm turn start pid=%s model=%s age=%.1fs timeout=%.0fs",
        proc.pid,
        model,
        time.time() - worker.created_at,
        _TURN_TIMEOUT_S,
    )

    try:
        await _write_user_event_and_close(proc, prompt)
        last_result = None
        async for event in _iter_stdout_events(
            proc,
            stop_after_first_schema_object=stop_after_first_schema_object,
            deadline=deadline,
            seen_events=seen_events,
        ):
            if event.get("event") == "result":
                last_result = event.get("result")
            yield event
        _raise_if_agy_result_error(last_result)
        record_runner_execution(model or "default", "stream-json-warm", "success", time.time() - t0)
        logger.info(
            "Warm turn success pid=%s events=%s elapsed=%.2fs",
            proc.pid,
            seen_events,
            time.time() - t0,
        )
    except (asyncio.CancelledError, GeneratorExit):
        logger.info(
            "Warm turn aborted by client/consumer pid=%s events=%s elapsed=%.2fs",
            proc.pid, seen_events, time.time() - t0,
        )
        raise
    except Exception as exc:
        stderr_tail = worker.stderr_tail()
        logger.error(
            "Warm turn failed pid=%s events=%s err=%s stderr_tail=%s",
            proc.pid,
            seen_events,
            exc,
            stderr_tail,
        )
        record_runner_execution(model or "default", "stream-json-warm", "error", time.time() - t0)
        if isinstance(exc, AgyTimeoutError) and stderr_tail:
            raise AgyTimeoutError(f"{exc}; stderr_tail={stderr_tail}") from exc
        raise
    finally:
        # Single-use warm worker: always terminate after turn to guarantee 100% context isolation
        await worker.close()


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
    Always consumes stream-json and returns the terminal `result` envelope.
    """
    if not extra_dirs:
        worker = await global_pool.acquire(model, json_schema=json_schema)
        if worker:
            logger.info("Using warm worker for run_agy_prompt (pid: %d)", worker.process.pid)
            final_result = None
            async for event in stream_agy_prompt_pooled(worker, prompt, model):
                if event.get("event") == "result":
                    final_result = event.get("result")
            if final_result is not None:
                return final_result

    inv = build_agy_invocation(
        prompt, model, output_format, extra_dirs=extra_dirs, json_schema=json_schema
    )
    _log_agy_cmd(inv, "command")
    t0 = time.time()
    process = None
    stderr_task = None
    try:
        process = await asyncio.create_subprocess_exec(
            *wrap_cmd(inv.cmd, inv.prompt_dir, extra_dirs),
            cwd=inv.prompt_dir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_agy_env(),
            limit=_STDOUT_LIMIT,
        )
        try:
            init_event = await _wait_for_init(process)
        except RuntimeError:
            err = b""
            if process.stderr is not None:
                try:
                    err = await asyncio.wait_for(process.stderr.read(), timeout=1.0)
                except Exception:
                    pass
            if process.returncode is None:
                process.kill()
                await process.wait()
            detail = err.decode(errors="replace").strip()
            if detail:
                raise RuntimeError(f"AGY CLI execution failed: {detail}") from None
            raise
        logger.info(
            "Cold run init ok pid=%s conversation_id=%s",
            process.pid,
            init_event.get("conversation_id"),
        )
        stderr_task = asyncio.create_task(_drain_stderr(process))
        await _write_user_event_and_close(process, prompt)

        final_result = None
        seen_events: list[str] = []
        deadline = asyncio.get_running_loop().time() + _TURN_TIMEOUT_S
        async for event in _iter_stdout_events(process, deadline=deadline, seen_events=seen_events):
            if event.get("event") == "result":
                final_result = event.get("result")

        await process.wait()
        stderr = await stderr_task
        if process.returncode not in (0, None) and final_result is None:
            error_msg = stderr.decode().strip()
            logger.error(f"AGY Error: {error_msg}")
            record_runner_execution(model or "default", "stream-json", "error", time.time() - t0)
            raise RuntimeError(f"AGY CLI execution failed: {error_msg}")
        record_runner_execution(model or "default", output_format, "success", time.time() - t0)
        _raise_if_agy_result_error(final_result)
        if final_result is not None:
            return final_result
        return {"text": ""}
    except Exception:
        record_runner_execution(model or "default", output_format, "error", time.time() - t0)
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


async def stream_agy_prompt(
    prompt: str,
    model: str = None,
    files: list[str] = None,
    extra_dirs: list[str] | None = None,
    json_schema: dict | str | None = None,
    stop_after_first_schema_object: bool = False,
):
    """Yield parsed NDJSON events from warm pool or cold spawn."""
    if not extra_dirs:
        worker = await global_pool.acquire(model, json_schema=json_schema)
        if worker:
            logger.info("Using warm worker for stream_agy_prompt (pid: %d)", worker.process.pid)
            async for event in stream_agy_prompt_pooled(
                worker, prompt, model, stop_after_first_schema_object=stop_after_first_schema_object
            ):
                yield event
            return

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
            *wrap_cmd(inv.cmd, inv.prompt_dir, extra_dirs),
            cwd=inv.prompt_dir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_agy_env(),
            limit=_STDOUT_LIMIT,
        )
        try:
            init_event = await _wait_for_init(process)
        except RuntimeError:
            err = b""
            if process.stderr is not None:
                try:
                    err = await asyncio.wait_for(process.stderr.read(), timeout=1.0)
                except Exception:
                    pass
            if process.returncode is None:
                process.kill()
                await process.wait()
            detail = err.decode(errors="replace").strip()
            if detail:
                raise RuntimeError(f"AGY CLI execution failed: {detail}") from None
            raise
        logger.info(
            "Cold stream init ok pid=%s conversation_id=%s",
            process.pid,
            init_event.get("conversation_id"),
        )
        yield init_event
        stderr_task = asyncio.create_task(_drain_stderr(process))
        await _write_user_event_and_close(process, prompt)

        saw_result = False
        last_result = None
        seen_events: list[str] = []
        deadline = asyncio.get_running_loop().time() + _TURN_TIMEOUT_S
        async for event in _iter_stdout_events(
            process,
            stop_after_first_schema_object=stop_after_first_schema_object,
            deadline=deadline,
            seen_events=seen_events,
        ):
            if event.get("event") == "result":
                saw_result = True
                last_result = event.get("result")
            yield event
        if stop_after_first_schema_object and not saw_result:
            stopped_early = True
            logger.info(
                "AGY cold turn early-stop (schema object complete) pid=%s events=%d elapsed=%.2fs",
                process.pid, len(seen_events), time.time() - t0,
            )
            if process.returncode is None:
                process.kill()

        await process.wait()
        stderr = await stderr_task
        if stopped_early:
            record_runner_execution(model or "default", "stream-json", "success", time.time() - t0)
            return
        if process.returncode not in (0, None):
            error_msg = stderr.decode().strip()
            logger.error(f"AGY Error: {error_msg}")
            record_runner_execution(model or "default", "stream-json", "error", time.time() - t0)
            raise RuntimeError(f"AGY CLI execution failed: {error_msg}")
        try:
            _raise_if_agy_result_error(last_result)
        except RuntimeError:
            record_runner_execution(model or "default", "stream-json", "error", time.time() - t0)
            raise
        record_runner_execution(model or "default", "stream-json", "success", time.time() - t0)
        logger.info("AGY cold turn done pid=%s events=%d elapsed=%.2fs", process.pid, len(seen_events), time.time() - t0)
    except (asyncio.CancelledError, GeneratorExit):
        logger.info("AGY cold turn aborted by client/consumer elapsed=%.2fs", time.time() - t0)
        raise
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
