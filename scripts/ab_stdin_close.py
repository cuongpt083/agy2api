import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["message", "tool_call"]},
        "content": {"type": "string"},
    },
    "required": ["kind"],
}
PROMPT = 'Reply with exactly one JSON object: {"kind":"message","content":"pong"}. Nothing else.'


async def run_case(name: str, close_stdin: bool, timeout: float = 90.0) -> dict:
    work = tempfile.mkdtemp(prefix=f"agy-ab-{name}-")
    schema_path = str(Path(work) / "schema.json")
    Path(schema_path).write_text(json.dumps(SCHEMA), encoding="utf-8")
    cmd = [
        "agy",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--dangerously-skip-permissions",
        "--add-dir",
        work,
        "--json-schema",
        schema_path,
        "--print-timeout",
        "60s",
    ]
    env = os.environ.copy()
    env["AGY_IS_API_CALL"] = "1"
    t0 = time.time()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=work,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        limit=16 * 1024 * 1024,
    )
    events = []
    stderr_chunks = []
    err = None

    async def drain_stderr():
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            stderr_chunks.append(line.decode(errors="replace").rstrip())

    stderr_task = asyncio.create_task(drain_stderr())
    try:
        init_line = await asyncio.wait_for(proc.stdout.readline(), timeout=60)
        if not init_line:
            err = "no init line"
        else:
            events.append(json.loads(init_line.decode(errors="replace")))
            payload = (
                json.dumps({"event": "user", "message": {"content": PROMPT}}, ensure_ascii=False)
                + "\n"
            ).encode("utf-8")
            proc.stdin.write(payload)
            await proc.stdin.drain()
            if close_stdin:
                proc.stdin.close()
                try:
                    await proc.stdin.wait_closed()
                except Exception:
                    pass
            deadline = time.time() + timeout
            while time.time() < deadline:
                remaining = max(0.1, deadline - time.time())
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
                except asyncio.TimeoutError:
                    err = "timeout waiting for stdout"
                    break
                if not line:
                    break
                text = line.decode(errors="replace").strip()
                if not text:
                    continue
                try:
                    ev = json.loads(text)
                except json.JSONDecodeError:
                    events.append({"raw": text[:300]})
                    continue
                events.append(ev)
                if ev.get("event") == "result":
                    break
            if not close_stdin and proc.stdin and not proc.stdin.is_closing():
                proc.stdin.close()
                try:
                    await proc.stdin.wait_closed()
                except Exception:
                    pass
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except Exception:
                pass
        if not stderr_task.done():
            stderr_task.cancel()
            try:
                await stderr_task
            except Exception:
                pass

    result_ev = next((e for e in events if isinstance(e, dict) and e.get("event") == "result"), None)
    return {
        "name": name,
        "close_stdin": close_stdin,
        "elapsed_s": round(time.time() - t0, 2),
        "returncode": proc.returncode,
        "error": err,
        "event_types": [
            e.get("event") if isinstance(e, dict) and "event" in e else ("raw" if "raw" in e else "?")
            for e in events
        ],
        "result": (result_ev or {}).get("result"),
        "stderr_tail": stderr_chunks[-20:],
        "workdir": work,
    }


async def main():
    a = await run_case("close", close_stdin=True)
    b = await run_case("keep", close_stdin=False)
    print(json.dumps({"A_close_stdin": a, "B_keep_stdin_open": b}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
