"""Warm spare process pool for Google Antigravity (agy) CLI.

Eliminates the 7-9s cold-start overhead (process launch + eligibility check)
by maintaining pre-warmed agy processes waiting on stdin.

Architecture: Take-and-Replenish (Single-Use Warm Spare) + Dynamic Auto-Warm
- Each warm process handles exactly ONE user request and is immediately terminated,
  guaranteeing complete context isolation between requests while delivering sub-4s response times.
- Dynamic Auto-Warm: When users call a new model, the request falls back to cold-spawn safely,
  while an async background task automatically pre-warms a worker for that model for future calls.
- Least-Recently-Used (LRU) Eviction: Bounded memory footprint by evicting idle model pools
  when exceeding AGY_POOL_MAX_DYNAMIC_MODELS.
- Flavors: each model is warmed twice — `plain` (no schema) and `tools` (`--json-schema`
  TOOLS_JSON_SCHEMA). OpenClaw always sends tools, so without a tools flavor every request
  would miss the pool.
"""

from __future__ import annotations

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

logger = logging.getLogger(__name__)

_STDOUT_LIMIT = 16 * 1024 * 1024
# Discard warm workers older than this (stale OAuth/eligibility risk).
_MAX_WORKER_AGE_S = float(os.environ.get("AGY_POOL_MAX_WORKER_AGE_SECONDS", "900"))
FLAVOR_PLAIN = "plain"
FLAVOR_TOOLS = "tools"


def _agy_env() -> dict[str, str]:
    env = os.environ.copy()
    env["AGY_IS_API_CALL"] = "1"
    return env


def pool_key(model: Optional[str], flavor: str) -> str:
    return f"{model or 'default'}||{flavor}"


def flavor_for_schema(json_schema) -> str:
    return FLAVOR_TOOLS if json_schema is not None else FLAVOR_PLAIN


@dataclass
class WarmWorker:
    process: asyncio.subprocess.Process
    prompt_dir: str
    model: Optional[str]
    created_at: float
    init_event: dict
    flavor: str = FLAVOR_PLAIN
    # Started at spawn so idle eligibility/diagnostics cannot fill the PIPE and stall agy.
    stderr_task: Optional[asyncio.Task] = None
    stderr_lines: Optional[list] = None

    def stderr_tail(self, max_chars: int = 500) -> str:
        lines = self.stderr_lines or []
        return "\n".join(lines)[-max_chars:]

    async def close(self) -> None:
        if self.stderr_task is not None and not self.stderr_task.done():
            self.stderr_task.cancel()
            try:
                await self.stderr_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            if self.process.returncode is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
        except Exception as e:
            logger.warning("Error terminating warm worker: %s", e)
        finally:
            shutil.rmtree(self.prompt_dir, ignore_errors=True)



async def _drain_worker_stderr(process: asyncio.subprocess.Process, lines: list, max_lines: int = 200) -> None:
    """Continuously read stderr so the OS pipe never fills while the worker is idle."""
    if process.stderr is None:
        return
    try:
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            text_line = line.decode(errors="replace").rstrip()
            if not text_line:
                continue
            lines.append(text_line)
            if len(lines) > max_lines:
                del lines[: len(lines) - max_lines]
            logger.debug("AGY warm stderr: %s", text_line)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug("AGY warm stderr drain ended: %s", e)


class AgyProcessPool:
    """Manages a pool of pre-warmed agy worker processes keyed by model||flavor."""

    def __init__(self, target_size: int = 1, default_model: Optional[str] = None):
        self.target_size = int(os.environ.get("AGY_POOL_SIZE", str(target_size)))
        # Counted in model||flavor keys (plain + tools). Default 6 ≈ 3 models.
        self.max_dynamic_models = int(os.environ.get("AGY_POOL_MAX_DYNAMIC_MODELS", "6"))

        env_models = os.environ.get("AGY_POOL_MODELS")
        if env_models:
            self.warm_models: list[str] = [m.strip() for m in env_models.split(",") if m.strip()]
        else:
            def_m = default_model or os.environ.get("AGY_DEFAULT_MODEL", "gemini-3.8-flash-high")
            self.warm_models = [def_m]

        self._pools: dict[str, asyncio.Queue[WarmWorker]] = {}
        self._last_used: dict[str, float] = {}
        self._running = False
        self._active_workers: list[WarmWorker] = []
        self._auto_warm_in_progress: set[str] = set()

    def _primary_keys(self) -> set[str]:
        if not self.warm_models:
            return set()
        m = self.warm_models[0]
        return {pool_key(m, FLAVOR_PLAIN), pool_key(m, FLAVOR_TOOLS)}

    async def start(self) -> None:
        """Start pre-warming workers in the background."""
        if self._running or self.target_size <= 0:
            return
        self._running = True
        logger.info(
            "Starting AgyProcessPool (target size: %d per model/flavor, initial models: %s, max dynamic keys: %d)",
            self.target_size,
            self.warm_models,
            self.max_dynamic_models,
        )
        for m in self.warm_models:
            for flavor in (FLAVOR_PLAIN, FLAVOR_TOOLS):
                key = pool_key(m, flavor)
                self._pools[key] = asyncio.Queue()
                self._last_used[key] = time.time()
                for _ in range(self.target_size):
                    asyncio.create_task(self._spawn_worker_safe(m, flavor))

    async def _spawn_one_worker(
        self, model: Optional[str] = None, flavor: str = FLAVOR_PLAIN
    ) -> Optional[WarmWorker]:
        prompt_dir = tempfile.mkdtemp(prefix="agy-pool-worker-")
        cmd = [
            "agy",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
            "--add-dir", prompt_dir,
        ]
        if flavor == FLAVOR_TOOLS:
            from app.core.tool_emulation import TOOLS_JSON_SCHEMA
            schema_path = str(Path(prompt_dir) / "schema.json")
            Path(schema_path).write_text(json.dumps(TOOLS_JSON_SCHEMA), encoding="utf-8")
            cmd.extend(["--json-schema", schema_path])
        if model and model != "default":
            cmd.extend(["--model", model])

        t0 = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=prompt_dir,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_agy_env(),
                limit=_STDOUT_LIMIT,
            )

            init_line = await proc.stdout.readline()
            if not init_line:
                stderr = await proc.stderr.read()
                logger.error(
                    "Warm worker failed to emit init event. Stderr: %s",
                    stderr.decode(errors="replace")[:300],
                )
                proc.kill()
                shutil.rmtree(prompt_dir, ignore_errors=True)
                return None

            init_event = json.loads(init_line.decode(errors="replace"))
            stderr_lines: list[str] = []
            stderr_task = asyncio.create_task(_drain_worker_stderr(proc, stderr_lines))
            worker = WarmWorker(
                process=proc,
                prompt_dir=prompt_dir,
                model=model,
                created_at=time.time(),
                init_event=init_event,
                flavor=flavor,
                stderr_task=stderr_task,
                stderr_lines=stderr_lines,
            )
            logger.info(
                "Warm worker ready in %.2fs (pid: %d, model: %s, flavor: %s)",
                time.time() - t0,
                proc.pid,
                model,
                flavor,
            )
            return worker
        except Exception as e:
            logger.error("Failed to spawn warm agy worker: %s", e)
            shutil.rmtree(prompt_dir, ignore_errors=True)
            return None

    async def _spawn_worker_safe(self, model: str, flavor: str = FLAVOR_PLAIN) -> None:
        if not self._running:
            return
        key = pool_key(model, flavor)
        worker = await self._spawn_one_worker(model, flavor)
        if worker:
            self._active_workers.append(worker)
            if key not in self._pools:
                self._pools[key] = asyncio.Queue()
            await self._pools[key].put(worker)
        else:
            await asyncio.sleep(5.0)
            if self._running and key in self._pools and self._pools[key].qsize() < self.target_size:
                asyncio.create_task(self._spawn_worker_safe(model, flavor))

    async def _evict_lru_model_if_needed(self, incoming_key: str) -> None:
        """Evicts the least-recently-used pool key if the dynamic limit is reached."""
        if incoming_key in self._pools or len(self._pools) < self.max_dynamic_models:
            return

        protected = self._primary_keys()
        candidates = [k for k in self._pools if k not in protected] or list(self._pools.keys())
        if not candidates:
            return

        lru_key = min(candidates, key=lambda k: self._last_used.get(k, 0.0))
        logger.info("Evicting LRU warm pool: %s to accommodate %s", lru_key, incoming_key)

        q = self._pools.pop(lru_key, None)
        self._last_used.pop(lru_key, None)
        if q:
            while not q.empty():
                try:
                    w = q.get_nowait()
                    if w in self._active_workers:
                        self._active_workers.remove(w)
                    await w.close()
                except Exception:
                    pass

    async def auto_warm_model(self, model: str, flavor: str = FLAVOR_PLAIN) -> None:
        """Dynamically registers and warms a new model/flavor when requested."""
        if not self._running or self.target_size <= 0 or not model:
            return
        key = pool_key(model, flavor)
        if key in self._pools or key in self._auto_warm_in_progress:
            return

        self._auto_warm_in_progress.add(key)
        try:
            await self._evict_lru_model_if_needed(key)
            self._pools[key] = asyncio.Queue()
            self._last_used[key] = time.time()
            logger.info("Auto-warming pool for %s", key)
            for _ in range(self.target_size):
                asyncio.create_task(self._spawn_worker_safe(model, flavor))
        finally:
            self._auto_warm_in_progress.discard(key)

    async def acquire(
        self, model: Optional[str] = None, json_schema=None
    ) -> Optional[WarmWorker]:
        """Get a warm worker matching model + schema flavor, else trigger auto-warm and return None."""
        if not self._running or self.target_size <= 0:
            logger.info("Pool MISS reason=pool_disabled running=%s target_size=%s", self._running, self.target_size)
            return None

        flavor = flavor_for_schema(json_schema)
        target_model = model or (self.warm_models[0] if self.warm_models else "gemini-3.8-flash-high")
        target_key = pool_key(target_model, flavor)

        if target_key not in self._pools:
            logger.info(
                "Pool MISS reason=no_pool key=%s requested_model=%r known_keys=%s",
                target_key, model, sorted(self._pools.keys()),
            )
            asyncio.create_task(self.auto_warm_model(target_model, flavor))
            return None

        q = self._pools[target_key]
        if q.empty():
            logger.info(
                "Pool MISS reason=empty_queue key=%s (worker still spawning or consumed by concurrent request)",
                target_key,
            )
            return None

        try:
            worker = q.get_nowait()
            self._last_used[target_key] = time.time()
            if worker in self._active_workers:
                self._active_workers.remove(worker)

            age = time.time() - worker.created_at
            if worker.process.returncode is not None or age > _MAX_WORKER_AGE_S:
                logger.warning(
                    "Pooled worker stale or died (%s age=%.1fs), discarding",
                    target_key,
                    age,
                )
                await worker.close()
                asyncio.create_task(self._spawn_worker_safe(target_model, flavor))
                return None

            logger.info(
                "Acquired warm worker %s pid=%s age=%.1fs",
                target_key,
                worker.process.pid,
                age,
            )
            asyncio.create_task(self._spawn_worker_safe(target_model, flavor))
            return worker
        except asyncio.QueueEmpty:
            logger.info("Pool MISS reason=queue_race key=%s", target_key)
            return None

    async def shutdown(self) -> None:
        """Terminate all pooled processes and clean up."""
        self._running = False
        logger.info("Shutting down AgyProcessPool...")
        for q in self._pools.values():
            while not q.empty():
                try:
                    worker = q.get_nowait()
                    await worker.close()
                except Exception:
                    pass
        for worker in list(self._active_workers):
            await worker.close()
        self._active_workers.clear()
        self._pools.clear()
        self._last_used.clear()


global_pool = AgyProcessPool()
