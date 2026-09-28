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
from typing import Optional

logger = logging.getLogger(__name__)


def _agy_env() -> dict[str, str]:
    env = os.environ.copy()
    env["AGY_IS_API_CALL"] = "1"
    return env


@dataclass
class WarmWorker:
    process: asyncio.subprocess.Process
    prompt_dir: str
    model: Optional[str]
    created_at: float
    init_event: dict

    async def close(self) -> None:
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


class AgyProcessPool:
    """Manages a pool of pre-warmed agy worker processes keyed by model."""

    def __init__(self, target_size: int = 1, default_model: Optional[str] = None):
        self.target_size = int(os.environ.get("AGY_POOL_SIZE", str(target_size)))
        self.max_dynamic_models = int(os.environ.get("AGY_POOL_MAX_DYNAMIC_MODELS", "3"))

        # Read configured models to pre-warm on startup
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

    async def start(self) -> None:
        """Start pre-warming workers in the background."""
        if self._running or self.target_size <= 0:
            return
        self._running = True
        logger.info(
            "Starting AgyProcessPool (target size: %d per model, initial models: %s, max dynamic: %d)",
            self.target_size,
            self.warm_models,
            self.max_dynamic_models,
        )
        for m in self.warm_models:
            self._pools[m] = asyncio.Queue()
            self._last_used[m] = time.time()
            for _ in range(self.target_size):
                asyncio.create_task(self._spawn_worker_safe(m))

    async def _spawn_one_worker(self, model: Optional[str] = None) -> Optional[WarmWorker]:
        prompt_dir = tempfile.mkdtemp(prefix="agy-pool-worker-")
        cmd = [
            "agy",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
            "--add-dir", prompt_dir,
        ]
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
            )

            # Read initial init event to verify readiness
            init_line = await proc.stdout.readline()
            if not init_line:
                stderr = await proc.stderr.read()
                logger.error("Warm worker failed to emit init event. Stderr: %s", stderr.decode(errors="replace")[:300])
                proc.kill()
                shutil.rmtree(prompt_dir, ignore_errors=True)
                return None

            init_event = json.loads(init_line.decode(errors="replace"))
            worker = WarmWorker(
                process=proc,
                prompt_dir=prompt_dir,
                model=model,
                created_at=time.time(),
                init_event=init_event,
            )
            logger.info("Warm worker ready in %.2fs (pid: %d, model: %s)", time.time() - t0, proc.pid, model)
            return worker
        except Exception as e:
            logger.error("Failed to spawn warm agy worker: %s", e)
            shutil.rmtree(prompt_dir, ignore_errors=True)
            return None

    async def _spawn_worker_safe(self, model: str) -> None:
        if not self._running:
            return
        worker = await self._spawn_one_worker(model)
        if worker:
            self._active_workers.append(worker)
            if model not in self._pools:
                self._pools[model] = asyncio.Queue()
            await self._pools[model].put(worker)
        else:
            await asyncio.sleep(5.0)
            if self._running and model in self._pools and self._pools[model].qsize() < self.target_size:
                asyncio.create_task(self._spawn_worker_safe(model))

    async def _evict_lru_model_if_needed(self, incoming_model: str) -> None:
        """Evicts the least-recently-used model queue if dynamic pool limit is reached."""
        if incoming_model in self._pools or len(self._pools) < self.max_dynamic_models:
            return

        # Find least recently used model (excluding default primary model if possible)
        primary_model = self.warm_models[0] if self.warm_models else None
        candidates = [m for m in self._pools if m != primary_model] or list(self._pools.keys())
        if not candidates:
            return

        lru_model = min(candidates, key=lambda m: self._last_used.get(m, 0.0))
        logger.info("Evicting LRU warm model pool: %s to accommodate %s", lru_model, incoming_model)

        q = self._pools.pop(lru_model, None)
        self._last_used.pop(lru_model, None)
        if q:
            while not q.empty():
                try:
                    w = q.get_nowait()
                    if w in self._active_workers:
                        self._active_workers.remove(w)
                    await w.close()
                except Exception:
                    pass

    async def auto_warm_model(self, model: str) -> None:
        """Dynamically registers and warms a new model when requested by users."""
        if not self._running or self.target_size <= 0 or not model:
            return
        if model in self._pools or model in self._auto_warm_in_progress:
            return

        self._auto_warm_in_progress.add(model)
        try:
            await self._evict_lru_model_if_needed(model)
            self._pools[model] = asyncio.Queue()
            self._last_used[model] = time.time()
            logger.info("Auto-warming pool for newly requested model: %s", model)
            for _ in range(self.target_size):
                asyncio.create_task(self._spawn_worker_safe(model))
        finally:
            self._auto_warm_in_progress.discard(model)

    async def acquire(self, model: Optional[str] = None) -> Optional[WarmWorker]:
        """Get a warm worker matching the exact model requested, else trigger auto-warm and return None."""
        if not self._running or self.target_size <= 0:
            return None

        # Resolve model key
        target_key = model or (self.warm_models[0] if self.warm_models else "gemini-3.8-flash-high")

        # If model is not yet in our warm pools, trigger dynamic auto-warm for subsequent requests
        if target_key not in self._pools:
            asyncio.create_task(self.auto_warm_model(target_key))
            return None

        q = self._pools[target_key]
        if q.empty():
            return None

        try:
            worker = q.get_nowait()
            self._last_used[target_key] = time.time()
            if worker in self._active_workers:
                self._active_workers.remove(worker)

            # Check if process is still alive and not stale (> 30 mins)
            if worker.process.returncode is not None or (time.time() - worker.created_at > 1800):
                logger.warning("Pooled worker stale or died (model: %s), discarding", target_key)
                await worker.close()
                asyncio.create_task(self._spawn_worker_safe(target_key))
                return None

            # Asynchronously spawn replacement immediately
            asyncio.create_task(self._spawn_worker_safe(target_key))
            return worker
        except asyncio.QueueEmpty:
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


# Global pool singleton
global_pool = AgyProcessPool()
