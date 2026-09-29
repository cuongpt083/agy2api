import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.process_pool import (
    FLAVOR_PLAIN,
    FLAVOR_TOOLS,
    AgyProcessPool,
    WarmWorker,
    pool_key,
)
from app.core.tool_emulation import TOOLS_JSON_SCHEMA


def _worker(model: str, proc, flavor: str = FLAVOR_PLAIN) -> WarmWorker:
    return WarmWorker(
        process=proc,
        prompt_dir="/tmp/test-worker",
        model=model,
        created_at=time.time(),
        init_event={"event": "init"},
        flavor=flavor,
    )


class TestProcessPool(unittest.TestCase):
    def test_pool_acquire_and_replenish(self):
        async def _run():
            pool = AgyProcessPool(target_size=1)
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.pid = 12345
            mock_proc.terminate = MagicMock()
            mock_proc.wait = AsyncMock(return_value=0)

            default_m = pool.warm_models[0]
            key = pool_key(default_m, FLAVOR_PLAIN)
            worker = _worker(default_m, mock_proc)

            pool._running = True
            pool._pools[key] = asyncio.Queue()
            await pool._pools[key].put(worker)

            with patch.object(pool, "_spawn_worker_safe", new_callable=AsyncMock) as mock_replenish:
                acquired = await pool.acquire(model=default_m)
                self.assertIsNotNone(acquired)
                self.assertEqual(acquired.process.pid, 12345)
                mock_replenish.assert_called_once_with(default_m, FLAVOR_PLAIN)

            second = await pool.acquire(model=default_m)
            self.assertIsNone(second)

            await worker.close()
            mock_proc.terminate.assert_called_once()
            await pool.shutdown()

        asyncio.run(_run())

    def test_pool_acquire_tools_flavor_when_json_schema_set(self):
        async def _run():
            pool = AgyProcessPool(target_size=1)
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.pid = 7
            mock_proc.terminate = MagicMock()
            mock_proc.wait = AsyncMock(return_value=0)

            default_m = pool.warm_models[0]
            key = pool_key(default_m, FLAVOR_TOOLS)
            worker = _worker(default_m, mock_proc, flavor=FLAVOR_TOOLS)

            pool._running = True
            pool._pools[key] = asyncio.Queue()
            await pool._pools[key].put(worker)

            with patch.object(pool, "_spawn_worker_safe", new_callable=AsyncMock):
                acquired = await pool.acquire(model=default_m, json_schema=TOOLS_JSON_SCHEMA)
                self.assertIsNotNone(acquired)
                self.assertEqual(acquired.flavor, FLAVOR_TOOLS)

            await worker.close()
            await pool.shutdown()

        asyncio.run(_run())

    def test_pool_discards_dead_process(self):
        async def _run():
            pool = AgyProcessPool(target_size=1)
            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.pid = 99999
            mock_proc.terminate = MagicMock()
            mock_proc.wait = AsyncMock(return_value=1)

            default_m = pool.warm_models[0]
            key = pool_key(default_m, FLAVOR_PLAIN)
            worker = _worker(default_m, mock_proc)

            pool._running = True
            pool._pools[key] = asyncio.Queue()
            await pool._pools[key].put(worker)

            with patch.object(pool, "_spawn_worker_safe", new_callable=AsyncMock) as mock_replenish:
                acquired = await pool.acquire(model=default_m)
                self.assertIsNone(acquired)
                mock_replenish.assert_called_once()

            await pool.shutdown()

        asyncio.run(_run())

    def test_dynamic_auto_warm_on_unseen_model(self):
        async def _run():
            pool = AgyProcessPool(target_size=1, default_model="gemini-3.8-flash-high")
            pool._running = True

            with patch.object(pool, "auto_warm_model", new_callable=AsyncMock) as mock_auto_warm:
                acquired = await pool.acquire(model="claude-sonnet-4-6")
                self.assertIsNone(acquired)
                mock_auto_warm.assert_called_once_with("claude-sonnet-4-6", FLAVOR_PLAIN)

            with patch.object(pool, "auto_warm_model", new_callable=AsyncMock) as mock_auto_warm:
                acquired = await pool.acquire(model="claude-sonnet-4-6", json_schema=TOOLS_JSON_SCHEMA)
                self.assertIsNone(acquired)
                mock_auto_warm.assert_called_once_with("claude-sonnet-4-6", FLAVOR_TOOLS)

            await pool.shutdown()

        asyncio.run(_run())

    def test_lru_model_eviction(self):
        async def _run():
            pool = AgyProcessPool(target_size=1, default_model="m1")
            pool.max_dynamic_models = 2
            pool._running = True
            k1 = pool_key("m1", FLAVOR_PLAIN)
            k2 = pool_key("m2", FLAVOR_PLAIN)
            pool._pools = {
                k1: asyncio.Queue(),
                k2: asyncio.Queue(),
            }
            pool._last_used = {
                k1: 100.0,
                k2: 50.0,
            }

            await pool._evict_lru_model_if_needed(pool_key("m3", FLAVOR_PLAIN))

            self.assertIn(k1, pool._pools)
            self.assertNotIn(k2, pool._pools)

            await pool.shutdown()

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
