import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.process_pool import AgyProcessPool, WarmWorker


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
            worker = WarmWorker(
                process=mock_proc,
                prompt_dir="/tmp/test-worker",
                model=default_m,
                created_at=time.time(),
                init_event={"event": "init"},
            )

            # Manually seed one worker
            pool._running = True
            if default_m not in pool._pools:
                pool._pools[default_m] = asyncio.Queue()
            await pool._pools[default_m].put(worker)

            # Acquire worker
            with patch.object(pool, "_spawn_worker_safe", new_callable=AsyncMock) as mock_replenish:
                acquired = await pool.acquire(model=default_m)
                self.assertIsNotNone(acquired)
                self.assertEqual(acquired.process.pid, 12345)
                # Verify replenishment triggered immediately
                mock_replenish.assert_called_once()

            # Second acquire should return None (pool empty)
            second = await pool.acquire(model=default_m)
            self.assertIsNone(second)

            await worker.close()
            mock_proc.terminate.assert_called_once()
            await pool.shutdown()

        asyncio.run(_run())

    def test_pool_discards_dead_process(self):
        async def _run():
            pool = AgyProcessPool(target_size=1)
            mock_proc = MagicMock()
            mock_proc.returncode = 1  # Process died
            mock_proc.pid = 99999

            default_m = pool.warm_models[0]
            worker = WarmWorker(
                process=mock_proc,
                prompt_dir="/tmp/test-dead-worker",
                model=default_m,
                created_at=time.time(),
                init_event={"event": "init"},
            )

            pool._running = True
            if default_m not in pool._pools:
                pool._pools[default_m] = asyncio.Queue()
            await pool._pools[default_m].put(worker)

            with patch.object(pool, "_spawn_worker_safe", new_callable=AsyncMock) as mock_replenish:
                acquired = await pool.acquire(model=default_m)
                # Dead process must be discarded and return None
                self.assertIsNone(acquired)
                mock_replenish.assert_called_once()

            await pool.shutdown()

        asyncio.run(_run())

    def test_dynamic_auto_warm_on_unseen_model(self):
        async def _run():
            pool = AgyProcessPool(target_size=1, default_model="gemini-3.8-flash-high")
            pool._running = True

            # Request unseen model 'claude-sonnet-4-6'
            with patch.object(pool, "auto_warm_model", new_callable=AsyncMock) as mock_auto_warm:
                acquired = await pool.acquire(model="claude-sonnet-4-6")
                # First request falls back to cold spawn safely
                self.assertIsNone(acquired)
                # But automatically triggers auto-warm in background
                mock_auto_warm.assert_called_once_with("claude-sonnet-4-6")

            await pool.shutdown()

        asyncio.run(_run())

    def test_lru_model_eviction(self):
        async def _run():
            pool = AgyProcessPool(target_size=1, default_model="m1")
            pool.max_dynamic_models = 2
            pool._running = True
            pool._pools = {
                "m1": asyncio.Queue(),
                "m2": asyncio.Queue(),
            }
            pool._last_used = {
                "m1": 100.0,
                "m2": 50.0,  # Least recently used
            }

            # Adding m3 should evict m2 (the LRU candidate)
            await pool._evict_lru_model_if_needed("m3")

            self.assertIn("m1", pool._pools)
            self.assertNotIn("m2", pool._pools)

            await pool.shutdown()

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
