import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core import model_manager
from app.core.process_pool import FLAVOR_PLAIN, AgyProcessPool, WarmWorker, pool_key


def _w(model, age):
    p = MagicMock()
    p.returncode = None
    p.pid = 1
    p.terminate = MagicMock()
    p.wait = AsyncMock(return_value=0)
    return WarmWorker(process=p, prompt_dir="/tmp/x", model=model, created_at=time.time() - age,
                      init_event={}, flavor=FLAVOR_PLAIN)


class TestSweep(unittest.TestCase):
    def test_sweep_replaces_stale_only(self):
        async def _run():
            pool = AgyProcessPool(target_size=2)
            pool._running = True
            key = pool_key("m", FLAVOR_PLAIN)
            pool._pools[key] = asyncio.Queue()
            pool._pools[key].put_nowait(_w("m", 99999))
            pool._pools[key].put_nowait(_w("m", 1))
            with patch.object(pool, "_spawn_worker_safe", new=AsyncMock()) as sp:
                await pool._sweep_once()
                await asyncio.sleep(0)
            self.assertEqual(pool._pools[key].qsize(), 1)
            self.assertEqual(sp.await_count, 1)
        asyncio.run(_run())

    def test_acquire_skips_stale_to_fresh(self):
        async def _run():
            pool = AgyProcessPool(target_size=2)
            pool._running = True
            key = pool_key("m", FLAVOR_PLAIN)
            pool._pools[key] = asyncio.Queue()
            pool._pools[key].put_nowait(_w("m", 99999))
            fresh = _w("m", 1)
            pool._pools[key].put_nowait(fresh)
            with patch.object(pool, "_spawn_worker_safe", new=AsyncMock()):
                got = await pool.acquire("m")
            self.assertIs(got, fresh)
        asyncio.run(_run())


class TestModelBackoff(unittest.TestCase):
    def test_failure_backoff_returns_stale_cache_without_refetch(self):
        async def _run():
            model_manager._CACHED_MODELS = [MagicMock()]
            model_manager._LAST_FETCH_TIME = 0
            model_manager._LAST_FAIL_TIME = 0
            with patch.object(model_manager, "fetch_models_from_cli", new=AsyncMock(return_value=[])) as f:
                await model_manager.get_available_models()
                await model_manager.get_available_models()
            self.assertEqual(f.await_count, 1)
        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()


class TestToolLoopGuard(unittest.TestCase):
    def test_aborts_after_too_many_builtin_tools(self):
        from app.core import agy_runner
        import json

        class R:
            def __init__(self, lines): self.lines = list(lines)
            async def readline(self): return self.lines.pop(0) if self.lines else b""

        class P:
            pid = 1
            def __init__(self, n):
                ev = {"event": "step_update", "step_update": {"step_type": "tool", "state": "ACTIVE", "tool_name": "run_command"}}
                self.stdout = R([(json.dumps(ev) + "\n").encode()] * n)

        async def _run(n):
            out = []
            async for e in agy_runner._iter_stdout_events(P(n), stop_after_first_schema_object=True):
                out.append(e)
            return out

        with patch.object(agy_runner, "_MAX_BUILTIN_TOOL_STEPS", 3):
            self.assertEqual(len(asyncio.run(_run(3))), 3)
            with self.assertRaises(agy_runner.AgyBuiltinToolLoopError):
                asyncio.run(_run(4))


class TestSandbox(unittest.TestCase):
    def test_wrap_cmd(self):
        from app.core import sandbox
        cmd = ["agy", "--model", "m"]
        with patch.object(sandbox, "_MODE", ""):
            self.assertEqual(sandbox.wrap_cmd(cmd, "/tmp/w"), cmd)
        with patch.object(sandbox, "_MODE", "bwrap"), patch("shutil.which", return_value="/usr/bin/bwrap"):
            out = sandbox.wrap_cmd(cmd, "/tmp/w")
            self.assertEqual(out[0], "/usr/bin/bwrap")
            self.assertIn("--clearenv", out)
            self.assertTrue(out[-3].endswith(".local/bin/agy"))
            self.assertEqual(out[-2:], ["--model", "m"])
