import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.agy_runner import AgyTimeoutError, stream_agy_prompt_pooled
from app.core.process_pool import FLAVOR_TOOLS, WarmWorker


class TestWarmTurnTimeout(unittest.TestCase):
    def test_readline_deadline_raises_agy_timeout(self):
        async def _run():
            proc = MagicMock()
            proc.pid = 42
            proc.returncode = None
            proc.stdin = MagicMock()
            proc.stdin.write = MagicMock()
            proc.stdin.drain = AsyncMock()
            proc.stdin.close = MagicMock()
            proc.stdin.wait_closed = AsyncMock()
            proc.stdin.is_closing = MagicMock(return_value=False)

            async def slow_readline():
                await asyncio.sleep(5)
                return b""

            proc.stdout = MagicMock()
            proc.stdout.readline = AsyncMock(side_effect=slow_readline)
            proc.stderr = MagicMock()
            proc.stderr.readline = AsyncMock(return_value=b"")

            worker = WarmWorker(
                process=proc,
                prompt_dir="/tmp/x",
                model="m",
                created_at=time.time(),
                init_event={"event": "init"},
                flavor=FLAVOR_TOOLS,
            )
            worker.close = AsyncMock()

            with patch("app.core.agy_runner._TURN_TIMEOUT_S", 0.05):
                with self.assertRaises(AgyTimeoutError) as ctx:
                    async for _ in stream_agy_prompt_pooled(worker, "hello", model="m"):
                        pass
            self.assertIn("timed out", str(ctx.exception))
            self.assertIn("pid=42", str(ctx.exception))
            worker.close.assert_awaited()

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
