import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.agy_runner import (
    _raise_if_agy_result_error,
    _wait_for_init,
    _write_user_event_and_close,
)


class TestAgyAuthSurfacing(unittest.TestCase):
    def test_raise_if_result_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            _raise_if_agy_result_error(
                {"status": "ERROR", "error": "authentication required. Run 'agy' to log in"}
            )
        self.assertIn("authentication required", str(ctx.exception))

    def test_success_result_noop(self):
        _raise_if_agy_result_error({"status": "SUCCESS", "response": "ok"})

    def test_write_broken_pipe_becomes_runtime_error(self):
        async def _run():
            proc = MagicMock()
            proc.pid = 99
            proc.stdin = MagicMock()
            proc.stdin.write = MagicMock()
            proc.stdin.drain = AsyncMock(side_effect=BrokenPipeError())
            proc.stdin.close = MagicMock()
            with self.assertRaises(RuntimeError) as ctx:
                await _write_user_event_and_close(proc, "hello")
            self.assertIn("authentication required", str(ctx.exception).lower() + "login")
            self.assertIn("closed stdin", str(ctx.exception).lower())

        asyncio.run(_run())

    def test_wait_for_init_empty_stdout(self):
        async def _run():
            proc = MagicMock()
            proc.pid = 7
            proc.stdout = MagicMock()
            proc.stdout.readline = AsyncMock(return_value=b"")
            with self.assertRaises(RuntimeError) as ctx:
                await _wait_for_init(proc, timeout=1)
            self.assertIn("before init", str(ctx.exception))

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
