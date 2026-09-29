import os
import unittest
from pathlib import Path

from app.core.agy_runner import build_agy_invocation, encode_user_event


class TestAgyInvocationArgv(unittest.TestCase):
    def test_huge_prompt_is_not_on_argv(self) -> None:
        huge = "chunk-payload " * 20_000  # ~280 KB, over Linux MAX_ARG_STRLEN
        inv = build_agy_invocation(huge, model="gemini-flash", output_format="json")
        try:
            self.assertTrue(all(len(arg.encode("utf-8")) < 4096 for arg in inv.cmd))
            self.assertNotIn(huge[:80], " ".join(inv.cmd))
            self.assertEqual(inv.cmd[0], "agy")
            self.assertNotIn("--print", inv.cmd)
            self.assertIn("--input-format", inv.cmd)
            self.assertEqual(inv.cmd[inv.cmd.index("--input-format") + 1], "stream-json")
            self.assertIn("--output-format", inv.cmd)
            self.assertEqual(inv.cmd[inv.cmd.index("--output-format") + 1], "stream-json")
            self.assertIn("--add-dir", inv.cmd)
            add_dir = inv.cmd[inv.cmd.index("--add-dir") + 1]
            self.assertEqual(add_dir, inv.prompt_dir)
            self.assertEqual(inv.prompt, huge)
            self.assertIsNone(inv.prompt_path)
            self.assertFalse(Path(inv.prompt_dir, "prompt.txt").exists())
        finally:
            inv.cleanup()
        self.assertFalse(os.path.exists(inv.prompt_dir))

    def test_short_prompt_uses_stdin_stream_json(self) -> None:
        inv = build_agy_invocation("hello", model=None, output_format="stream-json")
        try:
            self.assertNotIn("hello", inv.cmd)
            self.assertNotIn("--print", inv.cmd)
            self.assertEqual(inv.prompt, "hello")
            self.assertEqual(inv.cmd[inv.cmd.index("--input-format") + 1], "stream-json")
            self.assertNotIn("--json-schema", inv.cmd)
            payload = encode_user_event("hello").decode("utf-8")
            self.assertTrue(payload.endswith("\n"))
            self.assertIn('"event": "user"', payload)
            self.assertIn('"content": "hello"', payload)
        finally:
            inv.cleanup()
