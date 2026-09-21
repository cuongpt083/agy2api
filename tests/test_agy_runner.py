import os
import unittest
from pathlib import Path

from app.core.agy_runner import build_agy_invocation


class TestAgyInvocationArgv(unittest.TestCase):
    def test_huge_prompt_is_written_to_file_not_argv(self) -> None:
        huge = "chunk-payload " * 20_000  # ~280 KB, over Linux MAX_ARG_STRLEN
        inv = build_agy_invocation(huge, model="gemini-flash", output_format="json")
        try:
            self.assertTrue(all(len(arg.encode("utf-8")) < 4096 for arg in inv.cmd))
            self.assertNotIn(huge[:80], " ".join(inv.cmd))
            self.assertEqual(inv.cmd[0], "agy")
            self.assertIn("--print", inv.cmd)
            self.assertIn("--add-dir", inv.cmd)
            add_dir = inv.cmd[inv.cmd.index("--add-dir") + 1]
            self.assertEqual(add_dir, inv.prompt_dir)
            prompt_file = Path(inv.prompt_path)
            self.assertTrue(prompt_file.is_file())
            self.assertEqual(prompt_file.read_text(encoding="utf-8"), huge)
            self.assertEqual(str(prompt_file.parent), add_dir)
        finally:
            inv.cleanup()
        self.assertFalse(os.path.exists(inv.prompt_path))
        self.assertFalse(os.path.exists(inv.prompt_dir))

    def test_short_prompt_also_uses_file(self) -> None:
        inv = build_agy_invocation("hello", model=None, output_format="stream-json")
        try:
            self.assertIn("--output-format", inv.cmd)
            self.assertEqual(inv.cmd[inv.cmd.index("--output-format") + 1], "stream-json")
            self.assertNotIn("hello", inv.cmd)
            self.assertEqual(Path(inv.prompt_path).read_text(encoding="utf-8"), "hello")
        finally:
            inv.cleanup()


def _effort_value(cmd: list[str]) -> str | None:
    if "--effort" not in cmd:
        return None
    return cmd[cmd.index("--effort") + 1]


class TestAgyModelEffort(unittest.TestCase):
    def test_omits_openai_minimal_when_model_slug_encodes_effort(self) -> None:
        inv = build_agy_invocation(
            "hello",
            model="gemini-3.8-flash-high",
            output_format="stream-json",
            effort="minimal",
        )
        try:
            self.assertIn("gemini-3.8-flash-high", inv.cmd)
            self.assertIsNone(_effort_value(inv.cmd))
        finally:
            inv.cleanup()

    def test_omits_effort_for_display_name_with_level(self) -> None:
        inv = build_agy_invocation(
            "hello",
            model="Gemini 3.8 Flash (High)",
            output_format="json",
            effort="minimal",
        )
        try:
            self.assertIn("Gemini 3.8 Flash (High)", inv.cmd)
            self.assertIsNone(_effort_value(inv.cmd))
        finally:
            inv.cleanup()

    def test_maps_minimal_to_low_when_model_has_no_effort(self) -> None:
        inv = build_agy_invocation(
            "hello",
            model="claude-sonnet-4-6",
            output_format="json",
            effort="minimal",
        )
        try:
            self.assertEqual(_effort_value(inv.cmd), "low")
        finally:
            inv.cleanup()

    def test_maps_xhigh_to_high(self) -> None:
        inv = build_agy_invocation(
            "hello",
            model="claude-sonnet-4-6",
            output_format="json",
            effort="xhigh",
        )
        try:
            self.assertEqual(_effort_value(inv.cmd), "high")
        finally:
            inv.cleanup()

    def test_passes_agy_effort_unchanged_without_model_suffix(self) -> None:
        inv = build_agy_invocation(
            "hello",
            model="claude-sonnet-4-6",
            output_format="json",
            effort="high",
        )
        try:
            self.assertEqual(_effort_value(inv.cmd), "high")
        finally:
            inv.cleanup()
