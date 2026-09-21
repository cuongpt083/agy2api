import os
import json
import tempfile
import unittest
from pathlib import Path
from app.api.models import ChoiceMessage, Choice, ChatCompletionResponse, Usage
from app.core.agy_runner import build_agy_invocation, extract_thinking_from_brain
from app.core import dataset_writer


class TestCotCaptureModels(unittest.TestCase):
    def test_choice_message_with_reasoning(self):
        msg = ChoiceMessage(
            role="assistant",
            content="Hello world",
            reasoning_content="I should greet the user warmly.",
        )
        data = msg.model_dump()
        self.assertEqual(data["role"], "assistant")
        self.assertEqual(data["content"], "Hello world")
        self.assertEqual(data["reasoning_content"], "I should greet the user warmly.")

    def test_usage_with_reasoning_tokens(self):
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            completion_tokens_details={"reasoning_tokens": 40},
        )
        data = usage.model_dump()
        self.assertEqual(data["prompt_tokens"], 100)
        self.assertEqual(data["completion_tokens_details"]["reasoning_tokens"], 40)


class TestAgyRunnerEffort(unittest.TestCase):
    def test_build_agy_invocation_with_effort(self):
        inv = build_agy_invocation(
            prompt="Hello",
            model="gemini-3.7-flash-high",
            output_format="json",
            effort="high",
        )
        try:
            self.assertNotIn("--effort", inv.cmd)
            self.assertIn("--model", inv.cmd)
            self.assertIn("gemini-3.7-flash-high", inv.cmd)
        finally:
            inv.cleanup()


class TestExtractThinkingFromBrain(unittest.TestCase):
    def test_extract_thinking_from_temp_brain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            conv_id = "test-conv-999"
            log_dir = Path(tmpdir) / conv_id / ".system_generated" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            transcript_path = log_dir / "transcript_full.jsonl"

            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"step_index": 0, "type": "USER_INPUT", "content": "What is 2+2?"}) + "\n")
                f.write(json.dumps({
                    "step_index": 1,
                    "type": "PLANNER_RESPONSE",
                    "content": "4",
                    "thinking": "Calculation: 2 + 2 = 4.",
                }) + "\n")

            orig_env = os.environ.get("AGY_BRAIN_DIR")
            try:
                os.environ["AGY_BRAIN_DIR"] = tmpdir
                thinking = extract_thinking_from_brain(conv_id)
                self.assertEqual(thinking, "Calculation: 2 + 2 = 4.")
            finally:
                if orig_env is not None:
                    os.environ["AGY_BRAIN_DIR"] = orig_env
                else:
                    os.environ.pop("AGY_BRAIN_DIR", None)


class TestDatasetWriter(unittest.TestCase):
    def test_save_cot_turn_writes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_dir = dataset_writer.DATASET_DIR
            try:
                dataset_writer.DATASET_DIR = Path(tmpdir)
                saved = dataset_writer.save_cot_turn(
                    messages=[{"role": "user", "content": "Solve x+1=2"}],
                    response_content="x = 1",
                    reasoning_content="Step 1: subtract 1 from both sides. x = 1.",
                    model="gemini-3.7-flash-medium",
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                    conversation_id="conv-12345",
                )
                self.assertTrue(saved)

                files = list(Path(tmpdir).glob("cot_dataset_*.jsonl"))
                self.assertEqual(len(files), 1)

                with open(files[0], "r", encoding="utf-8") as f:
                    lines = f.readlines()
                self.assertEqual(len(lines), 1)
                record = json.loads(lines[0])
                self.assertEqual(record["id"], "conv-12345")
                self.assertEqual(record["reasoning_content"], "Step 1: subtract 1 from both sides. x = 1.")
                self.assertEqual(record["content"], "x = 1")
                self.assertEqual(len(record["messages"]), 2)
                self.assertEqual(record["messages"][1]["reasoning_content"], "Step 1: subtract 1 from both sides. x = 1.")
            finally:
                dataset_writer.DATASET_DIR = orig_dir

    def test_save_cot_turn_skips_empty_reasoning(self):
        saved = dataset_writer.save_cot_turn(
            messages="Hello",
            response_content="Hi there",
            reasoning_content="",
            model="gemini-3.7-flash",
        )
        self.assertFalse(saved)


if __name__ == "__main__":
    unittest.main()
