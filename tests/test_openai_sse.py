import unittest
from app.core.openai_sse import events_to_sse_bytes, next_text_delta


class TestNextTextDelta(unittest.TestCase):
    def test_incremental_fragments(self):
        sent = ""
        piece, sent = next_text_delta("Hello", sent)
        self.assertEqual(piece, "Hello")
        piece, sent = next_text_delta(" world", sent)
        self.assertEqual(piece, " world")
        self.assertEqual(sent, "Hello world")

    def test_cumulative_done_text(self):
        sent = "Hello"
        piece, sent = next_text_delta("Hello world", sent)
        self.assertEqual(piece, " world")
        self.assertEqual(sent, "Hello world")

    def test_duplicate_prefix(self):
        sent = "Hello world"
        piece, sent = next_text_delta("Hello", sent)
        self.assertEqual(piece, "")
        self.assertEqual(sent, "Hello world")


class TestEventsToSse(unittest.TestCase):
    def test_maps_text_delta_and_done(self):
        events = [
            {"event": "init", "init": {}},
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "agent_response",
                    "state": "ACTIVE",
                    "text_delta": "Git ",
                },
            },
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "agent_response",
                    "state": "DONE",
                    "text_delta": "Git rebase.\n",
                },
            },
            {
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "response": "Git rebase.\n",
                    "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
                },
            },
        ]
        frames = list(events_to_sse_bytes(events, "chatcmpl-test", 1, "test-model"))
        decoded = [f.decode("utf-8") for f in frames]
        self.assertTrue(decoded[0].startswith("data: "))
        self.assertIn('"content": "Git "', decoded[0])
        self.assertIn('"role": "assistant"', decoded[0])
        self.assertIn('"content": "rebase.\\n"', decoded[1])
        self.assertIn('"finish_reason": "stop"', decoded[2])
        self.assertIn('"prompt_tokens": 10', decoded[2])
        self.assertEqual(decoded[-1], "data: [DONE]\n\n")

    def test_skips_tool_steps(self):
        events = [
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "tool",
                    "tool_name": "run_command",
                    "text_delta": "should not leak",
                },
            },
            {
                "event": "result",
                "result": {"status": "SUCCESS", "response": "ok", "usage": {}},
            },
        ]
        decoded = [f.decode("utf-8") for f in events_to_sse_bytes(events, "id", 1, "m")]
        self.assertTrue(any('"content": "ok"' in d for d in decoded))
        self.assertFalse(any("should not leak" in d for d in decoded))


if __name__ == "__main__":
    unittest.main()
