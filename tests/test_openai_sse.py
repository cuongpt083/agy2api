import json
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


    def test_maps_reasoning_content(self):
        events = [
            {"event": "init", "conversation_id": "test-conv-123"},
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "agent_response",
                    "state": "DONE",
                    "text_delta": "The answer is 4.",
                },
            },
            {
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "response": "The answer is 4.",
                    "reasoning_content": "Step 1: 2+2=4. Step 2: verify.",
                    "usage": {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 8, "total_tokens": 15},
                },
            },
        ]
        frames = list(events_to_sse_bytes(events, "chatcmpl-test", 1, "test-model"))
        decoded = [f.decode("utf-8") for f in frames]
        # Verify content delta
        self.assertTrue(any('"content": "The answer is 4."' in d for d in decoded))
        # Verify reasoning_content chunk is emitted
        self.assertTrue(any('"reasoning_content": "Step 1: 2+2=4. Step 2: verify."' in d for d in decoded))
        # Verify reasoning_tokens in usage
        self.assertTrue(any('"reasoning_tokens": 8' in d for d in decoded))

    def test_content_arriving_first_still_emits_reasoning_before_content(self):
        events = [
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "agent_response",
                    "state": "ACTIVE",
                    "text_delta": "The answer is 4.",
                },
            },
            {
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "response": "The answer is 4.",
                    "reasoning_content": "I added 2 and 2.",
                },
            },
        ]
        decoded = [f.decode("utf-8") for f in events_to_sse_bytes(events, "id", 1, "m")]
        reasoning_idx = next(i for i, d in enumerate(decoded) if "reasoning_content" in d)
        content_idx = next(i for i, d in enumerate(decoded) if '"content":' in d)
        self.assertLess(reasoning_idx, content_idx)
        self.assertFalse(any('"content":' in d for d in decoded[:reasoning_idx]))

    def test_thinking_delta_emits_reasoning_before_content(self):
        events = [
            {"event": "thinking_delta", "thinking_delta": "I will add 2 and 2."},
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "agent_response",
                    "state": "ACTIVE",
                    "text_delta": "4",
                },
            },
            {
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "response": "4",
                    "reasoning_content": "I will add 2 and 2.",
                },
            },
        ]
        decoded = [f.decode("utf-8") for f in events_to_sse_bytes(events, "id", 1, "m")]
        reasoning_idx = next(i for i, d in enumerate(decoded) if "reasoning_content" in d)
        content_idx = next(i for i, d in enumerate(decoded) if '"content": "4"' in d)
        self.assertLess(reasoning_idx, content_idx)
        reasoning_frames = [d for d in decoded if "reasoning_content" in d]
        self.assertEqual(len(reasoning_frames), 1)
        self.assertIn("I will add 2 and 2.", reasoning_frames[0])

    def test_later_thinking_delta_does_not_repeat_prefix(self):
        events = [
            {"event": "thinking_delta", "thinking_delta": "First. "},
            {"event": "thinking_delta", "thinking_delta": "First. Second."},
            {
                "event": "result",
                "result": {"status": "SUCCESS", "response": "ok", "reasoning_content": "First. Second."},
            },
        ]
        decoded = [f.decode("utf-8") for f in events_to_sse_bytes(events, "id", 1, "m")]
        pieces = []
        for line in decoded:
            if "reasoning_content" not in line:
                continue
            payload = line[len("data: "):].strip()
            chunk = json.loads(payload)
            pieces.append(chunk["choices"][0]["delta"]["reasoning_content"])
        self.assertEqual(pieces, ["First. ", "Second."])


if __name__ == "__main__":
    unittest.main()
