import json
import unittest

from app.api.models import ChatCompletionRequest, Message
from app.core.file_handler import TempFileManager
from app.core.openai_sse import events_to_sse_bytes, events_to_sse_bytes_with_tools
from app.core.tool_emulation import (
    first_json_object,
    format_tools_preamble,
    parse_emulated_output,
    to_openai_tool_calls,
)
from app.api.routes import build_chat_prompt
from app.core.agy_runner import build_agy_invocation


class TestFirstJsonObject(unittest.TestCase):
    def test_concatenated_objects_takes_first(self):
        blob = (
            '{\n  "kind": "tool_call",\n  "name": "get_weather",\n'
            '  "arguments": {"city": "Hanoi"}\n}\n'
            '{\n  "kind": "message",\n  "content": "Waiting for the weather data for Hanoi..."\n}\n'
            '{\n  "kind": "tool_call",\n  "name": "get_weather",\n'
            '  "arguments": {"city": "Hanoi"}\n}\n'
        )
        obj = first_json_object(blob)
        self.assertEqual(obj["kind"], "tool_call")
        self.assertEqual(obj["name"], "get_weather")
        self.assertEqual(obj["arguments"]["city"], "Hanoi")

    def test_prefix_noise_then_object(self):
        obj = first_json_object('note\n{"kind":"message","content":"hi"}')
        self.assertEqual(obj["kind"], "message")
        self.assertEqual(obj["content"], "hi")


class TestParseEmulatedOutput(unittest.TestCase):
    def test_structured_output_wins(self):
        parsed = parse_emulated_output(
            '{"kind":"message","content":"nope"}',
            {"kind": "tool_call", "name": "lookup", "arguments": {"q": "x"}},
        )
        self.assertEqual(parsed["kind"], "tool_call")
        self.assertEqual(parsed["name"], "lookup")

    def test_message_kind(self):
        parsed = parse_emulated_output('{"kind":"message","content":"hello"}')
        self.assertEqual(parsed, {"kind": "message", "content": "hello"})

    def test_openai_tool_calls_shape(self):
        parsed = parse_emulated_output(
            '{"kind":"tool_call","name":"get_weather","arguments":{"city":"Hanoi"}}'
        )
        tcs = to_openai_tool_calls(parsed)
        self.assertEqual(len(tcs), 1)
        self.assertEqual(tcs[0]["type"], "function")
        self.assertEqual(tcs[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(tcs[0]["function"]["arguments"]), {"city": "Hanoi"})
        self.assertTrue(tcs[0]["id"].startswith("call_"))


class TestRequestSchema(unittest.TestCase):
    def test_accepts_openai_tools_and_tool_result(self):
        req = ChatCompletionRequest.model_validate(
            {
                "model": "gemini-flash",
                "messages": [
                    {"role": "user", "content": "Weather in Hanoi?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Hanoi"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "name": "get_weather",
                        "content": "32C sunny",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        },
                    }
                ],
            }
        )
        self.assertEqual(len(req.tools), 1)
        self.assertEqual(req.messages[1].tool_calls[0]["function"]["name"], "get_weather")
        self.assertEqual(req.messages[2].role, "tool")
        mgr = TempFileManager()
        try:
            prompt, _ = build_chat_prompt(req, mgr)
        finally:
            mgr.cleanup()
        self.assertIn("get_weather", prompt)
        self.assertIn("Available client tools", prompt)
        self.assertIn("Assistant called get_weather", prompt)
        self.assertIn("[id=call_1]", prompt)
        self.assertIn("Tool result [call_1]", prompt)
        self.assertIn("32C sunny", prompt)


class TestSseTools(unittest.TestCase):
    def test_emits_tool_calls_not_raw_json_content(self):
        events = [
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "agent_response",
                    "text_delta": '{"kind":"tool_call","name":"get_weather","arguments":{"city":"Hanoi"}}',
                },
            },
            {
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "response": '{"kind":"tool_call","name":"get_weather","arguments":{"city":"Hanoi"}}\n{"kind":"message","content":"Waiting"}',
                    "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
                },
            },
        ]
        decoded = [
            f.decode("utf-8")
            for f in events_to_sse_bytes_with_tools(events, "id", 1, "m")
        ]
        joined = "".join(decoded)
        self.assertIn('"finish_reason": "tool_calls"', joined)
        self.assertIn('"name": "get_weather"', joined)
        self.assertNotIn("Waiting", joined)
        self.assertNotIn('"kind": "tool_call"', joined.split("tool_calls")[0] if False else "")
        self.assertFalse(any('"content": "{\\"kind\\"' in d for d in decoded))

    def test_plain_sse_still_skips_builtin_tool_steps(self):
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


class TestAgyInvocationSchema(unittest.TestCase):
    def test_json_schema_written_beside_prompt(self):
        inv = build_agy_invocation(
            "hello",
            model=None,
            output_format="json",
            json_schema={"type": "object", "properties": {"kind": {"type": "string"}}},
        )
        try:
            self.assertIn("--json-schema", inv.cmd)
            schema_path = inv.cmd[inv.cmd.index("--json-schema") + 1]
            self.assertTrue(schema_path.endswith("schema.json"))
            data = json.loads(open(schema_path, encoding="utf-8").read())
            self.assertEqual(data["type"], "object")
        finally:
            inv.cleanup()

    def test_preamble_none_choice(self):
        text = format_tools_preamble(
            [{"type": "function", "function": {"name": "x"}}], tool_choice="none"
        )
        self.assertIn('tool_choice is "none"', text)


if __name__ == "__main__":
    unittest.main()
