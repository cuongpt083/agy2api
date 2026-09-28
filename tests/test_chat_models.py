import unittest
from app.api.models import ChatCompletionRequest, Message, ChoiceMessage


class TestChatModels(unittest.TestCase):
    def test_message_with_null_content_and_tool_calls(self):
        payload = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "test_func", "arguments": '{"param": 1}'}
                }
            ]
        }
        msg = Message.model_validate(payload)
        self.assertEqual(msg.role, "assistant")
        self.assertIsNone(msg.content)
        self.assertEqual(len(msg.tool_calls), 1)
        self.assertEqual(msg.tool_calls[0]["function"]["name"], "test_func")

    def test_message_tool_role(self):
        payload = {
            "role": "tool",
            "name": "test_func",
            "tool_call_id": "call_abc",
            "content": '{"status": "ok"}'
        }
        msg = Message.model_validate(payload)
        self.assertEqual(msg.role, "tool")
        self.assertEqual(msg.tool_call_id, "call_abc")
        self.assertEqual(msg.name, "test_func")
        self.assertEqual(msg.content, '{"status": "ok"}')

    def test_chat_request_with_agentic_fields_and_extra_fields(self):
        payload = {
            "model": "gemini-3.8-flash-medium",
            "messages": [
                {"role": "user", "content": "hello"}
            ],
            "tools": [
                {"type": "function", "function": {"name": "test_func", "description": "test"}}
            ],
            "tool_choice": "auto",
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 500,
            "extra_field_from_client": 123
        }
        req = ChatCompletionRequest.model_validate(payload)
        self.assertEqual(req.model, "gemini-3.8-flash-medium")
        self.assertEqual(len(req.messages), 1)
        self.assertEqual(len(req.tools), 1)
        self.assertEqual(req.tool_choice, "auto")
        self.assertEqual(req.max_tokens, 500)


if __name__ == "__main__":
    unittest.main()
