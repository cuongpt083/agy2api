import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.core.security import API_KEY
from app.main import app


GEMINI_FUNCTION_CALL = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "functionCall": {
                            "id": "call_read_1",
                            "name": "read",
                            "args": {"path": "src/foo.ts"},
                        },
                        "thoughtSignature": "sig-route",
                    }
                ]
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 12,
        "candidatesTokenCount": 4,
        "totalTokenCount": 16,
    },
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


class TestChatCompletionsToolsRoute(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.headers = {"Authorization": f"Bearer {API_KEY}"}

    async def test_tools_request_returns_openai_tool_calls_not_agy(self):
        os.environ["GEMINI_API_KEY"] = "test-gemini-key"
        generate = AsyncMock(return_value=GEMINI_FUNCTION_CALL)
        agy = AsyncMock(return_value={"text": "should-not-use-agy"})
        with patch("app.core.gemini_client.generate_content", generate), patch(
            "app.api.routes.run_agy_prompt", agy
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    headers=self.headers,
                    json={
                        "model": "gemini-3.7-flash-high",
                        "messages": [{"role": "user", "content": "read src/foo.ts"}],
                        "tools": TOOLS,
                    },
                )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        tool_call = choice["message"]["tool_calls"][0]
        self.assertEqual(tool_call["function"]["name"], "read")
        self.assertEqual(json.loads(tool_call["function"]["arguments"])["path"], "src/foo.ts")
        self.assertEqual(
            tool_call["extra_content"]["google"]["thought_signature"],
            "sig-route",
        )
        generate.assert_awaited_once()
        agy.assert_not_awaited()

    async def test_non_gemini_model_with_tools_is_rejected(self):
        os.environ["GEMINI_API_KEY"] = "test-gemini-key"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=self.headers,
                json={
                    "model": "claude-sonnet-4-6",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": TOOLS,
                },
            )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Gemini API", resp.json()["error"]["message"])

    async def test_tools_without_gemini_key_returns_openai_error(self):
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=self.headers,
                json={
                    "model": "gemini-2.5-flash",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": TOOLS,
                },
            )
        self.assertEqual(resp.status_code, 400)
        err = resp.json()["error"]
        self.assertIn("GEMINI_API_KEY", err["message"])
        self.assertEqual(err["type"], "invalid_request_error")

    async def test_request_without_tools_still_uses_agy(self):
        agy = AsyncMock(return_value={"text": "from-agy", "usage": {}})
        generate = AsyncMock()
        with patch("app.api.routes.run_agy_prompt", agy), patch(
            "app.core.gemini_client.generate_content", generate
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    headers=self.headers,
                    json={
                        "model": "gemini-3.7-flash-high",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["choices"][0]["message"]["content"], "from-agy")
        agy.assert_awaited_once()
        generate.assert_not_awaited()

    async def test_stream_tools_emits_tool_call_sse(self):
        os.environ["GEMINI_API_KEY"] = "test-gemini-key"

        async def fake_stream(*args, **kwargs):
            yield {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "id": "call_1",
                                        "name": "bash",
                                        "args": {"command": "ls"},
                                    },
                                    "thoughtSignature": "s",
                                }
                            ]
                        }
                    }
                ]
            }
            yield {"candidates": [{"finishReason": "STOP"}]}

        with patch("app.core.gemini_client.stream_generate_content", fake_stream), patch(
            "app.api.routes.stream_agy_prompt"
        ) as agy_stream:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    headers=self.headers,
                    json={
                        "model": "gemini-2.5-flash",
                        "stream": True,
                        "messages": [{"role": "user", "content": "ls"}],
                        "tools": TOOLS,
                    },
                )
        self.assertEqual(resp.status_code, 200, resp.text)
        text = resp.text
        self.assertIn("tool_calls", text)
        self.assertIn("bash", text)
        self.assertIn('"finish_reason": "tool_calls"', text)
        self.assertIn("data: [DONE]", text)
        agy_stream.assert_not_called()
