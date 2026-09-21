import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.core.security import API_KEY
from app.main import app


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
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)

    async def test_tools_request_uses_agy_without_gemini_key(self):
        agy = AsyncMock(return_value={"text": "from-agy", "usage": {}})
        with patch("app.api.routes.run_agy_prompt", agy):
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
        self.assertEqual(resp.json()["choices"][0]["message"]["content"], "from-agy")
        self.assertIsNone(resp.json()["choices"][0]["message"].get("tool_calls"))
        agy.assert_awaited_once()

    async def test_non_gemini_model_with_tools_uses_agy(self):
        agy = AsyncMock(return_value={"text": "from-agy-claude", "usage": {}})
        with patch("app.api.routes.run_agy_prompt", agy):
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
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["choices"][0]["message"]["content"], "from-agy-claude")
        agy.assert_awaited_once()

    async def test_request_without_tools_still_uses_agy(self):
        agy = AsyncMock(return_value={"text": "from-agy", "usage": {}})
        with patch("app.api.routes.run_agy_prompt", agy):
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

    async def test_stream_with_tools_uses_agy(self):
        async def fake_stream(*args, **kwargs):
            yield {
                "event": "step_update",
                "step_update": {
                    "step_type": "agent_response",
                    "state": "ACTIVE",
                    "text_delta": "hello",
                },
            }
            yield {
                "event": "result",
                "result": {"response": "hello", "usage": {}},
            }

        with patch("app.api.routes.stream_agy_prompt", fake_stream):
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
        self.assertIn("hello", text)
        self.assertNotIn("tool_calls", text)
        self.assertIn("data: [DONE]", text)
