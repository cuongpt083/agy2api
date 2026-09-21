import json
import unittest

import httpx

from app.core.gemini_client import GeminiAPIError, generate_content, stream_generate_content


def _handler_for(status: int, payload: dict | list[dict], stream: bool = False):
    def handler(request: httpx.Request) -> httpx.Response:
        if stream:
            chunks = []
            for item in payload:
                chunks.append(f"data: {json.dumps(item)}\n\n".encode("utf-8"))
            return httpx.Response(status, content=b"".join(chunks))
        return httpx.Response(status, json=payload)

    return handler


class TestGenerateContent(unittest.IsolatedAsyncioTestCase):
    async def test_posts_generate_content_with_api_key_header(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["key"] = request.headers.get("x-goog-api-key")
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}
                    ]
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await generate_content(
            "gemini-2.5-flash",
            {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            api_key="secret-key",
            client=client,
        )
        self.assertIn("/models/gemini-2.5-flash:generateContent", captured["url"])
        self.assertEqual(captured["key"], "secret-key")
        self.assertEqual(captured["body"]["contents"][0]["parts"][0]["text"], "hi")
        self.assertEqual(result["candidates"][0]["content"]["parts"][0]["text"], "ok")

    async def test_raises_openai_shaped_error_on_gemini_400(self):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                _handler_for(
                    400,
                    {
                        "error": {
                            "code": 400,
                            "message": "Function call is missing a thought_signature",
                            "status": "INVALID_ARGUMENT",
                        }
                    },
                )
            )
        )
        with self.assertRaises(GeminiAPIError) as ctx:
            await generate_content(
                "gemini-3.7-flash",
                {"contents": []},
                api_key="k",
                client=client,
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("thought_signature", ctx.exception.message)


class TestStreamGenerateContent(unittest.IsolatedAsyncioTestCase):
    async def test_parses_sse_data_frames(self):
        events = [
            {"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]},
            {"candidates": [{"finishReason": "STOP"}]},
        ]
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(_handler_for(200, events, stream=True))
        )
        received = []
        async for event in stream_generate_content(
            "gemini-2.5-flash",
            {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            api_key="k",
            client=client,
        ):
            received.append(event)
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0]["candidates"][0]["content"]["parts"][0]["text"], "Hi")
        self.assertEqual(received[1]["candidates"][0]["finishReason"], "STOP")
