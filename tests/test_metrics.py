import unittest
import asyncio
import httpx
from app.main import app
from app.core.metrics import (
    record_http_request,
    record_chat_completion,
    record_tokens,
    record_runner_execution,
    record_image_generation,
    record_speech_request,
    expose_metrics,
)


class TestMetrics(unittest.TestCase):
    def test_metrics_recording_helpers(self):
        record_http_request("POST", "/v1/chat/completions", 200, 1.234)
        record_chat_completion("gemini-flash", stream=True, status="success", duration=2.5, first_token_duration=0.5)
        record_tokens("gemini-flash", prompt_tokens=50, completion_tokens=25, total_tokens=75, cached_tokens=10)
        record_runner_execution("gemini-flash", "stream-json", "success", 2.1)
        record_image_generation("success")
        record_speech_request("success")

        body, content_type = expose_metrics()
        text = body.decode("utf-8")
        self.assertIn("agy_http_requests_total", text)
        self.assertIn("agy_chat_completions_total", text)
        self.assertIn("agy_chat_tokens_total", text)
        self.assertIn("agy_runner_execution_seconds", text)
        self.assertIn("agy_image_generations_total", text)
        self.assertIn("agy_speech_requests_total", text)

    def test_get_metrics_endpoint(self):
        async def _run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/metrics")
                self.assertEqual(resp.status_code, 200)
                self.assertIn("text/plain", resp.headers["content-type"])
                text = resp.text
                self.assertIn("agy_http_requests_total", text)
                # Check process metrics are present
                self.assertIn("process_cpu_seconds_total", text)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
