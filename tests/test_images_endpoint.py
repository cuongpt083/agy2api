import asyncio
import base64
import json
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.core.security import get_api_key

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.dependency_overrides[get_api_key] = lambda: "test"
    return app


class TestImagesEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(_app())

    def test_returns_data_uri_from_workspace_out_dir(self):
        async def fake_run(prompt, model=None, output_format="json", files=None, extra_dirs=None):
            out = Path(extra_dirs[0]) / "out"
            (out / "output-1.png").write_bytes(PNG_BYTES)
            return {"response": "wrote output-1.png"}

        with patch("app.api.routes.run_agy_prompt", new=AsyncMock(side_effect=fake_run)):
            res = self.client.post(
                "/v1/images/generations",
                json={"prompt": "a cat", "response_format": "url"},
            )

        self.assertEqual(res.status_code, 200)
        item = res.json()["data"][0]
        self.assertTrue(item["url"].startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(item["url"].split(",", 1)[1]), PNG_BYTES)
        self.assertIsNone(item.get("b64_json"))

    def test_returns_502_when_agy_writes_no_image(self):
        async def fake_run(prompt, model=None, output_format="json", files=None, extra_dirs=None):
            return {"response": "I could not generate an image"}

        with patch("app.api.routes.run_agy_prompt", new=AsyncMock(side_effect=fake_run)):
            res = self.client.post(
                "/v1/images/generations",
                json={"prompt": "a cat"},
            )

        self.assertEqual(res.status_code, 502)
        body = res.json()
        self.assertEqual(body["error"]["type"], "image_generation_error")
        self.assertTrue(body["error"]["message"])

    def test_honors_n_two_files(self):
        async def fake_run(prompt, model=None, output_format="json", files=None, extra_dirs=None):
            out = Path(extra_dirs[0]) / "out"
            (out / "output-1.png").write_bytes(PNG_BYTES)
            (out / "output-2.png").write_bytes(PNG_BYTES + b"x")
            return {"response": "ok"}

        with patch("app.api.routes.run_agy_prompt", new=AsyncMock(side_effect=fake_run)):
            res = self.client.post(
                "/v1/images/generations",
                json={"prompt": "a cat", "n": 2, "response_format": "b64_json"},
            )

        self.assertEqual(res.status_code, 200)
        data = res.json()["data"]
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["b64_json"], base64.b64encode(PNG_BYTES).decode("ascii"))
        self.assertEqual(data[1]["b64_json"], base64.b64encode(PNG_BYTES + b"x").decode("ascii"))

    def test_passes_model_and_size_into_agy(self):
        captured = {}

        async def fake_run(prompt, model=None, output_format="json", files=None, extra_dirs=None):
            captured["model"] = model
            captured["prompt"] = prompt
            out = Path(extra_dirs[0]) / "out"
            (out / "output-1.png").write_bytes(PNG_BYTES)
            return {"response": "ok"}

        with patch("app.api.routes.run_agy_prompt", new=AsyncMock(side_effect=fake_run)):
            res = self.client.post(
                "/v1/images/generations",
                json={
                    "prompt": "a cat",
                    "model": "gemini-3.8-flash-high",
                    "size": "9:16",
                },
            )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(captured["model"], "gemini-3.8-flash-high")
        self.assertIn("9:16", captured["prompt"])
        self.assertIn("a cat", captured["prompt"])

    def test_caps_n_at_four_in_prompt(self):
        captured = {}

        async def fake_run(prompt, model=None, output_format="json", files=None, extra_dirs=None):
            captured["prompt"] = prompt
            out = Path(extra_dirs[0]) / "out"
            for i in range(1, 5):
                (out / f"output-{i}.png").write_bytes(PNG_BYTES)
            return {"response": "ok"}

        with patch("app.api.routes.run_agy_prompt", new=AsyncMock(side_effect=fake_run)):
            res = self.client.post(
                "/v1/images/generations",
                json={"prompt": "a cat", "n": 10},
            )

        self.assertEqual(res.status_code, 200)
        self.assertIn("output-4.png", captured["prompt"])
        self.assertNotIn("output-5.png", captured["prompt"])
        self.assertEqual(len(res.json()["data"]), 4)

    def test_writes_reference_images_into_workspace_refs(self):
        captured = {}

        async def fake_run(prompt, model=None, output_format="json", files=None, extra_dirs=None):
            captured["prompt"] = prompt
            refs = list((Path(extra_dirs[0]) / "refs").iterdir())
            captured["ref_count"] = len(refs)
            captured["ref_suffix"] = refs[0].suffix
            captured["ref_bytes"] = refs[0].read_bytes()
            captured["ref_path"] = str(refs[0])
            out = Path(extra_dirs[0]) / "out"
            (out / "output-1.png").write_bytes(PNG_BYTES)
            return {"response": "ok"}

        ref = "data:image/jpeg;base64," + base64.b64encode(b"jpeg-bytes").decode("ascii")
        with patch("app.api.routes.run_agy_prompt", new=AsyncMock(side_effect=fake_run)):
            res = self.client.post(
                "/v1/images/generations",
                json={"prompt": "cyberpunk this", "reference_images": [ref]},
            )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(captured["ref_count"], 1)
        self.assertEqual(captured["ref_suffix"], ".jpg")
        self.assertEqual(captured["ref_bytes"], b"jpeg-bytes")
        self.assertIn(captured["ref_path"], captured["prompt"])

    def test_does_not_return_filesystem_path_as_url(self):
        async def fake_run(prompt, model=None, output_format="json", files=None, extra_dirs=None):
            return {"response": "C:\\\\Users\\\\agy\\\\cat.png"}

        with patch("app.api.routes.run_agy_prompt", new=AsyncMock(side_effect=fake_run)):
            res = self.client.post(
                "/v1/images/generations",
                json={"prompt": "a cat"},
            )

        self.assertEqual(res.status_code, 502)
        self.assertNotIn("url", res.json().get("data", [{}])[0] if res.json().get("data") else {})


def _sse_payloads(body: str) -> list:
    frames = []
    for block in body.split("\n\n"):
        line = block.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            frames.append("[DONE]")
        else:
            frames.append(json.loads(payload))
    return frames


class TestImagesEndpointStream(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(_app())

    def test_stream_emits_status_then_image_before_agy_finishes(self):
        hung = {"entered": False}

        async def fake_stream(prompt, model=None, files=None, extra_dirs=None):
            out = Path(extra_dirs[0]) / "out"
            yield {"event": "init"}
            (out / "output-1.png").write_bytes(PNG_BYTES)
            yield {"event": "wrote"}
            hung["entered"] = True
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                return
            yield {"event": "too_late"}

        t0 = time.time()
        with (
            patch("app.api.routes.stream_agy_prompt", new=fake_stream),
            patch(
                "app.api.routes.run_agy_prompt",
                new=AsyncMock(side_effect=AssertionError("non-stream path")),
            ),
        ):
            res = self.client.post(
                "/v1/images/generations",
                json={"prompt": "a cat", "stream": True, "response_format": "url"},
            )
        elapsed = time.time() - t0

        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.headers["content-type"].startswith("text/event-stream"))
        self.assertLess(elapsed, 5)
        frames = _sse_payloads(res.text)
        types = [f["type"] if isinstance(f, dict) else f for f in frames]
        self.assertEqual(types[0], "status")
        self.assertEqual(frames[0]["stage"], "started")
        self.assertIn("generating", [f.get("stage") for f in frames if isinstance(f, dict)])
        image_frames = [f for f in frames if isinstance(f, dict) and f.get("type") == "image"]
        self.assertEqual(len(image_frames), 1)
        url = image_frames[0]["data"][0]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), PNG_BYTES)
        self.assertIn("done", types)
        self.assertIn("[DONE]", types)
        self.assertTrue(hung["entered"])

    def test_stream_emits_error_when_agy_writes_no_image(self):
        async def fake_stream(prompt, model=None, files=None, extra_dirs=None):
            yield {"event": "result", "result": {"response": "nope"}}

        with (
            patch("app.api.routes.stream_agy_prompt", new=fake_stream),
            patch(
                "app.api.routes.run_agy_prompt",
                new=AsyncMock(side_effect=AssertionError("non-stream path")),
            ),
        ):
            res = self.client.post(
                "/v1/images/generations",
                json={"prompt": "a cat", "stream": True},
            )

        self.assertEqual(res.status_code, 200)
        frames = _sse_payloads(res.text)
        errors = [f for f in frames if isinstance(f, dict) and f.get("type") == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error"]["type"], "image_generation_error")
