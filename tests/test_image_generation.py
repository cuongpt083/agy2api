import os
import tempfile
import unittest
from pathlib import Path

from app.core.image_generation import (
    ImageWorkspace,
    agy_text_from_response,
    build_image_prompt,
    clamp_n,
    collect_generated_images,
    encode_image_objects,
    snapshot_image_sizes,
    stable_output_images,
)
from app.core.agy_runner import build_agy_invocation

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class TestCollectGeneratedImages(unittest.TestCase):
    def test_returns_png_written_to_out_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "out")
            os.makedirs(out_dir)
            target = os.path.join(out_dir, "output-1.png")
            Path(target).write_bytes(PNG_BYTES)

            paths = collect_generated_images(out_dir, agy_text="done", n=1)

            self.assertEqual(paths, [target])

    def test_copies_fallback_path_from_agy_text_when_out_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "out")
            os.makedirs(out_dir)
            stray = os.path.join(tmp, "elsewhere.png")
            Path(stray).write_bytes(PNG_BYTES)

            paths = collect_generated_images(
                out_dir,
                agy_text=f"Saved the image to {stray}",
                n=1,
            )

            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].startswith(out_dir))
            self.assertTrue(os.path.isfile(paths[0]))
            self.assertEqual(Path(paths[0]).read_bytes(), PNG_BYTES)


class TestEncodeImageObjects(unittest.TestCase):
    def test_url_format_uses_data_uri_with_jpeg_mime(self):
        import base64

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shot.jpg")
            Path(path).write_bytes(b"jpeg-bytes")
            objects = encode_image_objects([path], "url")
            self.assertEqual(len(objects), 1)
            self.assertEqual(
                objects[0]["url"],
                "data:image/jpeg;base64," + base64.b64encode(b"jpeg-bytes").decode("ascii"),
            )
            self.assertIsNone(objects[0].get("b64_json"))

    def test_b64_json_format_returns_raw_base64(self):
        import base64

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shot.png")
            Path(path).write_bytes(PNG_BYTES)
            objects = encode_image_objects([path], "b64_json")
            self.assertEqual(objects[0]["b64_json"], base64.b64encode(PNG_BYTES).decode("ascii"))
            self.assertIsNone(objects[0].get("url"))


class TestClampN(unittest.TestCase):
    def test_caps_at_four_and_floors_at_one(self):
        self.assertEqual(clamp_n(None), 1)
        self.assertEqual(clamp_n(0), 1)
        self.assertEqual(clamp_n(2), 2)
        self.assertEqual(clamp_n(10), 4)


class TestBuildImagePrompt(unittest.TestCase):
    def test_instructs_agy_to_write_named_files_in_out_dir(self):
        text = build_image_prompt(
            prompt="orange cat",
            out_dir="/tmp/job/out",
            n=2,
            size="9:16",
            ref_paths=["/tmp/job/refs/ref_0.png"],
        )
        self.assertIn("orange cat", text)
        self.assertIn("/tmp/job/out", text)
        self.assertIn("output-1.png", text)
        self.assertIn("output-2.png", text)
        self.assertIn("9:16", text)
        self.assertIn("/tmp/job/refs/ref_0.png", text)


class TestAgyTextFromResponse(unittest.TestCase):
    def test_prefers_response_then_text(self):
        self.assertEqual(agy_text_from_response({"response": "a", "text": "b"}), "a")
        self.assertEqual(agy_text_from_response({"text": "b"}), "b")
        self.assertEqual(agy_text_from_response("plain"), "plain")


class TestImageWorkspace(unittest.TestCase):
    def test_writes_reference_and_exposes_out_dir(self):
        import base64

        ws = ImageWorkspace()
        try:
            self.assertTrue(os.path.isdir(ws.out_dir))
            self.assertTrue(os.path.isdir(ws.refs_dir))
            payload = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
            path = ws.add_reference(payload, 0)
            self.assertTrue(path.startswith(ws.refs_dir))
            self.assertEqual(Path(path).read_bytes(), PNG_BYTES)
        finally:
            root = ws.root
            ws.cleanup()
            self.assertFalse(os.path.exists(root))


class TestBuildAgyInvocationExtraDirs(unittest.TestCase):
    def test_adds_extra_add_dir_without_putting_prompt_on_argv(self):
        extra = tempfile.mkdtemp(prefix="agy2api-extra-")
        inv = build_agy_invocation("secret-prompt", "gemini-flash", "json", extra_dirs=[extra])
        try:
            add_dirs = [inv.cmd[i + 1] for i, a in enumerate(inv.cmd) if a == "--add-dir"]
            self.assertIn(inv.prompt_dir, add_dirs)
            self.assertIn(extra, add_dirs)
            self.assertNotIn("secret-prompt", " ".join(inv.cmd))
            self.assertEqual(inv.cmd[inv.cmd.index("--model") + 1], "gemini-flash")
        finally:
            inv.cleanup()
            os.rmdir(extra)


class TestStableOutputImages(unittest.TestCase):
    def test_requires_unchanged_size_across_two_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "out")
            os.makedirs(out_dir)
            target = os.path.join(out_dir, "output-1.png")
            Path(target).write_bytes(PNG_BYTES)

            first = snapshot_image_sizes(out_dir)
            self.assertEqual(stable_output_images(out_dir, n=1, previous_sizes={}), [])
            self.assertEqual(stable_output_images(out_dir, n=1, previous_sizes=first), [target])

    def test_growing_file_is_not_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "out")
            os.makedirs(out_dir)
            target = os.path.join(out_dir, "output-1.png")
            Path(target).write_bytes(PNG_BYTES[:10])
            prev = snapshot_image_sizes(out_dir)
            Path(target).write_bytes(PNG_BYTES)
            self.assertEqual(stable_output_images(out_dir, n=1, previous_sizes=prev), [])
