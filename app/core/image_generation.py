import base64
import os
import re
import shutil
import tempfile
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
_PATH_RE = re.compile(
    r"(?:[A-Za-z]:)?(?:[/\\][^\s'\"`<>|*?]+\.(?:png|jpe?g|webp|gif))",
    re.IGNORECASE,
)


def _scan_out_dir(out_dir: str) -> list[str]:
    paths: list[str] = []
    if not os.path.isdir(out_dir):
        return paths
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path) and Path(path).suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
    return paths


def snapshot_image_sizes(out_dir: str) -> dict[str, int]:
    return {path: os.path.getsize(path) for path in _scan_out_dir(out_dir)}


def stable_output_images(out_dir: str, n: int, previous_sizes: dict[str, int]) -> list[str]:
    current = snapshot_image_sizes(out_dir)
    stable = [
        path
        for path, size in current.items()
        if size > 0 and previous_sizes.get(path) == size
    ]
    if len(stable) >= n:
        return stable[:n]
    return []


def collect_generated_images(out_dir: str, agy_text: str, n: int) -> list[str]:
    n = max(n, 0)
    paths = _scan_out_dir(out_dir)
    if len(paths) >= n:
        return paths[:n]

    os.makedirs(out_dir, exist_ok=True)
    seen = {os.path.normcase(os.path.abspath(p)) for p in paths}
    for match in _PATH_RE.findall(agy_text or ""):
        src = os.path.abspath(match)
        if not os.path.isfile(src) or Path(src).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = os.path.normcase(src)
        if key in seen:
            continue
        dest = os.path.join(out_dir, Path(src).name)
        if os.path.abspath(src) != os.path.abspath(dest):
            shutil.copy2(src, dest)
            src = dest
        paths.append(src)
        seen.add(os.path.normcase(os.path.abspath(src)))
        if len(paths) >= n:
            break
    return paths[:n]


def encode_image_objects(paths: list[str], response_format: str) -> list[dict]:
    objects: list[dict] = []
    for path in paths:
        with open(path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("ascii")
        if response_format == "b64_json":
            objects.append({"b64_json": b64, "url": None})
        else:
            ext = Path(path).suffix.lower()
            mime = MIME_BY_EXT.get(ext, "image/png")
            objects.append({"url": f"data:{mime};base64,{b64}", "b64_json": None})
    return objects


MAX_N = 4
MAX_REFERENCE_IMAGES = 3


def clamp_n(n: int | None) -> int:
    if n is None or n < 1:
        return 1
    return min(n, MAX_N)


def agy_text_from_response(agy_response) -> str:
    if isinstance(agy_response, dict):
        return str(
            agy_response.get("response")
            or agy_response.get("text")
            or agy_response.get("content")
            or ""
        )
    return str(agy_response)


def build_image_prompt(
    prompt: str,
    out_dir: str,
    n: int,
    size: str | None,
    ref_paths: list[str],
) -> str:
    names = ", ".join(f"output-{i}.png" for i in range(1, n + 1))
    lines = [
        "Generate image(s) for this prompt:",
        prompt,
        "",
        f"Write exactly {n} image file(s) into this output directory: {out_dir}",
        f"Use these exact filenames: {names}",
        "Do not write images anywhere else.",
        "After writing the files, reply with a short confirmation. Do not return the image as text.",
    ]
    if size:
        lines.append(f"Target size or aspect ratio: {size}")
    if ref_paths:
        lines.append("Use these reference images:")
        lines.extend(f"- {p}" for p in ref_paths)
    return "\n".join(lines)


class ImageWorkspace:
    def __init__(self) -> None:
        self.root = tempfile.mkdtemp(prefix="agy2api-img-")
        self.refs_dir = os.path.join(self.root, "refs")
        self.out_dir = os.path.join(self.root, "out")
        os.makedirs(self.refs_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

    def add_reference(self, data_uri: str, index: int) -> str:
        payload = data_uri
        ext = ".png"
        if payload.startswith("data:"):
            header, _, payload = payload.partition(",")
            mime = header[5:].split(";")[0].strip().lower()
            if mime in ("image/jpeg", "image/jpg"):
                ext = ".jpg"
            elif mime == "image/webp":
                ext = ".webp"
            elif mime == "image/gif":
                ext = ".gif"
        path = os.path.join(self.refs_dir, f"ref_{index}{ext}")
        Path(path).write_bytes(base64.b64decode(payload))
        return path

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
