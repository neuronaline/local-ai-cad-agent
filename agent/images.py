"""Safe normalization of user-provided visual references."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from uuid import uuid4

from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage

ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1600
# Hard ceiling on decoded pixels (header + post-decode). The previous
# 100 MP ceiling was a decompression-bomb risk: a maliciously crafted
# PNG could compress to <10 MB on disk yet decode to ~400 MB of RGB
# pixels, and ``image.draft()`` is a no-op for PNG/WebP so the old
# code path had no second-stage downsize. 10 MP RGB is ~30 MB per
# image — comfortably within budget for the 5-image cap (max ~150 MB)
# while still accepting legitimate reference photos that will be
# thumbnailed down to ``MAX_IMAGE_DIMENSION`` before persistence.
MAX_IMAGE_PIXELS = 10_000_000
# Reject image bombs at decode time across the process.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def store_images(files: list[FileStorage], project_dir: Path) -> list[Path]:
    if len(files) > 5:
        raise ValueError("Upload at most five reference images per message.")
    stored: list[Path] = []
    try:
        (project_dir / "inputs").mkdir(exist_ok=True)
        for upload in files:
            if not upload.filename:
                continue
            if upload.mimetype not in ALLOWED_MIME_TYPES:
                raise ValueError("Only PNG, JPEG, and WebP images are accepted.")
            raw = upload.read(MAX_IMAGE_BYTES + 1)
            if len(raw) > MAX_IMAGE_BYTES:
                raise ValueError("Each image must be 10 MB or smaller.")
            try:
                image = Image.open(io.BytesIO(raw), formats=["PNG", "JPEG", "WEBP"])
                # The header is parsed but pixels are not decoded yet:
                # reject any image whose declared dimensions would blow
                # past the per-image pixel cap before we even call
                # ``load()``. ``DecompressionBombError`` raised inside
                # ``load()`` covers the rarer case where header
                # dimensions undersell the actual decoded size.
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError(
                        f"Image dimensions exceed {MAX_IMAGE_PIXELS // 1_000_000} megapixels."
                    )
                # ``Image.draft`` is JPEG-only (no-op for PNG/WebP),
                # so we cannot rely on it for size reduction. Load the
                # pixels with the tight ``MAX_IMAGE_PIXELS`` guard in
                # effect (Pillow raises ``DecompressionBombError`` from
                # inside ``load()`` if the decoded buffer exceeds it).
                image.load()
                # Real downsize — works for every supported format and
                # shrinks in place to fit inside the
                # ``MAX_IMAGE_DIMENSION`` square before we hand the
                # buffer to the PNG encoder.
                image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
            except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
                raise ValueError(f"Invalid image: {upload.filename}") from error
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            target = project_dir / "inputs" / f"{uuid4().hex}.png"
            image.save(target, format="PNG", optimize=True)
            stored.append(target)
    except Exception:
        for path in stored:
            path.unlink(missing_ok=True)
        raise
    return stored


def as_chat_image(path: Path) -> dict[str, object]:
    """Encode an image as an OpenAI-compatible ``image_url`` chat content part.

    Both OpenRouter and OpenAI Chat Completions accept the same
    ``{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}``
    shape, so a single helper serves both providers.
    """
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
