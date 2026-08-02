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
                image = Image.open(io.BytesIO(raw))
                image.load()
            except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
                raise ValueError(f"Invalid image: {upload.filename}") from error
            image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
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


def as_openrouter_image(path: Path) -> dict[str, object]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
