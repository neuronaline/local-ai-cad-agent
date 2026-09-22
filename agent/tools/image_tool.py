"""Model-requested view image retrieval.

Hosts the ``get_view_images`` tool. The tool never runs a sandbox build;
the rendered per-view PNGs already exist on disk after a successful
``cad_build_and_verify`` (``<project>/.cad-agent/reviews/<sha256>/views/*.png``
plus ``manifest.json``). The tool validates each requested view against
the manifest's ``image_sha256`` and, when a ``crop`` is requested, derives a
cached sub-image with Pillow so repeated requests are free.

Returning the resulting paths in the success envelope lets the dispatcher
attach them as ``image_url`` content parts via ``relocate_tool_images``,
giving the model a stable, append-only permanent visual memory without
ever rewriting the cached prompt prefix.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

from PIL import Image

from agent.io import atomic_write_bytes
from agent.review_paths import review_dir
from agent.revisions import compute_model_sha256
from agent.tool_results import _is_png, _view_hash

# Canonical eight views from ``renderer.py:VIEWS`` mirrored as a tuple of
# ids so an unknown value can be rejected with a single membership test.
# The rendered renderer always writes ``<review_dir>/views/<view_id>.png``
# for each id in this list (see ``cad_scripts/renderer.py:VIEWS``).
CANONICAL_VIEW_IDS: tuple[str, ...] = (
    "x_positive",
    "x_negative",
    "y_positive",
    "y_negative",
    "z_positive",
    "z_negative",
    "isometric_positive",
    "isometric_negative",
)

# Rendered views are always 512x512 per ``renderer.py:_WIDTH``/``_HEIGHT``.
VIEW_PIXEL_WIDTH = 512
VIEW_PIXEL_HEIGHT = 512

# Crops whose longer side drops below this threshold are rejected: tiny
# crops are useless as model memory and signal a degenerate geometry.
_MIN_CROP_PIXELS = 8

# Crops are cached under ``<review_dir>/crops/`` with normalized
# coordinates in the filename (4 decimals) so identical requests reuse
# the same file instead of regenerating it.
_CROPS_DIRNAME = "crops"
_VIEWS_DIRNAME = "views"
_MANIFEST_FILENAME = "manifest.json"


class ImageTool:
    """Resolve ``get_view_images`` requests to cached PNGs on disk."""

    def __init__(
        self,
        project_dir: Path,
        publish: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.project_dir = project_dir.resolve()
        self._publish = publish
        self._call_id = ""

    # ------------------------------------------------------------------
    # Mirror ``CadTool.with_call_id`` so the dispatcher can propagate the
    # call id into any future publish() / activity-log entries without
    # restating the same plumbing here.
    # ------------------------------------------------------------------
    def with_call_id(self, call_id: str) -> ImageTool:
        self._call_id = call_id or ""
        return self

    # ------------------------------------------------------------------
    # Public tool entry point
    # ------------------------------------------------------------------
    def get_view_images(self, requests: object) -> dict[str, object]:
        """Return the success-envelope data for ``get_view_images``.

        ``requests`` is a list of dicts shaped like the JSON-schema items
        (``{"view": "<id>", "crop": {"x": ..., "y": ..., "width": ...,
        "height": ...}}`` or the full view without ``crop``). Raises
        :class:`ValueError` on bad input; the dispatcher converts that
        into the standard ``ok:false`` envelope via ``tool_failure``.
        """
        items = _coerce_requests(requests)
        review_root = self._current_review_dir()
        if review_root is None:
            raise ValueError(
                "No review artifacts exist for the current model.scad. "
                "Run cad_build_and_verify first."
            )
        manifest = _read_manifest(review_root)
        if manifest is None:
            raise ValueError(
                "Review directory is missing manifest.json. "
                "Run cad_build_and_verify again."
            )

        # De-duplicate identical (view, crop) requests so the model can
        # ask for the same view twice in one call without producing
        # duplicate image_url parts (which would inflate every later
        # payload without adding information).
        deduped = _deduplicate(items)

        out_images: list[dict[str, object]] = []
        for item in deduped:
            entry = self._resolve_one(review_root, manifest, item)
            if entry is not None:
                out_images.append(entry)

        if not out_images:
            raise ValueError(
                "None of the requested views could be produced. "
                "Run cad_build_and_verify to refresh the review."
            )

        review_sha = review_root.name
        return {
            "images": out_images,
            "review_sha256": review_sha,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _current_review_dir(self) -> Path | None:
        """Return the review dir for the *current* ``model.scad``.

        Hashing the file means a stale review after a model edit
        surfaces as a clean ``ValueError`` rather than silently
        returning PNGs of the previous geometry.
        """
        model_sha = compute_model_sha256(self.project_dir)
        if model_sha is None:
            return None
        candidate = review_dir(self.project_dir, model_sha)
        if not candidate.is_dir():
            return None
        return candidate

    def _resolve_one(
        self,
        review_root: Path,
        manifest: dict,
        item: dict[str, object],
    ) -> dict[str, object] | None:
        view_id = str(item.get("view") or "")
        if view_id not in CANONICAL_VIEW_IDS:
            raise ValueError(
                f"Unknown view id: {view_id!r}. "
                f"Must be one of {list(CANONICAL_VIEW_IDS)}."
            )
        view_entry_expected_sha = _view_hash(manifest, view_id)
        if view_entry_expected_sha is None:
            # Manifest does not know about this view id (older review
            # or partial render). Skip rather than fail the whole call.
            return None
        expected_sha = view_entry_expected_sha
        view_path = review_root / _VIEWS_DIRNAME / f"{view_id}.png"
        if not _is_png(view_path, expected_sha):
            return None

        crop = item.get("crop")
        if crop is None:
            return {
                "view": view_id,
                "cropped": False,
                "path": f"{_VIEWS_DIRNAME}/{view_id}.png",
                "sha256": expected_sha,
                "width": VIEW_PIXEL_WIDTH,
                "height": VIEW_PIXEL_HEIGHT,
            }

        crop_box, normalized = _normalize_crop(crop)
        crops_dir = review_root / _CROPS_DIRNAME
        crops_dir.mkdir(parents=True, exist_ok=True)
        crop_filename = (
            f"{view_id}-{normalized[0]:.4f}-{normalized[1]:.4f}-"
            f"{normalized[2]:.4f}-{normalized[3]:.4f}.png"
        )
        crop_path = crops_dir / crop_filename

        # Cheap existence probe; a full decode/hash check on a missing
        # file would crash before ``_is_png`` returned False.
        if not crop_path.is_file() or not _is_png(crop_path, _hash_file(crop_path)):
            _write_crop(view_path, crop_box, crop_path)

        sha = _hash_file(crop_path)
        width = crop_box[2] - crop_box[0]
        height = crop_box[3] - crop_box[1]
        return {
            "view": view_id,
            "cropped": True,
            "crop": {
                "x": normalized[0],
                "y": normalized[1],
                "width": normalized[2],
                "height": normalized[3],
            },
            "path": f"{_CROPS_DIRNAME}/{crop_filename}",
            "sha256": sha,
            "width": width,
            "height": height,
        }


# ---------------------------------------------------------------------------
# Module-level helpers (pure, no I/O dependencies on the tool instance)
# ---------------------------------------------------------------------------


def _coerce_requests(requests: object) -> list[dict[str, object]]:
    if requests is None:
        return []
    if not isinstance(requests, list):
        raise ValueError("'images' must be a list of view requests.")
    if not requests:
        raise ValueError("'images' must contain at least one view request.")
    if len(requests) > 4:
        raise ValueError("'images' accepts at most 4 view requests per call.")
    items: list[dict[str, object]] = []
    for index, raw in enumerate(requests):
        if not isinstance(raw, dict):
            raise ValueError(
                f"images[{index}] must be an object with a 'view' field."
            )
        items.append(raw)
    return items


def _deduplicate(items: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple] = set()
    out: list[dict[str, object]] = []
    for item in items:
        key = _item_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _item_key(item: dict[str, object]) -> tuple:
    crop = item.get("crop")
    if not isinstance(crop, dict):
        return (str(item.get("view") or ""), None)
    return (
        str(item.get("view") or ""),
        (
            float(crop.get("x") or 0.0),
            float(crop.get("y") or 0.0),
            float(crop.get("width") or 0.0),
            float(crop.get("height") or 0.0),
        ),
    )


def _normalize_crop(
    crop: object,
) -> tuple[tuple[int, int, int, int], tuple[float, float, float, float]]:
    """Return ``((left, top, right, bottom), (x, y, w, h))``.

    Clamps the requested normalized rectangle into the 512x512 canvas
    and rejects degenerate rectangles (< 8 px on either side).
    """
    if not isinstance(crop, dict):
        raise ValueError("crop must be an object with x, y, width, height.")
    try:
        x = float(crop.get("x"))
        y = float(crop.get("y"))
        width = float(crop.get("width"))
        height = float(crop.get("height"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "crop requires numeric x, y, width, height fields."
        ) from error
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError("crop x and y must lie in [0, 1].")
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        raise ValueError("crop width and height must lie in (0, 1].")
    if x + width > 1.0:
        width = max(0.0, 1.0 - x)
    if y + height > 1.0:
        height = max(0.0, 1.0 - y)

    left = round(x * VIEW_PIXEL_WIDTH)
    top = round(y * VIEW_PIXEL_HEIGHT)
    right = round((x + width) * VIEW_PIXEL_WIDTH)
    bottom = round((y + height) * VIEW_PIXEL_HEIGHT)
    # Clamp into the canvas so a single rounded pixel off-by-one does
    # not escape the image bounds.
    left = max(0, min(left, VIEW_PIXEL_WIDTH))
    top = max(0, min(top, VIEW_PIXEL_HEIGHT))
    right = max(left, min(right, VIEW_PIXEL_WIDTH))
    bottom = max(top, min(bottom, VIEW_PIXEL_HEIGHT))
    if (right - left) < _MIN_CROP_PIXELS or (bottom - top) < _MIN_CROP_PIXELS:
        raise ValueError(
            "Requested crop is degenerate (< 8 px on a side); "
            "widen the area or omit the crop."
        )
    normalized = (x, y, width, height)
    return (left, top, right, bottom), normalized


def _write_crop(
    source: Path,
    box: tuple[int, int, int, int],
    target: Path,
) -> None:
    """Pillow-crop ``source`` to ``box`` and atomically persist to ``target``."""
    with Image.open(source, formats=["PNG"]) as image:
        image.load()
        cropped = image.crop(box)
        buffer_bytes = _png_bytes(cropped)
    atomic_write_bytes(target, buffer_bytes)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _read_manifest(review_root: Path) -> dict | None:
    path = review_root / _MANIFEST_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
