"""Model-requested view image retrieval.

Hosts the ``get_view_images`` tool. The tool never runs a sandbox build;
the rendered per-view PNGs already exist on disk after a successful
``cad_build`` (``<project>/.cad-agent/reviews/<sha256>/views/*.png``
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
import io
import json
import math
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from agent.prompt import TOOL_HINTS
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

VIEW_ALIASES: dict[str, str] = {
    "all": "all",
    "sheet": "all",
    "contact": "all",
    "contact_sheet": "all",
    "review-sheet": "all",
    "review_sheet": "all",
    "overview": "all",
    "isometric": "isometric_positive",
    "iso": "isometric_positive",
    "iso+": "isometric_positive",
    "iso_pos": "isometric_positive",
    "iso_positive": "isometric_positive",
    "isometric+": "isometric_positive",
    "isometric_positive": "isometric_positive",
    "iso-": "isometric_negative",
    "iso_neg": "isometric_negative",
    "iso_negative": "isometric_negative",
    "isometric-": "isometric_negative",
    "isometric_negative": "isometric_negative",
    "top": "z_positive",
    "+z": "z_positive",
    "bottom": "z_negative",
    "-z": "z_negative",
    "front": "y_negative",
    "-y": "y_negative",
    "back": "y_positive",
    "+y": "y_positive",
    "right": "x_positive",
    "+x": "x_positive",
    "left": "x_negative",
    "-x": "x_negative",
    "x_positive": "x_positive",
    "x_negative": "x_negative",
    "y_positive": "y_positive",
    "y_negative": "y_negative",
    "z_positive": "z_positive",
    "z_negative": "z_negative",
}

_MAX_VIEWS_PER_CALL = 8

# Rendered views are always 512x512 per ``renderer.py:_WIDTH``/``_HEIGHT``.
VIEW_PIXEL_WIDTH = 512
VIEW_PIXEL_HEIGHT = 512

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
    def get_view_images(self, requests: object = None) -> dict[str, object]:
        """Return the success-envelope data for ``get_view_images``.

        Supports flexible input: empty/None (defaults to 'all' contact sheet),
        ``{"views": ["all", "top", ...]}``, single strings, raw lists, or
        legacy ``{"images": [...]}`` view requests. Raises :class:`ValueError`
        on bad input; the dispatcher converts that into the standard ``ok:false``
        envelope via ``tool_failure``.
        """
        items = _coerce_requests(requests)
        review_root = self._current_review_dir()
        if review_root is None:
            raise ValueError(
                "No review artifacts exist for the current model.scad. "
                f"{TOOL_HINTS['IMAGE_RUN_CAD_BUILD_FIRST']}"
            )
        manifest = _read_manifest(review_root)
        if manifest is None:
            raise ValueError(
                "Review directory is missing manifest.json. "
                f"{TOOL_HINTS['IMAGE_RUN_CAD_BUILD_AGAIN']}"
            )

        # De-duplicate identical view requests so the model can
        # ask for the same view twice in one call without producing
        # duplicate image_url parts (which would inflate every later
        # payload without adding information).
        deduped = _deduplicate(items)
        if len(deduped) > _MAX_VIEWS_PER_CALL:
            raise ValueError(
                f"Too many view requests ({len(deduped)}). At most {_MAX_VIEWS_PER_CALL} views can be requested per call."
            )

        out_images: list[dict[str, object]] = []
        for item in deduped:
            entry = self._resolve_one(review_root, manifest, item)
            if entry is not None:
                out_images.append(entry)

        if not out_images:
            raise ValueError(
                "None of the requested views could be produced. "
                f"{TOOL_HINTS['IMAGE_REFRESH_REVIEW']}"
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

    def _crop_view(
        self,
        source_path: Path,
        view_id: str,
        crop: list[float],
        review_root: Path,
        rel_dir: str = "",
    ) -> dict[str, object] | None:
        try:
            with Image.open(source_path) as img:
                img_w, img_h = img.size
                ymin, xmin, ymax, xmax = crop
                box = (
                    max(0, round(xmin * img_w)),
                    max(0, round(ymin * img_h)),
                    min(img_w, round(xmax * img_w)),
                    min(img_h, round(ymax * img_h)),
                )
                if box[2] <= box[0] or box[3] <= box[1]:
                    return None
                cropped = img.crop(box)
                buffer = io.BytesIO()
                cropped.save(buffer, format="PNG", compress_level=1)
                crop_bytes = buffer.getvalue()
                crop_sha = hashlib.sha256(crop_bytes).hexdigest()
                crop_filename = f"{view_id}_crop_{crop_sha[:12]}.png"
                rel_path = f"{rel_dir}/{crop_filename}" if rel_dir else crop_filename
                target_path = review_root / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if not target_path.is_file() or target_path.stat().st_size == 0:
                    target_path.write_bytes(crop_bytes)
                return {
                    "view": f"{view_id}_crop",
                    "cropped": True,
                    "crop": crop,
                    "path": rel_path,
                    "sha256": crop_sha,
                    "width": cropped.width,
                    "height": cropped.height,
                }
        except (OSError, ValueError):
            return None

    def _resolve_one(
        self,
        review_root: Path,
        manifest: dict,
        item: dict[str, object],
    ) -> dict[str, object] | None:
        raw_view = str(item.get("view") or "").strip().lower()
        canonical = VIEW_ALIASES.get(raw_view)
        if canonical is None:
            raise ValueError(
                f"Unknown view: {item.get('view')!r}. "
                "Valid views: 'all', 'isometric', 'top', 'bottom', 'front', 'back', 'left', 'right', 'isometric_negative'."
            )

        crop = _validate_crop(item.get("crop"))

        if canonical == "all":
            contact_entry = manifest.get("contact_sheet") if isinstance(manifest, dict) else None
            expected_sha = (
                contact_entry.get("image_sha256")
                if isinstance(contact_entry, dict)
                else None
            )
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                expected_sha = None
            sheet_path = review_root / "review-sheet.png"
            if not _is_png(sheet_path, expected_sha):
                return None
            if not expected_sha:
                expected_sha = _hash_file(sheet_path)
            width = (
                int(contact_entry.get("width", 2048))
                if isinstance(contact_entry, dict) and "width" in contact_entry
                else 2048
            )
            height = (
                int(contact_entry.get("height", 1024))
                if isinstance(contact_entry, dict) and "height" in contact_entry
                else 1024
            )

            if crop:
                return self._crop_view(
                    source_path=sheet_path,
                    view_id="all",
                    crop=crop,
                    review_root=review_root,
                    rel_dir="",
                )

            return {
                "view": "all",
                "cropped": False,
                "path": "review-sheet.png",
                "sha256": expected_sha,
                "width": width,
                "height": height,
            }

        view_id = canonical
        view_entry_expected_sha = _view_hash(manifest, view_id)
        if view_entry_expected_sha is None:
            # Manifest does not know about this view id (older review
            # or partial render). Skip rather than fail the whole call.
            return None
        expected_sha = view_entry_expected_sha
        view_path = review_root / _VIEWS_DIRNAME / f"{view_id}.png"
        if not _is_png(view_path, expected_sha):
            return None

        if crop:
            return self._crop_view(
                source_path=view_path,
                view_id=view_id,
                crop=crop,
                review_root=review_root,
                rel_dir=_VIEWS_DIRNAME,
            )

        actual_w = VIEW_PIXEL_WIDTH
        actual_h = VIEW_PIXEL_HEIGHT
        if isinstance(manifest, dict):
            for v in manifest.get("views", []):
                if isinstance(v, dict) and v.get("view_id") == view_id:
                    actual_w = int(v.get("width", VIEW_PIXEL_WIDTH) or VIEW_PIXEL_WIDTH)
                    actual_h = int(v.get("height", VIEW_PIXEL_HEIGHT) or VIEW_PIXEL_HEIGHT)
                    break

        return {
            "view": view_id,
            "cropped": False,
            "path": f"{_VIEWS_DIRNAME}/{view_id}.png",
            "sha256": expected_sha,
            "width": actual_w,
            "height": actual_h,
        }


# ---------------------------------------------------------------------------
# Module-level helpers (pure, no I/O dependencies on the tool instance)
# ---------------------------------------------------------------------------


def _validate_crop(crop_val: object) -> list[float] | None:
    if crop_val is None:
        return None
    if not isinstance(crop_val, (list, tuple)) or len(crop_val) != 4:
        raise ValueError("crop must be a list of 4 numbers [ymin, xmin, ymax, xmax].")
    try:
        ymin, xmin, ymax, xmax = [float(v) for v in crop_val]
    except (ValueError, TypeError):
        raise ValueError("crop coordinates must be numeric [ymin, xmin, ymax, xmax].")
    if math.isnan(ymin) or math.isnan(xmin) or math.isnan(ymax) or math.isnan(xmax):
        raise ValueError("crop coordinates must not contain NaN.")
    if not (0.0 <= ymin < ymax <= 1.0) or not (0.0 <= xmin < xmax <= 1.0):
        raise ValueError(
            f"Invalid crop bounds {[ymin, xmin, ymax, xmax]}; coordinates must satisfy 0.0 <= ymin < ymax <= 1.0 and 0.0 <= xmin < xmax <= 1.0."
        )
    if (ymax - ymin < 0.002) or (xmax - xmin < 0.002):
        raise ValueError(
            f"Crop box {[ymin, xmin, ymax, xmax]} is too small to contain visible pixels."
        )
    return [round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)]


def _coerce_list(raw_list: list, label: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for index, item in enumerate(raw_list):
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                items.append({"view": cleaned})
        elif isinstance(item, dict) and "view" in item:
            entry = dict(item)
            if "crop" in entry:
                entry["crop"] = _validate_crop(entry["crop"])
            items.append(entry)
        else:
            raise ValueError(
                f"{label}[{index}] must be a view name string or object with 'view'."
            )
    return items if items else [{"view": "all"}]


def _coerce_requests(requests: object) -> list[dict[str, object]]:
    """Coerce various input shapes into a normalized list of view requests."""
    if requests is None or requests == "" or requests == [] or requests == {}:
        return [{"view": "all"}]

    if isinstance(requests, str):
        cleaned = requests.strip()
        return [{"view": cleaned}] if cleaned else [{"view": "all"}]

    if isinstance(requests, list):
        return _coerce_list(requests, "Request item")

    if isinstance(requests, dict):
        if not requests:
            return [{"view": "all"}]

        global_crop = _validate_crop(requests.get("crop"))

        items: list[dict[str, object]] = []
        # 1. Handle "views" list/string
        if "views" in requests:
            raw_views = requests["views"]
            if raw_views is None or raw_views == [] or raw_views == "":
                items = [{"view": "all"}]
            elif isinstance(raw_views, str):
                cleaned = raw_views.strip()
                items = [{"view": cleaned}] if cleaned else [{"view": "all"}]
            elif isinstance(raw_views, list):
                items = _coerce_list(raw_views, "views")
            else:
                raise ValueError("'views' must be a list of view names or a single view name.")
        # 2. Handle "images" list (legacy schema compatibility)
        elif "images" in requests:
            raw_images = requests["images"]
            if raw_images is None or raw_images == [] or raw_images == "":
                items = [{"view": "all"}]
            elif not isinstance(raw_images, list):
                raise ValueError("'images' must be a list of view requests.")
            else:
                items = _coerce_list(raw_images, "images")
        # 3. Handle single "view" key
        elif "view" in requests:
            raw_view = requests["view"]
            if not isinstance(raw_view, str) or not raw_view.strip():
                raise ValueError("'view' must be a non-empty string.")
            items = [{"view": raw_view.strip()}]
        # 4. Handle crop-only request (defaults to contact sheet "all")
        elif global_crop:
            items = [{"view": "all"}]
        else:
            raise ValueError(
                "Invalid arguments for get_view_images. Pass 'views' array or leave empty for contact sheet."
            )

        if global_crop:
            for item in items:
                if "crop" not in item:
                    item["crop"] = global_crop

        return items

    raise ValueError("Invalid request format for get_view_images.")


def _deduplicate(items: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for item in items:
        key = _item_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _item_key(item: dict[str, object]) -> str:
    raw_view = str(item.get("view") or "").strip().lower()
    canonical = VIEW_ALIASES.get(raw_view, raw_view)
    crop = item.get("crop")
    if crop and isinstance(crop, (list, tuple)) and len(crop) == 4:
        return f"{canonical}:{tuple(crop)}"
    return canonical


def _read_manifest(review_root: Path) -> dict | None:
    path = review_root / _MANIFEST_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
