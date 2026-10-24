"""Deterministic 3D print profile recommendations from mesh printability telemetry."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from agent.revisions import compute_model_sha256
from agent.tools.cad_scripts.renderer import load_stl
from agent.tools.cad_scripts.runner import mesh_metrics

ALLOWED_MATERIALS: tuple[str, ...] = ("PLA", "PETG", "ASA", "ABS", "TPU")
ALLOWED_INFILL_PATTERNS: tuple[str, ...] = (
    "gyroid",
    "grid",
    "cubic",
    "adaptivecubic",
    "honeycomb",
    "lightning",
)
ALLOWED_SUPPORT_TYPES: tuple[str, ...] = ("normal", "tree")
ALLOWED_PRIORITIES: tuple[str, ...] = ("auto", "strength", "quality", "speed")

_DEFAULTS: dict[str, Any] = {
    "material": "PLA",
    "layer_height_mm": 0.20,
    "first_layer_height_mm": 0.20,
    "wall_loops": 3,
    "top_shell_layers": 4,
    "bottom_shell_layers": 4,
    "infill_density_percent": 20,
    "infill_pattern": "gyroid",
    "support_enabled": False,
    "support_type": "normal",
    "support_threshold_angle": 45,
    "brim_enabled": False,
    "brim_width_mm": 6,
    "part_category": "General Purpose",
    "rationale": "",
}

# Design-intent keywords matched as whole words in model.scad and user messages.
_MECHANICAL_WORDS = (
    "bracket", "mount", "gear", "hinge", "clamp", "hook", "load", "bearing", "screw",
    "bolt", "torque", "stress", "chassis", "housing", "flange", "shaft", "adapter",
    "spacer", "bushing", "pulley", "lever", "arm", "frame", "joint",
)
_DETAIL_WORDS = (
    "miniature", "figurine", "statue", "detailed", "art", "sculpture", "jewelry",
    "ornament", "logo", "badge", "coin", "keychain",
)
_ENCLOSURE_WORDS = (
    "box", "case", "enclosure", "cover", "lid", "tray", "holder", "organizer", "stand",
    "cup", "funnel", "bin", "drawer",
)
_FLEXIBLE_WORDS = (
    "flexible", "tpu", "bendable", "rubber", "gasket", "seal", "bumper", "dampener",
    "grip", "cushion", "tire", "tyre", "oring",
)
# ABS/ASA mentions also map to PETG: similar heat/UV resistance without the
# enclosure-dependent warping. ABS/ASA stay available via the material override.
_PETG_WORDS = (
    "outdoor", "sunlight", "weather", "waterproof", "weatherproof", "petg", "uv", "heat",
    "abs", "asa",
)

_PRIORITY_PRESETS: dict[str, dict[str, Any]] = {
    "strength": {
        "layer_height_mm": 0.20, "wall_loops": 4, "top_shell_layers": 5, "bottom_shell_layers": 5,
        "infill_density_percent": 40, "infill_pattern": "gyroid",
        "part_category": "Mechanical / Load-Bearing",
    },
    "quality": {
        "layer_height_mm": 0.12, "wall_loops": 3, "top_shell_layers": 5, "bottom_shell_layers": 4,
        "infill_density_percent": 20, "infill_pattern": "gyroid",
        "part_category": "Detailed / Aesthetic",
    },
    "speed": {
        "layer_height_mm": 0.28, "wall_loops": 2, "top_shell_layers": 3, "bottom_shell_layers": 3,
        "infill_density_percent": 15, "infill_pattern": "grid",
        "part_category": "Draft / Rapid Prototype",
    },
    "standard": {
        "layer_height_mm": 0.20, "wall_loops": 3, "top_shell_layers": 4, "bottom_shell_layers": 4,
        "infill_density_percent": 20, "infill_pattern": "gyroid",
        "part_category": "General Purpose",
    },
}


def _number(raw: dict[str, Any], key: str, lo: float, hi: float) -> float:
    """Read ``raw[key]`` as a finite float clamped to ``[lo, hi]`` (default on bad input)."""
    try:
        value = float(raw.get(key, _DEFAULTS[key]))
    except (TypeError, ValueError):
        value = float(_DEFAULTS[key])
    if not math.isfinite(value):
        value = float(_DEFAULTS[key])
    return min(hi, max(lo, value))


def _choice(raw: dict[str, Any], key: str, allowed: tuple[str, ...]) -> str:
    value = str(raw.get(key, _DEFAULTS[key])).strip()
    value = value.upper() if key == "material" else value.lower()
    return value if value in allowed else _DEFAULTS[key]


def _text(raw: dict[str, Any], key: str, limit: int) -> str:
    return re.sub(r"\s+", " ", str(raw.get(key) or "")).strip()[:limit]


def sanitize_print_profile(raw: Any) -> dict[str, Any]:
    """Clamp and validate a (possibly untrusted) print profile into safe slicer bounds.

    Every value written into 3MF slicer configs passes through here, so the
    serializers never see free-form input beyond the two text fields.
    """
    if not isinstance(raw, dict):
        raw = {}
    brim_enabled = raw.get("brim_enabled") is True
    return {
        "material": _choice(raw, "material", ALLOWED_MATERIALS),
        "layer_height_mm": round(_number(raw, "layer_height_mm", 0.08, 0.28), 2),
        "first_layer_height_mm": round(_number(raw, "first_layer_height_mm", 0.16, 0.28), 2),
        "wall_loops": round(_number(raw, "wall_loops", 2, 6)),
        "top_shell_layers": round(_number(raw, "top_shell_layers", 3, 7)),
        "bottom_shell_layers": round(_number(raw, "bottom_shell_layers", 3, 7)),
        "infill_density_percent": round(_number(raw, "infill_density_percent", 0, 100) / 5) * 5,
        "infill_pattern": _choice(raw, "infill_pattern", ALLOWED_INFILL_PATTERNS),
        "support_enabled": raw.get("support_enabled") is True,
        "support_type": _choice(raw, "support_type", ALLOWED_SUPPORT_TYPES),
        "support_threshold_angle": round(_number(raw, "support_threshold_angle", 30, 60)),
        "brim_enabled": brim_enabled,
        "brim_width_mm": round(_number(raw, "brim_width_mm", 4, 15)) if brim_enabled else 0,
        "part_category": _text(raw, "part_category", 60) or _DEFAULTS["part_category"],
        "rationale": _text(raw, "rationale", 500),
    }


def _has_any_word(text: str, words: tuple[str, ...]) -> bool:
    # ``_`` separates words in SCAD identifiers (``case_width`` -> ``case``).
    pattern = r"\b(" + "|".join(re.escape(w) for w in words) + r")\b"
    return bool(re.search(pattern, text.replace("_", " ")))


def build_heuristic_print_profile(
    metrics: dict[str, Any],
    code_snippet: str = "",
    priority: str = "auto",
    material_override: str = "auto",
    user_context: str = "",
) -> dict[str, Any]:
    """Deterministically compute tuned print settings from geometry telemetry and design intent."""
    pri = priority.strip().lower()
    if pri not in ALLOWED_PRIORITIES:
        pri = "auto"

    user_text = user_context.lower()
    code_text = code_snippet.lower()
    combined_text = f"{code_text} {user_text}"
    is_mechanical = _has_any_word(combined_text, _MECHANICAL_WORDS)
    is_detail = _has_any_word(combined_text, _DETAIL_WORDS)
    is_enclosure = _has_any_word(combined_text, _ENCLOSURE_WORDS)
    is_flexible = _has_any_word(combined_text, _FLEXIBLE_WORDS)

    requested_material = material_override.strip().upper()
    if requested_material in ALLOWED_MATERIALS:
        material = requested_material
    elif is_flexible:
        material = "TPU"
    elif _has_any_word(user_text, _PETG_WORDS) or _has_any_word(code_text, ("petg", "weatherproof")):
        material = "PETG"
    else:
        material = "PLA"

    dims = metrics.get("dimensions_mm") or {}
    extents = [float(dims.get(axis, 0.0)) for axis in ("x", "y", "z")]
    max_dim, min_dim = max(extents), min(extents)
    volume = float(metrics.get("volume_mm3", 0.0))
    printability = metrics.get("printability") or {}
    is_small = 0.0 < max_dim <= 30.0 and volume <= 8000.0

    if pri != "auto":
        effective_priority = pri
    elif is_mechanical:
        effective_priority = "strength"
    elif is_detail or is_small:
        effective_priority = "quality"
    else:
        effective_priority = "standard"

    preset = dict(_PRIORITY_PRESETS[effective_priority])
    if effective_priority == "standard":
        if is_enclosure:
            preset["part_category"] = "Enclosure / Container"
        # Auto mode: tune layer height to the part's scale.
        if 0.0 < max_dim <= 30.0:
            preset["layer_height_mm"] = 0.12
        elif 0.0 < max_dim <= 50.0 and 0.0 < min_dim <= 8.0:
            preset["layer_height_mm"] = 0.16
        elif max_dim >= 140.0 and volume >= 80000.0:
            preset["layer_height_mm"] = 0.24
            preset["infill_pattern"] = "adaptivecubic"

    # Thin-walled geometry (high surface-to-volume ratio) needs more perimeters.
    surface_area = float(printability.get("surface_area_mm2", 0.0))
    if volume > 0.0 and surface_area / volume > 1.2:
        preset["wall_loops"] = max(3, preset["wall_loops"])

    if material == "TPU":
        preset["wall_loops"] = max(3, preset["wall_loops"])
        preset["infill_density_percent"] = max(25, preset["infill_density_percent"])
        preset["infill_pattern"] = "gyroid"
        if preset["part_category"] == "General Purpose":
            preset["part_category"] = "Flexible / Elastomeric"

    overhang_area_45 = float(printability.get("overhang_area_45deg_mm2", 0.0))
    needs_support = bool(printability.get("needs_support"))
    tree_support = (
        float(printability.get("overhang_ratio", 0.0)) > 0.08
        or float(printability.get("overhang_area_60deg_mm2", 0.0)) > 50.0
        or is_detail
        or _has_any_word(combined_text, ("organic", "tree"))
    )
    slenderness = float(printability.get("slenderness_ratio", 1.0))
    # ABS/ASA always warp without a brim.
    needs_brim = material in {"ABS", "ASA"} or bool(printability.get("needs_brim"))
    brim_width_mm = 8 if material in {"ABS", "ASA"} or slenderness >= 3.5 else 6

    support_type = "tree" if tree_support else "normal"
    reasons = [
        f"{preset['part_category']} profile using {material} ({preset['layer_height_mm']} mm layer height).",
        f"{preset['infill_density_percent']}% {preset['infill_pattern']} infill with {preset['wall_loops']} perimeters.",
        (
            f"Support enabled ({support_type}) for {overhang_area_45:.1f} mm² downward overhangs."
            if needs_support
            else "No support required (angles within safe overhang threshold)."
        ),
        (
            f"Brim ({brim_width_mm} mm) enabled to stabilize bed contact."
            if needs_brim
            else "Stable bed contact footprint, no brim needed."
        ),
    ]
    return sanitize_print_profile(
        {
            **preset,
            "material": material,
            "first_layer_height_mm": 0.20,
            "support_enabled": needs_support,
            "support_type": support_type,
            "support_threshold_angle": 45,
            "brim_enabled": needs_brim,
            "brim_width_mm": brim_width_mm,
            "rationale": " ".join(reasons),
        }
    )


def _load_metrics(project_dir: Path) -> dict[str, Any]:
    """Mesh metrics with printability telemetry for the current build.

    Uses ``.cad_metrics.json`` when it matches ``model.scad`` and already
    carries printability data; otherwise recomputes from ``preview.stl``.
    """
    preview_stl = project_dir / "preview.stl"
    if not preview_stl.is_file() or preview_stl.stat().st_size == 0:
        raise FileNotFoundError("Build the model before requesting print settings.")
    try:
        cached = json.loads((project_dir / ".cad_metrics.json").read_text(encoding="utf-8"))
        metrics = cached.get("metrics")
        if (
            cached.get("model_sha256") == compute_model_sha256(project_dir)
            and isinstance(metrics, dict)
            and "printability" in metrics
        ):
            return metrics
    except (OSError, ValueError, AttributeError):
        pass
    vertices, triangles = load_stl(preview_stl)
    return mesh_metrics(vertices, triangles)


def _read_scad(project_dir: Path, limit: int = 4000) -> str:
    try:
        return (project_dir / "model.scad").read_text(encoding="utf-8")[:limit]
    except (OSError, UnicodeDecodeError):
        return ""


def _read_recent_user_context(project_dir: Path, max_messages: int = 5) -> str:
    """Join the most recent plain-text user messages to capture design intent."""
    user_texts: list[str] = []
    try:
        with (project_dir / "conversation.jsonl").open(encoding="utf-8") as conversation:
            for line in conversation:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                text = msg.get("content") if isinstance(msg, dict) and msg.get("role") == "user" else None
                if isinstance(text, str) and text.strip():
                    user_texts.append(text.strip())
    except OSError:
        return ""
    return " | ".join(user_texts[-max_messages:])


def recommend_print_profile(
    project_dir: Path,
    *,
    priority: str = "auto",
    material_override: str = "auto",
    extra_context: str = "",
) -> dict[str, Any]:
    """Recommend a print profile for the project's current build.

    Raises FileNotFoundError when the model has not been built yet.
    """
    metrics = _load_metrics(project_dir)
    profile = build_heuristic_print_profile(
        metrics,
        code_snippet=_read_scad(project_dir),
        priority=priority,
        material_override=material_override,
        user_context=f"{_read_recent_user_context(project_dir)} {extra_context}".strip(),
    )
    return {
        "profile": profile,
        "printability": metrics.get("printability") or {},
        "dimensions_mm": metrics.get("dimensions_mm") or {},
        "volume_mm3": float(metrics.get("volume_mm3", 0.0)),
    }


def format_print_command_prompt(project_dir: Path, extra_notes: str = "") -> str:
    """Build the ``/print`` chat prompt: exact mesh telemetry plus the heuristic baseline."""
    result = recommend_print_profile(project_dir, extra_context=extra_notes)
    p = result["printability"]
    dims = result["dimensions_mm"]
    baseline = result["profile"]

    support_desc = (
        f"Enabled ({baseline['support_type']} supports, {baseline['support_threshold_angle']}° threshold)"
        if baseline["support_enabled"]
        else "Disabled (none required)"
    )
    brim_desc = f"Enabled ({baseline['brim_width_mm']} mm)" if baseline["brim_enabled"] else "Disabled / None"
    notes = extra_notes.strip() or "Standard production print."

    return (
        "Please analyze the current CAD model and provide a comprehensive 3D printing and slicing recommendation.\n\n"
        "### Exact Model Telemetry:\n"
        f"- Dimensions (Bounding Box): {dims.get('x', 0.0):.1f} × {dims.get('y', 0.0):.1f} × {dims.get('z', 0.0):.1f} mm\n"
        f"- Volume: {result['volume_mm3']:.1f} mm³ | Surface Area: {p.get('surface_area_mm2', 0.0):.1f} mm²\n"
        f"- Bed Contact Footprint: {p.get('bed_contact_area_mm2', 0.0):.1f} mm² "
        f"({p.get('bed_contact_ratio', 0.0) * 100:.1f}% of base)\n"
        f"- Downward Overhang Area (>45°): {p.get('overhang_area_45deg_mm2', 0.0):.1f} mm² "
        f"({p.get('overhang_ratio', 0.0) * 100:.1f}%)\n"
        f"- Downward Overhang Area (>60°): {p.get('overhang_area_60deg_mm2', 0.0):.1f} mm²\n"
        f"- Slenderness / Aspect Ratio (Z / min(X,Y)): {p.get('slenderness_ratio', 1.0):.2f}\n"
        f"- Automated Telemetry Flags: needs_support={bool(p.get('needs_support'))}, "
        f"needs_brim={bool(p.get('needs_brim'))}\n\n"
        "### Heuristic Baseline Settings:\n"
        f"- Part Category: {baseline['part_category']}\n"
        f"- Material: {baseline['material']}\n"
        f"- Layer Height: {baseline['layer_height_mm']} mm (First layer: {baseline['first_layer_height_mm']} mm)\n"
        f"- Perimeters (Wall Loops): {baseline['wall_loops']}\n"
        f"- Shells: {baseline['top_shell_layers']} top / {baseline['bottom_shell_layers']} bottom\n"
        f"- Infill: {baseline['infill_density_percent']}% {baseline['infill_pattern']}\n"
        f"- Supports: {support_desc}\n"
        f"- Bed Adhesion: {brim_desc}\n\n"
        f"### User Requirements & Specific Notes:\n{notes}\n\n"
        "Please provide detailed, professional advice covering:\n"
        "1. **Slicer Profile Recommendations:** Optimal material, layer heights, wall counts, infill pattern/density, and temperature/cooling advice.\n"
        "2. **Build Plate Orientation:** How to orient the part on the print bed to balance mechanical strength (layer line shear), overhangs, and surface finish.\n"
        "3. **Support & Adhesion Strategy:** Tree vs. standard supports, critical overhang areas, contact z-distance, and brim requirements.\n"
        "4. **Manufacturing Warnings & Tolerances:** Warping risks, dimensional tolerances for fitting parts, and post-processing suggestions."
    )
