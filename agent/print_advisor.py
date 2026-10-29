"""Hardware-, telemetry-, and OpenSCAD-code-aware 3MF print profile configurator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import is_dataclass, replace
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import threading
from typing import Any

from agent.llm_base import extract_usage_summary
from agent.revisions import compute_model_sha256
from agent.settings import with_reasoning_effort

_LOG = logging.getLogger(__name__)

ALLOWED_MATERIALS: tuple[str, ...] = ("PLA", "PETG", "ASA", "ABS", "TPU")
ALLOWED_INFILL_PATTERNS: tuple[str, ...] = (
    "gyroid",
    "grid",
    "cubic",
    "adaptivecubic",
    "honeycomb",
    "3dhoneycomb",
    "crosshatch",
    "rectilinear",
    "zig-zag",
    "zigzag",
    "concentric",
    "lightning",
)
ALLOWED_SURFACE_PATTERNS: tuple[str, ...] = (
    "monotonicline",
    "monotonic",
    "concentric",
    "rectilinear",
)
ALLOWED_SUPPORT_TYPES: tuple[str, ...] = ("normal", "tree")
ALLOWED_BRIM_TYPES: tuple[str, ...] = (
    "no_brim",
    "outer_only",
    "inner_only",
    "outer_and_inner",
    "auto_brim",
)
ALLOWED_WALL_GENERATORS: tuple[str, ...] = ("arachne", "classic")
ALLOWED_SEAM_POSITIONS: tuple[str, ...] = ("aligned", "back", "rear", "random", "nearest")
ALLOWED_WALL_SEQUENCES: tuple[str, ...] = (
    "inner/outer",
    "outer/inner",
    "inner/outer/inner",
)
ALLOWED_IRONING_TYPES: tuple[str, ...] = ("no iron", "no ironing", "top", "topmost")
ALLOWED_PRIORITIES: tuple[str, ...] = ("auto", "strength", "quality", "speed")

_DEFAULTS: dict[str, Any] = {
    "material": "PLA",
    "nozzle_diameter_mm": 0.4,
    "layer_height_mm": 0.20,
    "first_layer_height_mm": 0.20,
    "line_width_mm": 0.42,
    "initial_layer_line_width_mm": 0.50,
    "outer_wall_line_width_mm": 0.42,
    "inner_wall_line_width_mm": 0.45,
    "top_surface_line_width_mm": 0.42,
    "sparse_infill_line_width_mm": 0.45,
    "internal_solid_infill_line_width_mm": 0.42,
    "support_line_width_mm": 0.38,
    "wall_loops": 3,
    "top_shell_layers": 4,
    "bottom_shell_layers": 4,
    "wall_generator": "arachne",
    "detect_thin_wall": True,
    "seam_position": "aligned",
    "wall_sequence": "inner/outer",
    "only_one_wall_top": False,
    "elefant_foot_compensation_mm": 0.10,
    "xy_hole_compensation_mm": 0.0,
    "xy_contour_compensation_mm": 0.0,
    "precise_z_height": True,
    "ironing_type": "no iron",
    "ironing_flow_percent": 15,
    "ironing_speed_mms": 30,
    "infill_density_percent": 20,
    "infill_pattern": "gyroid",
    "top_surface_pattern": "monotonicline",
    "bottom_surface_pattern": "monotonic",
    "infill_wall_overlap_percent": 15,
    "support_enabled": False,
    "support_type": "normal",
    "support_threshold_angle": 45,
    "support_on_build_plate_only": False,
    "support_top_z_distance_mm": 0.20,
    "support_bottom_z_distance_mm": 0.20,
    "support_interface_top_layers": 2,
    "bridge_flow": 0.95,
    "thick_bridges": False,
    "brim_enabled": False,
    "brim_type": "no_brim",
    "brim_width_mm": 0,
    "nozzle_temp_initial_c": 220,
    "nozzle_temp_c": 215,
    "bed_temp_initial_c": 60,
    "bed_temp_c": 60,
    "fan_min_percent": 100,
    "fan_max_percent": 100,
    "overhang_fan_percent": 100,
    "slow_down_for_layer_cooling": True,
    "max_volumetric_speed_mm3s": 16.0,
    "initial_layer_speed_mms": 50,
    "initial_layer_infill_speed_mms": 80,
    "outer_wall_speed_mms": 120,
    "inner_wall_speed_mms": 180,
    "sparse_infill_speed_mms": 200,
    "internal_solid_infill_speed_mms": 180,
    "top_surface_speed_mms": 100,
    "bridge_speed_mms": 40,
    "support_speed_mms": 120,
    "travel_speed_mms": 350,
    "slow_down_overhangs": True,
    "bed_x_mm": 256.0,
    "bed_y_mm": 256.0,
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
# ABS/ASA mentions also map to PETG unless overridden: similar heat/UV resistance
# without enclosure-dependent warping. ABS/ASA stay available via material override.
_PETG_WORDS = (
    "outdoor", "sunlight", "weather", "waterproof", "weatherproof", "petg", "uv", "heat",
    "abs", "asa",
)

_THREAD_WORDS = (
    "thread", "threads", "threaded", "helix", "m2", "m3", "m4", "m5",
    "m6", "m8", "m10", "m12", "nut", "bolt", "trapezoidal", "acme",
)
_GEAR_WORDS = (
    "gear", "teeth", "tooth", "involute", "pinion", "rack", "sprocket", "spline",
    "pulley", "gt2", "timing", "planetary", "herringbone", "bevel",
)
_MATING_WORDS = (
    "tolerance", "snap", "latch", "detent", "hinge", "knuckle",
    "pivot", "bearing", "bushing", "dovetail", "bayonet",
    "mating", "interlock", "captive",
)
_FINE_DETAIL_WORDS = (
    "knurl", "knurling", "emboss", "engrave", "logo", "monogram",
    "relief", "inscription",
)
_THIN_WALL_WORDS = (
    "thin", "fin", "fins", "blade", "rib", "ribs", "membrane", "vane", "lattice", "mesh",
)
_INTERNAL_CAVITY_WORDS = (
    "channel", "duct", "tunnel", "cavity", "port", "bore", "internal", "manifold",
    "passage", "conduit", "funnel", "pipe", "tube",
)
_FLAT_TOP_WORDS = (
    "lid", "cover", "plate", "plaque", "badge", "coaster", "tray", "sign", "panel",
    "faceplate", "nameplate", "tile", "tag",
)

_HIGH_SPEED_PRINTER_PATTERNS = (
    r"\bbambu\b",
    r"\bx1[ce]?\b",
    r"\bp1[ps]\b",
    r"\ba1(\s*mini)?\b",
    r"\bh2d\b",
    r"\bk1(\s*max)?\b",
    r"\bk2(\s*plus)?\b",
    r"\bvoron\b",
    r"\bratrig\b",
    r"\bvzbot\b",
    r"\bmk4\b",
    r"\bcore\s*one\b",
    r"\bprusa\s*xl\b",
    r"\bflsun\b",
    r"\bv400\b",
    r"\bqidi\b",
    r"\bankermake\b",
    r"\badventurer\s*5\b",
    r"\bcentauri\b",
    r"\bsv08\b",
)

_MATERIAL_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "PLA": {
        "nozzle_temp": (180.0, 245.0),
        "bed_temp": (40.0, 75.0),
        "max_volumetric": (2.0, 28.0),
    },
    "PETG": {
        "nozzle_temp": (215.0, 265.0),
        "bed_temp": (60.0, 95.0),
        "max_volumetric": (2.0, 20.0),
    },
    "ASA": {
        "nozzle_temp": (230.0, 280.0),
        "bed_temp": (80.0, 115.0),
        "max_volumetric": (2.0, 24.0),
    },
    "ABS": {
        "nozzle_temp": (225.0, 275.0),
        "bed_temp": (80.0, 115.0),
        "max_volumetric": (2.0, 24.0),
    },
    "TPU": {
        "nozzle_temp": (200.0, 245.0),
        "bed_temp": (25.0, 65.0),
        "max_volumetric": (1.0, 8.0),
    },
}


def normalize_material(raw_material: str | None, default: str = "PLA") -> str:
    """Normalize user-entered filament strings (e.g. 'PLA+', 'PETG HF', 'TPU 95A', 'ABS-GF') to standard base material."""
    text = (raw_material or "").strip().upper()
    if not text or text == "AUTO":
        return default
    if "TPU" in text or "FLEX" in text:
        return "TPU"
    if "PETG" in text or "PET-G" in text or text == "PET" or text.startswith("PET-") or "PET " in text:
        return "PETG"
    if "ASA" in text:
        return "ASA"
    if "ABS" in text:
        return "ABS"
    if "PLA" in text:
        return "PLA"
    return default


def _number(
    raw: dict[str, Any],
    key: str,
    lo: float,
    hi: float,
    fallback: float | None = None,
) -> float:
    """Read ``raw[key]`` as a finite float clamped to ``[lo, hi]``."""
    default_val = _DEFAULTS[key] if fallback is None else fallback
    try:
        value = float(raw.get(key, default_val))
    except (TypeError, ValueError):
        value = float(default_val)
    if not math.isfinite(value):
        value = float(default_val)
    return min(hi, max(lo, value))


def _bool(raw: dict[str, Any], key: str, fallback: bool | None = None) -> bool:
    default_val = bool(_DEFAULTS[key] if fallback is None else fallback)
    if key not in raw or raw[key] is None:
        return default_val
    val = raw[key]
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        lowered = val.strip().lower()
        if lowered in ("1", "true", "yes", "on", "enabled"):
            return True
        if lowered in ("0", "false", "no", "off", "disabled"):
            return False
    return default_val


def _choice(
    raw: dict[str, Any],
    key: str,
    allowed: tuple[str, ...],
    fallback: str | None = None,
) -> str:
    default_val = _DEFAULTS[key] if fallback is None else fallback
    value = str(raw.get(key, default_val)).strip()
    value = value.upper() if key == "material" else value.lower()
    return value if value in allowed else default_val


def _text(raw: dict[str, Any], key: str, limit: int) -> str:
    return re.sub(r"\s+", " ", str(raw.get(key) or "")).strip()[:limit]


def parse_nozzle_spec(nozzle_str: str | None) -> dict[str, Any]:
    """Parse user nozzle string into diameter (mm), material, and clean description."""
    text = (nozzle_str or "").strip()
    dia = 0.4
    if text:
        for pattern in (
            r"(?<![A-Za-z#])(\d+(?:\.\d+)?)\s*mm\b",
            r"(?<![A-Za-z0-9.#])(\d+\.\d+)(?!\d)",
            r"(?<![A-Za-z#])(\d+(?:\.\d+)?)\b",
        ):
            found = False
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    parsed_dia = float(match.group(1))
                    if 0.15 <= parsed_dia <= 1.2:
                        dia = parsed_dia
                        found = True
                        break
                except (ValueError, TypeError):
                    continue
            if found:
                break
    lower = text.lower()
    material = ""
    if "hardened" in lower:
        material = "Hardened Steel"
    elif "stainless" in lower:
        material = "Stainless Steel"
    elif "brass" in lower:
        material = "Brass"
    elif "ruby" in lower or "diamond" in lower:
        material = "Abrasion-Resistant"

    return {
        "diameter_mm": round(dia, 2),
        "material": material,
        "raw": text,
    }


def parse_printer_spec(printer_str: str | None) -> dict[str, Any]:
    """Infer kinematics speed tier and build plate dimensions (mm) from printer string."""
    text = (printer_str or "").strip()
    lower = text.lower()
    # Sidewinder X1/Genius and Ender-3 S1 are standard bedslingers, not high-speed CoreXY
    is_slow_override = any(tok in lower for tok in ("sidewinder", "ender 3 s1", "ender-3 s1"))
    is_high_speed = not is_slow_override and any(
        re.search(pat, lower) for pat in _HIGH_SPEED_PRINTER_PATTERNS
    )
    speed_tier = "high_speed" if is_high_speed else "standard"

    bed_x, bed_y = 256.0, 256.0
    dim_match = re.search(r"\b(\d{3})\s*[x×]\s*(\d{3})\b", lower)
    if dim_match:
        bx, by = float(dim_match.group(1)), float(dim_match.group(2))
        if 100.0 <= bx <= 600.0 and 100.0 <= by <= 600.0:
            bed_x, bed_y = bx, by
    elif "ender 2" in lower or "ender-2" in lower:
        bed_x, bed_y = 165.0, 165.0
    elif "a1 mini" in lower or "prusa mini" in lower or "kp3s" in lower:
        bed_x, bed_y = 180.0, 180.0
    elif "voron 0" in lower or "v0" in lower:
        bed_x, bed_y = 120.0, 120.0
    elif "mk3" in lower or "mk4" in lower or "core one" in lower:
        bed_x, bed_y = 250.0, 210.0
    elif "prusa xl" in lower:
        bed_x, bed_y = 360.0, 360.0
    elif "ender 5 plus" in lower or "ender-5 plus" in lower or "sv08" in lower or ("350" in lower and "voron" in lower):
        bed_x, bed_y = 350.0, 350.0
    elif "sidewinder" in lower or "k1 max" in lower or "k2 plus" in lower or "300" in lower:
        bed_x, bed_y = 300.0, 300.0
    elif "max" in lower and ("neptune" in lower or "kobra" in lower):
        bed_x, bed_y = 420.0, 420.0
    elif "plus" in lower and ("neptune" in lower or "kobra" in lower):
        bed_x, bed_y = 320.0, 320.0
    elif any(k in lower for k in ("ender", "k1", "kobra", "neptune", "sovol", "vyper", "artillery")):
        bed_x, bed_y = 220.0, 220.0

    return {
        "name": text,
        "speed_tier": speed_tier,
        "bed_x_mm": bed_x,
        "bed_y_mm": bed_y,
    }


def _material_thermal_and_flow_defaults(
    material: str,
    nozzle_dia: float,
    speed_tier: str,
    layer_height_mm: float = 0.20,
) -> dict[str, Any]:
    """Return baseline thermal, cooling, volumetric flow, and speed parameters for a material."""
    hi = speed_tier == "high_speed"
    nozzle_scale = min(1.5, max(0.7, nozzle_dia / 0.4))
    z_gap_base = round(max(0.14, min(0.32, layer_height_mm)), 2)

    presets: dict[str, dict[str, Any]] = {
        "PLA": {
            "nozzle_temp_initial_c": 220,
            "nozzle_temp_c": 215,
            "bed_temp_initial_c": 60,
            "bed_temp_c": 60,
            "fan_min_percent": 100,
            "fan_max_percent": 100,
            "overhang_fan_percent": 100,
            "slow_down_for_layer_cooling": True,
            "max_volumetric_speed_mm3s": round((21.0 if hi else 15.0) * nozzle_scale, 1),
            "support_top_z_distance_mm": z_gap_base,
            "support_bottom_z_distance_mm": z_gap_base,
            "initial_layer_speed_mms": 50 if hi else 35,
            "initial_layer_infill_speed_mms": 85 if hi else 60,
            "outer_wall_speed_mms": 150 if hi else 60,
            "inner_wall_speed_mms": 220 if hi else 90,
            "sparse_infill_speed_mms": 250 if hi else 100,
            "internal_solid_infill_speed_mms": 200 if hi else 90,
            "top_surface_speed_mms": 120 if hi else 50,
            "bridge_speed_mms": 45 if hi else 30,
            "support_speed_mms": 140 if hi else 70,
            "travel_speed_mms": 400 if hi else 180,
        },
        "PETG": {
            "nozzle_temp_initial_c": 240,
            "nozzle_temp_c": 240,
            "bed_temp_initial_c": 75,
            "bed_temp_c": 75,
            "fan_min_percent": 40,
            "fan_max_percent": 85,
            "overhang_fan_percent": 100,
            "slow_down_for_layer_cooling": True,
            "max_volumetric_speed_mm3s": round((13.0 if hi else 10.0) * nozzle_scale, 1),
            "support_top_z_distance_mm": round(max(0.22, z_gap_base + 0.04), 2),
            "support_bottom_z_distance_mm": round(max(0.22, z_gap_base + 0.04), 2),
            "initial_layer_speed_mms": 35 if hi else 25,
            "initial_layer_infill_speed_mms": 65 if hi else 45,
            "outer_wall_speed_mms": 100 if hi else 45,
            "inner_wall_speed_mms": 150 if hi else 70,
            "sparse_infill_speed_mms": 170 if hi else 80,
            "internal_solid_infill_speed_mms": 150 if hi else 70,
            "top_surface_speed_mms": 80 if hi else 40,
            "bridge_speed_mms": 30 if hi else 25,
            "support_speed_mms": 100 if hi else 60,
            "travel_speed_mms": 350 if hi else 160,
        },
        "ASA": {
            "nozzle_temp_initial_c": 260,
            "nozzle_temp_c": 255,
            "bed_temp_initial_c": 100,
            "bed_temp_c": 100,
            "fan_min_percent": 15,
            "fan_max_percent": 35,
            "overhang_fan_percent": 60,
            "slow_down_for_layer_cooling": True,
            "max_volumetric_speed_mm3s": round((16.0 if hi else 12.0) * nozzle_scale, 1),
            "support_top_z_distance_mm": round(max(0.20, z_gap_base + 0.02), 2),
            "support_bottom_z_distance_mm": round(max(0.20, z_gap_base + 0.02), 2),
            "initial_layer_speed_mms": 40 if hi else 30,
            "initial_layer_infill_speed_mms": 70 if hi else 50,
            "outer_wall_speed_mms": 120 if hi else 55,
            "inner_wall_speed_mms": 180 if hi else 80,
            "sparse_infill_speed_mms": 200 if hi else 90,
            "internal_solid_infill_speed_mms": 170 if hi else 80,
            "top_surface_speed_mms": 90 if hi else 45,
            "bridge_speed_mms": 35 if hi else 25,
            "support_speed_mms": 110 if hi else 60,
            "travel_speed_mms": 380 if hi else 170,
        },
        "ABS": {
            "nozzle_temp_initial_c": 255,
            "nozzle_temp_c": 250,
            "bed_temp_initial_c": 100,
            "bed_temp_c": 100,
            "fan_min_percent": 15,
            "fan_max_percent": 35,
            "overhang_fan_percent": 50,
            "slow_down_for_layer_cooling": True,
            "max_volumetric_speed_mm3s": round((16.0 if hi else 12.0) * nozzle_scale, 1),
            "support_top_z_distance_mm": round(max(0.20, z_gap_base + 0.02), 2),
            "support_bottom_z_distance_mm": round(max(0.20, z_gap_base + 0.02), 2),
            "initial_layer_speed_mms": 40 if hi else 30,
            "initial_layer_infill_speed_mms": 70 if hi else 50,
            "outer_wall_speed_mms": 120 if hi else 55,
            "inner_wall_speed_mms": 180 if hi else 80,
            "sparse_infill_speed_mms": 200 if hi else 90,
            "internal_solid_infill_speed_mms": 170 if hi else 80,
            "top_surface_speed_mms": 90 if hi else 45,
            "bridge_speed_mms": 35 if hi else 25,
            "support_speed_mms": 110 if hi else 60,
            "travel_speed_mms": 380 if hi else 170,
        },
        "TPU": {
            "nozzle_temp_initial_c": 225,
            "nozzle_temp_c": 225,
            "bed_temp_initial_c": 50,
            "bed_temp_c": 50,
            "fan_min_percent": 80,
            "fan_max_percent": 100,
            "overhang_fan_percent": 100,
            "slow_down_for_layer_cooling": True,
            "max_volumetric_speed_mm3s": 3.5,
            "support_top_z_distance_mm": round(max(0.26, z_gap_base + 0.06), 2),
            "support_bottom_z_distance_mm": round(max(0.26, z_gap_base + 0.06), 2),
            "initial_layer_speed_mms": 25,
            "initial_layer_infill_speed_mms": 35,
            "outer_wall_speed_mms": 40,
            "inner_wall_speed_mms": 50,
            "sparse_infill_speed_mms": 55,
            "internal_solid_infill_speed_mms": 50,
            "top_surface_speed_mms": 40,
            "bridge_speed_mms": 25,
            "support_speed_mms": 45,
            "travel_speed_mms": 200,
        },
    }
    return dict(presets.get(material.upper(), presets["PLA"]))


def sanitize_print_profile(
    raw: Any,
    *,
    nozzle_diameter_mm: float = 0.4,
    user_printer: str = "",
    fallback_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Clamp and validate a (possibly untrusted) print profile into safe slicer bounds.

    Every value written into 3MF slicer configs passes through here, so the
    serializers never see free-form input beyond the bounded text fields.
    """
    if not isinstance(raw, dict):
        raw = {}
    fb = fallback_profile or _DEFAULTS
    raw_mat = raw.get("material", fb.get("material", "PLA"))
    material = normalize_material(str(raw_mat), fb.get("material", "PLA"))
    if material not in ALLOWED_MATERIALS:
        material = "PLA"

    raw_dia = raw.get("nozzle_diameter_mm")
    try:
        parsed_dia = float(raw_dia) if raw_dia is not None else float(nozzle_diameter_mm or 0.4)
    except (ValueError, TypeError):
        parsed_dia = parse_nozzle_spec(str(raw_dia))["diameter_mm"] if raw_dia else float(nozzle_diameter_mm or 0.4)
    nozzle_dia = max(0.15, min(1.2, parsed_dia))

    target_printer = _text(raw, "target_printer", 80) or _text({"p": user_printer}, "p", 80)
    printer_spec = parse_printer_spec(target_printer)

    min_layer = round(max(0.05, nozzle_dia * 0.15), 2)
    max_layer = round(min(0.80, nozzle_dia * 0.75), 2)
    min_first = round(max(0.10, nozzle_dia * 0.25), 2)
    max_first = round(min(0.60, nozzle_dia * 0.75), 2)

    layer_height = round(
        _number(raw, "layer_height_mm", min_layer, max_layer, fb.get("layer_height_mm", round(nozzle_dia * 0.50, 2))),
        2,
    )
    first_layer = round(
        _number(
            raw,
            "first_layer_height_mm",
            min_first,
            max_first,
            fb.get("first_layer_height_mm", round(min(0.40, max(0.20, nozzle_dia * 0.50)), 2)),
        ),
        2,
    )

    mat_defaults = _material_thermal_and_flow_defaults(
        material, nozzle_dia, printer_spec["speed_tier"], layer_height
    )
    mat_fb = fb if fb.get("material") == material else mat_defaults
    bounds = _MATERIAL_BOUNDS.get(material, _MATERIAL_BOUNDS["PLA"])
    nozzle_min, nozzle_max = bounds["nozzle_temp"]
    bed_min, bed_max = bounds["bed_temp"]
    vol_min, vol_max = bounds["max_volumetric"]

    min_lw = round(max(0.12, nozzle_dia * 0.85), 2)
    max_lw = round(min(1.80, nozzle_dia * 1.60), 2)
    def_lw = round(nozzle_dia * 1.05, 2)
    def_init_lw = round(nozzle_dia * 1.25, 2)
    def_inner_lw = round(nozzle_dia * 1.12, 2)
    def_top_lw = round(nozzle_dia * 1.00, 2)
    def_sup_lw = round(nozzle_dia * 0.95, 2)

    brim_enabled = _bool(raw, "brim_enabled", fb.get("brim_enabled", False))
    raw_brim_type = _choice(
        raw,
        "brim_type",
        ALLOWED_BRIM_TYPES,
        fb.get("brim_type", "outer_only" if brim_enabled else "no_brim"),
    )
    if not brim_enabled:
        brim_type = "no_brim"
        brim_width = 0
    else:
        brim_type = "outer_only" if raw_brim_type == "no_brim" else raw_brim_type
        brim_width = round(_number(raw, "brim_width_mm", 4, 15, fb.get("brim_width_mm", 6)))

    support_enabled = _bool(raw, "support_enabled", fb.get("support_enabled", False))

    sanitized: dict[str, Any] = {
        "material": material,
        "nozzle_diameter_mm": round(nozzle_dia, 2),
        "layer_height_mm": layer_height,
        "first_layer_height_mm": first_layer,
        "line_width_mm": round(_number(raw, "line_width_mm", min_lw, max_lw, fb.get("line_width_mm", def_lw)), 2),
        "initial_layer_line_width_mm": round(
            _number(raw, "initial_layer_line_width_mm", min_lw, max_lw, fb.get("initial_layer_line_width_mm", def_init_lw)), 2
        ),
        "outer_wall_line_width_mm": round(
            _number(raw, "outer_wall_line_width_mm", min_lw, max_lw, fb.get("outer_wall_line_width_mm", def_lw)), 2
        ),
        "inner_wall_line_width_mm": round(
            _number(raw, "inner_wall_line_width_mm", min_lw, max_lw, fb.get("inner_wall_line_width_mm", def_inner_lw)), 2
        ),
        "top_surface_line_width_mm": round(
            _number(raw, "top_surface_line_width_mm", min_lw, max_lw, fb.get("top_surface_line_width_mm", def_top_lw)), 2
        ),
        "sparse_infill_line_width_mm": round(
            _number(raw, "sparse_infill_line_width_mm", min_lw, max_lw, fb.get("sparse_infill_line_width_mm", def_inner_lw)), 2
        ),
        "internal_solid_infill_line_width_mm": round(
            _number(raw, "internal_solid_infill_line_width_mm", min_lw, max_lw, fb.get("internal_solid_infill_line_width_mm", def_lw)), 2
        ),
        "support_line_width_mm": round(
            _number(raw, "support_line_width_mm", min_lw, max_lw, fb.get("support_line_width_mm", def_sup_lw)), 2
        ),
        "wall_loops": round(_number(raw, "wall_loops", 2, 8, fb.get("wall_loops", 3))),
        "top_shell_layers": round(_number(raw, "top_shell_layers", 2, 10, fb.get("top_shell_layers", 4))),
        "bottom_shell_layers": round(_number(raw, "bottom_shell_layers", 2, 10, fb.get("bottom_shell_layers", 4))),
        "wall_generator": _choice(raw, "wall_generator", ALLOWED_WALL_GENERATORS, fb.get("wall_generator", "arachne")),
        "detect_thin_wall": _bool(raw, "detect_thin_wall", fb.get("detect_thin_wall", True)),
        "seam_position": _choice(raw, "seam_position", ALLOWED_SEAM_POSITIONS, fb.get("seam_position", "aligned")),
        "wall_sequence": _choice(raw, "wall_sequence", ALLOWED_WALL_SEQUENCES, fb.get("wall_sequence", "inner/outer")),
        "only_one_wall_top": _bool(raw, "only_one_wall_top", fb.get("only_one_wall_top", False)),
        "elefant_foot_compensation_mm": round(
            _number(raw, "elefant_foot_compensation_mm", 0.0, 0.30, fb.get("elefant_foot_compensation_mm", 0.10)), 2
        ),
        "xy_hole_compensation_mm": round(
            _number(raw, "xy_hole_compensation_mm", -0.20, 0.30, fb.get("xy_hole_compensation_mm", 0.0)), 2
        ),
        "xy_contour_compensation_mm": round(
            _number(raw, "xy_contour_compensation_mm", -0.20, 0.20, fb.get("xy_contour_compensation_mm", 0.0)), 2
        ),
        "precise_z_height": _bool(raw, "precise_z_height", fb.get("precise_z_height", True)),
        "ironing_type": _choice(raw, "ironing_type", ALLOWED_IRONING_TYPES, fb.get("ironing_type", "no iron")),
        "ironing_flow_percent": round(_number(raw, "ironing_flow_percent", 5, 30, fb.get("ironing_flow_percent", 15))),
        "ironing_speed_mms": round(_number(raw, "ironing_speed_mms", 15, 60, fb.get("ironing_speed_mms", 30))),
        "infill_density_percent": round(_number(raw, "infill_density_percent", 0, 100, fb.get("infill_density_percent", 20)) / 5) * 5,
        "infill_pattern": _choice(raw, "infill_pattern", ALLOWED_INFILL_PATTERNS, fb.get("infill_pattern", "gyroid")),
        "top_surface_pattern": _choice(raw, "top_surface_pattern", ALLOWED_SURFACE_PATTERNS, fb.get("top_surface_pattern", "monotonicline")),
        "bottom_surface_pattern": _choice(
            raw, "bottom_surface_pattern", ALLOWED_SURFACE_PATTERNS, fb.get("bottom_surface_pattern", "monotonic")
        ),
        "infill_wall_overlap_percent": round(
            _number(raw, "infill_wall_overlap_percent", 10, 35, fb.get("infill_wall_overlap_percent", 25 if material == "TPU" else 15))
        ),
        "support_enabled": support_enabled,
        "support_type": _choice(raw, "support_type", ALLOWED_SUPPORT_TYPES, fb.get("support_type", "normal")),
        "support_threshold_angle": round(_number(raw, "support_threshold_angle", 25, 60, fb.get("support_threshold_angle", 45))),
        "support_on_build_plate_only": _bool(raw, "support_on_build_plate_only", fb.get("support_on_build_plate_only", False)),
        "support_top_z_distance_mm": round(
            _number(raw, "support_top_z_distance_mm", 0.10, 0.40, fb.get("support_top_z_distance_mm", mat_defaults["support_top_z_distance_mm"])),
            2,
        ),
        "support_bottom_z_distance_mm": round(
            _number(raw, "support_bottom_z_distance_mm", 0.10, 0.40, fb.get("support_bottom_z_distance_mm", mat_defaults["support_bottom_z_distance_mm"])),
            2,
        ),
        "support_interface_top_layers": round(
            _number(raw, "support_interface_top_layers", 0, 5, fb.get("support_interface_top_layers", 2))
        ),
        "bridge_flow": round(_number(raw, "bridge_flow", 0.80, 1.15, fb.get("bridge_flow", 0.95)), 2),
        "thick_bridges": _bool(raw, "thick_bridges", fb.get("thick_bridges", False)),
        "brim_enabled": brim_enabled,
        "brim_type": brim_type,
        "brim_width_mm": brim_width,
        "nozzle_temp_initial_c": round(
            _number(raw, "nozzle_temp_initial_c", nozzle_min, nozzle_max, mat_fb.get("nozzle_temp_initial_c", mat_defaults["nozzle_temp_initial_c"]))
        ),
        "nozzle_temp_c": round(
            _number(raw, "nozzle_temp_c", nozzle_min, nozzle_max, mat_fb.get("nozzle_temp_c", mat_defaults["nozzle_temp_c"]))
        ),
        "bed_temp_initial_c": round(
            _number(raw, "bed_temp_initial_c", bed_min, bed_max, mat_fb.get("bed_temp_initial_c", mat_defaults["bed_temp_initial_c"]))
        ),
        "bed_temp_c": round(
            _number(raw, "bed_temp_c", bed_min, bed_max, mat_fb.get("bed_temp_c", mat_defaults["bed_temp_c"]))
        ),
        "fan_min_percent": round(
            _number(raw, "fan_min_percent", 0, 100, mat_fb.get("fan_min_percent", mat_defaults["fan_min_percent"]))
        ),
        "fan_max_percent": round(
            _number(raw, "fan_max_percent", 0, 100, mat_fb.get("fan_max_percent", mat_defaults["fan_max_percent"]))
        ),
        "overhang_fan_percent": round(
            _number(raw, "overhang_fan_percent", 0, 100, mat_fb.get("overhang_fan_percent", mat_defaults["overhang_fan_percent"]))
        ),
        "slow_down_for_layer_cooling": _bool(
            raw, "slow_down_for_layer_cooling", mat_fb.get("slow_down_for_layer_cooling", mat_defaults["slow_down_for_layer_cooling"])
        ),
        "max_volumetric_speed_mm3s": round(
            _number(raw, "max_volumetric_speed_mm3s", vol_min, vol_max, mat_fb.get("max_volumetric_speed_mm3s", mat_defaults["max_volumetric_speed_mm3s"])),
            1,
        ),
        "initial_layer_speed_mms": round(
            _number(raw, "initial_layer_speed_mms", 15, 100, mat_fb.get("initial_layer_speed_mms", mat_defaults["initial_layer_speed_mms"]))
        ),
        "initial_layer_infill_speed_mms": round(
            _number(raw, "initial_layer_infill_speed_mms", 20, 150, mat_fb.get("initial_layer_infill_speed_mms", mat_defaults["initial_layer_infill_speed_mms"]))
        ),
        "outer_wall_speed_mms": round(
            _number(raw, "outer_wall_speed_mms", 20, 250, mat_fb.get("outer_wall_speed_mms", mat_defaults["outer_wall_speed_mms"]))
        ),
        "inner_wall_speed_mms": round(
            _number(raw, "inner_wall_speed_mms", 25, 350, mat_fb.get("inner_wall_speed_mms", mat_defaults["inner_wall_speed_mms"]))
        ),
        "sparse_infill_speed_mms": round(
            _number(raw, "sparse_infill_speed_mms", 25, 400, mat_fb.get("sparse_infill_speed_mms", mat_defaults["sparse_infill_speed_mms"]))
        ),
        "internal_solid_infill_speed_mms": round(
            _number(raw, "internal_solid_infill_speed_mms", 25, 350, mat_fb.get("internal_solid_infill_speed_mms", mat_defaults["internal_solid_infill_speed_mms"]))
        ),
        "top_surface_speed_mms": round(
            _number(raw, "top_surface_speed_mms", 20, 200, mat_fb.get("top_surface_speed_mms", mat_defaults["top_surface_speed_mms"]))
        ),
        "bridge_speed_mms": round(
            _number(raw, "bridge_speed_mms", 15, 80, mat_fb.get("bridge_speed_mms", mat_defaults["bridge_speed_mms"]))
        ),
        "support_speed_mms": round(
            _number(raw, "support_speed_mms", 25, 220, mat_fb.get("support_speed_mms", mat_defaults["support_speed_mms"]))
        ),
        "travel_speed_mms": round(
            _number(raw, "travel_speed_mms", 80, 600, mat_fb.get("travel_speed_mms", mat_defaults["travel_speed_mms"]))
        ),
        "slow_down_overhangs": _bool(raw, "slow_down_overhangs", fb.get("slow_down_overhangs", True)),
        "bed_x_mm": round(_number(raw, "bed_x_mm", 100.0, 600.0, fb.get("bed_x_mm", printer_spec["bed_x_mm"])), 1),
        "bed_y_mm": round(_number(raw, "bed_y_mm", 100.0, 600.0, fb.get("bed_y_mm", printer_spec["bed_y_mm"])), 1),
        "part_category": _text(raw, "part_category", 60) or fb.get("part_category", _DEFAULTS["part_category"]),
        "rationale": _text(raw, "rationale", 500) or fb.get("rationale", ""),
    }
    if sanitized["fan_min_percent"] > sanitized["fan_max_percent"]:
        sanitized["fan_min_percent"] = sanitized["fan_max_percent"]
    if target_printer:
        sanitized["target_printer"] = target_printer
    return sanitized


def _has_any_word(text: str, words: tuple[str, ...]) -> bool:
    # ``_`` separates words in SCAD identifiers (``case_width`` -> ``case``).
    pattern = r"\b(" + "|".join(re.escape(w) for w in words) + r")\b"
    return bool(re.search(pattern, text.replace("_", " ")))


def _analyze_scad_features(code_snippet: str, user_context: str = "") -> dict[str, bool]:
    """Detect engineering and geometric constructs from OpenSCAD code and user prompt."""
    code_lower = (code_snippet or "").lower()
    combined = f"{code_lower} {(user_context or '').lower()}"
    has_threads = _has_any_word(combined, _THREAD_WORDS)
    has_gears = _has_any_word(combined, _GEAR_WORDS)
    has_mating_fits = _has_any_word(combined, _MATING_WORDS)
    has_fine_detail = (
        _has_any_word(combined, _FINE_DETAIL_WORDS)
        or bool(re.search(r"\btext\s*\(", code_lower))
    )
    has_thin_walls = _has_any_word(combined, _THIN_WALL_WORDS)
    has_internal_cavities = _has_any_word(combined, _INTERNAL_CAVITY_WORDS) or has_threads
    is_rotational = bool(
        re.search(r"\b(rotate_extrude|cylinder|sphere)\s*\(", code_lower)
        and not re.search(r"\bcube\s*\(", code_lower)
    )
    has_flat_top = _has_any_word(combined, _FLAT_TOP_WORDS)
    return {
        "has_threads": has_threads,
        "has_gears": has_gears,
        "has_mating_fits": has_mating_fits,
        "has_fine_detail": has_fine_detail,
        "has_thin_walls": has_thin_walls,
        "has_internal_cavities": has_internal_cavities,
        "is_rotational": is_rotational,
        "has_flat_top": has_flat_top,
    }


def _extract_json_block(text: str) -> dict[str, Any] | None:
    """Extract and parse the first valid JSON object from a markdown response."""
    decoder = json.JSONDecoder()
    profile_keys = set(_DEFAULTS) | {"advice_markdown"}
    fallback_dict: dict[str, Any] | None = None

    # 1. Try decoding JSON objects starting right after fenced code block headers first.
    for fence_match in re.finditer(r"```(?:json)?\s*\{", text, re.IGNORECASE):
        brace_pos = fence_match.end() - 1
        try:
            parsed, _ = decoder.raw_decode(text, brace_pos)
            if isinstance(parsed, dict):
                if any(k in parsed for k in profile_keys):
                    return parsed
                if fallback_dict is None:
                    fallback_dict = parsed
        except (ValueError, TypeError):
            continue

    # 2. Scan all '{' positions in the text using raw_decode so nested braces or
    # surrounding prose/code snippets never corrupt the slice boundaries.
    idx = text.find("{")
    while idx != -1:
        try:
            parsed, _ = decoder.raw_decode(text, idx)
            if isinstance(parsed, dict):
                if any(k in parsed for k in profile_keys):
                    return parsed
                if fallback_dict is None:
                    fallback_dict = parsed
        except (ValueError, TypeError):
            pass
        idx = text.find("{", idx + 1)

    return fallback_dict


def _format_heuristic_advice_markdown(
    profile: dict[str, Any],
    printer: str,
    nozzle: str,
) -> str:
    nozzle_str = f" with {nozzle}" if nozzle else ""
    printer_str = f" on {printer}" if printer else ""
    support_str = (
        f"Enabled ({profile['support_type']} supports, {profile['support_threshold_angle']}° threshold)"
        if profile["support_enabled"]
        else "Disabled (none required)"
    )
    brim_str = f"Enabled ({profile['brim_width_mm']} mm brim)" if profile["brim_enabled"] else "Disabled (stable base contact)"

    return (
        f"**Slicer Profile for {profile['material']}{printer_str}{nozzle_str}:**\n\n"
        f"- **Layer Height:** {profile['layer_height_mm']} mm (First layer: {profile['first_layer_height_mm']} mm)\n"
        f"- **Perimeters & Shells:** {profile['wall_loops']} wall loops, {profile['top_shell_layers']} top / {profile['bottom_shell_layers']} bottom\n"
        f"- **Infill:** {profile['infill_density_percent']}% {profile['infill_pattern']}\n"
        f"- **Supports & Adhesion:** {support_str} | {brim_str}\n\n"
        f"**Engineering Rationale:**\n{profile['rationale']}"
    )


def build_heuristic_print_profile(
    metrics: dict[str, Any],
    code_snippet: str = "",
    priority: str = "auto",
    material_override: str = "auto",
    user_context: str = "",
    *,
    user_printer: str = "",
    user_nozzle: str = "",
    default_material: str = "PLA",
) -> dict[str, Any]:
    """Deterministically compute tuned print settings from geometry telemetry and design intent."""
    nozzle_spec = parse_nozzle_spec(user_nozzle)
    nozzle_dia = nozzle_spec["diameter_mm"]
    nozzle_mat = nozzle_spec["material"]
    printer_spec = parse_printer_spec(user_printer)

    pri = priority.strip().lower()
    if pri not in ALLOWED_PRIORITIES:
        pri = "auto"

    user_text = user_context.lower()
    code_text = code_snippet.lower()
    combined_text = f"{code_text} {user_text}"
    is_mechanical = _has_any_word(combined_text, _MECHANICAL_WORDS)
    is_detail = _has_any_word(combined_text, _DETAIL_WORDS)
    is_enclosure = _has_any_word(combined_text, _ENCLOSURE_WORDS)
    features = _analyze_scad_features(code_snippet, user_context)

    norm_override = normalize_material(material_override, "")
    norm_default = normalize_material(default_material, "PLA")
    wants_tpu = _has_any_word(user_text, ("tpu", "flex", "flexible filament"))
    wants_petg = _has_any_word(user_text, ("petg", "weatherproof", "outdoor", "uv"))
    wants_abs = _has_any_word(user_text, ("abs", "asa"))

    if norm_override in ALLOWED_MATERIALS:
        material = norm_override
    elif wants_tpu:
        material = "TPU"
    elif wants_petg:
        material = "PETG"
    elif wants_abs:
        material = "ABS" if "abs" in user_text else "ASA"
    else:
        material = norm_default

    dims = metrics.get("dimensions_mm") or {}
    extents = [float(dims.get(axis, 0.0)) for axis in ("x", "y", "z")]
    max_dim, min_dim = max(extents), min(extents)
    volume = float(metrics.get("volume_mm3", 0.0))
    printability = metrics.get("printability") or {}
    is_small = 0.0 < max_dim <= 30.0 and volume <= 8000.0

    if pri != "auto":
        effective_priority = pri
    elif is_mechanical or features["has_gears"]:
        effective_priority = "strength"
    elif is_detail or is_small or features["has_fine_detail"]:
        effective_priority = "quality"
    else:
        effective_priority = "standard"

    std_lh = round(nozzle_dia * 0.50, 2)
    fine_lh = round(max(0.08, nozzle_dia * 0.40), 2)
    quality_lh = round(max(0.08, nozzle_dia * 0.30), 2)
    speed_lh = round(min(0.65, nozzle_dia * 0.65), 2)
    first_layer_lh = round(min(0.40, max(0.20, nozzle_dia * 0.50)), 2)

    priority_presets: dict[str, dict[str, Any]] = {
        "strength": {
            "layer_height_mm": std_lh,
            "wall_loops": 4,
            "top_shell_layers": 5,
            "bottom_shell_layers": 5,
            "infill_density_percent": 40,
            "infill_pattern": "gyroid",
            "part_category": "Mechanical / Load-Bearing",
        },
        "quality": {
            "layer_height_mm": quality_lh,
            "wall_loops": 3,
            "top_shell_layers": 5,
            "bottom_shell_layers": 4,
            "infill_density_percent": 20,
            "infill_pattern": "gyroid",
            "part_category": "Detailed / Aesthetic",
        },
        "speed": {
            "layer_height_mm": speed_lh,
            "wall_loops": 2,
            "top_shell_layers": 3,
            "bottom_shell_layers": 3,
            "infill_density_percent": 15,
            "infill_pattern": "grid",
            "part_category": "Draft / Rapid Prototype",
        },
        "standard": {
            "layer_height_mm": std_lh,
            "wall_loops": 3,
            "top_shell_layers": 4,
            "bottom_shell_layers": 4,
            "infill_density_percent": 20,
            "infill_pattern": "gyroid",
            "part_category": "General Purpose",
        },
    }

    preset = dict(priority_presets[effective_priority])
    if effective_priority == "standard":
        if is_enclosure:
            preset["part_category"] = "Enclosure / Container"
        if 0.0 < max_dim <= 30.0:
            preset["layer_height_mm"] = quality_lh
        elif 0.0 < max_dim <= 50.0 and 0.0 < min_dim <= 8.0:
            preset["layer_height_mm"] = fine_lh
        elif max_dim >= 140.0 and volume >= 80000.0:
            preset["layer_height_mm"] = speed_lh
            preset["infill_pattern"] = "adaptivecubic"

    # Screw threads and gear teeth need finer Z resolution so flanks mesh smoothly.
    if (features["has_threads"] or features["has_gears"]) and pri not in ("speed", "quality"):
        preset["layer_height_mm"] = min(preset["layer_height_mm"], fine_lh)
        if features["has_threads"] and preset["part_category"] == "General Purpose":
            preset["part_category"] = "Threaded / Precision Fit"

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
    overhang_ratio = float(printability.get("overhang_ratio", 0.0))
    needs_support = bool(printability.get("needs_support"))
    tree_support = (
        overhang_ratio > 0.08
        or float(printability.get("overhang_area_60deg_mm2", 0.0)) > 50.0
        or is_detail
        or features["has_internal_cavities"]
        or _has_any_word(combined_text, ("organic", "tree"))
    )
    slenderness = float(printability.get("slenderness_ratio", 1.0))
    needs_brim = material in {"ABS", "ASA"} or bool(printability.get("needs_brim"))
    brim_width_mm = 8 if material in {"ABS", "ASA"} or slenderness >= 3.5 else 6
    support_type = "tree" if tree_support else "normal"

    mat_defaults = _material_thermal_and_flow_defaults(
        material, nozzle_dia, printer_spec["speed_tier"], preset["layer_height_mm"]
    )

    # Code- and geometry-driven precision & surface adjustments
    precision_part = (
        features["has_threads"]
        or features["has_gears"]
        or features["has_mating_fits"]
        or is_mechanical
    )
    wall_sequence = (
        "inner/outer/inner"
        if (
            precision_part
            and not needs_support
            and overhang_ratio <= 0.04
            and preset["wall_loops"] >= 3
        )
        else "inner/outer"
    )
    seam_position = (
        "random"
        if (features["has_threads"] or features["has_gears"] or features["is_rotational"])
        else ("back" if is_detail else "aligned")
    )
    elefant_foot = 0.15 if precision_part else 0.10
    xy_hole_comp = (
        0.08
        if precision_part and material in ("PETG", "ABS", "ASA")
        else (0.04 if (features["has_threads"] or features["has_mating_fits"]) else 0.0)
    )
    outer_wall_lw = (
        round(nozzle_dia * 1.00, 2)
        if (features["has_fine_detail"] or features["has_threads"] or features["has_gears"] or effective_priority == "quality")
        else round(nozzle_dia * 1.05, 2)
    )
    only_one_wall_top = bool(features["has_flat_top"] or features["has_fine_detail"] or is_enclosure)
    ironing_type = (
        "topmost"
        if (features["has_flat_top"] and effective_priority == "quality" and material in ("PLA", "PETG"))
        else "no iron"
    )
    top_surface_pattern = "concentric" if features["is_rotational"] else "monotonicline"
    bottom_surface_pattern = "concentric" if features["is_rotational"] else "monotonic"
    support_buildplate_only = bool(
        support_enabled := needs_support
    ) and bool(features["has_internal_cavities"] or features["has_threads"])

    outer_wall_speed = mat_defaults["outer_wall_speed_mms"]
    if features["has_threads"] or features["has_gears"] or effective_priority == "quality":
        outer_wall_speed = max(30, round(outer_wall_speed * 0.70))

    hw_label = f" on {user_printer}" if user_printer else ""
    nozzle_label = f" ({nozzle_spec['raw']})" if nozzle_spec.get("raw") else f" ({nozzle_dia} mm nozzle)"

    reasons = [
        f"{preset['part_category']} profile using {material}{hw_label}{nozzle_label} ({preset['layer_height_mm']} mm layer height).",
        f"{preset['infill_density_percent']}% {preset['infill_pattern']} infill with {preset['wall_loops']} perimeters ({wall_sequence}).",
        (
            f"Support enabled ({support_type}{', build-plate only' if support_buildplate_only else ''}) for {overhang_area_45:.1f} mm² downward overhangs."
            if needs_support
            else "No support required (angles within safe overhang threshold)."
        ),
        (
            f"Brim ({brim_width_mm} mm) enabled to stabilize bed contact."
            if needs_brim
            else "Stable bed contact footprint, no brim needed."
        ),
    ]

    dx = float(dims.get("x", 0.0))
    dy = float(dims.get("y", 0.0))
    bx = float(printer_spec["bed_x_mm"])
    by = float(printer_spec["bed_y_mm"])
    exceeds_bed = (dx > bx or dy > by) and (dx > by or dy > bx)
    if exceeds_bed:
        reasons.append(
            f"WARNING: Model footprint ({dx:.1f} × {dy:.1f} mm) exceeds printer build plate ({bx:.0f} × {by:.0f} mm)."
        )

    is_abrasive = (
        _has_any_word(combined_text, ("cf", "carbon", "gf", "glass", "glow", "wood", "metal"))
        or any(tok in (default_material or "").lower() for tok in ("cf", "carbon", "gf", "glass", "glow", "wood", "metal"))
        or any(tok in (material_override or "").lower() for tok in ("cf", "carbon", "gf", "glass", "glow", "wood", "metal"))
    )
    if is_abrasive and nozzle_mat in ("Stainless Steel", "Brass"):
        reasons.append(
            f"Caution: Abrasive filament detected with {nozzle_mat} nozzle; a hardened steel nozzle is strongly advised."
        )

    profile_dict: dict[str, Any] = {
        **mat_defaults,
        **preset,
        "material": material,
        "nozzle_diameter_mm": nozzle_dia,
        "first_layer_height_mm": first_layer_lh,
        "outer_wall_line_width_mm": outer_wall_lw,
        "wall_generator": "arachne",
        "detect_thin_wall": True,
        "seam_position": seam_position,
        "wall_sequence": wall_sequence,
        "only_one_wall_top": only_one_wall_top,
        "elefant_foot_compensation_mm": elefant_foot,
        "xy_hole_compensation_mm": xy_hole_comp,
        "xy_contour_compensation_mm": 0.0,
        "precise_z_height": True,
        "ironing_type": ironing_type,
        "top_surface_pattern": top_surface_pattern,
        "bottom_surface_pattern": bottom_surface_pattern,
        "infill_wall_overlap_percent": 25 if (material == "TPU" or effective_priority == "strength") else 15,
        "support_enabled": needs_support,
        "support_type": support_type,
        "support_threshold_angle": 45,
        "support_on_build_plate_only": support_buildplate_only,
        "support_interface_top_layers": 3 if material == "PETG" else 2,
        "brim_enabled": needs_brim,
        "brim_type": "outer_only" if needs_brim else "no_brim",
        "brim_width_mm": brim_width_mm if needs_brim else 0,
        "outer_wall_speed_mms": outer_wall_speed,
        "bed_x_mm": printer_spec["bed_x_mm"],
        "bed_y_mm": printer_spec["bed_y_mm"],
        "rationale": " ".join(reasons),
    }
    if user_printer:
        profile_dict["target_printer"] = user_printer
    return sanitize_print_profile(
        profile_dict,
        nozzle_diameter_mm=nozzle_dia,
        user_printer=user_printer,
    )


def _load_metrics(project_dir: Path) -> dict[str, Any]:
    """Mesh metrics with printability telemetry for the current build.

    Uses ``.cad_metrics.json`` when it matches ``model.scad`` and already
    carries printability data; otherwise recomputes from ``preview.stl``.
    Raises ``FileNotFoundError`` if ``.cad_metrics.json`` belongs to an older
    revision than ``model.scad`` (unbuilt edits).
    """
    preview_stl = project_dir / "preview.stl"
    if not preview_stl.is_file() or preview_stl.stat().st_size == 0:
        raise FileNotFoundError("Build the model before requesting print settings.")
    current_sha = compute_model_sha256(project_dir)
    metrics_path = project_dir / ".cad_metrics.json"
    if metrics_path.is_file():
        try:
            cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = None
        if isinstance(cached, dict):
            recorded_sha = cached.get("model_sha256")
            if (
                isinstance(recorded_sha, str)
                and recorded_sha
                and current_sha
                and recorded_sha != current_sha
            ):
                raise FileNotFoundError(
                    "Model has unbuilt changes; build the model before requesting print settings."
                )
            metrics = cached.get("metrics")
            if (
                recorded_sha == current_sha
                and isinstance(metrics, dict)
                and "printability" in metrics
            ):
                return metrics
    from agent.tools.cad_scripts.renderer import load_stl
    from agent.tools.cad_scripts.runner import mesh_metrics

    vertices, triangles = load_stl(preview_stl)
    return mesh_metrics(vertices, triangles)


def _read_scad(project_dir: Path, limit: int = 4000) -> str:
    try:
        return (project_dir / "model.scad").read_text(encoding="utf-8")[:limit]
    except (OSError, UnicodeDecodeError):
        return ""


def _clean_user_message_text(text: str) -> str:
    """Strip synthetic prefixes and auto-generated /print templates from user message text."""
    trimmed = text.strip()
    if not trimmed:
        return ""
    if "Please analyze the current CAD model and provide a comprehensive 3D printing and slicing recommendation" in trimmed:
        match = re.search(r"### User Requirements & Specific Notes:\s*(.*?)(?:\s*###|\Z)", trimmed, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            if extracted and extracted != "Standard production print.":
                return extracted
        return ""
    return trimmed


def _read_recent_user_context(project_dir: Path, max_messages: int = 5) -> str:
    """Join the most recent plain-text and multimodal user messages to capture design intent."""
    user_texts: list[str] = []
    try:
        with (project_dir / "conversation.jsonl").open(encoding="utf-8") as conversation:
            for line in conversation:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(msg, dict) or msg.get("role") != "user":
                    continue
                if msg.get("synthetic"):
                    continue
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    cleaned = _clean_user_message_text(content)
                    if cleaned:
                        user_texts.append(cleaned)
                elif isinstance(content, list):
                    parts = [
                        _clean_user_message_text(str(part.get("text")).strip())
                        for part in content
                        if isinstance(part, dict)
                        and isinstance(part.get("text"), str)
                        and str(part.get("text")).strip()
                    ]
                    parts = [p for p in parts if p]
                    if parts:
                        user_texts.append(" ".join(parts))
    except OSError:
        return ""
    return " | ".join(user_texts[-max_messages:])


def _prepare_advisor_settings(settings: Any) -> Any:
    """Return lightweight settings for the AI print advisor (none/low effort and token budget)."""
    advisor_settings = settings
    try:
        effort = getattr(advisor_settings, "reasoning_effort", None)
        if effort in ("medium", "high", "xhigh", "max"):
            advisor_settings = with_reasoning_effort(advisor_settings, "none")
        if is_dataclass(advisor_settings) and hasattr(advisor_settings, "llm_max_completion_tokens"):
            advisor_settings = replace(advisor_settings, llm_max_completion_tokens=4096)
    except Exception:  # noqa: BLE001
        return settings
    return advisor_settings


def _publish_advisor_usage(
    publish: Callable[..., None] | None,
    project_name: str,
    usage: dict[str, Any] | None,
) -> None:
    if not callable(publish) or not isinstance(usage, dict) or not usage:
        return
    summary = extract_usage_summary(usage)
    p_val = summary["prompt_tokens"]
    c_val = summary["completion_tokens"]
    cached_tokens = summary["cached_tokens"]
    cost_val = summary["cost"]
    try:
        publish(
            "agent_usage",
            {
                "project": project_name,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cached_tokens": cached_tokens,
                "cache_write_tokens": summary["cache_write_tokens"],
                "cache_hit_rate": summary["cache_hit_rate"],
                "cost": cost_val,
                "total_cost": None,
                "total_prompt_tokens": p_val,
                "total_completion_tokens": c_val,
                "total_cached_tokens": cached_tokens,
                "advisor_call": True,
            },
        )
    except Exception:  # noqa: BLE001
        pass


def generate_ai_print_profile(
    project_dir: Path,
    settings: Any,
    *,
    priority: str = "auto",
    material_override: str = "auto",
    extra_context: str = "",
    stop_event: threading.Event | None = None,
    publish: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Query the active LLM to configure a complete 3MF slicer profile."""
    metrics = _load_metrics(project_dir)
    user_printer = (getattr(settings, "user_printer", "") or "").strip()
    user_nozzle = (getattr(settings, "user_nozzle", "") or "").strip()
    user_filament = (getattr(settings, "user_filament", "PLA") or "PLA").strip()
    user_additional_info = " ".join(str(getattr(settings, "user_additional_info", "") or "").split()).strip()
    nozzle_spec = parse_nozzle_spec(user_nozzle)
    nozzle_dia = nozzle_spec["diameter_mm"]

    recent_user_context = _read_recent_user_context(project_dir)
    code_text = _read_scad(project_dir)
    combined_context = f"{recent_user_context} {extra_context}".strip()

    baseline = build_heuristic_print_profile(
        metrics,
        code_snippet=code_text,
        priority=priority,
        material_override=material_override,
        user_context=combined_context,
        user_printer=user_printer,
        user_nozzle=user_nozzle,
        default_material=user_filament,
    )
    scad_features = _analyze_scad_features(code_text, combined_context)
    active_features = [k for k, v in scad_features.items() if v]

    dims = metrics.get("dimensions_mm") or {}
    p = metrics.get("printability") or {}
    vol = float(metrics.get("volume_mm3", 0.0))

    model_sha = compute_model_sha256(project_dir) or ""
    if not model_sha:
        try:
            model_sha = hashlib.sha256((project_dir / "preview.stl").read_bytes()).hexdigest()
        except OSError:
            model_sha = ""
    active_provider = getattr(settings, "llm_provider", "") or ""
    active_model = getattr(settings, f"{active_provider}_model", "") or getattr(settings, "model", "") or ""
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "schema_v": 4,
                "provider": active_provider,
                "model": active_model,
                "model_sha256": model_sha,
                "printer": user_printer,
                "nozzle": user_nozzle,
                "filament": user_filament,
                "additional_info": user_additional_info,
                "priority": priority,
                "material_override": material_override,
                "extra_context": extra_context,
                "user_context": recent_user_context,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cache_path = project_dir / ".cad-agent" / "print_profile_cache.json"
    if cache_path.is_file():
        try:
            cached_entry = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                isinstance(cached_entry, dict)
                and cached_entry.get("cache_key") == cache_key
                and isinstance(cached_entry.get("profile"), dict)
            ):
                cached_profile = sanitize_print_profile(
                    {**baseline, **cached_entry["profile"]},
                    nozzle_diameter_mm=nozzle_dia,
                    user_printer=user_printer,
                    fallback_profile=baseline,
                )
                cached_advice = (
                    str(cached_entry.get("advice_markdown") or "").strip()
                    or _format_heuristic_advice_markdown(
                        cached_profile, user_printer, user_nozzle
                    )
                )
                return {
                    "profile": cached_profile,
                    "advice_markdown": cached_advice,
                    "printability": p,
                    "dimensions_mm": dims,
                    "volume_mm3": vol,
                    "hardware": {
                        "printer": user_printer,
                        "nozzle": user_nozzle,
                        "filament": user_filament,
                        "additional_info": user_additional_info,
                    },
                    "source": "ai",
                    "cached": True,
                }
        except (OSError, ValueError):
            pass

    system_prompt = (
        "You are an expert additive manufacturing (3D printing) slicing engineer.\n"
        "Your output is embedded directly into a 3MF project archive (OrcaSlicer / Bambu Studio / PrusaSlicer) "
        "so the user can open the 3MF and immediately start printing without manual adjustments.\n"
        "Analyze the OpenSCAD code, exact geometry telemetry, detected mechanical features, and target hardware.\n"
        "Output ONLY a valid JSON object wrapped in ```json ... ``` with these keys:\n"
        "{\n"
        '  "material": "PLA" | "PETG" | "ASA" | "ABS" | "TPU",\n'
        '  "layer_height_mm": number,\n'
        '  "first_layer_height_mm": number,\n'
        '  "outer_wall_line_width_mm": number,\n'
        '  "inner_wall_line_width_mm": number,\n'
        '  "wall_loops": integer (2 to 8),\n'
        '  "top_shell_layers": integer (2 to 10),\n'
        '  "bottom_shell_layers": integer (2 to 10),\n'
        '  "wall_generator": "arachne" | "classic",\n'
        '  "seam_position": "aligned" | "back" | "random" | "nearest",\n'
        '  "wall_sequence": "inner/outer" | "outer/inner" | "inner/outer/inner",\n'
        '  "only_one_wall_top": boolean,\n'
        '  "elefant_foot_compensation_mm": number (0.0 to 0.30),\n'
        '  "xy_hole_compensation_mm": number (-0.20 to 0.30),\n'
        '  "ironing_type": "no iron" | "top" | "topmost",\n'
        '  "infill_density_percent": integer (multiple of 5, 0 to 100),\n'
        '  "infill_pattern": "gyroid" | "grid" | "cubic" | "adaptivecubic" | "honeycomb" | "3dhoneycomb" | "crosshatch" | "rectilinear" | "concentric" | "lightning",\n'
        '  "top_surface_pattern": "monotonicline" | "monotonic" | "concentric" | "rectilinear",\n'
        '  "bottom_surface_pattern": "monotonic" | "monotonicline" | "concentric",\n'
        '  "support_enabled": boolean,\n'
        '  "support_type": "normal" | "tree",\n'
        '  "support_threshold_angle": integer (25 to 60),\n'
        '  "support_on_build_plate_only": boolean,\n'
        '  "support_top_z_distance_mm": number (0.10 to 0.40),\n'
        '  "brim_enabled": boolean,\n'
        '  "brim_type": "no_brim" | "outer_only" | "inner_only" | "outer_and_inner" | "auto_brim",\n'
        '  "brim_width_mm": integer (0 to 15),\n'
        '  "nozzle_temp_initial_c": integer,\n'
        '  "nozzle_temp_c": integer,\n'
        '  "bed_temp_initial_c": integer,\n'
        '  "bed_temp_c": integer,\n'
        '  "fan_min_percent": integer (0 to 100),\n'
        '  "fan_max_percent": integer (0 to 100),\n'
        '  "max_volumetric_speed_mm3s": number,\n'
        '  "outer_wall_speed_mms": integer,\n'
        '  "inner_wall_speed_mms": integer,\n'
        '  "sparse_infill_speed_mms": integer,\n'
        '  "top_surface_speed_mms": integer,\n'
        '  "initial_layer_speed_mms": integer,\n'
        '  "part_category": string,\n'
        '  "rationale": string (concise 1-2 sentence technical summary)\n'
        "}"
    )

    hw_lines = [
        f"- 3D Printer: {user_printer or 'Standard FDM'} (Bed: {baseline['bed_x_mm']:.0f}×{baseline['bed_y_mm']:.0f} mm)",
        f"- Nozzle: {user_nozzle or '0.4 mm'}",
        f"- Default Filament: {user_filament}",
    ]
    if user_additional_info:
        hw_lines.append(f"- Additional Instructions: {user_additional_info}")
    hw_section = "\n".join(hw_lines)

    user_prompt = (
        f"### Target Hardware & Setup:\n"
        f"{hw_section}\n\n"
        f"### Exact Model Telemetry:\n"
        f"- Dimensions: {dims.get('x', 0):.1f} × {dims.get('y', 0):.1f} × {dims.get('z', 0):.1f} mm\n"
        f"- Volume: {vol:.1f} mm³ | Surface Area: {float(p.get('surface_area_mm2', 0)):.1f} mm²\n"
        f"- Bed Contact Footprint: {float(p.get('bed_contact_area_mm2', 0)):.1f} mm² ({float(p.get('bed_contact_ratio', 0))*100:.1f}%)\n"
        f"- Overhang Area (>45°): {float(p.get('overhang_area_45deg_mm2', 0)):.1f} mm² (Overhang ratio: {float(p.get('overhang_ratio', 0))*100:.1f}%)\n"
        f"- Overhang Area (>60°): {float(p.get('overhang_area_60deg_mm2', 0)):.1f} mm²\n"
        f"- Slenderness Ratio: {float(p.get('slenderness_ratio', 1.0)):.2f}\n"
        f"- Needs Support: {bool(p.get('needs_support'))} | Needs Brim: {bool(p.get('needs_brim'))}\n"
        f"- Detected Code Features: {', '.join(active_features) if active_features else 'standard solid geometry'}\n\n"
        f"### Heuristic Baseline JSON:\n```json\n{json.dumps(baseline, indent=2)}\n```\n\n"
        f"### User Context & Intent:\n{combined_context or 'Standard CAD part'}\n\n"
        f"### OpenSCAD Code:\n```scad\n{code_text[:2000]}\n```\n"
    )

    ai_error: str | None = None
    try:
        from agent.llm_base import create_llm_client

        advisor_settings = _prepare_advisor_settings(settings)
        client = create_llm_client(advisor_settings)
        if stop_event is not None:
            client.stop_event = stop_event
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            resp = client.chat(messages, max_attempts=2)
        except TypeError:
            resp = client.chat(messages)
        _publish_advisor_usage(publish, project_dir.name, getattr(client, "last_usage", None))
        content = ""
        if isinstance(resp, dict):
            choices = resp.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(msg, dict):
                    content = str(msg.get("content") or "")
            if not content:
                content = str(resp.get("content") or "")
        else:
            content = str(getattr(resp, "content", "") or "")
        if isinstance(content, str) and content.strip():
            json_block = _extract_json_block(content)
            if json_block and isinstance(json_block, dict):
                ai_mat_raw = json_block.get("material")
                ai_mat = normalize_material(str(ai_mat_raw), baseline["material"]) if ai_mat_raw else baseline["material"]
                if ai_mat != baseline["material"]:
                    new_mat_defaults = _material_thermal_and_flow_defaults(
                        ai_mat, nozzle_dia, printer_spec["speed_tier"], baseline["layer_height_mm"]
                    )
                    effective_baseline = {**baseline, **new_mat_defaults, "material": ai_mat}
                else:
                    effective_baseline = baseline

                merged_raw = {**effective_baseline, **json_block}
                if user_printer:
                    merged_raw["target_printer"] = user_printer
                ai_profile = sanitize_print_profile(
                    merged_raw,
                    nozzle_diameter_mm=nozzle_dia,
                    user_printer=user_printer,
                    fallback_profile=effective_baseline,
                )
                ai_advice = json_block.get("advice_markdown")
                final_advice = (
                    ai_advice.strip()
                    if isinstance(ai_advice, str) and ai_advice.strip()
                    else _format_heuristic_advice_markdown(
                        ai_profile, user_printer, user_nozzle
                    )
                )
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps(
                            {
                                "cache_key": cache_key,
                                "profile": ai_profile,
                                "advice_markdown": final_advice,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                except OSError:
                    pass
                return {
                    "profile": ai_profile,
                    "advice_markdown": final_advice,
                    "printability": p,
                    "dimensions_mm": dims,
                    "volume_mm3": vol,
                    "hardware": {
                        "printer": user_printer,
                        "nozzle": user_nozzle,
                        "filament": user_filament,
                        "additional_info": user_additional_info,
                    },
                    "source": "ai",
                    "cached": False,
                }
            ai_error = "AI response did not contain a valid JSON print profile."
        else:
            ai_error = "AI response was empty."
    except Exception as exc:  # noqa: BLE001
        if stop_event is not None and stop_event.is_set():
            raise
        _LOG.warning("AI 3MF profile generation failed; falling back to heuristic: %s", exc)
        ai_error = str(exc)

    advice_md = _format_heuristic_advice_markdown(baseline, user_printer, user_nozzle)
    fallback_result: dict[str, Any] = {
        "profile": baseline,
        "advice_markdown": advice_md,
        "printability": p,
        "dimensions_mm": dims,
        "volume_mm3": vol,
        "hardware": {
            "printer": user_printer,
            "nozzle": user_nozzle,
            "filament": user_filament,
            "additional_info": user_additional_info,
        },
        "source": "heuristic",
        "cached": False,
    }
    if ai_error:
        fallback_result["fallback"] = True
        fallback_result["ai_error"] = ai_error
    return fallback_result


def recommend_print_profile(
    project_dir: Path,
    *,
    priority: str = "auto",
    material_override: str = "auto",
    extra_context: str = "",
    settings: Any | None = None,
    use_ai: bool = False,
    stop_event: threading.Event | None = None,
    publish: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Recommend a print profile for the project's current build.

    Raises FileNotFoundError when the model has not been built yet.
    """
    if use_ai and settings is not None:
        return generate_ai_print_profile(
            project_dir,
            settings,
            priority=priority,
            material_override=material_override,
            extra_context=extra_context,
            stop_event=stop_event,
            publish=publish,
        )

    metrics = _load_metrics(project_dir)
    user_printer = (getattr(settings, "user_printer", "") or "").strip() if settings else ""
    user_nozzle = (getattr(settings, "user_nozzle", "") or "").strip() if settings else ""
    user_filament = (getattr(settings, "user_filament", "PLA") or "PLA").strip() if settings else "PLA"
    user_additional_info = (
        " ".join(str(getattr(settings, "user_additional_info", "") or "").split()).strip()
        if settings
        else ""
    )

    profile = build_heuristic_print_profile(
        metrics,
        code_snippet=_read_scad(project_dir),
        priority=priority,
        material_override=material_override,
        user_context=f"{_read_recent_user_context(project_dir)} {extra_context}".strip(),
        user_printer=user_printer,
        user_nozzle=user_nozzle,
        default_material=user_filament,
    )
    dims = metrics.get("dimensions_mm") or {}
    p = metrics.get("printability") or {}
    return {
        "profile": profile,
        "advice_markdown": _format_heuristic_advice_markdown(profile, user_printer, user_nozzle),
        "printability": p,
        "dimensions_mm": dims,
        "volume_mm3": float(metrics.get("volume_mm3", 0.0)),
        "hardware": {
            "printer": user_printer,
            "nozzle": user_nozzle,
            "filament": user_filament,
            "additional_info": user_additional_info,
        },
        "source": "heuristic",
        "cached": False,
    }


def format_print_command_prompt(
    project_dir: Path, extra_notes: str = "", settings: Any | None = None
) -> str:
    """Build the ``/print`` chat prompt: exact mesh telemetry plus the heuristic baseline."""
    result = recommend_print_profile(project_dir, extra_context=extra_notes, settings=settings)
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

    hw_lines = []
    if settings:
        printer = str(getattr(settings, "user_printer", "") or "").strip()
        nozzle = str(getattr(settings, "user_nozzle", "") or "").strip()
        filament = str(getattr(settings, "user_filament", "") or "").strip()
        additional_info = " ".join(
            str(getattr(settings, "user_additional_info", "") or "").split()
        ).strip()
        if printer:
            hw_lines.append(f"- Target 3D Printer: {printer}")
        if nozzle:
            hw_lines.append(f"- Nozzle: {nozzle}")
        if filament:
            hw_lines.append(f"- Default Filament: {filament}")
        if additional_info:
            hw_lines.append(f"- Additional Instructions: {additional_info}")
    hw_block = ("### User Hardware Setup:\n" + "\n".join(hw_lines) + "\n\n") if hw_lines else ""

    return (
        "Please analyze the current CAD model and provide a comprehensive 3D printing and slicing recommendation.\n\n"
        f"{hw_block}"
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
        "### Baseline Print Settings:\n"
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
        "1. **Slicer Profile Recommendations:** Optimal material, layer heights, wall counts, infill pattern/density, and temperature/cooling advice for the target printer and nozzle.\n"
        "2. **Build Plate Orientation:** How to orient the part on the print bed to balance mechanical strength (layer line shear), overhangs, and surface finish.\n"
        "3. **Support & Adhesion Strategy:** Tree vs. standard supports, critical overhang areas, contact z-distance, and brim requirements.\n"
        "4. **Manufacturing Warnings & Tolerances:** Warping risks, dimensional tolerances for fitting parts, and post-processing suggestions."
    )
