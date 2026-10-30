"""LLM-driven 3MF print profile generator and schema sanitizer."""

from __future__ import annotations

from collections.abc import Callable
import functools
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

from agent.llm_base import extract_usage_summary
from agent.revisions import compute_model_sha256

_LOG = logging.getLogger(__name__)

ALLOWED_MATERIALS: tuple[str, ...] = ("PA-CF", "PETG", "HIPS", "PLA", "ASA", "ABS", "TPU", "PVA", "PA", "PC")
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
    "concentric",
    "lightning",
)
ALLOWED_SURFACE_PATTERNS: tuple[str, ...] = (
    "monotonicline",
    "monotonic",
    "concentric",
    "rectilinear",
    "zig-zag",
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
ALLOWED_SEAM_POSITIONS: tuple[str, ...] = ("aligned", "back", "random", "nearest")
ALLOWED_WALL_SEQUENCES: tuple[str, ...] = (
    "inner/outer",
    "outer/inner",
    "inner/outer/inner",
)
ALLOWED_IRONING_TYPES: tuple[str, ...] = ("no iron", "top", "topmost")
ALLOWED_PRINTER_STRUCTURES: tuple[str, ...] = ("corexy", "i3", "delta")

_CHOICE_ALIASES: dict[str, dict[str, str]] = {
    "infill_pattern": {"zigzag": "zig-zag", "line": "rectilinear"},
    "top_surface_pattern": {"monotoniclines": "monotonicline", "zigzag": "zig-zag"},
    "bottom_surface_pattern": {"monotoniclines": "monotonicline", "zigzag": "zig-zag"},
    "support_type": {
        "tree(auto)": "tree",
        "organic": "tree",
        "normal(auto)": "normal",
        "grid": "normal",
    },
    "seam_position": {"rear": "back"},
    "wall_sequence": {
        "inner wall/outer wall": "inner/outer",
        "outer wall/inner wall": "outer/inner",
        "inner-outer-inner wall": "inner/outer/inner",
    },
    "ironing_type": {
        "no ironing": "no iron",
        "none": "no iron",
        "off": "no iron",
        "all top": "top",
        "alltop": "top",
    },
}

_DEFAULTS: dict[str, Any] = {
    "material": "PLA",
    "filament_density_g_cm3": 1.24,
    "filament_cost_per_kg": 20.0,
    "filament_flow_ratio": 0.98,
    "filament_id": "GFL99",
    "printer_model": "Bambu Lab A1",
    "printer_structure": "i3",
    "bambu_printer_settings_id": "Bambu Lab A1 0.4 nozzle",
    "bambu_print_settings_id": "0.20mm Standard @BBL A1",
    "bambu_filament_settings_id": "Generic PLA @BBL A1",
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
    "support_line_width_mm": 0.42,
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
    "default_acceleration_mms2": 6000,
    "outer_wall_acceleration_mms2": 3000,
    "inner_wall_acceleration_mms2": 6000,
    "top_surface_acceleration_mms2": 2000,
    "initial_layer_acceleration_mms2": 500,
    "travel_acceleration_mms2": 10000,
    "retraction_length_mm": 0.8,
    "retraction_speed_mms": 30,
    "z_hop_mm": 0.4,
    "bed_x_mm": 256.0,
    "bed_y_mm": 256.0,
    "bed_z_mm": 256.0,
    "part_category": "General Purpose",
    "rationale": "",
}


def normalize_material(raw_material: str | None, default: str = "PLA") -> str:
    """Clean a material label while preserving standard filament identifiers."""
    text = (raw_material or "").strip().upper()
    if not text or text == "AUTO":
        return default
    if "PET-G" in text:
        return "PETG"
    for mat in ALLOWED_MATERIALS:
        if re.search(rf"(?<![A-Z0-9]){re.escape(mat)}(?![A-Z0-9])", text):
            return mat
    if "FLEX" in text:
        return "TPU"
    if "NYLON" in text:
        return "PA"
    cleaned = re.sub(r"[^A-Z0-9+-]", "", text)[:16]
    return cleaned or default


def _number(
    raw: dict[str, Any],
    key: str,
    lo: float,
    hi: float,
    fallback: float | None = None,
) -> float:
    """Read ``raw[key]`` as a finite float clamped to ``[lo, hi]``."""
    default_val = _DEFAULTS[key] if fallback is None else fallback
    val = raw.get(key)
    if val is None or isinstance(val, bool):
        value = float(default_val)
    else:
        try:
            value = float(val)
        except (TypeError, ValueError):
            value = float(default_val)
    if not math.isfinite(value):
        value = float(default_val)
    return min(hi, max(lo, value))


def _bool(raw: dict[str, Any], key: str, fallback: bool | None = None) -> bool:
    default_val = bool(_DEFAULTS[key] if fallback is None else fallback)
    val = raw.get(key)
    if val is None:
        return default_val
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
    val = raw.get(key)
    if val is None:
        return default_val
    value = str(val).strip().lower()
    value = _CHOICE_ALIASES.get(key, {}).get(value, value)
    return value if value in allowed else default_val


def _text(raw: dict[str, Any], key: str, limit: int) -> str:
    return re.sub(r"\s+", " ", str(raw.get(key) or "")).strip()[:limit]


def parse_nozzle_spec(nozzle_str: str | None) -> dict[str, Any]:
    """Extract numeric nozzle diameter (mm) from a user nozzle string."""
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
    return {
        "diameter_mm": round(dia, 2),
        "raw": text,
    }


def parse_printer_spec(printer_str: str | None) -> dict[str, Any]:
    """Parse explicit build plate dimensions (e.g. '256x256') if present in printer string."""
    text = (printer_str or "").strip()
    bed_x, bed_y = 256.0, 256.0
    dim_match = re.search(r"\b(\d{3})\s*[x×]\s*(\d{3})\b", text.lower())
    if dim_match:
        bx, by = float(dim_match.group(1)), float(dim_match.group(2))
        if 100.0 <= bx <= 600.0 and 100.0 <= by <= 600.0:
            bed_x, bed_y = bx, by
    return {
        "name": text,
        "bed_x_mm": bed_x,
        "bed_y_mm": bed_y,
    }


def sanitize_print_profile(
    raw: Any,
    *,
    nozzle_diameter_mm: float = 0.4,
    user_printer: str = "",
) -> dict[str, Any]:
    """Validate types and clamp an LLM-generated print profile into safe physical bounds."""
    if not isinstance(raw, dict):
        raw = {}
    else:
        raw = dict(raw)
        if "first_layer_height_mm" not in raw and "initial_layer_height_mm" in raw:
            raw["first_layer_height_mm"] = raw["initial_layer_height_mm"]
        if "elefant_foot_compensation_mm" not in raw and "elephant_foot_compensation_mm" in raw:
            raw["elefant_foot_compensation_mm"] = raw["elephant_foot_compensation_mm"]
        if "support_threshold_angle" not in raw and "support_threshold_angle_deg" in raw:
            raw["support_threshold_angle"] = raw["support_threshold_angle_deg"]

    material = normalize_material(str(raw.get("material") or "PLA"), "PLA")

    raw_dia = raw.get("nozzle_diameter_mm")
    try:
        parsed_dia = float(raw_dia) if raw_dia is not None and not isinstance(raw_dia, bool) else float(nozzle_diameter_mm or 0.4)
    except (ValueError, TypeError):
        parsed_dia = parse_nozzle_spec(str(raw_dia))["diameter_mm"] if raw_dia else float(nozzle_diameter_mm or 0.4)
    nozzle_dia = max(0.15, min(1.2, parsed_dia))

    printer_model = (
        _text(raw, "printer_model", 80)
        or _text(raw, "target_printer", 80)
        or _text({"p": user_printer}, "p", 80)
        or _DEFAULTS["printer_model"]
    )
    printer_spec = parse_printer_spec(printer_model)

    min_lw = round(max(0.15, nozzle_dia), 2)
    max_lw = round(min(1.80, nozzle_dia * 1.60), 2)
    def_lw = round(nozzle_dia * 1.05, 2)
    line_width = round(_number(raw, "line_width_mm", min_lw, max_lw, def_lw), 2)
    lw_fields = {
        key: round(_number(raw, key, min_lw, max_lw, line_width), 2)
        for key in (
            "initial_layer_line_width_mm",
            "outer_wall_line_width_mm",
            "inner_wall_line_width_mm",
            "top_surface_line_width_mm",
            "sparse_infill_line_width_mm",
            "internal_solid_infill_line_width_mm",
            "support_line_width_mm",
        )
    }
    min_active_lw = min(
        line_width,
        *(val for key, val in lw_fields.items() if key != "initial_layer_line_width_mm"),
    )

    min_layer = round(max(0.04, nozzle_dia * 0.20), 2)
    max_layer = max(min_layer, round(min(0.80, nozzle_dia * 0.70, min_active_lw * 0.75), 2))
    min_first = round(max(0.08, nozzle_dia * 0.25), 2)
    max_first = max(
        min_first,
        round(min(0.80, nozzle_dia * 0.70, lw_fields["initial_layer_line_width_mm"] * 0.75), 2),
    )
    def_layer = min(max_layer, round(nozzle_dia * 0.50, 2))

    layer_height = round(_number(raw, "layer_height_mm", min_layer, max_layer, def_layer), 2)
    first_layer = round(_number(raw, "first_layer_height_mm", min_first, max_first, def_layer), 2)

    raw_brim_type = _choice(raw, "brim_type", ALLOWED_BRIM_TYPES)
    raw_brim_width = _number(raw, "brim_width_mm", 0, 20, 0)
    if "brim_enabled" in raw and raw["brim_enabled"] is not None:
        brim_enabled = _bool(raw, "brim_enabled")
    else:
        brim_enabled = raw_brim_type != "no_brim" or raw_brim_width > 0
    if not brim_enabled:
        brim_type = "no_brim"
        brim_width = 0
    else:
        brim_type = "outer_only" if raw_brim_type == "no_brim" else raw_brim_type
        brim_width = round(min(20.0, max(2.0, raw_brim_width if raw_brim_width > 0 else 6.0)))

    bed_x = round(_number(raw, "bed_x_mm", 100.0, 600.0, printer_spec["bed_x_mm"]), 1)
    bed_y = round(_number(raw, "bed_y_mm", 100.0, 600.0, printer_spec["bed_y_mm"]), 1)
    bed_z = round(_number(raw, "bed_z_mm", 100.0, 600.0, min(bed_x, bed_y)), 1)

    nozzle_tag = f"{nozzle_dia:g}"
    printer_tag = re.sub(r"^Bambu Lab\s+", "BBL ", printer_model, flags=re.IGNORECASE)
    nozzle_suffix = "" if abs(nozzle_dia - 0.4) < 1e-3 else f" {nozzle_tag} nozzle"
    bambu_printer_id = (
        _text(raw, "bambu_printer_settings_id", 100)
        or f"{printer_model} {nozzle_tag} nozzle"
    )
    bambu_print_id = (
        _text(raw, "bambu_print_settings_id", 100)
        or f"{layer_height:.2f}mm Standard @{printer_tag}{nozzle_suffix}"
    )
    bambu_filament_id = (
        _text(raw, "bambu_filament_settings_id", 100)
        or f"Generic {material} @{printer_tag}"
    )

    max_elefant = round(min(0.40, nozzle_dia * 0.50), 2)
    min_sup_z = round(max(0.05, layer_height * 0.50), 2)
    max_sup_z = round(min(0.50, layer_height * 1.50), 2)

    sanitized: dict[str, Any] = {
        "material": material,
        "filament_density_g_cm3": round(_number(raw, "filament_density_g_cm3", 0.50, 3.00), 2),
        "filament_cost_per_kg": round(_number(raw, "filament_cost_per_kg", 1.0, 500.0), 2),
        "filament_flow_ratio": round(_number(raw, "filament_flow_ratio", 0.80, 1.20), 3),
        "filament_id": _text(raw, "filament_id", 20) or _DEFAULTS["filament_id"],
        "printer_model": printer_model,
        "printer_structure": _choice(raw, "printer_structure", ALLOWED_PRINTER_STRUCTURES),
        "bambu_printer_settings_id": bambu_printer_id,
        "bambu_print_settings_id": bambu_print_id,
        "bambu_filament_settings_id": bambu_filament_id,
        "nozzle_diameter_mm": round(nozzle_dia, 2),
        "layer_height_mm": layer_height,
        "first_layer_height_mm": first_layer,
        "line_width_mm": line_width,
        **lw_fields,
        "wall_loops": round(_number(raw, "wall_loops", 1, 10)),
        "top_shell_layers": round(_number(raw, "top_shell_layers", 1, 12)),
        "bottom_shell_layers": round(_number(raw, "bottom_shell_layers", 1, 12)),
        "wall_generator": _choice(raw, "wall_generator", ALLOWED_WALL_GENERATORS),
        "detect_thin_wall": _bool(raw, "detect_thin_wall"),
        "seam_position": _choice(raw, "seam_position", ALLOWED_SEAM_POSITIONS),
        "wall_sequence": _choice(raw, "wall_sequence", ALLOWED_WALL_SEQUENCES),
        "only_one_wall_top": _bool(raw, "only_one_wall_top"),
        "elefant_foot_compensation_mm": round(_number(raw, "elefant_foot_compensation_mm", 0.0, max_elefant), 2),
        "xy_hole_compensation_mm": round(_number(raw, "xy_hole_compensation_mm", -0.50, 0.50), 2),
        "xy_contour_compensation_mm": round(_number(raw, "xy_contour_compensation_mm", -0.50, 0.50), 2),
        "precise_z_height": _bool(raw, "precise_z_height"),
        "ironing_type": _choice(raw, "ironing_type", ALLOWED_IRONING_TYPES),
        "ironing_flow_percent": round(_number(raw, "ironing_flow_percent", 5, 40)),
        "ironing_speed_mms": round(_number(raw, "ironing_speed_mms", 10, 100)),
        "infill_density_percent": round(_number(raw, "infill_density_percent", 0, 100)),
        "infill_pattern": _choice(raw, "infill_pattern", ALLOWED_INFILL_PATTERNS),
        "top_surface_pattern": _choice(raw, "top_surface_pattern", ALLOWED_SURFACE_PATTERNS),
        "bottom_surface_pattern": _choice(raw, "bottom_surface_pattern", ALLOWED_SURFACE_PATTERNS),
        "infill_wall_overlap_percent": round(_number(raw, "infill_wall_overlap_percent", 5, 50)),
        "support_enabled": _bool(raw, "support_enabled"),
        "support_type": _choice(raw, "support_type", ALLOWED_SUPPORT_TYPES),
        "support_threshold_angle": round(_number(raw, "support_threshold_angle", 15, 75)),
        "support_on_build_plate_only": _bool(raw, "support_on_build_plate_only"),
        "support_top_z_distance_mm": round(_number(raw, "support_top_z_distance_mm", min_sup_z, max_sup_z, layer_height), 2),
        "support_bottom_z_distance_mm": round(_number(raw, "support_bottom_z_distance_mm", min_sup_z, max_sup_z, layer_height), 2),
        "support_interface_top_layers": round(_number(raw, "support_interface_top_layers", 0, 6)),
        "bridge_flow": round(_number(raw, "bridge_flow", 0.70, 1.30), 2),
        "thick_bridges": _bool(raw, "thick_bridges"),
        "brim_enabled": brim_enabled,
        "brim_type": brim_type,
        "brim_width_mm": brim_width,
        "nozzle_temp_initial_c": round(_number(raw, "nozzle_temp_initial_c", 170, 320)),
        "nozzle_temp_c": round(_number(raw, "nozzle_temp_c", 170, 320)),
        "bed_temp_initial_c": round(_number(raw, "bed_temp_initial_c", 25, 130)),
        "bed_temp_c": round(_number(raw, "bed_temp_c", 25, 130)),
        "fan_min_percent": round(_number(raw, "fan_min_percent", 0, 100)),
        "fan_max_percent": round(_number(raw, "fan_max_percent", 0, 100)),
        "overhang_fan_percent": round(_number(raw, "overhang_fan_percent", 0, 100)),
        "slow_down_for_layer_cooling": _bool(raw, "slow_down_for_layer_cooling"),
        "max_volumetric_speed_mm3s": round(_number(raw, "max_volumetric_speed_mm3s", 1.0, 40.0), 1),
        "initial_layer_speed_mms": round(_number(raw, "initial_layer_speed_mms", 10, 150)),
        "initial_layer_infill_speed_mms": round(_number(raw, "initial_layer_infill_speed_mms", 10, 250)),
        "outer_wall_speed_mms": round(_number(raw, "outer_wall_speed_mms", 10, 400)),
        "inner_wall_speed_mms": round(_number(raw, "inner_wall_speed_mms", 10, 500)),
        "sparse_infill_speed_mms": round(_number(raw, "sparse_infill_speed_mms", 10, 600)),
        "internal_solid_infill_speed_mms": round(_number(raw, "internal_solid_infill_speed_mms", 10, 500)),
        "top_surface_speed_mms": round(_number(raw, "top_surface_speed_mms", 10, 300)),
        "bridge_speed_mms": round(_number(raw, "bridge_speed_mms", 5, 150)),
        "support_speed_mms": round(_number(raw, "support_speed_mms", 10, 300)),
        "travel_speed_mms": round(_number(raw, "travel_speed_mms", 50, 1000)),
        "slow_down_overhangs": _bool(raw, "slow_down_overhangs"),
        "default_acceleration_mms2": round(_number(raw, "default_acceleration_mms2", 100, 30000)),
        "outer_wall_acceleration_mms2": round(_number(raw, "outer_wall_acceleration_mms2", 100, 20000)),
        "inner_wall_acceleration_mms2": round(_number(raw, "inner_wall_acceleration_mms2", 100, 30000)),
        "top_surface_acceleration_mms2": round(_number(raw, "top_surface_acceleration_mms2", 100, 15000)),
        "initial_layer_acceleration_mms2": round(_number(raw, "initial_layer_acceleration_mms2", 100, 5000)),
        "travel_acceleration_mms2": round(_number(raw, "travel_acceleration_mms2", 500, 30000)),
        "retraction_length_mm": round(_number(raw, "retraction_length_mm", 0.0, 10.0), 2),
        "retraction_speed_mms": round(_number(raw, "retraction_speed_mms", 5, 120)),
        "z_hop_mm": round(_number(raw, "z_hop_mm", 0.0, 2.0), 2),
        "bed_x_mm": bed_x,
        "bed_y_mm": bed_y,
        "bed_z_mm": bed_z,
        "part_category": _text(raw, "part_category", 60) or _DEFAULTS["part_category"],
        "rationale": _text(raw, "rationale", 500),
    }

    # Universal FDM cross-field physical invariants
    if sanitized["infill_density_percent"] == 100 and sanitized["infill_pattern"] not in (
        "rectilinear",
        "zig-zag",
        "concentric",
    ):
        sanitized["infill_pattern"] = "rectilinear"
    sanitized["nozzle_temp_initial_c"] = max(
        sanitized["nozzle_temp_initial_c"], sanitized["nozzle_temp_c"]
    )
    sanitized["bed_temp_initial_c"] = max(
        sanitized["bed_temp_initial_c"], sanitized["bed_temp_c"]
    )
    sanitized["fan_min_percent"] = min(
        sanitized["fan_min_percent"], sanitized["fan_max_percent"]
    )
    sanitized["outer_wall_speed_mms"] = min(
        sanitized["outer_wall_speed_mms"], sanitized["inner_wall_speed_mms"]
    )
    sanitized["initial_layer_speed_mms"] = min(
        sanitized["initial_layer_speed_mms"], sanitized["outer_wall_speed_mms"]
    )
    sanitized["outer_wall_acceleration_mms2"] = min(
        sanitized["outer_wall_acceleration_mms2"], sanitized["inner_wall_acceleration_mms2"]
    )
    sanitized["initial_layer_acceleration_mms2"] = min(
        sanitized["initial_layer_acceleration_mms2"], sanitized["outer_wall_acceleration_mms2"]
    )
    return sanitized


def _extract_json_block(text: str) -> dict[str, Any] | None:
    """Extract and parse the first valid JSON object from a markdown response."""
    decoder = json.JSONDecoder()
    profile_keys = set(_DEFAULTS) | {"advice_markdown"}
    fallback_dict: dict[str, Any] | None = None

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


def _format_profile_summary_markdown(
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


def _load_metrics(project_dir: Path) -> dict[str, Any]:
    """Mesh metrics with printability telemetry for the current build."""
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


def _extract_llm_content(resp: Any) -> str:
    if isinstance(resp, dict):
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(msg, dict):
                content = str(msg.get("content") or "")
                if content:
                    return content
        return str(resp.get("content") or "")
    return str(getattr(resp, "content", "") or "")


_MINIMAL_3MF_MODEL_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
    "\t<resources>\n"
    '\t\t<object id="1" type="model">\n'
    "\t\t\t<mesh>\n"
    "\t\t\t\t<vertices>\n"
    '\t\t\t\t\t<vertex x="0" y="0" z="0" />\n'
    '\t\t\t\t\t<vertex x="10" y="0" z="0" />\n'
    '\t\t\t\t\t<vertex x="10" y="10" z="0" />\n'
    '\t\t\t\t\t<vertex x="0" y="10" z="0" />\n'
    '\t\t\t\t\t<vertex x="0" y="0" z="10" />\n'
    '\t\t\t\t\t<vertex x="10" y="0" z="10" />\n'
    '\t\t\t\t\t<vertex x="10" y="10" z="10" />\n'
    '\t\t\t\t\t<vertex x="0" y="10" z="10" />\n'
    "\t\t\t\t</vertices>\n"
    "\t\t\t\t<triangles>\n"
    '\t\t\t\t\t<triangle v1="0" v2="2" v3="1" />\n'
    '\t\t\t\t\t<triangle v1="0" v2="3" v3="2" />\n'
    '\t\t\t\t\t<triangle v1="4" v2="5" v3="6" />\n'
    '\t\t\t\t\t<triangle v1="4" v2="6" v3="7" />\n'
    '\t\t\t\t\t<triangle v1="0" v2="1" v3="5" />\n'
    '\t\t\t\t\t<triangle v1="0" v2="5" v3="4" />\n'
    '\t\t\t\t\t<triangle v1="1" v2="2" v3="6" />\n'
    '\t\t\t\t\t<triangle v1="1" v2="6" v3="5" />\n'
    '\t\t\t\t\t<triangle v1="2" v2="3" v3="7" />\n'
    '\t\t\t\t\t<triangle v1="2" v2="7" v3="6" />\n'
    '\t\t\t\t\t<triangle v1="3" v2="0" v3="4" />\n'
    '\t\t\t\t\t<triangle v1="3" v2="4" v3="7" />\n'
    "\t\t\t\t</triangles>\n"
    "\t\t\t</mesh>\n"
    "\t\t</object>\n"
    "\t</resources>\n"
    "\t<build>\n"
    '\t\t<item objectid="1" />\n'
    "\t</build>\n"
    "</model>\n"
)


@functools.lru_cache(maxsize=1)
def _discover_slicer_cli() -> tuple[str, list[str]] | None:
    """Locate an installed headless slicer CLI (OrcaSlicer, PrusaSlicer/SuperSlicer, or Bambu Studio)."""
    for bin_name, kind in (
        ("orca-slicer", "orca_bambu"),
        ("orcaslicer", "orca_bambu"),
        ("prusa-slicer", "prusa"),
        ("prusaslicer", "prusa"),
        ("superslicer", "prusa"),
        ("bambu-studio", "orca_bambu"),
    ):
        found = shutil.which(bin_name)
        if found:
            return kind, [found]
    if shutil.which("flatpak"):
        user_flatpak = Path.home() / ".local" / "share" / "flatpak" / "app"
        sys_flatpak = Path("/var/lib/flatpak/app")
        for app_id, kind in (
            ("io.github.softfever.OrcaSlicer", "orca_bambu"),
            ("com.prusa3d.PrusaSlicer", "prusa"),
            ("com.bambulab.BambuStudio", "orca_bambu"),
        ):
            if (sys_flatpak / app_id).is_dir() or (user_flatpak / app_id).is_dir():
                return kind, ["flatpak", "run", app_id]
    return None


def _validate_profile_universal(
    profile: dict[str, Any], project_dir: Path | None = None
) -> str | None:
    """Validate 3MF OPC/XML structure, dual slicer dialects, and headless CLI slicing."""
    from agent.converter import (
        BAMBU_CONFIG_ENTRY,
        MODEL_SETTINGS_ENTRY,
        PRUSA_CONFIG_ENTRY,
        SLICE_INFO_ENTRY,
        enrich_3mf_archive,
    )

    resolved_proj = project_dir.resolve() if project_dir is not None else None
    if resolved_proj is not None and not str(resolved_proj).startswith("/tmp/"):
        work_parent = resolved_proj / ".cad-agent"
    else:
        work_parent = Path(__file__).resolve().parent.parent / "projects"
    work_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=work_parent, prefix=".tmp_slice_val_"))
    try:
        base_3mf = tmp_dir / "base.3mf"
        in_3mf = tmp_dir / "in.3mf"
        with zipfile.ZipFile(base_3mf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "[Content_Types].xml",
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
                '\t<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />\n'
                '\t<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />\n'
                "</Types>\n",
            )
            zf.writestr(
                "_rels/.rels",
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
                '\t<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />\n'
                "</Relationships>\n",
            )
            zf.writestr("3D/3dmodel.model", _MINIMAL_3MF_MODEL_XML)

        enrich_3mf_archive(
            base_3mf,
            in_3mf,
            profile=profile,
            project_name=(resolved_proj.name if resolved_proj is not None else "") or "model",
            thumbnail=None,
        )

        # Stage 1: Universal 3MF XML & dual-dialect structural validation
        bad_tokens = ("none", "nan", "inf", "-inf")
        with zipfile.ZipFile(in_3mf, "r") as zin:
            for xml_entry in (
                "[Content_Types].xml",
                "_rels/.rels",
                "3D/3dmodel.model",
                MODEL_SETTINGS_ENTRY,
                SLICE_INFO_ENTRY,
            ):
                try:
                    ET.fromstring(zin.read(xml_entry))
                except Exception as exc:  # noqa: BLE001
                    return f"Malformed XML in {xml_entry}: {exc}"

            prusa_text = zin.read(PRUSA_CONFIG_ENTRY).decode("utf-8")
            for line in prusa_text.splitlines()[1:]:
                if not line.strip():
                    continue
                _, sep, rhs = line[1:].partition("=")
                if not re.match(r"^;\s*[a-z0-9_]+\s*=.*$", line) or not sep or rhs.strip().lower() in bad_tokens:
                    return f"Invalid PrusaSlicer config line: {line}"

            sent_cfg = json.loads(zin.read(BAMBU_CONFIG_ENTRY).decode("utf-8"))
            for key, val in sent_cfg.items():
                if isinstance(val, list) and key != "printable_area":
                    if len(val) != 1 or str(val[0]).strip().lower() in (*bad_tokens, ""):
                        return f"Invalid extruder/filament array value for {key}: {val!r}"
                elif str(val).strip().lower() in bad_tokens:
                    return f"Invalid scalar value for {key}: {val!r}"

        # Stage 2: Headless slicer CLI verification (if a slicer is installed on host)
        slicer_cli = _discover_slicer_cli()
        if slicer_cli is None:
            return None
        kind, cmd = slicer_cli
        if kind == "prusa":
            prusa_ini = tmp_dir / "config.ini"
            prusa_ini.write_text(
                "\n".join(
                    line[2:] if line.startswith("; ") else line
                    for line in prusa_text.splitlines()[1:]
                )
                + "\n",
                encoding="utf-8",
            )
            proc = subprocess.run(
                [*cmd, "--load", str(prusa_ini), "--export-gcode", "--output", str(tmp_dir / "out.gcode"), str(in_3mf)],
                capture_output=True,
                text=True,
                timeout=25,
                check=False,
                cwd=str(tmp_dir),
            )
            if proc.returncode < 0 or proc.returncode in (126, 127, 134, 139):
                return None
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout).strip()[-400:]
                return f"Slicer CLI validation failed: {err}"
            return None

        out_3mf = tmp_dir / "out.3mf"
        proc = subprocess.run(
            [*cmd, "--slice", "0", "--export-3mf", str(out_3mf), str(in_3mf)],
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
            cwd=str(tmp_dir),
        )
        if proc.returncode < 0 or proc.returncode in (126, 127, 134, 139):
            return None
        if proc.returncode != 0:
            err_lines = [
                ln.strip()
                for ln in (proc.stdout + "\n" + proc.stderr).splitlines()
                if ln.strip()
                and "OpenGL" not in ln
                and "GLFW" not in ln
                and "Initializing StaticPrintConfigs" not in ln
            ]
            return f"Slicer CLI validation failed: {' | '.join(err_lines[-3:]) or f'exit code {proc.returncode}'}"

        if out_3mf.is_file():
            with zipfile.ZipFile(out_3mf, "r") as zout:
                if BAMBU_CONFIG_ENTRY in zout.namelist():
                    saved_cfg = json.loads(zout.read(BAMBU_CONFIG_ENTRY).decode("utf-8"))
                    replaced: list[str] = []
                    for key, sent_val in sent_cfg.items():
                        if key not in saved_cfg:
                            continue
                        saved_val = saved_cfg[key]
                        norm_sent = sent_val[0] if isinstance(sent_val, list) and len(sent_val) == 1 else sent_val
                        norm_saved = saved_val[0] if isinstance(saved_val, list) and len(saved_val) == 1 else saved_val
                        if norm_sent != norm_saved:
                            replaced.append(f"{key} ({norm_sent!r} -> {norm_saved!r})")
                    if replaced:
                        return f"Slicer rejected and replaced values: {', '.join(replaced)}"
        return None
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("Headless slicer validation skipped: %s", exc)
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
    """Query the active LLM to generate a complete 3MF slicer profile."""
    metrics = _load_metrics(project_dir)
    user_printer = (getattr(settings, "user_printer", "") or "").strip()
    user_nozzle = (getattr(settings, "user_nozzle", "") or "").strip()
    user_filament = (getattr(settings, "user_filament", "PLA") or "PLA").strip()
    user_additional_info = " ".join(str(getattr(settings, "user_additional_info", "") or "").split()).strip()
    nozzle_spec = parse_nozzle_spec(user_nozzle)
    nozzle_dia = nozzle_spec["diameter_mm"]
    priority = (priority or "auto").strip().lower()
    material_override = (material_override or "auto").strip()
    target_filament = (
        material_override
        if material_override.lower() != "auto"
        else user_filament
    )

    recent_user_context = _read_recent_user_context(project_dir)
    code_text = _read_scad(project_dir)
    combined_context = f"{recent_user_context} {extra_context}".strip()

    dims = metrics.get("dimensions_mm") or {}
    p = metrics.get("printability") or {}
    vol = float(metrics.get("volume_mm3", 0.0))
    surf = float(p.get("surface_area_mm2", 0.0))

    model_sha = compute_model_sha256(project_dir) or ""
    if not model_sha:
        try:
            model_sha = hashlib.sha256((project_dir / "preview.stl").read_bytes()).hexdigest()
        except OSError:
            model_sha = ""
    active_provider = getattr(settings, "llm_provider", "") or ""
    active_model = (
        getattr(settings, "llm_model", "")
        or getattr(settings, f"{active_provider}_model", "")
        or getattr(settings, "model", "")
        or ""
    )
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "schema_v": 9,
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
                    cached_entry["profile"],
                    nozzle_diameter_mm=nozzle_dia,
                    user_printer=user_printer,
                )
                cached_advice = (
                    str(cached_entry.get("advice_markdown") or "").strip()
                    or _format_profile_summary_markdown(
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
        "Your JSON output is translated 1:1 into a 3MF project archive for Bambu Studio, OrcaSlicer, and PrusaSlicer "
        "with zero post-processing heuristics. Every value you produce is written directly into the slicer configs.\n"
        "Analyze the user's printer hardware, nozzle, filament, mesh telemetry, and OpenSCAD source code, "
        "and output ONLY a valid JSON object wrapped in ```json ... ``` with ALL of the following keys:\n"
        "{\n"
        '  "printer_model": string (canonical printer model, e.g. "Bambu Lab A1", "Bambu Lab P1S", "Prusa MK4"),\n'
        '  "printer_structure": "corexy" | "i3" | "delta",\n'
        '  "bed_x_mm": number (build plate X width in mm),\n'
        '  "bed_y_mm": number (build plate Y depth in mm),\n'
        '  "bed_z_mm": number (max printable Z height in mm),\n'
        '  "nozzle_diameter_mm": number (installed nozzle orifice diameter in mm),\n'
        '  "bambu_printer_settings_id": string (exact Bambu/Orca machine preset name, e.g. "Bambu Lab A1 0.6 nozzle"),\n'
        '  "bambu_print_settings_id": string (exact Bambu/Orca process preset name; 0.4 nozzle has no nozzle suffix e.g. "0.20mm Standard @BBL A1", 0.6 nozzle uses "0.18mm Fine @BBL A1 0.6 nozzle" / "0.24mm Optimal @BBL A1 0.6 nozzle" / "0.30mm Standard @BBL A1 0.6 nozzle"),\n'
        '  "bambu_filament_settings_id": string (exact Bambu/Orca filament preset name without 0.4/0.6/0.8 nozzle suffix, e.g. "Generic PETG @BBL A1" or "Generic PLA"),\n'
        '  "material": string (e.g. "PLA", "PETG", "ASA", "ABS", "TPU"),\n'
        '  "filament_density_g_cm3": number (material density in g/cm³, e.g. 1.24 for PLA, 1.27 for PETG, 1.04 for ABS, 1.07 for ASA, 1.22 for TPU),\n'
        '  "filament_cost_per_kg": number (filament cost per kg, e.g. 20.0),\n'
        '  "filament_flow_ratio": number (extrusion multiplier, e.g. 0.98 for PLA/TPU, 0.95 for PETG/ABS/ASA),\n'
        '  "filament_id": string (Bambu filament code, e.g. "GFL99" for PLA, "GFG99" for PETG, "GFB99" for ABS, "GFB98" for ASA, "GFU99" for TPU),\n'
        '  "layer_height_mm": number,\n'
        '  "first_layer_height_mm": number,\n'
        '  "line_width_mm": number (must be >= nozzle_diameter_mm),\n'
        '  "initial_layer_line_width_mm": number (must be >= nozzle_diameter_mm),\n'
        '  "outer_wall_line_width_mm": number (must be >= nozzle_diameter_mm),\n'
        '  "inner_wall_line_width_mm": number (must be >= nozzle_diameter_mm),\n'
        '  "top_surface_line_width_mm": number (must be >= nozzle_diameter_mm),\n'
        '  "sparse_infill_line_width_mm": number (must be >= nozzle_diameter_mm),\n'
        '  "internal_solid_infill_line_width_mm": number (must be >= nozzle_diameter_mm),\n'
        '  "support_line_width_mm": number (must be >= nozzle_diameter_mm),\n'
        '  "wall_loops": integer,\n'
        '  "top_shell_layers": integer,\n'
        '  "bottom_shell_layers": integer,\n'
        '  "wall_generator": "arachne" | "classic",\n'
        '  "detect_thin_wall": boolean,\n'
        '  "seam_position": "aligned" | "back" | "random" | "nearest",\n'
        '  "wall_sequence": "inner/outer" | "outer/inner" | "inner/outer/inner",\n'
        '  "only_one_wall_top": boolean,\n'
        '  "elefant_foot_compensation_mm": number,\n'
        '  "xy_hole_compensation_mm": number,\n'
        '  "xy_contour_compensation_mm": number,\n'
        '  "precise_z_height": boolean,\n'
        '  "ironing_type": "no iron" | "top" | "topmost",\n'
        '  "ironing_flow_percent": integer,\n'
        '  "ironing_speed_mms": integer,\n'
        '  "infill_density_percent": integer,\n'
        '  "infill_pattern": "gyroid" | "grid" | "cubic" | "adaptivecubic" | "honeycomb" | "3dhoneycomb" | "crosshatch" | "rectilinear" | "zig-zag" | "concentric" | "lightning",\n'
        '  "top_surface_pattern": "monotonicline" | "monotonic" | "concentric" | "rectilinear" | "zig-zag",\n'
        '  "bottom_surface_pattern": "monotonic" | "monotonicline" | "concentric" | "rectilinear" | "zig-zag",\n'
        '  "infill_wall_overlap_percent": integer,\n'
        '  "support_enabled": boolean,\n'
        '  "support_type": "normal" | "tree",\n'
        '  "support_threshold_angle": integer,\n'
        '  "support_on_build_plate_only": boolean,\n'
        '  "support_top_z_distance_mm": number,\n'
        '  "support_bottom_z_distance_mm": number,\n'
        '  "support_interface_top_layers": integer,\n'
        '  "bridge_flow": number,\n'
        '  "thick_bridges": boolean,\n'
        '  "brim_enabled": boolean,\n'
        '  "brim_type": "no_brim" | "outer_only" | "inner_only" | "outer_and_inner" | "auto_brim",\n'
        '  "brim_width_mm": integer,\n'
        '  "nozzle_temp_initial_c": integer,\n'
        '  "nozzle_temp_c": integer,\n'
        '  "bed_temp_initial_c": integer,\n'
        '  "bed_temp_c": integer,\n'
        '  "fan_min_percent": integer,\n'
        '  "fan_max_percent": integer,\n'
        '  "overhang_fan_percent": integer,\n'
        '  "slow_down_for_layer_cooling": boolean,\n'
        '  "max_volumetric_speed_mm3s": number,\n'
        '  "initial_layer_speed_mms": integer,\n'
        '  "initial_layer_infill_speed_mms": integer,\n'
        '  "outer_wall_speed_mms": integer,\n'
        '  "inner_wall_speed_mms": integer,\n'
        '  "sparse_infill_speed_mms": integer,\n'
        '  "internal_solid_infill_speed_mms": integer,\n'
        '  "top_surface_speed_mms": integer,\n'
        '  "bridge_speed_mms": integer,\n'
        '  "support_speed_mms": integer,\n'
        '  "travel_speed_mms": integer,\n'
        '  "slow_down_overhangs": boolean,\n'
        '  "default_acceleration_mms2": integer,\n'
        '  "outer_wall_acceleration_mms2": integer,\n'
        '  "inner_wall_acceleration_mms2": integer,\n'
        '  "top_surface_acceleration_mms2": integer,\n'
        '  "initial_layer_acceleration_mms2": integer,\n'
        '  "travel_acceleration_mms2": integer,\n'
        '  "retraction_length_mm": number,\n'
        '  "retraction_speed_mms": integer,\n'
        '  "z_hop_mm": number,\n'
        '  "part_category": string,\n'
        '  "rationale": string (concise 1-2 sentence technical summary)\n'
        "}"
    )

    hw_lines = [
        f"- 3D Printer: {user_printer or 'Standard FDM'}",
        f"- Nozzle: {user_nozzle or f'{nozzle_dia:.2f} mm'}",
        f"- Filament: {target_filament}",
    ]
    if priority and priority.lower() != "auto":
        hw_lines.append(f"- Priority: {priority}")
    if user_additional_info:
        hw_lines.append(f"- Additional Instructions: {user_additional_info}")
    hw_section = "\n".join(hw_lines)

    user_prompt = (
        f"### Target Hardware & Setup:\n"
        f"{hw_section}\n\n"
        f"### Exact Model Telemetry:\n"
        f"- Bounding Box Dimensions: {dims.get('x', 0):.1f} × {dims.get('y', 0):.1f} × {dims.get('z', 0):.1f} mm\n"
        f"- Volume: {vol:.1f} mm³ | Surface Area: {surf:.1f} mm²\n"
        f"- Bed Contact Footprint: {float(p.get('bed_contact_area_mm2', 0)):.1f} mm² ({float(p.get('bed_contact_ratio', 0))*100:.1f}%)\n"
        f"- Downward Overhang Area (>45°): {float(p.get('overhang_area_45deg_mm2', 0)):.1f} mm² (Overhang ratio: {float(p.get('overhang_ratio', 0))*100:.1f}%)\n"
        f"- Downward Overhang Area (>60°): {float(p.get('overhang_area_60deg_mm2', 0)):.1f} mm²\n"
        f"- Slenderness Ratio (Z / min(X,Y)): {float(p.get('slenderness_ratio', 1.0)):.2f}\n"
        f"- Mesh Printability Flags: needs_support={bool(p.get('needs_support'))}, needs_brim={bool(p.get('needs_brim'))}\n\n"
        f"### User Context & Intent:\n{combined_context or 'Standard functional CAD part'}\n\n"
        f"### OpenSCAD Code (`model.scad`):\n```scad\n{code_text}\n```\n"
    )

    try:
        from agent.llm_base import create_llm_client

        client = create_llm_client(settings)
        if stop_event is not None:
            client.stop_event = stop_event
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        def _query_llm(max_attempts: int) -> str:
            try:
                resp = client.chat(messages, max_attempts=max_attempts)
            except TypeError:
                resp = client.chat(messages)
            _publish_advisor_usage(publish, project_dir.name, getattr(client, "last_usage", None))
            return _extract_llm_content(resp)

        content = _query_llm(2)
        if not content.strip():
            raise RuntimeError("LLM returned an empty response while generating the print profile.")

        json_block = _extract_json_block(content)
        if not json_block or not isinstance(json_block, dict):
            raise RuntimeError("LLM response did not contain a valid JSON print profile.")

        raw_profile = dict(json_block)
        if not raw_profile.get("material"):
            raw_profile["material"] = target_filament

        ai_profile = sanitize_print_profile(
            raw_profile,
            nozzle_diameter_mm=nozzle_dia,
            user_printer=user_printer,
        )
        val_err = _validate_profile_universal(ai_profile, project_dir)
        if val_err:
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"3MF slicer validation failed: {val_err}\n"
                        "Please fix the offending parameters and output the complete corrected JSON object."
                    ),
                }
            )
            try:
                retry_block = _extract_json_block(_query_llm(1))
                if retry_block and isinstance(retry_block, dict):
                    raw_retry = {**raw_profile, **retry_block}
                    json_block = raw_retry
                    if not raw_retry.get("material"):
                        raw_retry["material"] = target_filament
                    ai_profile = sanitize_print_profile(
                        raw_retry,
                        nozzle_diameter_mm=nozzle_dia,
                        user_printer=user_printer,
                    )
            except Exception:  # noqa: BLE001
                if stop_event is not None and stop_event.is_set():
                    raise

        ai_advice = json_block.get("advice_markdown")
        final_advice = (
            ai_advice.strip()
            if isinstance(ai_advice, str) and ai_advice.strip()
            else _format_profile_summary_markdown(
                ai_profile, user_printer, user_nozzle
            )
        )
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=cache_path.parent,
                prefix=".tmp_profile_",
                suffix=".json",
                encoding="utf-8",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                json.dump(
                    {
                        "cache_key": cache_key,
                        "profile": ai_profile,
                        "advice_markdown": final_advice,
                    },
                    tmp,
                    indent=2,
                )
            try:
                tmp_path.replace(cache_path)
            finally:
                tmp_path.unlink(missing_ok=True)
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
    except Exception as exc:
        if stop_event is not None and stop_event.is_set():
            raise
        _LOG.error("LLM 3MF print profile generation failed: %s", exc)
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Failed to generate AI print profile: {exc}") from exc


def recommend_print_profile(
    project_dir: Path,
    *,
    priority: str = "auto",
    material_override: str = "auto",
    extra_context: str = "",
    settings: Any | None = None,
    stop_event: threading.Event | None = None,
    publish: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Generate an LLM-configured print profile for the project's current build."""
    if settings is None:
        from agent.settings import load_settings

        settings = load_settings()
    return generate_ai_print_profile(
        project_dir,
        settings,
        priority=priority,
        material_override=material_override,
        extra_context=extra_context,
        stop_event=stop_event,
        publish=publish,
    )


def format_print_command_prompt(
    project_dir: Path, extra_notes: str = "", settings: Any | None = None
) -> str:
    """Build the ``/print`` chat prompt from exact mesh telemetry and user hardware setup."""
    metrics = _load_metrics(project_dir)
    p = metrics.get("printability") or {}
    dims = metrics.get("dimensions_mm") or {}
    vol = float(metrics.get("volume_mm3", 0.0))
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
        f"- Volume: {vol:.1f} mm³ | Surface Area: {p.get('surface_area_mm2', 0.0):.1f} mm²\n"
        f"- Bed Contact Footprint: {p.get('bed_contact_area_mm2', 0.0):.1f} mm² "
        f"({p.get('bed_contact_ratio', 0.0) * 100:.1f}% of base)\n"
        f"- Downward Overhang Area (>45°): {p.get('overhang_area_45deg_mm2', 0.0):.1f} mm² "
        f"({p.get('overhang_ratio', 0.0) * 100:.1f}%)\n"
        f"- Downward Overhang Area (>60°): {p.get('overhang_area_60deg_mm2', 0.0):.1f} mm²\n"
        f"- Slenderness / Aspect Ratio (Z / min(X,Y)): {p.get('slenderness_ratio', 1.0):.2f}\n"
        f"- Automated Telemetry Flags: needs_support={bool(p.get('needs_support'))}, "
        f"needs_brim={bool(p.get('needs_brim'))}\n\n"
        f"### User Requirements & Specific Notes:\n{notes}\n\n"
        "Please provide detailed, professional advice covering:\n"
        "1. **Slicer Profile Recommendations:** Optimal material, layer heights, line widths, wall counts, infill pattern/density, and temperature/cooling advice for the target printer and nozzle.\n"
        "2. **Build Plate Orientation:** How to orient the part on the print bed to balance mechanical strength (layer line shear), overhangs, and surface finish.\n"
        "3. **Support & Adhesion Strategy:** Tree vs. standard supports, critical overhang areas, contact z-distance, and brim requirements.\n"
        "4. **Manufacturing Warnings & Tolerances:** Warping risks, dimensional tolerances for fitting parts, and post-processing suggestions."
    )
