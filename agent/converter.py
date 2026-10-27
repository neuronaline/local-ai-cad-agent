"""CAD model mesh and format conversion utilities."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any
from xml.sax.saxutils import escape as xml_escape, unescape as xml_unescape
import zipfile

from agent.revisions import compute_model_sha256
from agent.sandbox import _BWRAP, command as sandbox_command
from agent.tools.cad_scripts.renderer import load_stl, render_iso
from agent.tools.cad_tool import _supports_manifold

SUPPORTED_EXPORT_FORMATS = frozenset({"stl", "scad", "3mf", "obj", "amf", "off", "csg"})

EXPORT_MIME_TYPES: dict[str, str] = {
    "stl": "model/stl",
    "scad": "text/x-scad",
    "3mf": "model/3mf",
    "obj": "model/obj",
    "amf": "application/x-amf",
    "off": "text/plain",
    "csg": "text/plain",
}


def find_openscad() -> str | None:
    """Locate the openscad executable on PATH or in local .venv."""
    direct = shutil.which("openscad")
    if direct:
        return direct
    repo_root = Path(__file__).resolve().parent.parent
    for candidate in (
        repo_root / ".venv" / "bin" / "openscad",
        repo_root / ".venv" / "openscad",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def stl_to_obj(stl_path: Path) -> str:
    """Convert an STL file (binary or ASCII) into a Wavefront OBJ string."""
    vertices, triangles = load_stl(stl_path)
    buf = io.StringIO()
    buf.write("# OBJ export from Local AI CAD Agent\n")
    for v in vertices:
        buf.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
    for t in triangles:
        # OBJ indices are 1-based
        buf.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
    return buf.getvalue()


# Archive entries written by :func:`enrich_3mf_archive`.
PRUSA_CONFIG_ENTRY = "Metadata/Slic3r_PE.config"
BAMBU_CONFIG_ENTRY = "Metadata/project_settings.config"
MODEL_SETTINGS_ENTRY = "Metadata/model_settings.config"
PROFILE_ENTRY = "Metadata/cad_agent_print_profile.json"
THUMBNAIL_ENTRY = "Metadata/thumbnail.png"
_THUMBNAIL_REL_TYPE = "http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail"

_PRUSA_INFILL_MAP: dict[str, str] = {
    "gyroid": "gyroid",
    "grid": "grid",
    "cubic": "cubic",
    "adaptivecubic": "adaptivecubic",
    "honeycomb": "honeycomb",
    "3dhoneycomb": "3dhoneycomb",
    "crosshatch": "gyroid",
    "rectilinear": "rectilinear",
    "concentric": "concentric",
    "lightning": "lightning",
}
_PRUSA_SURFACE_MAP: dict[str, str] = {
    "monotonicline": "monotoniclines",
    "monotonic": "monotonic",
    "concentric": "concentric",
    "rectilinear": "rectilinear",
}
_BAMBU_WALL_SEQUENCE_MAP: dict[str, str] = {
    "inner/outer": "inner wall/outer wall",
    "outer/inner": "outer wall/inner wall",
    "inner/outer/inner": "inner-outer-inner wall",
    "inner wall/outer wall": "inner wall/outer wall",
    "outer wall/inner wall": "outer wall/inner wall",
    "inner-outer-inner wall": "inner-outer-inner wall",
}


def build_prusa_slicer_config(profile: dict[str, Any]) -> str:
    """Generate a PrusaSlicer/SuperSlicer ``Slic3r_PE.config`` print config.

    PrusaSlicer parses this entry as G-code comments, so every setting line
    must use the ``; key = value`` form; bare ``key = value`` lines are ignored.
    """
    nozzle_dia = float(profile.get("nozzle_diameter_mm", 0.4))
    lh = float(profile.get("layer_height_mm", 0.20))
    first_lh = float(profile.get("first_layer_height_mm", 0.20))
    brim_enabled = bool(profile.get("brim_enabled"))
    brim_width = profile.get("brim_width_mm", 6) if brim_enabled else 0
    support = 1 if profile.get("support_enabled") else 0
    material = str(profile.get("material", "PLA"))
    infill_pat = _PRUSA_INFILL_MAP.get(
        str(profile.get("infill_pattern", "gyroid")), "gyroid"
    )
    top_pat = _PRUSA_SURFACE_MAP.get(
        str(profile.get("top_surface_pattern", "monotonicline")), "monotoniclines"
    )
    bot_pat = _PRUSA_SURFACE_MAP.get(
        str(profile.get("bottom_surface_pattern", "monotonic")), "monotonic"
    )
    ironing_type = str(profile.get("ironing_type", "no iron"))
    ironing_on = 0 if ironing_type == "no iron" else 1
    wall_seq = str(profile.get("wall_sequence", "inner/outer"))

    settings = {
        "nozzle_diameter": nozzle_dia,
        "layer_height": lh,
        "first_layer_height": first_lh,
        "extrusion_width": profile.get("line_width_mm", round(nozzle_dia * 1.05, 2)),
        "first_layer_extrusion_width": profile.get(
            "initial_layer_line_width_mm", round(nozzle_dia * 1.25, 2)
        ),
        "external_perimeter_extrusion_width": profile.get(
            "outer_wall_line_width_mm", round(nozzle_dia * 1.05, 2)
        ),
        "perimeter_extrusion_width": profile.get(
            "inner_wall_line_width_mm", round(nozzle_dia * 1.12, 2)
        ),
        "top_infill_extrusion_width": profile.get(
            "top_surface_line_width_mm", round(nozzle_dia * 1.00, 2)
        ),
        "infill_extrusion_width": profile.get(
            "sparse_infill_line_width_mm", round(nozzle_dia * 1.12, 2)
        ),
        "solid_infill_extrusion_width": profile.get(
            "internal_solid_infill_line_width_mm", round(nozzle_dia * 1.05, 2)
        ),
        "support_material_extrusion_width": profile.get(
            "support_line_width_mm", round(nozzle_dia * 0.95, 2)
        ),
        "perimeters": profile["wall_loops"],
        "top_solid_layers": profile["top_shell_layers"],
        "bottom_solid_layers": profile["bottom_shell_layers"],
        "perimeter_generator": profile.get("wall_generator", "arachne"),
        "thin_walls": 1 if profile.get("detect_thin_wall", True) else 0,
        "seam_position": profile.get("seam_position", "aligned"),
        "external_perimeters_first": 1 if wall_seq == "outer/inner" else 0,
        "elefant_foot_compensation": profile.get("elefant_foot_compensation_mm", 0.10),
        "xy_size_compensation": profile.get("xy_contour_compensation_mm", 0.0),
        "ironing": ironing_on,
        "ironing_type": "top" if ironing_type == "topmost" else ("alltop" if ironing_on else "top"),
        "ironing_flowrate": f"{profile.get('ironing_flow_percent', 15)}%",
        "ironing_speed": profile.get("ironing_speed_mms", 30),
        "fill_density": f"{profile['infill_density_percent']}%",
        "fill_pattern": infill_pat,
        "top_fill_pattern": top_pat,
        "bottom_fill_pattern": bot_pat,
        "infill_overlap": f"{profile.get('infill_wall_overlap_percent', 15)}%",
        "support_material": support,
        # Without ``support_material_auto`` PrusaSlicer only honours painted supports.
        "support_material_auto": support,
        "support_material_style": "organic" if profile["support_type"] == "tree" else "grid",
        "support_material_threshold": profile["support_threshold_angle"],
        "support_material_buildplate_only": (
            1 if profile.get("support_on_build_plate_only", False) else 0
        ),
        "support_material_contact_distance": profile.get("support_top_z_distance_mm", 0.20),
        "support_material_bottom_contact_distance": profile.get(
            "support_bottom_z_distance_mm", 0.20
        ),
        "support_material_interface_layers": profile.get("support_interface_top_layers", 2),
        "bridge_flow_ratio": profile.get("bridge_flow", 0.95),
        "thick_bridges": 1 if profile.get("thick_bridges", False) else 0,
        "brim_width": brim_width,
        "skirts": 0 if brim_enabled else 1,
        "filament_type": material,
        "first_layer_temperature": profile.get("nozzle_temp_initial_c", 220),
        "temperature": profile.get("nozzle_temp_c", 215),
        "first_layer_bed_temperature": profile.get("bed_temp_initial_c", 60),
        "bed_temperature": profile.get("bed_temp_c", 60),
        "min_fan_speed": profile.get("fan_min_percent", 100),
        "max_fan_speed": profile.get("fan_max_percent", 100),
        "bridge_fan_speed": profile.get("overhang_fan_percent", 100),
        "cooling": 1 if profile.get("slow_down_for_layer_cooling", True) else 0,
        "max_volumetric_speed": profile.get("max_volumetric_speed_mm3s", 16.0),
        "first_layer_speed": profile.get("initial_layer_speed_mms", 50),
        "external_perimeter_speed": profile.get("outer_wall_speed_mms", 120),
        "perimeter_speed": profile.get("inner_wall_speed_mms", 180),
        "infill_speed": profile.get("sparse_infill_speed_mms", 200),
        "solid_infill_speed": profile.get("internal_solid_infill_speed_mms", 180),
        "top_solid_infill_speed": profile.get("top_surface_speed_mms", 100),
        "bridge_speed": profile.get("bridge_speed_mms", 40),
        "support_material_speed": profile.get("support_speed_mms", 120),
        "travel_speed": profile.get("travel_speed_mms", 350),
        "enable_dynamic_overhang_speeds": 1 if profile.get("slow_down_overhangs", True) else 0,
    }
    lines = ["; generated by Local AI CAD Agent"]
    lines += [f"; {key} = {value}" for key, value in settings.items()]
    return "\n".join(lines) + "\n"


def build_bambu_orca_config(profile: dict[str, Any]) -> str:
    """Generate a Bambu Studio / OrcaSlicer ``project_settings.config`` (JSON, string values)."""
    nozzle_dia = float(profile.get("nozzle_diameter_mm", 0.4))
    lh = float(profile.get("layer_height_mm", 0.20))
    first_lh = float(profile.get("first_layer_height_mm", 0.20))
    top_layers = int(profile.get("top_shell_layers", 4))
    bot_layers = int(profile.get("bottom_shell_layers", 4))
    brim_enabled = bool(profile.get("brim_enabled"))
    brim_type = str(profile.get("brim_type", "outer_only" if brim_enabled else "no_brim"))
    if not brim_enabled:
        brim_type = "no_brim"
    brim_width = profile.get("brim_width_mm", 6) if brim_enabled else 0
    material = str(profile.get("material", "PLA"))
    nozzle_temp_init = str(profile.get("nozzle_temp_initial_c", 220))
    nozzle_temp = str(profile.get("nozzle_temp_c", 215))
    bed_temp_init = str(profile.get("bed_temp_initial_c", 60))
    bed_temp = str(profile.get("bed_temp_c", 60))

    config: dict[str, Any] = {
        "nozzle_diameter": [str(nozzle_dia)],
        "layer_height": str(lh),
        "initial_layer_print_height": str(first_lh),
        "line_width": str(profile.get("line_width_mm", round(nozzle_dia * 1.05, 2))),
        "initial_layer_line_width": str(
            profile.get("initial_layer_line_width_mm", round(nozzle_dia * 1.25, 2))
        ),
        "outer_wall_line_width": str(
            profile.get("outer_wall_line_width_mm", round(nozzle_dia * 1.05, 2))
        ),
        "inner_wall_line_width": str(
            profile.get("inner_wall_line_width_mm", round(nozzle_dia * 1.12, 2))
        ),
        "top_surface_line_width": str(
            profile.get("top_surface_line_width_mm", round(nozzle_dia * 1.00, 2))
        ),
        "sparse_infill_line_width": str(
            profile.get("sparse_infill_line_width_mm", round(nozzle_dia * 1.12, 2))
        ),
        "internal_solid_infill_line_width": str(
            profile.get("internal_solid_infill_line_width_mm", round(nozzle_dia * 1.05, 2))
        ),
        "support_line_width": str(
            profile.get("support_line_width_mm", round(nozzle_dia * 0.95, 2))
        ),
        "wall_loops": str(profile["wall_loops"]),
        "top_shell_layers": str(top_layers),
        "bottom_shell_layers": str(bot_layers),
        "top_shell_thickness": str(round(max(0.6, top_layers * lh), 2)),
        "bottom_shell_thickness": str(round(max(0.6, bot_layers * lh), 2)),
        "wall_generator": str(profile.get("wall_generator", "arachne")),
        "detect_thin_wall": "1" if profile.get("detect_thin_wall", True) else "0",
        "seam_position": str(profile.get("seam_position", "aligned")),
        "wall_sequence": _BAMBU_WALL_SEQUENCE_MAP.get(
            str(profile.get("wall_sequence", "inner/outer")),
            "inner wall/outer wall",
        ),
        "only_one_wall_top": "1" if profile.get("only_one_wall_top", False) else "0",
        "elefant_foot_compensation": str(profile.get("elefant_foot_compensation_mm", 0.10)),
        "xy_hole_compensation": str(profile.get("xy_hole_compensation_mm", 0.0)),
        "xy_contour_compensation": str(profile.get("xy_contour_compensation_mm", 0.0)),
        "precise_z_height": "1" if profile.get("precise_z_height", True) else "0",
        "ironing_type": str(profile.get("ironing_type", "no iron")),
        "ironing_flow": f"{profile.get('ironing_flow_percent', 15)}%",
        "ironing_speed": str(profile.get("ironing_speed_mms", 30)),
        "sparse_infill_density": f"{profile['infill_density_percent']}%",
        "sparse_infill_pattern": str(profile["infill_pattern"]),
        "top_surface_pattern": str(profile.get("top_surface_pattern", "monotonicline")),
        "bottom_surface_pattern": str(profile.get("bottom_surface_pattern", "monotonic")),
        "infill_wall_overlap": f"{profile.get('infill_wall_overlap_percent', 15)}%",
        "enable_support": "1" if profile["support_enabled"] else "0",
        "support_type": "tree(auto)" if profile["support_type"] == "tree" else "normal(auto)",
        "support_threshold_angle": str(profile["support_threshold_angle"]),
        "support_on_build_plate_only": (
            "1" if profile.get("support_on_build_plate_only", False) else "0"
        ),
        "support_top_z_distance": str(profile.get("support_top_z_distance_mm", 0.20)),
        "support_bottom_z_distance": str(profile.get("support_bottom_z_distance_mm", 0.20)),
        "support_interface_top_layers": str(profile.get("support_interface_top_layers", 2)),
        "bridge_flow": str(profile.get("bridge_flow", 0.95)),
        "thick_bridges": "1" if profile.get("thick_bridges", False) else "0",
        "brim_type": brim_type,
        "brim_width": str(brim_width),
        "skirt_loops": "0" if brim_enabled else "1",
        "filament_type": [material],
        "nozzle_temperature_initial_layer": [nozzle_temp_init],
        "nozzle_temperature": [nozzle_temp],
        "hot_plate_temp_initial_layer": [bed_temp_init],
        "hot_plate_temp": [bed_temp],
        "textured_plate_temp_initial_layer": [bed_temp_init],
        "textured_plate_temp": [bed_temp],
        "eng_plate_temp_initial_layer": [bed_temp_init],
        "eng_plate_temp": [bed_temp],
        "cool_plate_temp_initial_layer": [bed_temp_init],
        "cool_plate_temp": [bed_temp],
        "fan_min_speed": [str(profile.get("fan_min_percent", 100))],
        "fan_max_speed": [str(profile.get("fan_max_percent", 100))],
        "overhang_fan_speed": [str(profile.get("overhang_fan_percent", 100))],
        "slow_down_for_layer_cooling": [
            "1" if profile.get("slow_down_for_layer_cooling", True) else "0"
        ],
        "filament_max_volumetric_speed": [
            str(profile.get("max_volumetric_speed_mm3s", 16.0))
        ],
        "initial_layer_speed": str(profile.get("initial_layer_speed_mms", 50)),
        "initial_layer_infill_speed": str(profile.get("initial_layer_infill_speed_mms", 80)),
        "outer_wall_speed": str(profile.get("outer_wall_speed_mms", 120)),
        "inner_wall_speed": str(profile.get("inner_wall_speed_mms", 180)),
        "sparse_infill_speed": str(profile.get("sparse_infill_speed_mms", 200)),
        "internal_solid_infill_speed": str(
            profile.get("internal_solid_infill_speed_mms", 180)
        ),
        "top_surface_speed": str(profile.get("top_surface_speed_mms", 100)),
        "bridge_speed": str(profile.get("bridge_speed_mms", 40)),
        "support_speed": str(profile.get("support_speed_mms", 120)),
        "travel_speed": str(profile.get("travel_speed_mms", 350)),
        "enable_overhang_speed": "1" if profile.get("slow_down_overhangs", True) else "0",
    }
    return json.dumps(config, indent=2) + "\n"


def build_bambu_model_settings_config(
    objects_meta: list[dict[str, Any]],
    *,
    has_thumbnail: bool = False,
) -> str:
    """Generate ``Metadata/model_settings.config`` for OrcaSlicer / Bambu Studio."""
    if not objects_meta:
        objects_meta = [{"id": 1, "name": "model", "transform": "1 0 0 0 1 0 0 0 1 128 128 0"}]

    obj_blocks: list[str] = []
    model_instances: list[str] = []
    assemble_items: list[str] = []
    for idx, obj in enumerate(objects_meta, start=1):
        oid = int(obj.get("id", idx))
        name = xml_escape(str(obj.get("name") or f"part_{oid}"))
        transform = xml_escape(str(obj.get("transform") or "1 0 0 0 1 0 0 0 1 128 128 0"))
        obj_blocks.append(
            f'  <object id="{oid}">\n'
            f'    <metadata key="name" value="{name}"/>\n'
            f'    <metadata key="extruder" value="1"/>\n'
            f'    <part id="1" subtype="normal_part">\n'
            f'      <metadata key="name" value="{name}"/>\n'
            f'      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
            f"    </part>\n"
            f"  </object>"
        )
        model_instances.append(
            f"    <model_instance>\n"
            f'      <metadata key="object_id" value="{oid}"/>\n'
            f'      <metadata key="instance_id" value="0"/>\n'
            f'      <metadata key="identify_id" value="{oid}"/>\n'
            f"    </model_instance>"
        )
        assemble_items.append(
            f'    <assemble_item object_id="{oid}" instance_id="0" '
            f'transform="{transform}" offset="0 0 0"/>'
        )

    thumb_line = (
        f'    <metadata key="thumbnail_file" value="{THUMBNAIL_ENTRY}"/>\n'
        if has_thumbnail
        else ""
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<config>\n"
        + "\n".join(obj_blocks)
        + "\n  <plate>\n"
        '    <metadata key="plater_id" value="1"/>\n'
        '    <metadata key="plater_name" value=""/>\n'
        '    <metadata key="locked" value="false"/>\n'
        + thumb_line
        + "\n".join(model_instances)
        + "\n  </plate>\n"
        "  <assemble>\n"
        + "\n".join(assemble_items)
        + "\n  </assemble>\n"
        "</config>\n"
    )


def _insert_before(xml: str, marker: str, insertion: str) -> str:
    return xml.replace(marker, insertion + marker, 1)


def _fmt_coord(val: float) -> str:
    rounded = round(val, 5) + 0.0
    text = f"{rounded:.5f}".rstrip("0").rstrip(".")
    return text or "0"


def _prepare_3mf_objects_and_layout(
    content: bytes,
    project_name: str,
    *,
    bed_x_mm: float = 256.0,
    bed_y_mm: float = 256.0,
) -> tuple[bytes, list[dict[str, Any]]]:
    """Split independent parts, ground each object to Z=0, and center on the build plate."""
    default_meta = [{"id": 1, "name": project_name or "part_1", "transform": "1 0 0 0 1 0 0 0 1 0 0 0"}]
    try:
        xml = content.decode("utf-8")
    except UnicodeDecodeError:
        return content, default_meta

    obj_matches = list(
        re.finditer(r"<object\b[^>]*>\s*<mesh>.*?</mesh>\s*</object>", xml, flags=re.DOTALL)
    )
    if len(obj_matches) != 1:
        if len(obj_matches) > 1:
            existing_meta: list[dict[str, Any]] = []
            for idx, om in enumerate(obj_matches, start=1):
                tag_head = om.group(0).split(">", 1)[0]
                id_m = re.search(r'\bid="(\d+)"', tag_head)
                name_m = re.search(r'\bname="([^"]*)"', tag_head)
                oid = int(id_m.group(1)) if id_m else idx
                oname = (
                    xml_unescape(name_m.group(1))
                    if name_m
                    else f"{project_name or 'part'}_{oid}"
                )
                item_tag_m = re.search(rf'<item\b[^>]*?\bobjectid="{oid}"[^>]*/>', xml)
                tm_m = (
                    re.search(r'\btransform="([^"]+)"', item_tag_m.group(0))
                    if item_tag_m
                    else None
                )
                t_str = (
                    tm_m.group(1)
                    if tm_m
                    else f"1 0 0 0 1 0 0 0 1 {_fmt_coord(bed_x_mm * 0.5)} {_fmt_coord(bed_y_mm * 0.5)} 0"
                )
                existing_meta.append({"id": oid, "name": oname, "transform": t_str})
            return content, existing_meta
        return content, default_meta

    obj_block = obj_matches[0].group(0)
    vert_tags = re.findall(r"<vertex\b[^>]*/\s*>", obj_block)
    tri_matches = list(
        re.finditer(
            r'<triangle\b([^>]*?\bv1="(\d+)"[^>]*?\bv2="(\d+)"[^>]*?\bv3="(\d+)"[^>]*)/>',
            obj_block,
        )
    )
    if len(vert_tags) < 3 or len(tri_matches) < 1:
        return content, default_meta

    coords: list[tuple[float, float, float]] = []
    coord_to_canon: dict[tuple[float, float, float], int] = {}
    canon_idx: list[int] = []
    for idx, vtag in enumerate(vert_tags):
        xm = re.search(r'\bx="([^"]+)"', vtag)
        ym = re.search(r'\by="([^"]+)"', vtag)
        zm = re.search(r'\bz="([^"]+)"', vtag)
        if not (xm and ym and zm):
            return content, default_meta
        try:
            vx, vy, vz = float(xm.group(1)), float(ym.group(1)), float(zm.group(1))
        except ValueError:
            return content, default_meta
        coords.append((vx, vy, vz))
        key = (round(vx, 5) + 0.0, round(vy, 5) + 0.0, round(vz, 5) + 0.0)
        canon = coord_to_canon.setdefault(key, idx)
        canon_idx.append(canon)

    n_verts = len(vert_tags)
    tris: list[tuple[int, int, int, str]] = []
    for tm in tri_matches:
        v1, v2, v3 = int(tm.group(2)), int(tm.group(3)), int(tm.group(4))
        if v1 >= n_verts or v2 >= n_verts or v3 >= n_verts:
            return content, default_meta
        tris.append((v1, v2, v3, tm.group(0)))

    parent = list(range(len(tris)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    edge_map: dict[tuple[int, int], int] = {}
    for t_idx, (v1, v2, v3, _) in enumerate(tris):
        c1, c2, c3 = canon_idx[v1], canon_idx[v2], canon_idx[v3]
        for a, b in ((c1, c2), (c2, c3), (c3, c1)):
            edge = (a, b) if a <= b else (b, a)
            prev = edge_map.get(edge)
            if prev is None:
                edge_map[edge] = t_idx
            else:
                union(prev, t_idx)

    raw_components: dict[int, list[int]] = {}
    for t_idx in range(len(tris)):
        raw_components.setdefault(find(t_idx), []).append(t_idx)

    # Merge components whose 2D XY bounding boxes overlap so nested or
    # print-in-place sub-components keep their relative Z clearances intact.
    if len(raw_components) > 1:
        comp_roots = list(raw_components.keys())
        comp_xy_bounds: dict[int, tuple[float, float, float, float]] = {}
        for root, tri_list in raw_components.items():
            v_indices = {v for t_i in tri_list for v in tris[t_i][:3]}
            xs = [coords[v][0] for v in v_indices]
            ys = [coords[v][1] for v in v_indices]
            comp_xy_bounds[root] = (min(xs), max(xs), min(ys), max(ys))
        for i, r1 in enumerate(comp_roots):
            x0_a, x1_a, y0_a, y1_a = comp_xy_bounds[r1]
            for r2 in comp_roots[i + 1 :]:
                x0_b, x1_b, y0_b, y1_b = comp_xy_bounds[r2]
                if (
                    x0_a < x1_b - 1e-3
                    and x1_a > x0_b + 1e-3
                    and y0_a < y1_b - 1e-3
                    and y1_a > y0_b + 1e-3
                ):
                    union(r1, r2)

    components: dict[int, list[int]] = {}
    for t_idx in range(len(tris)):
        components.setdefault(find(t_idx), []).append(t_idx)

    all_used_verts = {v for v1, v2, v3, _ in tris for v in (v1, v2, v3)}
    all_xs = [coords[v][0] for v in all_used_verts]
    all_ys = [coords[v][1] for v in all_used_verts]
    assembly_cx = (min(all_xs) + max(all_xs)) * 0.5
    assembly_cy = (min(all_ys) + max(all_ys)) * 0.5
    plate_cx = bed_x_mm * 0.5
    plate_cy = bed_y_mm * 0.5

    safe_prefix = xml_escape(project_name or "part")
    multi_part = len(components) > 1
    new_objects: list[str] = []
    build_items: list[str] = []
    objects_meta: list[dict[str, Any]] = []

    for obj_id, tri_indices in enumerate(components.values(), start=1):
        used_verts = sorted({v for t_i in tri_indices for v in tris[t_i][:3]})
        remap = {old_v: new_v for new_v, old_v in enumerate(used_verts)}
        xs = [coords[v][0] for v in used_verts]
        ys = [coords[v][1] for v in used_verts]
        zs = [coords[v][2] for v in used_verts]
        comp_cx = (min(xs) + max(xs)) * 0.5
        comp_cy = (min(ys) + max(ys)) * 0.5
        comp_min_z = min(zs)

        comp_vert_lines = "\n".join(
            f'\t\t\t\t\t<vertex x="{_fmt_coord(coords[old_v][0] - comp_cx)}" '
            f'y="{_fmt_coord(coords[old_v][1] - comp_cy)}" '
            f'z="{_fmt_coord(coords[old_v][2] - comp_min_z)}" />'
            for old_v in used_verts
        )
        comp_tri_lines_list: list[str] = []
        for t_i in tri_indices:
            v1, v2, v3, raw_tag = tris[t_i]
            updated = re.sub(r'\bv1="\d+"', f'v1="{remap[v1]}"', raw_tag, count=1)
            updated = re.sub(r'\bv2="\d+"', f'v2="{remap[v2]}"', updated, count=1)
            updated = re.sub(r'\bv3="\d+"', f'v3="{remap[v3]}"', updated, count=1)
            comp_tri_lines_list.append(f"\t\t\t\t\t{updated}")
        comp_tri_lines = "\n".join(comp_tri_lines_list)

        obj_name_raw = f"{project_name or 'part'}_part_{obj_id}" if multi_part else (project_name or "part_1")
        obj_name_escaped = f"{safe_prefix}_part_{obj_id}" if multi_part else safe_prefix
        tx = plate_cx + (comp_cx - assembly_cx)
        ty = plate_cy + (comp_cy - assembly_cy)
        transform_str = f"1 0 0 0 1 0 0 0 1 {_fmt_coord(tx)} {_fmt_coord(ty)} 0"

        new_objects.append(
            f'\t\t<object id="{obj_id}" name="{obj_name_escaped}" type="model">\n'
            f"\t\t\t<mesh>\n"
            f"\t\t\t\t<vertices>\n{comp_vert_lines}\n\t\t\t\t</vertices>\n"
            f"\t\t\t\t<triangles>\n{comp_tri_lines}\n\t\t\t\t</triangles>\n"
            f"\t\t\t</mesh>\n"
            f"\t\t</object>"
        )
        build_items.append(
            f'\t\t<item objectid="{obj_id}" transform="{transform_str}" printable="1" />'
        )
        objects_meta.append(
            {
                "id": obj_id,
                "name": obj_name_raw,
                "transform": transform_str,
            }
        )

    start, end = obj_matches[0].span()
    xml = xml[:start] + "\n".join(new_objects) + xml[end:]
    xml = re.sub(
        r"<build\b[^>]*>.*?</build>",
        "<build>\n" + "\n".join(build_items) + "\n\t</build>",
        xml,
        count=1,
        flags=re.DOTALL,
    )
    return xml.encode("utf-8"), objects_meta


def _split_disconnected_3mf_objects(
    content: bytes,
    project_name: str,
    *,
    bed_x_mm: float = 256.0,
    bed_y_mm: float = 256.0,
) -> bytes:
    """Split disconnected multi-body objects, ground to Z=0, and center on the build plate."""
    updated, _ = _prepare_3mf_objects_and_layout(
        content, project_name, bed_x_mm=bed_x_mm, bed_y_mm=bed_y_mm
    )
    return updated


def enrich_3mf_archive(
    base_3mf_path: Path,
    output_path: Path,
    *,
    profile: dict[str, Any] | None,
    project_name: str,
    thumbnail: bytes | None,
) -> None:
    """Write ``base_3mf_path`` plus slicer configs, core metadata and thumbnail to ``output_path``.

    ``profile`` must already be sanitized (see ``sanitize_print_profile``).
    """
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, prefix=".tmp_enrich_", suffix=".3mf", delete=False
    ) as tmp:
        temp_file = Path(tmp.name)

    bed_x = float(profile.get("bed_x_mm", 256.0)) if profile else 256.0
    bed_y = float(profile.get("bed_y_mm", 256.0)) if profile else 256.0

    added: dict[str, bytes] = {}
    if profile:
        added[PRUSA_CONFIG_ENTRY] = build_prusa_slicer_config(profile).encode("utf-8")
        added[BAMBU_CONFIG_ENTRY] = build_bambu_orca_config(profile).encode("utf-8")
        added[PROFILE_ENTRY] = (json.dumps(profile, indent=2) + "\n").encode("utf-8")
    if thumbnail:
        added[THUMBNAIL_ENTRY] = thumbnail
    content_types = {"config": "text/plain", "json": "application/json"} if profile else {}
    if thumbnail:
        content_types["png"] = "image/png"

    reserved_names = set(added) | ({MODEL_SETTINGS_ENTRY} if profile else set())
    objects_meta: list[dict[str, Any]] = []

    try:
        with zipfile.ZipFile(base_3mf_path, "r") as zin, zipfile.ZipFile(
            temp_file, "w", compression=zipfile.ZIP_DEFLATED
        ) as zout:
            for item in zin.infolist():
                # Duplicate central-directory names break slicer OPC readers.
                if item.filename in reserved_names:
                    continue
                content = zin.read(item.filename)
                if item.filename == "3D/3dmodel.model":
                    content, objects_meta = _prepare_3mf_objects_and_layout(
                        content,
                        project_name,
                        bed_x_mm=bed_x,
                        bed_y_mm=bed_y,
                    )
                    if profile:
                        content = _with_core_metadata(content, project_name, profile)
                elif item.filename == "[Content_Types].xml":
                    xml = content.decode("utf-8")
                    extra = "".join(
                        f'\t<Default Extension="{ext}" ContentType="{ctype}" />\n'
                        for ext, ctype in content_types.items()
                        if f'Extension="{ext}"' not in xml
                    )
                    content = _insert_before(xml, "</Types>", extra).encode("utf-8")
                elif item.filename == "_rels/.rels" and thumbnail:
                    xml = content.decode("utf-8")
                    if _THUMBNAIL_REL_TYPE not in xml:
                        rel = (
                            f'\t<Relationship Type="{_THUMBNAIL_REL_TYPE}" '
                            f'Target="/{THUMBNAIL_ENTRY}" Id="rel-cad-thumbnail" />\n'
                        )
                        xml = _insert_before(xml, "</Relationships>", rel)
                    content = xml.encode("utf-8")
                zout.writestr(item, content)
            if profile:
                added[MODEL_SETTINGS_ENTRY] = build_bambu_model_settings_config(
                    objects_meta, has_thumbnail=bool(thumbnail)
                ).encode("utf-8")
            for name, data in added.items():
                zout.writestr(name, data)
        temp_file.replace(output_path)
    finally:
        temp_file.unlink(missing_ok=True)


def _with_core_metadata(content: bytes, project_name: str, profile: dict[str, Any]) -> bytes:
    """Replace Title/Application/Designer/Description/Printer/BambuStudio:3mfVersion ``<metadata>``."""
    xml = content.decode("utf-8")
    xml = re.sub(
        r'\s*<metadata\s+name="(Title|Application|Designer|Description|Printer|BambuStudio:3mfVersion)"(?:\s[^>]*?/>|(?:\s[^>]*?(?<!/))?>.*?</metadata>)',
        "",
        xml,
        flags=re.DOTALL,
    )
    hardware_parts = []
    if profile.get("target_printer"):
        hardware_parts.append(f"Printer: {profile['target_printer']}")
    if profile.get("nozzle_diameter_mm"):
        hardware_parts.append(f"Nozzle: {profile['nozzle_diameter_mm']} mm")
    hw_info = f" | {' '.join(hardware_parts)}" if hardware_parts else ""
    description = (
        f"Material: {profile['material']} | Category: {profile['part_category']}{hw_info} | "
        f"{profile['rationale']}"
    ).strip(" |")
    meta: dict[str, str] = {
        "Title": project_name,
        "Application": "BambuStudio-01.09.05.51",
        "BambuStudio:3mfVersion": "1",
        "Designer": "Local AI CAD Agent",
        "Description": description,
    }
    if profile.get("target_printer"):
        meta["Printer"] = str(profile["target_printer"])
    tags = "".join(
        f'\t<metadata name="{name}">{xml_escape(value)}</metadata>\n' for name, value in meta.items()
    )
    # 3MF core spec: <metadata> elements precede <resources>.
    match = re.search(r"<resources[\s/>]", xml)
    if match:
        xml = xml[: match.start()] + tags + xml[match.start() :]
    else:
        xml = _insert_before(xml, "</model>", tags)
    return xml.encode("utf-8")


def _cached(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _ensure_preview_matches_model(project_dir: Path, model_sha256: str) -> None:
    """Reject stale ``preview.stl`` when ``.cad_metrics.json`` belongs to an older ``model.scad``."""
    metrics_path = project_dir / ".cad_metrics.json"
    if not metrics_path.is_file():
        return
    try:
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(cached, dict):
        recorded_sha = cached.get("model_sha256")
        if isinstance(recorded_sha, str) and recorded_sha and recorded_sha != model_sha256:
            raise FileNotFoundError(
                "Model has unbuilt changes; build the model before exporting or requesting print settings."
            )


def _synthesize_stl_thumbnail(preview_path: Path, cache_thumb_path: Path) -> bytes | None:
    """Render an isometric PNG from ``preview_path`` and cache it at ``cache_thumb_path``."""
    if _cached(cache_thumb_path):
        try:
            return cache_thumb_path.read_bytes()
        except OSError:
            pass
    if not _cached(preview_path):
        return None
    try:
        vertices, triangles = load_stl(preview_path)
        if len(vertices) == 0 or len(triangles) == 0:
            return None
        with tempfile.NamedTemporaryFile(
            dir=cache_thumb_path.parent, prefix=".tmp_thumb_", suffix=".png", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            render_iso(vertices, triangles, tmp_path)
            if _cached(tmp_path):
                data = tmp_path.read_bytes()
                tmp_path.replace(cache_thumb_path)
                return data
        finally:
            tmp_path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        return None
    return None


def _resolve_3mf_thumbnail(project_dir: Path, exports_dir: Path, model_sha256: str) -> bytes | None:
    """Return validated ``render.png`` bytes or synthesize a cached thumbnail from ``preview.stl``."""
    render_path = project_dir / "render.png"
    metrics_path = project_dir / ".cad_metrics.json"
    preview_path = project_dir / "preview.stl"
    cache_thumb_path = exports_dir / f"{model_sha256}_thumb.png"

    if not metrics_path.is_file():
        if _cached(render_path):
            return render_path.read_bytes()
        return _synthesize_stl_thumbnail(preview_path, cache_thumb_path)

    try:
        build = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(build, dict) or build.get("model_sha256") != model_sha256:
        return None

    if _cached(render_path):
        try:
            manifest = build.get("review_manifest") or {}
            expected_sha = (manifest.get("single_render") or {}).get("image_sha256")
            render_bytes = render_path.read_bytes()
            if (
                manifest.get("model_sha256") == model_sha256
                and expected_sha
                and hashlib.sha256(render_bytes).hexdigest() == expected_sha
            ):
                return render_bytes
        except OSError:
            pass

    return _synthesize_stl_thumbnail(preview_path, cache_thumb_path)


def _prune_stale_exports(exports_dir: Path, current_model_sha256: str) -> None:
    """Remove cached export files from earlier model revisions."""
    try:
        for entry in exports_dir.iterdir():
            if not entry.is_file() or entry.name.startswith(".tmp_"):
                continue
            if not entry.name.startswith(current_model_sha256):
                entry.unlink(missing_ok=True)
    except OSError:
        pass


def export_project_model(
    project_dir: Path,
    fmt: str,
    *,
    timeout_seconds: int = 120,
    print_profile: dict[str, Any] | None = None,
) -> Path:
    """Export the project model to the specified format with caching.

    3MF exports embed ``render.png`` as the package thumbnail and, when
    ``print_profile`` (already sanitized) is given, slicer configs for
    PrusaSlicer and Bambu Studio / OrcaSlicer.

    Returns the path to the exported file on disk.
    Raises ValueError for unsupported formats or empty models.
    Raises FileNotFoundError for missing model files.
    Raises RuntimeError if external conversion fails.
    """
    fmt = fmt.lower().strip()
    if fmt not in SUPPORTED_EXPORT_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_EXPORT_FORMATS))
        raise ValueError(f"Unsupported export format '{fmt}'. Supported formats: {supported}")

    model_path = project_dir / "model.scad"
    if not model_path.is_file():
        raise FileNotFoundError("No model source file found.")
    if model_path.stat().st_size == 0:
        raise ValueError("The model source file is empty.")

    if fmt == "scad":
        return model_path

    model_sha256 = compute_model_sha256(project_dir) or hashlib.sha256(model_path.read_bytes()).hexdigest()

    preview_path = project_dir / "preview.stl"
    if fmt in {"stl", "obj"}:
        if not preview_path.is_file():
            raise FileNotFoundError("No preview has been generated.")
        if preview_path.stat().st_size == 0:
            raise ValueError("The generated preview is empty.")
        _ensure_preview_matches_model(project_dir, model_sha256)
        if fmt == "stl":
            return preview_path

    exports_dir = project_dir / ".cad-agent" / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    _prune_stale_exports(exports_dir, model_sha256)
    base_file = exports_dir / f"{model_sha256}.{fmt}"

    if not _cached(base_file):
        if fmt == "obj":
            _write_obj(preview_path, base_file, exports_dir, model_sha256)
        else:
            # OpenSCAD-backed formats: 3mf, amf, off, csg
            _run_openscad_export(model_path, base_file, exports_dir, model_sha256, timeout_seconds)

    if fmt != "3mf":
        return base_file

    thumbnail = _resolve_3mf_thumbnail(project_dir, exports_dir, model_sha256)
    variant = hashlib.sha256(
        json.dumps(print_profile, sort_keys=True).encode("utf-8") + (thumbnail or b"")
    ).hexdigest()[:16]
    enriched_file = exports_dir / f"{model_sha256}_{variant}.3mf"
    if not _cached(enriched_file):
        enrich_3mf_archive(
            base_file,
            enriched_file,
            profile=print_profile,
            project_name=project_dir.name,
            thumbnail=thumbnail,
        )
    return enriched_file


def _write_obj(preview_path: Path, target: Path, exports_dir: Path, model_sha256: str) -> None:
    obj_content = stl_to_obj(preview_path)
    with tempfile.NamedTemporaryFile(
        dir=exports_dir,
        prefix=f".tmp_{model_sha256}_",
        suffix=".obj",
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as tmp:
        tmp.write(obj_content)
        temp_file = Path(tmp.name)
    try:
        temp_file.replace(target)
    finally:
        temp_file.unlink(missing_ok=True)


def _run_openscad_export(
    model_path: Path, target: Path, exports_dir: Path, model_sha256: str, timeout_seconds: int
) -> None:
    openscad_bin = find_openscad()
    if not openscad_bin:
        raise RuntimeError("OpenSCAD executable not found on server.")

    with tempfile.NamedTemporaryFile(
        dir=exports_dir,
        prefix=f".tmp_{model_sha256}_",
        suffix=target.suffix,
        delete=False,
    ) as tmp:
        temp_file = Path(tmp.name)

    project_dir = model_path.parent.resolve()
    try:
        temp_file.resolve().relative_to(project_dir)
        model_path.resolve().relative_to(project_dir)
        in_workspace = True
    except ValueError:
        in_workspace = False

    scad_args: list[str] = ["-o", str(temp_file.resolve())]
    if _supports_manifold():
        scad_args.append("--enable=manifold")
    scad_args.append(str(model_path.resolve()))

    seccomp_fd: int | None = None
    try:
        if _BWRAP and in_workspace:
            cmd, seccomp_fd = sandbox_command(
                project_dir,
                scad_args,
                writable=True,
                timeout_seconds=timeout_seconds,
                executable="openscad",
            )
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                cwd=str(project_dir),
                pass_fds=(seccomp_fd,),
            )
        else:
            proc = subprocess.run(
                [openscad_bin, *scad_args],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                cwd=str(project_dir) if in_workspace else None,
            )
        if proc.returncode != 0 or not _cached(temp_file):
            err_msg = (proc.stderr or proc.stdout).strip()
            raise RuntimeError(f"OpenSCAD export failed:\n{err_msg or 'No output produced.'}")
        temp_file.replace(target)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"OpenSCAD export to {target.suffix[1:]} timed out.") from exc
    finally:
        if seccomp_fd is not None:
            try:
                os.close(seccomp_fd)
            except OSError:
                pass
        temp_file.unlink(missing_ok=True)
