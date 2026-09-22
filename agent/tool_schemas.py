"""Small, operation-specific schemas exposed to the model.

The schemas hide every fixed infrastructure detail (file paths, SHA-256
digests, sandbox limits) so the model focuses on the operation it is
trying to perform. Each tool returns a common ``ok`` envelope — see
:mod:`agent.tool_results` — so the model never has to guess which fields
a tool produced.
"""

from __future__ import annotations

from typing import Any


def _tool(
    name: str, description: str, properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    _tool(
        "read_file",
        "Read the current model.scad. Returns whether the file exists and its contents. "
        "Optionally provide offset and limit to inspect specific line ranges.",
        {
            "offset": {
                "type": "integer",
                "minimum": 1,
                "default": 1,
                "description": "1-indexed starting line number (defaults to 1).",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 2000,
                "description": "Maximum number of lines to return (1-2000).",
            },
        },
        [],
    ),
    _tool(
        "write_file",
        "Create model.scad, or deliberately replace its entire contents. Use this "
        "ONLY for the initial creation of model.scad or a deliberate full rewrite; "
        "for localized changes, prefer edit_file.",
        {
            "content": {"type": "string", "description": "Complete file contents."},
        },
        ["content"],
    ),
    _tool(
        "edit_file",
        "Apply exact replacements to model.scad. Pass either old_string and new_string "
        "for a single replacement, or an edits array for multiple atomic replacements.",
        {
            "old_string": {
                "type": "string",
                "description": "Exact text to find, copied verbatim from read_file. Used for single edit.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text for old_string (may be empty to delete).",
            },
            "edits": {
                "type": "array",
                "minItems": 1,
                "maxItems": 16,
                "description": "Optional atomic edit list for multiple replacements at once. Each entry must have old_string and new_string.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "old_string": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Exact text to find, copied verbatim from read_file. Must occur exactly once across the file.",
                        },
                        "new_string": {
                            "type": "string",
                            "description": "Replacement text; may be empty to delete the block.",
                        },
                    },
                    "required": ["old_string", "new_string"],
                },
            },
        },
        [],
    ),
    _tool(
        "cad_build_and_verify",
        "Build the latest model.scad, validate basic geometry, export preview.stl, "
        "and render the canonical isometric PNG plus the eight-view review manifest "
        "and contact sheet. Automatically reports dimensions, solid count, volume, "
        "and numeric UPPER_CASE parameters.",
        {},
        [],
    ),
    _tool(
        "get_view_images",
        "Retrieve rendered image(s) of the current model for permanent visual memory. "
        "Each request selects one canonical view and optionally a cropped area within "
        "it (normalized 0-1 coordinates, origin top-left). Request all areas you need "
        "in one call (max 4). Requires a successful cad_build_and_verify first.",
        {
            "images": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "description": "Images to retrieve in this call.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "view": {
                            "type": "string",
                            "enum": [
                                "x_positive", "x_negative",
                                "y_positive", "y_negative",
                                "z_positive", "z_negative",
                                "isometric_positive", "isometric_negative",
                            ],
                            "description": "Canonical rendered view to take the image from.",
                        },
                        "crop": {
                            "type": "object",
                            "additionalProperties": False,
                            "description": "Optional sub-area of the 512x512 view. "
                                           "Omit for the full view.",
                            "properties": {
                                "x":      {"type": "number", "minimum": 0, "maximum": 1},
                                "y":      {"type": "number", "minimum": 0, "maximum": 1},
                                "width":  {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                                "height": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                            },
                            "required": ["x", "y", "width", "height"],
                        },
                    },
                    "required": ["view"],
                },
            },
        },
        ["images"],
    ),
    _tool(
        "question",
        "Ask all blocking clarification questions together, then stop and wait. ``input_type`` defaults to ``text``; pass ``select`` or ``multiselect`` for choice questions. ``required`` defaults to ``true``.",
        {
            "title": {"type": "string", "description": "Optional short heading."},
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "description": "Blocking questions to present in one form (max 3 per batch).",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Short key, such as hole_diameter.",
                        },
                        "question": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Direct user-facing question.",
                        },
                        "input_type": {
                            "type": "string",
                            "enum": ["text", "select", "number", "multiselect"],
                            "description": "Answer control; defaults to text.",
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "minItems": 2,
                            "description": "Required for select and multiselect.",
                        },
                        "required": {
                            "type": "boolean",
                            "description": "Whether an answer is mandatory; defaults to true.",
                        },
                    },
                    "required": ["id", "question"],
                },
            },
        },
        ["questions"],
    ),
]
