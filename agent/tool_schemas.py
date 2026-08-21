"""Small, operation-specific schemas exposed to the model."""

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


_FILENAME = {
    "type": "string",
    "enum": ["model.py"],
    "description": "Project file to read or edit.",
}
_TIMEOUT = {
    "type": "integer",
    "minimum": 1,
    "maximum": 120,
    "description": "Maximum runtime in seconds (1-120).",
}

TOOL_SCHEMAS = [
    _tool(
        "read_file",
        "Read model.py. The response always includes exists and "
        "the current SHA-256 (when the file exists); pass that digest as "
        "expected_sha256 to write_file or edit_file. A missing file reports "
        "exists=false and must be created with write_file.",
        {
            "filename": _FILENAME,
            "offset": {
                "type": "integer",
                "minimum": 1,
                "description": "One-based starting line. Defaults to 1.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 2000,
                "description": "Maximum lines to return. Omit for the complete file.",
            },
            "known_sha256": {
                "type": "string",
                "minLength": 64,
                "maxLength": 64,
                "description": "Previously returned SHA-256; returns unchanged metadata if it still matches.",
            },
        },
        ["filename"],
    ),
    _tool(
        "write_file",
        "Create model.py, or deliberately replace its entire contents. Use this "
        "ONLY for the initial creation of model.py or a deliberate full rewrite; "
        "for any small localized change (≤ ~10 lines, a single parameter, a "
        "block with a clear old/new boundary), prefer edit_file to avoid wasting "
        "tokens and breaking the revision history. ``expected_sha256`` is "
        "optional; omit it unless you want strict conflict detection.",
        {
            "filename": _FILENAME,
            "content": {"type": "string", "description": "Complete file contents."},
            "expected_sha256": {
                "type": "string",
                "minLength": 64,
                "maxLength": 64,
                "description": "Optional. SHA-256 from a recent read_file; a stale digest is rejected. Omit when you want an unconditional overwrite.",
            },
        },
        ["filename", "content"],
    ),
    _tool(
        "edit_file",
        "Replace one exact, uniquely matching block in an existing file. Prefer "
        "this over write_file for any small localized change (single parameter, "
        "narrow bug fix, ≤ ~10 lines). ``expected_sha256`` is optional; omit it "
        "unless you are guarding against a concurrent external edit.",
        {
            "filename": _FILENAME,
            "old_string": {
                "type": "string",
                "minLength": 1,
                "description": "Exact text to find, copied verbatim from read_file.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text; may be empty to delete the block.",
            },
            "expected_sha256": {
                "type": "string",
                "minLength": 64,
                "maxLength": 64,
                "description": "Optional. Current SHA-256 from read_file; pass only when you want to reject stale edits. Omit for an unconditional edit.",
            },
        },
        ["filename", "old_string", "new_string"],
    ),
    _tool(
        "insert_file",
        "Insert a new block immediately before or after one exact, uniquely matching short anchor. Use this for substantial feature additions so you do not repeat a large existing block in edit_file.old_string.",
        {
            "filename": _FILENAME,
            "anchor": {
                "type": "string",
                "minLength": 1,
                "description": "Short exact anchor copied from read_file; it must occur once.",
            },
            "content": {
                "type": "string",
                "minLength": 1,
                "description": "New content to insert, including intentional newlines.",
            },
            "position": {"type": "string", "enum": ["before", "after"]},
            "expected_sha256": {
                "type": "string",
                "minLength": 64,
                "maxLength": 64,
            },
        },
        ["filename", "anchor", "content", "position"],
    ),
    _tool(
        "cad_build_and_verify",
        "Build the latest model.py revision, validate basic geometry, export preview.stl, and (when render=true) rasterise the canonical eight views plus a labelled contact sheet. Does NOT trigger review automatically — call cad_review separately if you want a verdict. Default is render=false: returns only metrics + preview.stl + model_sha256 + preview_sha256 (cheap, cache-friendly). Pass render=true only for the final verification before declaring the task ready.",
        {
            "render": {
                "type": "boolean",
                "default": False,
                "description": "Generate canonical eight-view rasterisation + contact sheet (true) or skip rendering and return only metrics + preview.stl (false, default). Use false during early iterations; final verification must use true.",
            },
            "parameter_checks": {
                "type": "array",
                "maxItems": 20,
                "description": "Final-build checks for explicit numeric parameters defined in model.py. Use these for every user-stated dimension, angle, clearance, or count represented by a named parameter. Each check must include at least one bound: equals, minimum, or maximum (the runner silently treats a bare name as a no-op pass, so always include a bound).",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "minimum": {"type": "number"},
                        "maximum": {"type": "number"},
                        "equals": {"type": "number"},
                        "tolerance": {"type": "number", "minimum": 0},
                    },
                    "required": ["name"],
                },
            },
        },
        [],
    ),
    _tool(
        "cad_screenshot",
        "Rasterise the latest model.py revision from one or more camera views without re-running build123d. Reuses the artifact cache produced by cad_build_and_verify(render=true) when the (model_sha256, sorted(views), quality) tuple matches; otherwise re-rasterises only the missing subset. Reserve this for complex or visually ambiguous work; do not use it for routine small edits that already pass cad_build_and_verify. Returns file paths and per-image SHA-256 digests; no base64 inline payloads.",
        {
            "views": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "x_positive",
                        "x_negative",
                        "y_positive",
                        "y_negative",
                        "z_positive",
                        "z_negative",
                        "isometric_positive",
                        "isometric_negative",
                    ],
                },
                "minItems": 1,
                "maxItems": 8,
                "description": "Subset of canonical view_ids to rasterise. Empty or omitted = the full canonical eight.",
            },
            "contact_sheet": {
                "type": "boolean",
                "default": True,
                "description": "If true, also return review-sheet.png combining the chosen views in canonical order. If false, return only per-view PNGs.",
            },
            "quality": {
                "type": "string",
                "enum": ["low", "standard", "high"],
                "default": "standard",
                "description": "low=256x256 + coarse tessellation (0.3 tol); standard=512x512 + 0.1 tol (matches cad_build_and_verify default); high=1024x1024 + 0.05 tol.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 120,
                "default": 30,
                "description": "Maximum runtime in seconds for the sandbox subprocess (1-120).",
            },
        },
        [],
    ),
    _tool(
        "cad_review",
        "Run the structured reviewer against the latest CAD build's visual + logical evidence. Deterministic checks (dimensions, volume, solid count, through-hole count, spec requirements) always run first; the multimodal LLM call only runs when visual evidence is present. If no artifact exists yet, cad_review internally calls cad_screenshot to produce one. Behavior is always strict: any blocking or major finding reclassifies the verdict to fail. This is optional: reserve it for complex, high-risk, visually ambiguous, fit-critical, or user-requested work; never use it as a routine final step for a small local edit. The agent loop does NOT auto-trigger review.",
        {
            "views": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "x_positive",
                        "x_negative",
                        "y_positive",
                        "y_negative",
                        "z_positive",
                        "z_negative",
                        "isometric_positive",
                        "isometric_negative",
                    ],
                },
                "description": "Subset of view_ids the multimodal reviewer should focus on. Empty = all canonical views. The deterministic layer always checks every face, so this only narrows the visual scan.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 120,
                "default": 60,
                "description": "Maximum runtime in seconds for the visual reviewer (1-120).",
            },
        },
        [],
    ),
    _tool(
        "question",
        "Ask all blocking clarification questions together, then stop and wait. Use only when the answer materially affects fit, function, or manufacturability.",
        {
            "title": {"type": "string", "description": "Optional short heading."},
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "description": "Blocking questions to present in one form.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Unique short key, such as hole_diameter.",
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
                    "additionalProperties": False,
                },
            },
        },
        ["questions"],
    ),
]
