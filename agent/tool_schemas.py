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
    "enum": ["model.py", "summary.md"],
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
        "file_read",
        "Read model.py or summary.md. Use offset and limit for a focused range; the ranged result includes a SHA-256 digest for guarded edits.",
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
        "file_write",
        "Replace model.py or summary.md with complete new contents. This overwrites the whole file; for an existing file, pass expected_sha256 from a ranged read to prevent overwriting a newer edit.",
        {
            "filename": _FILENAME,
            "content": {"type": "string", "description": "Complete file contents."},
            "expected_sha256": {
                "type": "string",
                "minLength": 64,
                "maxLength": 64,
                "description": "Optional SHA-256 from file_read; rejects stale writes.",
            },
        },
        ["filename", "content"],
    ),
    _tool(
        "file_replace",
        "Replace only the first exact text match. Optionally require the expected total match count and file SHA-256 to avoid editing an ambiguous or stale target.",
        {
            "filename": _FILENAME,
            "old": {
                "type": "string",
                "minLength": 1,
                "description": "Exact text to find.",
            },
            "new": {
                "type": "string",
                "description": "Replacement text; may be empty to delete the match.",
            },
            "expected_sha256": {
                "type": "string",
                "minLength": 64,
                "maxLength": 64,
                "description": "Optional SHA-256 from file_read; rejects stale edits.",
            },
            "expected_matches": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
                "description": "Optional total occurrences old must have before the first is replaced.",
            },
        },
        ["filename", "old", "new"],
    ),
    _tool(
        "file_regex_replace",
        "Apply a bounded Python regex replacement with DOTALL enabled. Use explicit anchors and non-greedy matching for multiline edits.",
        {
            "filename": _FILENAME,
            "pattern": {
                "type": "string",
                "minLength": 1,
                "maxLength": 500,
                "description": "Python regular expression; DOTALL is enabled.",
            },
            "replacement": {
                "type": "string",
                "maxLength": 10000,
                "description": "Python re.sub replacement text.",
            },
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 1,
                "description": "Maximum replacements; defaults to 1.",
            },
            "expected_sha256": {
                "type": "string",
                "minLength": 64,
                "maxLength": 64,
                "description": "Optional SHA-256 from file_read; rejects stale edits.",
            },
        },
        ["filename", "pattern", "replacement"],
    ),
    _tool(
        "cad_build_and_verify",
        "Build the latest model.py revision, validate basic geometry, export preview.stl, and (when render=true, the default) rasterise the canonical eight views plus a labelled contact sheet. Does NOT trigger review automatically — call cad_review separately if you want a verdict. Set render=false to skip rendering during early iteration; the final verification must use render=true.",
        {
            "render": {
                "type": "boolean",
                "default": True,
                "description": "Generate canonical eight-view rasterisation + contact sheet (true, default) or skip rendering and return only metrics + preview.stl (false). Use false during early iterations; final verification must use true.",
            },
        },
        [],
    ),
    _tool(
        "cad_screenshot",
        "Rasterise the latest model.py revision from one or more camera views without re-running build123d. Reuses the artifact cache produced by cad_build_and_verify(render=true) when the (model_sha256, sorted(views), quality) tuple matches; otherwise re-rasterises only the missing subset. Use this when you need a single angle, a subset of canonical views, or a higher-resolution look before deciding whether to call cad_review. Returns file paths and per-image SHA-256 digests; no base64 inline payloads.",
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
        "Run the structured reviewer against the latest CAD build's visual + logical evidence. Deterministic checks (dimensions, volume, solid count, through-hole count, spec requirements) always run first; the multimodal LLM call only runs when visual evidence is present. If no artifact exists yet, cad_review internally calls cad_screenshot to produce one. Behavior is always strict: any blocking or major finding reclassifies the verdict to fail. Call this once you want a verdict — not after every model.py edit. The agent loop does NOT auto-trigger review.",
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
