"""Small, operation-specific schemas exposed to the model.

The schemas hide every fixed infrastructure detail (file paths, SHA-256
digests, sandbox limits) so the model focuses on the operation it is
trying to perform. Each tool returns a common ``ok`` envelope — see
:mod:`agent.tool_results` — so the model never has to guess which fields
a tool produced.
"""

from __future__ import annotations

from typing import Any

from agent.tools.file_tool import MAX_READ_LINES
from agent.tools.process_runner import MAX_SANDBOX_TIMEOUT_SECONDS


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


_TIMEOUT = {
    "type": "integer",
    "minimum": 1,
    "maximum": MAX_SANDBOX_TIMEOUT_SECONDS,
    "description": f"Maximum runtime in seconds (1-{MAX_SANDBOX_TIMEOUT_SECONDS}).",
}

CANONICAL_VIEWS = (
    "x_positive",
    "x_negative",
    "y_positive",
    "y_negative",
    "z_positive",
    "z_negative",
    "isometric_positive",
    "isometric_negative",
)

_QUALITY_DESCRIPTION = (
    "low=256x256 + coarse tessellation (0.3 tol); "
    "standard=512x512 + 0.1 tol (matches cad_build_and_verify default); "
    "high=1024x1024 + 0.05 tol."
)

TOOL_SCHEMAS = [
    _tool(
        "read_file",
        "Read the current model.py. The response always includes ``exists`` and, "
        "when the file exists, the current SHA-256. A missing file reports "
        "``exists=false`` and must be created with write_file.",
        {
            "offset": {
                "type": "integer",
                "minimum": 1,
                "description": "One-based starting line. Defaults to 1.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_READ_LINES,
                "description": f"Maximum lines to return. Omit for the complete file (capped at {MAX_READ_LINES}).",
            },
            "known_sha256": {
                "type": "string",
                "minLength": 64,
                "maxLength": 64,
                "description": "Previously returned SHA-256; returns unchanged metadata when it still matches.",
            },
        },
        [],
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
            "content": {"type": "string", "description": "Complete file contents."},
            "expected_sha256": {
                "type": "string",
                "minLength": 64,
                "maxLength": 64,
                "description": "Optional. SHA-256 from a recent read_file; a stale digest is rejected. Omit when you want an unconditional overwrite.",
            },
        },
        ["content"],
    ),
    _tool(
        "edit_file",
        "Apply one or more exact, atomic replacements to model.py. Pass an array "
        "of {old_string, new_string} edits; every match is verified first, then "
        "the edits are applied together and the final source is re-validated. "
        "Use the single-pair shape for one fix, the array shape when changing "
        "several related parameters at once. ``expected_sha256`` is optional; "
        "omit it unless you are guarding against a concurrent external edit.",
        {
            "edits": {
                "type": "array",
                "minItems": 1,
                "maxItems": 16,
                "description": "Atomic edit list. Each entry must have old_string and new_string.",
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
            "expected_sha256": {
                "type": "string",
                "minLength": 64,
                "maxLength": 64,
                "description": "Optional. Current SHA-256 from read_file; pass only when you want to reject stale edits. Omit for an unconditional edit.",
            },
        },
        ["edits"],
    ),
    _tool(
        "insert_file",
        "Insert a new block immediately before or after one exact, uniquely matching short anchor. Use this for substantial feature additions so you do not repeat a large existing block in edit_file.edits[].old_string.",
        {
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
        ["anchor", "content", "position"],
    ),
    _tool(
        "cad_build_and_verify",
        "Build the latest model.py revision, validate basic geometry, and export "
        "preview.stl. It renders canonical eight-view evidence by default and "
        "automatically reports numeric UPPER_CASE parameters read from the "
        "initial model.py AST block. Set ``render=false`` only for a cheap "
        "iteration where visual evidence is not needed.\n"
        "Does NOT trigger review automatically — call cad_review separately if "
        "you want a verdict.",
        {
            "render": {
                "type": "boolean",
                "default": True,
                "description": "True (default) renders canonical evidence inline after geometry validation. False returns metrics and preview only for a quick iteration.",
            },
        },
        [],
    ),
    _tool(
        "cad_screenshot",
        "Rasterise the latest model.py revision from one or more camera views without re-running build123d. Reuses the artifact cache produced by a rendered cad_build_and_verify call when the (model_sha256, sorted(views), quality) tuple matches; otherwise re-rasterises only the missing subset. Attach the requested views inline so you can inspect them in-band; reserve this for complex or visually ambiguous work, not routine small edits that already pass cad_build_and_verify.",
        {
            "views": {
                "type": "array",
                "items": {"type": "string", "enum": list(CANONICAL_VIEWS)},
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
                "description": _QUALITY_DESCRIPTION,
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_SANDBOX_TIMEOUT_SECONDS,
                "default": 30,
                "description": f"Maximum runtime in seconds for the sandbox subprocess (1-{MAX_SANDBOX_TIMEOUT_SECONDS}).",
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
                "items": {"type": "string", "enum": list(CANONICAL_VIEWS)},
                "description": "Subset of view_ids the multimodal reviewer should focus on. Empty = all canonical views. The deterministic layer always checks every face, so this only narrows the visual scan.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_SANDBOX_TIMEOUT_SECONDS,
                "default": 60,
                "description": f"Maximum runtime in seconds for the visual reviewer (1-{MAX_SANDBOX_TIMEOUT_SECONDS}).",
            },
        },
        [],
    ),
    _tool(
        "question",
        "Ask all blocking clarification questions together, then stop and wait. Use only when the answer materially affects fit, function, or manufacturability. Each item's input_type defaults to ``text``; pass ``select`` (single choice) or ``multiselect`` to add an ``options`` array. ``required`` defaults to ``true``; pass ``false`` only for genuinely optional questions.",
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
