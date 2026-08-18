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
_TAGS = {
    "type": "array",
    "items": {"type": "string", "minLength": 1, "maxLength": 50},
    "maxItems": 10,
    "description": "Up to 10 concise lowercase technical tags.",
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
        "terminal_run",
        "Execute an existing Python script from the project root inside a writable sandbox. Use this when you need to run project-specific code that already exists as a file; do NOT use it for one-off inspections (use terminal_check) or shell commands (use terminal_bash). arguments MUST be exactly ['python', 'script.py']; inline code, modules, subdirectories, and script arguments are rejected.",
        {
            "arguments": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
                "description": 'Exactly ["python", "script.py"].',
            },
            "timeout_seconds": _TIMEOUT,
        },
        ["arguments"],
    ),
    _tool(
        "terminal_check",
        "Run a read-only Python inspection, pytest, or pip query in the sandbox. Use this for quick verification (e.g. python -c 'expr', python -m pytest, python -m pip list). NOT for project scripts (use terminal_run) or shell commands (use terminal_bash). arguments must start with python and use -c or -m.",
        {
            "arguments": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "description": 'Command tokens, e.g. ["python", "-m", "pytest", "-q"].',
            },
            "timeout_seconds": _TIMEOUT,
        },
        ["arguments"],
    ),
    _tool(
        "terminal_bash",
        "Run one read-only shell-style inspection in the sandbox (e.g. rg, git status, ls). Use this for fast repo navigation when Python would be overkill. NOT for project scripts (use terminal_run) or Python evaluations (use terminal_check). Shell operators, redirects, substitutions, writes, and network access are rejected.",
        {
            "command": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2000,
                "description": "One read-only command, such as 'rg Cylinder model.py' or 'git status'.",
            },
            "timeout_seconds": _TIMEOUT,
        },
        ["command"],
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
    _tool(
        "experience_search",
        "Search verified cross-project CAD lessons. Use concise technical failure, feature, and API terms rather than the full conversation.",
        {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "Concise space-separated technical search terms.",
            }
        },
        ["query"],
    ),
    _tool(
        "experience_get",
        "Read the complete verified lesson for a record ID from the past-issues context index or experience_search results.",
        {
            "id": {
                "type": "string",
                "minLength": 1,
                "description": "Exact record ID from the past-issues context index or experience_search.",
            }
        },
        ["id"],
    ),
    _tool(
        "experience_add",
        "Store a general, newly verified CAD problem-solution lesson after checking for duplicates. Never store source code, secrets, or identifying details.",
        {
            "title": {
                "type": "string",
                "minLength": 1,
                "maxLength": 120,
                "description": "Short, single-line technical title shown in the past-issues index.",
            },
            "problem": {
                "type": "string",
                "minLength": 1,
                "maxLength": 500,
                "description": "Generalized technical problem.",
            },
            "solution": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2000,
                "description": "Tested solution and the conditions where it applies.",
            },
            "tags": _TAGS,
        },
        ["title", "problem", "solution"],
    ),
    _tool(
        "experience_update",
        "Correct or enrich an existing verified CAD lesson. Provide only fields that should change.",
        {
            "id": {
                "type": "string",
                "minLength": 1,
                "description": "Record ID from experience_search or the past-issues index.",
            },
            "title": {
                "type": "string",
                "minLength": 1,
                "maxLength": 120,
                "description": "Corrected short technical title.",
            },
            "problem": {
                "type": "string",
                "minLength": 1,
                "maxLength": 500,
                "description": "Corrected generalized problem.",
            },
            "solution": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2000,
                "description": "Corrected verified solution.",
            },
            "tags": _TAGS,
        },
        ["id"],
    ),
]
