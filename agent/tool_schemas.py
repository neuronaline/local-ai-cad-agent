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
        "Read the current complete contents of model.py or summary.md. Use before a targeted replacement when the exact text is unknown.",
        {"filename": _FILENAME},
        ["filename"],
    ),
    _tool(
        "file_write",
        "Replace model.py or summary.md with complete new contents. This overwrites the whole file; use a replacement tool for a localized edit.",
        {
            "filename": _FILENAME,
            "content": {"type": "string", "description": "Complete file contents."},
        },
        ["filename", "content"],
    ),
    _tool(
        "file_replace",
        "Replace only the first exact text match. The call fails without changing the file when old is not found.",
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
        },
        ["filename", "pattern", "replacement"],
    ),
    _tool(
        "cad_build_and_verify",
        "Build the latest model.py revision and return geometry metrics plus a render. Call after editing model.py; do not repeat unless the source changes.",
        {},
        [],
    ),
    _tool(
        "terminal_run",
        'Run one existing Python script from the project root in a writable sandbox. Use arguments ["python", "script.py"]; inline code, modules, subdirectories, and script arguments are rejected.',
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
        "Run a read-only Python inspection, pytest, or pip query in the sandbox. arguments must start with python and use -c or -m; use terminal_run for a project script.",
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
        "experience_add",
        "Store a general, newly verified CAD problem-solution lesson after checking for duplicates. Never store source code, secrets, or identifying details.",
        {
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
        ["problem", "solution"],
    ),
    _tool(
        "experience_update",
        "Correct or enrich an existing verified CAD lesson. Provide only fields that should change.",
        {
            "id": {
                "type": "string",
                "minLength": 1,
                "description": "Record id returned by experience_search.",
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
