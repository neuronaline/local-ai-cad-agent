"""Small, operation-specific schemas exposed to the model."""

from __future__ import annotations

from typing import Any

from agent.tools.question_tool import QuestionTool


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


_FILENAME = {"type": "string", "enum": ["model.py", "summary.md"]}
_TIMEOUT = {"type": "integer", "minimum": 1, "maximum": 120}

TOOL_SCHEMAS = [
    _tool(
        "file_read",
        "Read model.py or summary.md.",
        {"filename": _FILENAME},
        ["filename"],
    ),
    _tool(
        "file_write",
        "Write the complete contents of model.py or summary.md.",
        {"filename": _FILENAME, "content": {"type": "string"}},
        ["filename", "content"],
    ),
    _tool(
        "file_replace",
        "Replace the first exact text match in model.py or summary.md.",
        {"filename": _FILENAME, "old": {"type": "string"}, "new": {"type": "string"}},
        ["filename", "old", "new"],
    ),
    _tool(
        "file_regex_replace",
        "Apply a bounded regular-expression replacement to model.py or summary.md.",
        {
            "filename": _FILENAME,
            "pattern": {"type": "string"},
            "replacement": {"type": "string"},
            "count": {"type": "integer", "minimum": 1, "maximum": 20, "default": 1},
        },
        ["filename", "pattern", "replacement"],
    ),
    _tool(
        "cad_build_and_verify",
        "Build model.py once and return validated geometry metrics, preview, and a rendered image for visual review.",
        {},
        [],
    ),
    _tool(
        "terminal_run",
        "Run one project-local Python script in the sandbox.",
        {
            "arguments": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "timeout_seconds": _TIMEOUT,
        },
        ["arguments"],
    ),
    _tool(
        "terminal_check",
        "Run a read-only Python, pytest, or pip inspection in the sandbox.",
        {
            "arguments": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "timeout_seconds": _TIMEOUT,
        },
        ["arguments"],
    ),
    QuestionTool.__tool_schema__,
    _tool(
        "experience_search",
        "Search verified problem-solution memory shared across projects.",
        {"query": {"type": "string", "minLength": 1}},
        ["query"],
    ),
    _tool(
        "experience_add",
        "Store a newly verified, generalized problem-solution pair.",
        {
            "problem": {"type": "string", "maxLength": 500},
            "solution": {"type": "string", "maxLength": 2000},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        },
        ["problem", "solution"],
    ),
    _tool(
        "experience_update",
        "Update an existing verified memory record by id.",
        {
            "id": {"type": "string", "minLength": 1},
            "problem": {"type": "string", "maxLength": 500},
            "solution": {"type": "string", "maxLength": 2000},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        },
        ["id"],
    ),
]
