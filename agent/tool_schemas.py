"""Small, operation-specific schemas exposed to the model.

The schemas hide every fixed infrastructure detail (file paths, SHA-256
digests, sandbox limits) so the model focuses on the operation it is
trying to perform. Each tool returns a common ``ok`` envelope — see
:mod:`agent.tool_results` — so the model never has to guess which fields
a tool produced.
"""

from __future__ import annotations

from typing import Any

from agent.prompt import TOOL_DESCRIPTIONS


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
        TOOL_DESCRIPTIONS["read_file"]["description"],
        {
            "offset": {
                "type": "integer",
                "minimum": 1,
                "default": 1,
                "description": TOOL_DESCRIPTIONS["read_file"]["offset"],
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 2000,
                "description": TOOL_DESCRIPTIONS["read_file"]["limit"],
            },
        },
        [],
    ),
    _tool(
        "write_file",
        TOOL_DESCRIPTIONS["write_file"]["description"],
        {
            "content": {
                "type": "string",
                "description": TOOL_DESCRIPTIONS["write_file"]["content"],
            },
        },
        ["content"],
    ),
    _tool(
        "edit_file",
        TOOL_DESCRIPTIONS["edit_file"]["description"],
        {
            "old_string": {
                "type": "string",
                "description": TOOL_DESCRIPTIONS["edit_file"]["old_string"],
            },
            "new_string": {
                "type": "string",
                "description": TOOL_DESCRIPTIONS["edit_file"]["new_string"],
            },
        },
        ["old_string", "new_string"],
    ),
    _tool(
        "cad_build",
        TOOL_DESCRIPTIONS["cad_build"]["description"],
        {
            "views": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "string",
                    "enum": [
                        "all",
                        "isometric",
                        "top",
                        "bottom",
                        "front",
                        "back",
                        "left",
                        "right",
                        "isometric_negative",
                    ],
                },
                "description": TOOL_DESCRIPTIONS["cad_build"]["views"],
            },
            "resolution": {
                "type": "integer",
                "enum": [512, 1024],
                "default": 512,
                "description": TOOL_DESCRIPTIONS["cad_build"]["resolution"],
            },
            "crop": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "number"},
                "description": TOOL_DESCRIPTIONS["cad_build"]["crop"],
            },
        },
        [],
    ),
    _tool(
        "get_view_images",
        TOOL_DESCRIPTIONS["get_view_images"]["description"],
        {
            "views": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "string",
                    "enum": [
                        "all",
                        "isometric",
                        "top",
                        "bottom",
                        "front",
                        "back",
                        "left",
                        "right",
                        "isometric_negative",
                    ],
                },
                "description": TOOL_DESCRIPTIONS["get_view_images"]["views"],
            },
            "crop": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "number"},
                "description": TOOL_DESCRIPTIONS["get_view_images"]["crop"],
            },
        },
        [],
    ),
    _tool(
        "question",
        TOOL_DESCRIPTIONS["question"]["description"],
        {
            "title": {
                "type": "string",
                "description": TOOL_DESCRIPTIONS["question"]["title"],
            },
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "description": TOOL_DESCRIPTIONS["question"]["questions"],
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {
                            "type": "string",
                            "minLength": 1,
                            "description": TOOL_DESCRIPTIONS["question"]["id"],
                        },
                        "question": {
                            "type": "string",
                            "minLength": 1,
                            "description": TOOL_DESCRIPTIONS["question"]["question"],
                        },
                        "input_type": {
                            "type": "string",
                            "enum": ["text", "select", "number", "multiselect"],
                            "description": TOOL_DESCRIPTIONS["question"]["input_type"],
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "minItems": 2,
                            "description": TOOL_DESCRIPTIONS["question"]["options"],
                        },
                        "required": {
                            "type": "boolean",
                            "description": TOOL_DESCRIPTIONS["question"]["required"],
                        },
                    },
                    "required": ["id", "question"],
                },
            },
        },
        ["questions"],
    ),
]
