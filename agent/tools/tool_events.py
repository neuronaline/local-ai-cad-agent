"""Shared SSE status-publish helper for the CAD tools.

Three tools (``cad_build_and_verify``, ``cad_screenshot``, ``cad_review``)
need to publish phase events to the live activity panel. Before this
module each one reimplemented the same callable-check + try/except
boilerplate and quietly diverged in field names (``result`` vs
``message``). Centralising the contract keeps the activity pill stable
and stops activity-publish failures from aborting the actual tool.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

_LOG = logging.getLogger(__name__)


def publish_tool_phase(
    publish: Callable[[str, dict[str, Any]], None] | None,
    *,
    project: str,
    tool: str,
    call_id: str,
    status: str,
    message: str,
) -> None:
    """Publish a ``tool_status`` (call_id present) or ``agent_status`` event.

    Tools that already have a ``call_id`` route the event to ``tool_status``
    so the activity panel can attach the event to the in-flight tool card.
    Tools without a ``call_id`` (e.g. status updates emitted from the host
    before a tool dispatch) fall back to ``agent_status``. Failures are
    logged and never propagate; status events are advisory and must not
    abort the real tool execution.
    """
    if not callable(publish):
        return
    try:
        if call_id:
            publish(
                "tool_status",
                {
                    "project": project,
                    "call_id": call_id,
                    "tool": tool,
                    "status": status,
                    "result": message,
                },
            )
        else:
            publish(
                "agent_status",
                {
                    "project": project,
                    "status": status,
                    "message": message,
                },
            )
    except Exception as error:  # noqa: BLE001 - status events must never fail the tool.
        _LOG.warning(
            "tool status publish failed (tool=%s, status=%s, project=%s): %s",
            tool,
            status,
            project,
            error,
            exc_info=_LOG.isEnabledFor(logging.DEBUG),
        )
