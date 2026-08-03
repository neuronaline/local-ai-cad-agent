"""Validation and artifact generation for a finished CAD project."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from agent.tool_results import is_failure
from agent.tools.cad_tool import CadTool

MIN_REASONABLE_DIMENSION_MM = 0.001
MAX_REASONABLE_DIMENSION_MM = 1_000_000.0

PLACEHOLDER_MARKER = "No model has been created yet."


def _validate_metrics(metrics: dict[str, Any]) -> None:
    dimensions = metrics["dimensions_mm"]
    if metrics["solid_count"] < 1:
        raise ValueError("Finalization requires at least one solid.")
    if not metrics["is_valid"]:
        raise ValueError("Finalization requires valid geometry.")
    dimension_values = [float(value) for value in dimensions.values()]
    if len(dimension_values) != 3 or any(
        not math.isfinite(value) for value in dimension_values
    ):
        raise ValueError("Finalization requires finite X, Y, and Z dimensions.")
    if any(
        value < MIN_REASONABLE_DIMENSION_MM or value > MAX_REASONABLE_DIMENSION_MM
        for value in dimension_values
    ):
        raise ValueError(
            "Bounding-box dimensions must be between "
            f"{MIN_REASONABLE_DIMENSION_MM:g} and {MAX_REASONABLE_DIMENSION_MM:g} mm."
        )
    volume = float(metrics["volume_mm3"])
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("Finalization requires a finite, positive solid volume.")


def _extract_limitations(summary: str) -> str | None:
    """Extract a `## Limitations` section from the summary if present."""
    match = re.search(
        r"##\s+Limitations\s*\n((?:(?!##\s).*\n?)+)", summary, re.IGNORECASE
    )
    if not match:
        return None
    content = match.group(1).strip()
    return content if content else None


def _parse_journey(project_dir: Path) -> dict[str, int | str]:
    """Scan conversation.jsonl for CAD iteration counts and token usage."""
    log_path = project_dir / "conversation.jsonl"
    cad_runs = 0
    cad_failures = 0
    tool_counts: dict[str, int] = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    last_assistant_message = ""
    seen_tool_calls: set[str] = set()

    if not log_path.is_file():
        return {}

    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue

        entry_type = entry.get("type", "")
        data = entry.get("data")

        # System events (agent_usage has token data)
        if entry_type == "agent_usage" and isinstance(data, dict):
            total_prompt_tokens += int(data.get("prompt_tokens", 0) or 0)
            total_completion_tokens += int(data.get("completion_tokens", 0) or 0)
            continue

        if entry_type == "tool_status" and isinstance(data, dict):
            status = data.get("status")
            call_id = str(data.get("call_id", ""))
            if status not in {"completed", "error"} or (
                call_id and call_id in seen_tool_calls
            ):
                continue
            if call_id:
                seen_tool_calls.add(call_id)
            name = str(data.get("tool", ""))
            if name:
                tool_counts[name] = tool_counts.get(name, 0) + 1
            arguments = data.get("arguments")
            is_build = name == "cad_build_and_verify" or (
                name == "cad"
                and isinstance(arguments, dict)
                and arguments.get("operation") == "run"
            )
            if is_build:
                if status == "error":
                    cad_failures += 1
                else:
                    cad_runs += 1
            continue

        # Tool events from _log
        content = entry.get("content")
        if isinstance(content, dict) and content.get("name") in {
            "cad",
            "cad_build_and_verify",
        }:
            call_id = str(content.get("call_id", ""))
            if call_id and call_id in seen_tool_calls:
                continue
            if content.get("operation") not in {None, "run"}:
                continue
            result = str(content.get("result", ""))
            if is_failure(result):
                cad_failures += 1
                tool_counts["cad_build_and_verify"] = (
                    tool_counts.get("cad_build_and_verify", 0) + 1
                )
            else:
                cad_runs += 1
                tool_counts["cad_build_and_verify"] = (
                    tool_counts.get("cad_build_and_verify", 0) + 1
                )
            continue

        # Tool content is a dict with name/result
        if isinstance(content, dict):
            call_id = str(content.get("call_id", ""))
            if call_id and call_id in seen_tool_calls:
                continue
            name = content.get("name", "")
            if name and name != "cad":
                tool_counts[name] = tool_counts.get(name, 0) + 1
            continue

        # Assistant messages for fallback summary
        if (
            entry.get("role") == "assistant"
            and isinstance(content, str)
            and content.strip()
        ):
            last_assistant_message = content.strip()

    total_tokens = total_prompt_tokens + total_completion_tokens
    journey: dict[str, int | str] = {}
    if cad_runs or cad_failures:
        journey["cad_runs"] = cad_runs
        journey["cad_failures"] = cad_failures
    if tool_counts:
        tool_parts = [f"{name}({count})" for name, count in sorted(tool_counts.items())]
        journey["tools"] = ", ".join(tool_parts)
    if total_tokens > 0:
        journey["total_tokens"] = total_tokens
    journey["_last_assistant"] = last_assistant_message
    return journey


def _build_report(
    summary: str,
    metrics: dict[str, Any],
    limitations: str | None,
    journey: dict[str, int | str],
) -> str:
    dimensions = metrics["dimensions_mm"]
    sections: list[str] = [
        "# CAD Finalization Report",
        "",
        "## Design summary",
        summary.strip() or "No project summary was provided.",
        "",
        "## Verified geometry",
        f"- Solid count: {metrics['solid_count']}",
        f"- Valid: {metrics['is_valid']}",
        f"- Volume: {metrics['volume_mm3']} mm³",
        f"- Bounding box: {dimensions['x']} × {dimensions['y']} × {dimensions['z']} mm",
    ]

    cad_runs = journey.get("cad_runs", 0)
    cad_failures = journey.get("cad_failures", 0)
    if cad_runs or cad_failures or journey.get("total_tokens"):
        journey_lines: list[str] = ["", "## Design journey"]
        if cad_runs or cad_failures:
            parts = []
            if cad_runs:
                parts.append(f"{cad_runs} succeeded")
            if cad_failures:
                parts.append(f"{cad_failures} failed")
            journey_lines.append(f"- CAD iterations: {', '.join(parts)}")
        tools_str = journey.get("tools")
        if tools_str:
            journey_lines.append(f"- Tools used: {tools_str}")
        total_tokens = journey.get("total_tokens")
        if total_tokens:
            journey_lines.append(f"- Total tokens: {total_tokens:,}")
        sections.extend(journey_lines)

    sections.extend(
        [
            "",
            "## Generated files",
            "- `model.step`",
            "- `model.stl`",
            "- `report.md`",
        ]
    )

    if limitations:
        sections.extend(
            [
                "",
                "## Known limitations",
                limitations,
            ]
        )

    return "\n".join(sections) + "\n"


def finalize_project(project_dir: Path) -> dict[str, Any]:
    cad = CadTool(project_dir)

    with tempfile.TemporaryDirectory(prefix=".finalize-", dir=project_dir) as temporary:
        staging_root = Path(temporary)
        relative_output = staging_root.relative_to(project_dir) / "output"
        result = cad.finalize(relative_output.as_posix())
        metrics = result["metrics"]

        _validate_metrics(metrics)

        render_path = project_dir / "render.png"
        if not render_path.is_file() or render_path.stat().st_size == 0:
            raise RuntimeError("CAD render was not generated.")

        journey = _parse_journey(project_dir)

        summary_path = project_dir / "summary.md"
        summary = (
            summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
        )
        if PLACEHOLDER_MARKER in summary:
            fallback = journey.pop("_last_assistant", "")
            if fallback:
                summary = f"_(Auto-extracted from agent's final message)_\n\n{fallback}"
            else:
                summary = "No project summary was provided (placeholder detected)."

        limitations = _extract_limitations(summary)
        report = _build_report(summary, metrics, limitations, journey)

        staged_output = project_dir / relative_output
        (staged_output / "report.md").write_text(report, encoding="utf-8")
        # Record the source digest so stale exports can be detected later.
        model_path = project_dir / "model.py"
        model_digest = (
            hashlib.sha256(model_path.read_bytes()).hexdigest()
            if model_path.is_file()
            else ""
        )
        (staged_output / ".finalize_meta.json").write_text(
            json.dumps({"model_sha256": model_digest}, ensure_ascii=False),
            encoding="utf-8",
        )
        expected = [
            staged_output / name for name in ("model.step", "model.stl", "report.md")
        ]
        if any(not path.is_file() or path.stat().st_size == 0 for path in expected):
            raise RuntimeError(
                "Finalization did not produce every required output artifact."
            )

        output_dir = project_dir / "output"
        backup_dir = staging_root / "previous-output"
        had_previous_output = output_dir.exists()
        try:
            if had_previous_output:
                output_dir.replace(backup_dir)
            staged_output.replace(output_dir)
        except OSError:
            if not output_dir.exists() and backup_dir.exists():
                backup_dir.replace(output_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

    return {
        "metrics": metrics,
        "exports": {
            "metrics": metrics,
            "step": "output/model.step",
            "stl": "output/model.stl",
        },
        "report": "output/report.md",
        "report_text": report,
    }
