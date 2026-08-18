"""Host-side orchestrator for the ``cad_review`` AI tool.

Composes a structured verdict for the active CAD build:

1. Confirm visual evidence exists (``review_manifest`` + contact sheet +
   single render); if not, internally call ``cad_screenshot`` to produce
   the canonical eight at ``standard`` quality (auto-screenshot).
2. Run ``agent.cad_review.review_cad`` which assembles the deterministic
   + visual findings and assembles the final verdict under the always-
   strict rule (blocking/major → fail; only minor → pass; no evidence →
   inconclusive).
3. Persist the verdict into ``<project>/.cad-agent/reviews/<sha>/result.json``
   and publish a ``review_updated`` event so the UI status pill updates.

The orchestrator never auto-triggers from ``cad_build_and_verify``. The
agent calls ``cad_review`` deliberately when it wants a verdict.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from agent.cad_review import (
    ALLOWED_SEVERITIES,
    review_cad,
    write_review_result,
)
from agent.tools.cad_screenshot_tool import CadScreenshotTool

_LOG = logging.getLogger(__name__)


class CadReviewTool:
    """Compose deterministic + visual findings for the active build."""

    # Sandbox timeout cap matches the schema's ``maximum: 120`` ceiling.
    _MAX_TIMEOUT = 120

    def __init__(
        self,
        project_dir: Path,
        publish: Any = None,
        settings: Any = None,
        stop_event: Any = None,
    ) -> None:
        """Wire the review orchestrator to the active project + publish bus.

        ``settings`` (optional) is forwarded to the visual layer's LLM
        factory; passing ``None`` keeps the visual layer skipped (useful for
        unit tests).
        """
        self.project_dir = project_dir.resolve()
        self._publish = publish
        self._settings = settings
        self._stop_event = stop_event
        self._call_id = ""
        # Local screenshot helper used for the auto-screenshot fallback.
        # Reuses the cache produced by ``cad_build_and_verify`` when possible.
        self._screenshot = CadScreenshotTool(project_dir, publish=publish)

    def with_call_id(self, call_id: str) -> CadReviewTool:
        self._call_id = call_id
        return self

    def stop(self) -> None:
        """Stop any in-flight screenshot subprocess (best-effort)."""
        try:
            self._screenshot.stop()
        except Exception:  # noqa: BLE001 - stop is best-effort.
            _LOG.warning("cad_review stop ignored an exception", exc_info=True)

    # ------------------------------------------------------------------ surface

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Public entry point invoked by ``AgentRunner._execute`` dispatch."""
        requested_views = CadScreenshotTool._normalize_views(arguments.get("views"))
        view_filter = requested_views if arguments.get("views") else ()
        timeout_seconds = min(
            max(int(arguments.get("timeout_seconds") or 60), 1),
            self._MAX_TIMEOUT,
        )
        metrics, feature_summary, validation_results = self._load_inputs()
        review_manifest, sheet_path, render_path, preview_sha = self._resolve_evidence()
        if (
            review_manifest is None
            or sheet_path is None
            or render_path is None
        ):
            self._emit_status(
                "screenshot_auto",
                "No visual evidence found; rendering canonical eight views before review.",
            )
            # Always include the full canonical set here. In particular, the
            # reviewer needs an isometric image as its standalone evidence;
            # a caller narrowing visual focus to an orthogonal view must not
            # make the fallback intrinsically unreviewable.
            try:
                self._screenshot.with_call_id(self._call_id).execute(
                    {
                        "views": list(CadScreenshotTool.SUBSET_VIEWS),
                        "quality": "standard",
                        "contact_sheet": True,
                        "timeout_seconds": timeout_seconds,
                    }
                )
            except Exception as error:
                raise RuntimeError(
                    f"cad_review auto-screenshot failed: {type(error).__name__}: {error}"
                ) from error
            review_manifest, sheet_path, render_path, preview_sha = self._resolve_evidence()
            if (
                review_manifest is None
                or sheet_path is None
                or render_path is None
            ):
                raise RuntimeError(
                    "cad_review could not locate visual evidence after auto-screenshot."
                )
        request_text = self._latest_user_request()
        model_source = self._load_model_source()
        # Filter the visual manifest to the agent's narrowed subset when the
        # agent asked for one; the deterministic layer keeps checking the full
        # model so narrowed views do not hide a real defect.
        if view_filter:
            narrowed_views = [
                entry
                for entry in review_manifest.get("views", []) or []
                if isinstance(entry, dict) and entry.get("view_id") in view_filter
            ]
            if narrowed_views:
                review_manifest = dict(review_manifest)
                review_manifest["views"] = narrowed_views
        # Pre-flight the manifest itself: the contact sheet + single render
        # must hash-match so the reviewer can't be tricked into visualising
        # a tampered file.
        self._verify_artifact_hashes(review_manifest, sheet_path, render_path)
        self._emit_status("reviewing", "Running deterministic + visual review…")
        # Direct LLM call (no subprocess) so the reviewer's stop_event is
        # honoured by the chat-completions cancel path.
        result = review_cad(
            settings=self._settings,
            request_text=request_text,
            model_source=model_source,
            metrics=metrics if isinstance(metrics, dict) else {},
            feature_summary=feature_summary if isinstance(feature_summary, dict) else {},
            review_manifest=review_manifest,
            sheet_path=sheet_path,
            single_render_path=render_path,
            validation_results=validation_results,
            stop_event=self._stop_event,
            stop_instruction=self._stop_instruction(),
        )
        # Persist the verdict next to the manifest so the UI status pill
        # can render without a separate API call.
        model_sha = (
            str(review_manifest.get("model_sha256") or "")
            if isinstance(review_manifest, dict)
            else ""
        )
        review_dir = self.project_dir / ".cad-agent" / "reviews" / model_sha
        if model_sha:
            try:
                write_review_result(review_dir, result)
            except OSError as error:
                _LOG.warning(
                    "cad_review could not persist result.json (%s): %s",
                    review_dir,
                    error,
                )
        # Update preview_sha in the payload (use the cached hash we already
        # computed) so the UI can match the verdict to the rendered state.
        result_payload = result.as_dict()
        if preview_sha:
            result_payload["preview_sha256"] = preview_sha
        # Per-layer breakdown so the UI can surface "deterministic X,
        # visual Y" details without re-parsing.
        deterministic = [
            finding.as_dict()
            for finding in result.findings
            if finding.source == "deterministic"
        ]
        visual_findings = [
            finding.as_dict()
            for finding in result.findings
            if finding.source == "visual"
        ]
        result_payload["deterministic"] = {
            "checks_run": _deterministic_checks_run(metrics, feature_summary),
            "findings": deterministic,
        }
        result_payload["visual_review"] = (
            {
                "invoked": True,
                "status": result.status,
                "findings_count": len(visual_findings),
            }
            if visual_findings or result.status != "inconclusive"
            else None
        )
        try:
            if callable(self._publish):
                self._publish(
                    "review_updated",
                    {
                        "project": self.project_dir.name,
                        "status": result.status,
                        "summary": result.summary,
                        "model_sha256": model_sha,
                        "preview_sha256": preview_sha,
                    },
                )
        except Exception:  # noqa: BLE001 - status events must never fail the tool.
            _LOG.warning(
                "cad_review status publish failed (project=%s): ignored",
                self.project_dir.name,
                exc_info=_LOG.isEnabledFor(logging.DEBUG),
            )
        return result_payload

    # ------------------------------------------------------------------ helpers

    def _emit_status(self, status: str, message: str) -> None:
        publish = self._publish
        if not callable(publish):
            return
        try:
            event: dict[str, Any] = {
                "project": self.project_dir.name,
                "status": status,
                "result": message,
            }
            if self._call_id:
                event.update({"call_id": self._call_id, "tool": "cad_review"})
                publish("tool_status", event)
            else:
                event["message"] = event.pop("result")
                publish("agent_status", event)
        except Exception:  # noqa: BLE001 - activity events must never fail the tool.
            _LOG.warning(
                "cad_review status publish failed (status=%s, project=%s): ignored",
                status,
                self.project_dir.name,
                exc_info=_LOG.isEnabledFor(logging.DEBUG),
            )

    def _load_inputs(self) -> tuple[Any, Any, list[dict[str, Any]] | None]:
        """Read deterministic evidence only when it matches ``model.py``."""
        model_path = self.project_dir / "model.py"
        try:
            model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        except OSError:
            return {}, {}, None
        metrics_path = self.project_dir / ".cad_metrics.json"
        metrics: Any = {}
        if metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metrics = {}
        if not isinstance(metrics, dict) or metrics.get("model_sha256") != model_sha:
            return {}, {}, None
        geometry_metrics = metrics.get("metrics")
        if not isinstance(geometry_metrics, dict):
            return {}, {}, None
        feature_summary = (
            metrics.get("feature_summary", {})
        )
        validation: list[dict[str, Any]] | None = None
        validation_path = self.project_dir / ".cad_validation.json"
        if validation_path.is_file():
            try:
                payload = json.loads(validation_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                    validation = [
                        entry
                        for entry in payload["results"]
                        if isinstance(entry, dict)
                    ]
            except (OSError, json.JSONDecodeError):
                validation = None
        return geometry_metrics, feature_summary, validation

    def _resolve_evidence(
        self,
    ) -> tuple[dict[str, Any] | None, Path | None, Path | None, str]:
        """Locate evidence for the current model, never a prior revision."""
        model_path = self.project_dir / "model.py"
        try:
            model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        except OSError:
            return None, None, None, ""
        review_dir = self.project_dir / ".cad-agent" / "reviews" / model_sha
        manifest_path = review_dir / "manifest.json"
        if not manifest_path.is_file():
            return None, None, None, ""
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None, None, None, ""
        if not isinstance(manifest, dict) or manifest.get("model_sha256") != model_sha:
            return None, None, None, ""
        sheet_path = review_dir / "review-sheet.png"
        single_render = manifest.get("single_render")
        if not isinstance(single_render, dict):
            return None, None, None, ""
        artifact_path = str(single_render.get("path") or "")
        if artifact_path == "render.png":
            render_path = self.project_dir / artifact_path
        else:
            render_path = (review_dir / artifact_path).resolve()
            if review_dir not in render_path.parents:
                return None, None, None, ""
        preview_sha = (
            str(manifest.get("preview_sha256") or "") if isinstance(manifest, dict) else ""
        )
        if not sheet_path.is_file() or not render_path.is_file():
            return None, None, None, preview_sha
        return manifest, sheet_path, render_path, preview_sha

    @staticmethod
    def _verify_artifact_hashes(
        manifest: dict[str, Any], sheet_path: Path, render_path: Path
    ) -> None:
        """Reject a manifest whose hashes do not match the on-disk artifacts."""
        contact = manifest.get("contact_sheet") or {}
        single = manifest.get("single_render") or {}
        if isinstance(contact, dict):
            expected = contact.get("image_sha256")
            if (
                isinstance(expected, str)
                and len(expected) == 64
                and sheet_path.is_file()
            ):
                actual = hashlib.sha256(sheet_path.read_bytes()).hexdigest()
                if actual != expected:
                    raise RuntimeError(
                        "cad_review: contact sheet hash does not match the manifest."
                    )
        if isinstance(single, dict):
            expected = single.get("image_sha256")
            if (
                isinstance(expected, str)
                and len(expected) == 64
                and render_path.is_file()
            ):
                actual = hashlib.sha256(render_path.read_bytes()).hexdigest()
                if actual != expected:
                    raise RuntimeError(
                        "cad_review: single render hash does not match the manifest."
                    )

    def _load_model_source(self) -> str:
        model_path = self.project_dir / "model.py"
        try:
            return model_path.read_text(encoding="utf-8") if model_path.is_file() else ""
        except OSError:
            return ""

    def _latest_user_request(self) -> str:
        history_path = self.project_dir / "conversation.jsonl"
        if not history_path.is_file():
            return ""
        try:
            lines = history_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return ""
        for raw in reversed(lines):
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict) or entry.get("role") != "user":
                continue
            content = entry.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                fragments: list[str] = []
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                    ):
                        fragments.append(part["text"])
                if fragments:
                    return "\n".join(fragments)
        return ""

    def _stop_instruction(self) -> str | None:
        return None


def _deterministic_checks_run(
    metrics: Any, feature_summary: Any
) -> list[str]:
    """Best-effort enumeration of which deterministic checks the layer ran.

    Used by the UI to render a short checklist next to the verdict. We
    keep the list short — the actual checks live in
    ``agent.cad_review._deterministic_review`` and may grow over time.
    """
    checks: list[str] = []
    if isinstance(metrics, dict):
        checks.append("solid_count")
        checks.append("is_valid")
        checks.append("dimensions")
        checks.append("volume")
    if isinstance(feature_summary, dict) and "through_hole_count" in feature_summary:
        checks.append("through_holes")
    # The orchestrator already loaded validation_results; surface the
    # spec line so the UI can label the result without knowing the
    # internals.
    checks.append("spec.requirements")
    return checks


def _coerce_finding_dict(raw: Any) -> dict[str, Any] | None:
    """Best-effort coercion of a single finding dict (defensive)."""
    if not isinstance(raw, dict):
        return None
    severity = raw.get("severity")
    if severity not in ALLOWED_SEVERITIES:
        return None
    category = raw.get("category")
    message = raw.get("message")
    if not isinstance(message, str) or not message.strip():
        return None
    out: dict[str, Any] = {
        "severity": severity,
        "category": category if isinstance(category, str) else "geometry",
        "message": message[:240],
        "source": raw.get("source") or "visual",
    }
    view = raw.get("view")
    if isinstance(view, str) and view:
        out["view"] = view
    hint = raw.get("repair_hint")
    if isinstance(hint, str) and hint:
        out["repair_hint"] = hint[:240]
    return out
