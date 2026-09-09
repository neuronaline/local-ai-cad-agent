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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.cad_review import (
    review_cad,
    write_review_result,
)
from agent.review_paths import review_dir as review_path_for
from agent.tools.cad_screenshot_tool import CadScreenshotTool
from agent.tools.process_runner import MAX_SANDBOX_TIMEOUT_SECONDS
from agent.tools.tool_events import publish_tool_phase

_LOG = logging.getLogger(__name__)


@dataclass
class Evidence:
    """Verified visual + manifest evidence for the current ``model.py``.

    All four fields are required and mutually consistent: the manifest
    must point at ``sheet_path`` and ``render_path`` and the
    ``preview_sha`` is the digest the manifest declared for the
    accompanying ``preview.stl``. Callers that fail to find complete
    evidence should return ``None`` from :meth:`_resolve_evidence`
    instead of constructing a partial :class:`Evidence`.
    """

    manifest: dict[str, Any]
    sheet_path: Path
    render_path: Path
    preview_sha: str


class CadReviewTool:
    """Compose deterministic + visual findings for the active build."""

    # Sandbox timeout cap matches the schema's ``maximum`` ceiling.
    _MAX_TIMEOUT = MAX_SANDBOX_TIMEOUT_SECONDS

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
        """Return a copy of this tool bound to a specific tool-call ID.

        Mirrors :meth:`FileTool.with_call_id` so concurrent tool calls
        cannot clobber each other's IDs. The clone shares ``project_dir``,
        ``_publish``, ``_settings`` and ``_stop_event``; the new
        ``_call_id`` is set on the fresh instance and the original is
        left untouched.
        """
        clone = CadReviewTool(
            self.project_dir,
            publish=self._publish,
            settings=self._settings,
            stop_event=self._stop_event,
        )
        clone._call_id = call_id
        return clone

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
        evidence = self._resolve_evidence()
        if evidence is None:
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
            evidence = self._resolve_evidence()
            if evidence is None:
                raise RuntimeError(
                    "cad_review could not locate visual evidence after auto-screenshot."
                )
        review_manifest = evidence.manifest
        sheet_path = evidence.sheet_path
        render_path = evidence.render_path
        preview_sha = evidence.preview_sha
        request_text = self._latest_user_request()
        model_source = self._load_model_source()
        # Filter the visual manifest to the agent's narrowed subset when the
        # agent asked for one; the deterministic layer keeps checking the full
        # model so narrowed views do not hide a real defect.
        if view_filter:
            existing_views = review_manifest.get("views") or []
            narrowed_views = [
                entry
                for entry in existing_views
                if isinstance(entry, dict) and entry.get("view_id") in view_filter
            ]
            # An empty narrowed set means the requested view ids were not
            # present in the manifest. Silently widening the scan to all
            # views (the previous behaviour) inverted the caller's intent;
            # fail the call so the agent learns its filter matched nothing.
            if not narrowed_views:
                available = sorted(
                    {
                        entry.get("view_id")
                        for entry in existing_views
                        if isinstance(entry, dict)
                        and isinstance(entry.get("view_id"), str)
                    }
                )
                raise RuntimeError(
                    "cad_review view filter matched no views: "
                    f"requested={sorted(view_filter)!r}, available={available!r}."
                )
            review_manifest = dict(review_manifest)
            review_manifest["views"] = narrowed_views
            # Keep the prompt's tile ordering consistent with the narrowed
            # ``allowed_view_ids`` enforced by ``_coerce_finding``; otherwise
            # the reviewer can attribute findings to a tile that the visual
            # layer then silently strips via ``view_id = None``.
            contact_sheet = review_manifest.get("contact_sheet")
            if isinstance(contact_sheet, dict):
                contact_sheet = dict(contact_sheet)
                contact_sheet["view_order"] = [
                    entry.get("view_id")
                    for entry in narrowed_views
                    if isinstance(entry, dict)
                    and isinstance(entry.get("view_id"), str)
                ]
                review_manifest["contact_sheet"] = contact_sheet
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
        )
        # Persist the verdict next to the manifest so the UI status pill
        # can render without a separate API call.
        model_sha = (
            str(review_manifest.get("model_sha256") or "")
            if isinstance(review_manifest, dict)
            else ""
        )
        review_dir_path = review_path_for(self.project_dir, model_sha)
        if model_sha:
            try:
                write_review_result(review_dir_path, result)
            except OSError as error:
                _LOG.warning(
                    "cad_review could not persist result.json (%s): %s",
                    review_dir_path,
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
        publish_tool_phase(
            self._publish,
            project=self.project_dir.name,
            tool="cad_review",
            call_id=self._call_id,
            status=status,
            message=message,
        )

    def _load_inputs(self) -> tuple[Any, Any, list[dict[str, Any]] | None]:
        """Read deterministic evidence only when it matches ``model.py``.

        Validation results live inside the sha-gated ``.cad_metrics.json``
        payload (``metrics.validation_results``), so there is a single
        source of truth: a previous ``.cad_validation.json`` sidecar read
        here was unverified (the runner writes it inside the sandbox but
        the host never copies it out), so a stale on-disk file could have
        flowed attacker/model-controlled ``severity`` strings into
        findings. The runner-written sidecar is intentionally ignored.
        """
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
        raw_validation = metrics.get("validation_results")
        if isinstance(raw_validation, list):
            validation = [
                entry
                for entry in raw_validation
                if isinstance(entry, dict)
            ]
        return geometry_metrics, feature_summary, validation

    def _resolve_evidence(
        self,
    ) -> Evidence | None:
        """Locate evidence for the current model, never a prior revision.

        Returns ``None`` when no usable evidence exists yet (the caller
        then triggers the auto-screenshot fallback). A partial manifest —
        e.g. parseable but missing the contact sheet or single render —
        also returns ``None``; the auto-screenshot path repopulates the
        artifacts before ``Evidence`` is reconstructed.
        """
        model_path = self.project_dir / "model.py"
        try:
            model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        except OSError:
            return None
        review_dir_path = review_path_for(self.project_dir, model_sha)
        manifest_path = review_dir_path / "manifest.json"
        preview_sha = ""
        preview_stl_path = self.project_dir / "preview.stl"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = None
            if isinstance(manifest, dict) and manifest.get("model_sha256") == model_sha:
                preview_sha = str(manifest.get("preview_sha256") or "")
                # The manifest's preview_sha is the digest the runner declared
                # for the preview.stl it produced alongside this build. The
                # project-root preview.stl is overwritten by every
                # ``cad_build_and_verify(render=true)`` — different model
                # sha — so an on-disk mismatch means the manifest is stale
                # or the file was tampered with. Returning ``None`` here
                # routes the caller through the auto-screenshot fallback
                # instead of letting the reviewer visualise stale geometry.
                if (
                    preview_sha
                    and len(preview_sha) == 64
                    and preview_stl_path.is_file()
                ):
                    actual_preview_sha = hashlib.sha256(
                        preview_stl_path.read_bytes()
                    ).hexdigest()
                    if actual_preview_sha != preview_sha:
                        return None
                sheet_path = review_dir_path / "review-sheet.png"
                single_render = manifest.get("single_render")
                if isinstance(single_render, dict):
                    artifact_path = str(single_render.get("path") or "")
                    if artifact_path == "render.png":
                        # Legacy build path: the single render lives at the
                        # project root, not inside the review directory.
                        # No traversal check is needed here because the
                        # literal string is whitelisted above.
                        render_path = self.project_dir / artifact_path
                    else:
                        # Reject any ``single_render.path`` that escapes
                        # the review directory via ``..`` segments or
                        # symlinks. ``Path.resolve()`` follows symlinks,
                        # and the membership test below pins the artifact
                        # to this review's directory.
                        candidate = (review_dir_path / artifact_path).resolve()
                        if review_dir_path in candidate.parents:
                            render_path = candidate
                        else:
                            render_path = None
                    if (
                        isinstance(render_path, Path)
                        and sheet_path.is_file()
                        and render_path.is_file()
                    ):
                        return Evidence(
                            manifest=manifest,
                            sheet_path=sheet_path,
                            render_path=render_path,
                            preview_sha=preview_sha,
                        )
        return None

    @staticmethod
    def _verify_artifact_hashes(
        manifest: dict[str, Any], sheet_path: Path, render_path: Path
    ) -> None:
        """Reject a manifest whose hashes do not match the on-disk artifacts.

        Mirrors the strict contract used by ``_verify_image_artifact`` in the
        visual layer: an existing artifact whose manifest hash is missing or
        not a 64-char SHA-256 is treated as evidence corruption and rejected
        with ``RuntimeError``. The visual layer would otherwise downgrade the
        same situation to ``inconclusive``; failing here keeps the two
        pre-flight checks aligned so a single corruption class always
        produces a single failure mode.
        """
        contact = manifest.get("contact_sheet") or {}
        single = manifest.get("single_render") or {}
        if isinstance(contact, dict) and sheet_path.is_file():
            expected = contact.get("image_sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise RuntimeError(
                    "cad_review: contact sheet manifest has no valid hash."
                )
            actual = hashlib.sha256(sheet_path.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(
                    "cad_review: contact sheet hash does not match the manifest."
                )
        if isinstance(single, dict) and render_path.is_file():
            expected = single.get("image_sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise RuntimeError(
                    "cad_review: single render manifest has no valid hash."
                )
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
            # Skip agent-generated nudges (e.g. "Call cad_build_and_verify
            # now.") — they are persisted with ``role: user`` for the
            # in-context loop but are not user-authored design intent.
            if entry.get("synthetic") is True:
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
