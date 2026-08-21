"""Build and render the project's CAD model inside the bubblewrap sandbox."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from agent.io import atomic_write_bytes
from agent.revisions import RevisionIntegrityError, RevisionStore
from agent.sandbox import command as sandbox_command
from agent.tools.file_tool import FileTool
from agent.tools.process_runner import _stream_with_limit, _TimedOut
from agent.tools.tool_events import publish_tool_phase

# Sandbox-side script files. The host copies these into the bubblewrap
# workspace so the runner can ``import renderer`` instead of relying on a
# source-string concatenation.
_SCRIPTS_DIR = Path(__file__).resolve().parent / "cad_scripts"
_RUNNER_FILENAME = "runner.py"
_RENDERER_FILENAME = "renderer.py"


# Hard validation bounds shared with the runner; enforced here so callers see
# a useful error before the sandbox subprocess spins up.
_MIN_DIMENSION_MM = 0.001
_MAX_DIMENSION_MM = 1_000_000.0
_MIN_VOLUME_MM3 = 0.0

# Review artifacts live under <project>/.cad-agent/reviews/<model_sha256>/.
# A new manifest only invalidates the previous review when ``preview_sha256``
# changes; same (model_sha256, preview_sha256) hits the on-disk cache.
_REVIEW_MANIFEST_NAME = "manifest.json"
_REVIEW_VIEWS_DIR = "views"
_REVIEW_SHEET_NAME = "review-sheet.png"


def _kill_process_group(process: subprocess.Popen[str], *, force: bool) -> None:
    """Send a signal to the sandbox process group so descendant pythons exit.

    The bubblewrap subprocess is launched with ``start_new_session=True`` and
    in turn spawns a python interpreter inside its namespace. ``Popen.terminate``
    only signals the direct child (bwrap); without killing the process group
    the inner python keeps running and holding memory/file handles.
    """
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass


def _project_name(project_dir: Path) -> str:
    """Return the workspace-relative project name used by SSE payloads."""
    try:
        return project_dir.resolve().name
    except OSError:
        return project_dir.name


class CadTool:
    def __init__(
        self,
        project_dir: Path,
        publish: Any = None,
        revisions: RevisionStore | None = None,
        review_render_workers: int = 4,
        review_required_views: int = 8,
        review_enabled: bool = True,
    ) -> None:
        self.project_dir = project_dir.resolve()
        self._publish = publish
        self._revisions = revisions or RevisionStore(project_dir)
        self._review_render_workers = max(1, int(review_render_workers))
        self._review_required_views = max(1, int(review_required_views))
        self._review_enabled = bool(review_enabled)
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._call_id = ""

    # ------------------------------------------------------------------ review

    def with_call_id(self, call_id: str) -> CadTool:
        self._call_id = call_id
        return self

    def _review_dir(self, model_sha256: str) -> Path:
        return self.project_dir / ".cad-agent" / "reviews" / model_sha256

    def _publish_status(
        self, status: str, message: str, call_id: str = ""
    ) -> None:
        """Emit a phase update for the active build tool.

        Used to surface the multi-view rendering phase so the UI activity
        drawer can label the step (``rendering_views``). The publish callback
        is optional so tests that instantiate ``CadTool`` without a publish
        function keep working. Implementation is delegated to the shared
        :func:`agent.tools.tool_events.publish_tool_phase` helper.
        """
        publish_tool_phase(
            self._publish,
            project=_project_name(self.project_dir),
            tool="cad_build_and_verify",
            call_id=call_id or self._call_id,
            status=status,
            message=message,
        )

    # ------------------------------------------------------------------ build

    def _runner_settings(
        self, render: bool, parameter_checks: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """JSON-kwarg payload forwarded to ``runner.main`` as ``argv[1]``.

        Replaces the legacy module-level globals (``_RENDER_VIEWS``,
        ``_WRITE_ISOMETRIC``, ``_RENDER_WORKERS``, ``_REQUIRED_VIEWS``) so
        ``runner.py`` uses a normal Python function call rather than reading
        injected globals.
        """
        if render:
            return {
                "render_views": self._review_enabled,
                "write_isometric": True,
                "render_workers": self._review_render_workers,
                "required_views": self._review_required_views,
                "parameter_checks": parameter_checks or [],
            }
        return {
            "render_views": False,
            "write_isometric": False,
            "render_workers": self._review_render_workers,
            "required_views": self._review_required_views,
            "parameter_checks": parameter_checks or [],
        }

    def _execute(
        self,
        render: bool = False,
        call_id: str = "",
        parameter_checks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        model_path = self.project_dir / "model.py"
        if not model_path.exists():
            raise ValueError("model.py does not exist yet.")
        model_code = model_path.read_text(encoding="utf-8")
        FileTool.validate_model(model_code)
        # Copy ``renderer.py`` and ``runner.py`` as siblings into the workspace
        # so the runner can ``import renderer`` at runtime. Pass the render
        # settings as a JSON kwargs payload on ``argv[1]`` instead of mutating
        # module-level globals; this restores a real module boundary between
        # the host and the sandbox.
        settings_payload = json.dumps(
            self._runner_settings(render, parameter_checks)
        )

        with tempfile.TemporaryDirectory(prefix="cad-agent-") as temporary:
            workspace = Path(temporary)
            (workspace / "model.py").write_text(model_code, encoding="utf-8")
            (workspace / _RUNNER_FILENAME).write_text(
                (_SCRIPTS_DIR / _RUNNER_FILENAME).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (workspace / _RENDERER_FILENAME).write_text(
                (_SCRIPTS_DIR / _RENDERER_FILENAME).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            command, seccomp_fd = sandbox_command(
                workspace,
                [_RUNNER_FILENAME, settings_payload],
                writable=True,
                timeout_seconds=120,
            )
            process: subprocess.Popen[str] | None = None
            try:
                with self._lock:
                    self._process = subprocess.Popen(
                        command,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        pass_fds=(seccomp_fd,),
                        start_new_session=True,
                    )
                    process = self._process
                try:
                    stdout, stderr = _stream_with_limit(process, timeout=120)
                except _TimedOut as error:
                    self._record_build_failure(
                        "CAD operation timed out after 120 seconds."
                    )
                    raise RuntimeError(
                        "CAD operation timed out after 120 seconds."
                    ) from error
                except RuntimeError as error:
                    detail = f"CAD subprocess failed: {error}"
                    self._record_build_failure(detail)
                    raise RuntimeError(detail) from error
                if process.returncode:
                    detail = self._failure_detail(stderr or stdout)
                    self._record_build_failure(detail)
                    raise RuntimeError(f"CAD execution failed:\n{detail}")
            finally:
                os.close(seccomp_fd)
                with self._lock:
                    self._process = None
                # If ``_stream_with_limit`` raised (timeout, memory-limit
                # overflow, or any other exception), ``process`` is still
                # alive and would otherwise be leaked: the outer reference
                # on ``self._process`` has already been cleared, so
                # ``stop()`` becomes a no-op for this child. Force-kill the
                # whole process group so the bubblewrap namespace and any
                # spawned python interpreter exit cleanly.
                if process is not None:
                    _kill_process_group(process, force=True)
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass

            metrics_path = workspace / ".cad_metrics.json"
            preview_path = workspace / "preview.stl"
            if not metrics_path.is_file() or not preview_path.is_file():
                error_msg = (
                    "CAD execution did not produce preview and geometry metrics."
                )
                self._record_build_failure(error_msg)
                raise RuntimeError(error_msg)
            cached = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not isinstance(cached, dict) or not isinstance(
                cached.get("metrics"), dict
            ):
                error_msg = "CAD geometry metrics are malformed."
                self._record_build_failure(error_msg)
                raise TypeError(error_msg)
            metrics = cached["metrics"]
            self._enforce_basic_geometry(metrics)
            self._atomic_copy(preview_path, self.project_dir / "preview.stl")
            review_manifest = (
                cached.get("review_manifest")
                if render and self._review_enabled
                else None
            )
            if render and self._review_enabled:
                render_path = workspace / "render.png"
                if not render_path.is_file() or render_path.stat().st_size == 0:
                    error_msg = "CAD execution did not produce a render."
                    self._record_build_failure(error_msg)
                    raise RuntimeError(error_msg)
                self._atomic_copy(render_path, self.project_dir / "render.png")
            self._atomic_copy(metrics_path, self.project_dir / ".cad_metrics.json")
            self._record_build_success(metrics)
            # The temp ``workspace`` is cleaned up when this ``with`` block exits,
            # so copy the multi-view review artifacts into a stable staging
            # location under ``project_dir`` before returning. ``promote_review``
            # reads from ``staging_views_dir``/``staging_sheet_path`` and
            # verifies each view's SHA against the manifest entry.
            staging_views_dir: str | None = None
            staging_sheet_path: str | None = None
            if render:
                # Surface the multi-view rasterisation phase so the UI can
                # label the activity drawer ("rendering_views"). The event is
                # only published when render=True because the cheap
                # ``CadTool.run`` path skips view rendering entirely.
                self._publish_status(
                    "rendering_views",
                    "Rendering review views…",
                    call_id,
                )
                staging_views_dir, staging_sheet_path = self._stage_review_artifacts(
                    workspace
                )
            return {
                "metrics": metrics,
                "feature_summary": cached.get("feature_summary") or {},
                "review_manifest": review_manifest,
                "review_views_dir": staging_views_dir,
                "review_sheet_path": staging_sheet_path,
            }

    def _stage_review_artifacts(
        self, workspace: Path
    ) -> tuple[str, str]:
        """Copy multi-view review artifacts out of the bubblewrap workspace.

        Returns the project-relative paths to the staged ``views/`` directory
        and the contact sheet PNG. Both paths live under
        ``project_dir/.review-staging/`` and survive the temp workspace
        teardown so ``promote_review`` can verify and promote them. The
        staging directory is recreated atomically each run; stale views from
        previous renders do not bleed into the new manifest.
        """
        staging_root = self.project_dir / ".review-staging"
        source_views = workspace / ".review-views"
        source_sheet = workspace / ".review-sheet.png"
        if not source_views.is_dir():
            raise RuntimeError("Review views directory was not produced by the sandbox.")
        if not source_sheet.is_file() or source_sheet.stat().st_size == 0:
            raise RuntimeError("Review contact sheet was not produced by the sandbox.")
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        staging_root.mkdir(parents=True, exist_ok=True)
        views_target = staging_root / "views"
        shutil.copytree(source_views, views_target)
        sheet_target = staging_root / "review-sheet.png"
        shutil.copyfile(source_sheet, sheet_target)
        return str(views_target), str(sheet_target)

    def _enforce_basic_geometry(self, metrics: dict[str, Any]) -> None:
        """Apply the basic geometry checks the runner pre-validates.

        Keeps the simple "valid geometry, at least one solid, positive volume,
        plausible bounding box" contract even when the quality verifiers are
        no longer in the loop.

        Flat parts (one dimension effectively zero — sheet metal, gaskets,
        planar faces) are explicitly out of scope: the multi-view rasteriser
        and STL preview are meaningless for 2D-projection geometry, so the
        ``math.isclose`` tolerance is anchored to ``_MIN_DIMENSION_MM``.
        """
        try:
            solid_count = int(metrics.get("solid_count", 0) or 0)
            is_valid = bool(metrics.get("is_valid"))
            dimensions = metrics.get("dimensions_mm") or {}
            volume = float(metrics.get("volume_mm3", 0.0) or 0.0)
        except (TypeError, ValueError):
            raise ValueError("CAD geometry metrics are malformed.") from None
        if solid_count < 1:
            raise ValueError("Build did not produce a solid.")
        if not is_valid:
            raise ValueError("CAD geometry is not valid.")
        try:
            dim_values = [float(dimensions[axis]) for axis in ("x", "y", "z")]
        except (KeyError, TypeError, ValueError):
            raise ValueError("CAD dimensions are missing or malformed.") from None
        for axis, value in zip(("x", "y", "z"), dim_values):
            # Use ``math.isclose`` so a tiny but non-zero thickness is accepted
            # while still rejecting genuine flat / negative parts. The
            # tolerance is the same as ``_MIN_DIMENSION_MM`` so the runner
            # (which uses the same constant) stays the source of truth.
            if math.isclose(value, 0.0, abs_tol=_MIN_DIMENSION_MM):
                raise ValueError(
                    f"CAD dimension {axis}={value} mm is effectively zero; "
                    "flat / 2D-projection parts are not supported."
                )
            if value < _MIN_DIMENSION_MM or value > _MAX_DIMENSION_MM:
                raise ValueError(
                    f"CAD dimension {axis}={value} mm is outside the plausible range."
                )
        if not (volume > _MIN_VOLUME_MM3):
            raise ValueError("CAD volume must be positive.")

    def _record_build_success(self, metrics: dict[str, Any]) -> None:
        """Record a successful build against the active revision."""
        try:
            head = self._revisions.head()
            if head is None:
                return
            self._revisions.record_build_success(
                head.id, metrics, self.project_dir / "preview.stl"
            )
        except (OSError, RevisionIntegrityError):
            pass  # Best-effort; don't block CAD results on revision issues.

    def _record_build_failure(self, error: str) -> None:
        """Record a failed build against the active revision."""
        try:
            head = self._revisions.head()
            if head is None:
                return
            self._revisions.record_build_failure(head.id, error)
        except (OSError, RevisionIntegrityError):
            pass  # Best-effort; don't block CAD errors on revision issues.

    @staticmethod
    def _atomic_copy(source: Path, target: Path) -> None:
        # Read the source into memory and delegate to the canonical byte
        # helper so the temp-cleanup + replace semantics match every other
        # atomic write in the agent (audit_272).
        atomic_write_bytes(target, source.read_bytes())

    # ------------------------------------------------------------------ review promotion

    def promote_review(
        self,
        review_manifest: dict[str, Any],
        *,
        sandbox_views_dir: str,
        sandbox_sheet_path: str,
    ) -> dict[str, Any]:
        """Copy the multi-view review artifacts from the sandbox into the project.

        Returns the persisted manifest payload so the caller can publish the
        review event without re-reading ``.cad-agent/reviews/.../manifest.json``.
        Raises ``RuntimeError`` if any required view is missing or its hash
        does not match the manifest entry; partial output is treated as a
        build failure by the calling agent.
        """
        if not isinstance(review_manifest, dict):
            raise TypeError("Review manifest is missing or malformed.")
        model_sha256 = review_manifest.get("model_sha256")
        preview_sha256 = review_manifest.get("preview_sha256")
        if not (isinstance(model_sha256, str) and isinstance(preview_sha256, str)):
            raise TypeError("Review manifest is missing model/preview hashes.")
        views = review_manifest.get("views")
        contact_sheet = review_manifest.get("contact_sheet")
        single_render = review_manifest.get("single_render")
        if (
            not isinstance(views, list)
            or not isinstance(contact_sheet, dict)
            or not isinstance(single_render, dict)
        ):
            raise TypeError("Review manifest is missing visual evidence metadata.")
        single_render_sha = single_render.get("image_sha256")
        if not isinstance(single_render_sha, str) or len(single_render_sha) != 64:
            raise RuntimeError("Review manifest has no valid single render hash.")

        views_dir = Path(sandbox_views_dir)
        sheet_path = Path(sandbox_sheet_path)
        if not views_dir.is_dir():
            raise RuntimeError("Review views directory was not produced by the sandbox.")
        if not sheet_path.is_file() or sheet_path.stat().st_size == 0:
            raise RuntimeError("Review contact sheet was not produced by the sandbox.")

        review_dir = self._review_dir(model_sha256)
        cache_hit = self._review_is_fresh(
            review_dir, preview_sha256, contact_sheet.get("image_sha256")
        )
        if cache_hit:
            return self._read_review_manifest(review_dir) or review_manifest

        staging = review_dir.with_suffix(review_dir.suffix + ".tmp")
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        views_target = staging / _REVIEW_VIEWS_DIR
        views_target.mkdir(exist_ok=True)
        try:
            for entry in views:
                view_id = entry.get("view_id")
                expected_sha = entry.get("image_sha256")
                expected_size = int(entry.get("image_bytes") or 0)
                if not (isinstance(view_id, str) and isinstance(expected_sha, str)):
                    raise TypeError("Review view entry is missing identifiers.")
                source = views_dir / f"{view_id}.png"
                if not source.is_file() or source.stat().st_size == 0:
                    raise RuntimeError(
                        f"Review view {view_id} was not produced by the sandbox."
                    )
                destination = views_target / f"{view_id}.png"
                shutil.copyfile(source, destination)
                if destination.stat().st_size != expected_size:
                    raise RuntimeError(
                        f"Review view {view_id} size mismatch."
                    )
                actual_sha = hashlib.sha256(destination.read_bytes()).hexdigest()
                if actual_sha != expected_sha:
                    raise RuntimeError(
                        f"Review view {view_id} hash mismatch: "
                        f"expected {expected_sha[:12]}, got {actual_sha[:12]}."
                    )
            sheet_target = staging / _REVIEW_SHEET_NAME
            shutil.copyfile(sheet_path, sheet_target)
            actual_sheet_sha = hashlib.sha256(sheet_target.read_bytes()).hexdigest()
            expected_sheet_sha = contact_sheet.get("image_sha256")
            if (
                expected_sheet_sha
                and actual_sheet_sha != expected_sheet_sha
            ):
                raise RuntimeError("Review contact sheet hash mismatch.")

            persisted_manifest = dict(review_manifest)
            persisted_manifest["artifact_dir"] = review_dir.name
            persisted_manifest["preview_sha256"] = preview_sha256
            (staging / _REVIEW_MANIFEST_NAME).write_text(
                json.dumps(persisted_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            backup = review_dir.with_suffix(review_dir.suffix + ".previous")
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            if review_dir.exists():
                os.replace(review_dir, backup)
            try:
                os.replace(staging, review_dir)
            except OSError:
                if backup.exists() and not review_dir.exists():
                    os.replace(backup, review_dir)
                raise
            shutil.rmtree(backup, ignore_errors=True)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        return self._read_review_manifest(review_dir) or persisted_manifest

    @staticmethod
    def _read_review_manifest(review_dir: Path) -> dict[str, Any] | None:
        manifest_path = review_dir / _REVIEW_MANIFEST_NAME
        if not manifest_path.is_file():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    @classmethod
    def _review_is_fresh(
        cls,
        review_dir: Path,
        preview_sha256: str,
        sheet_sha: object,
    ) -> bool:
        if not review_dir.is_dir():
            return False
        manifest = cls._read_review_manifest(review_dir)
        if manifest is None:
            return False
        if manifest.get("preview_sha256") != preview_sha256:
            return False
        if sheet_sha and manifest.get("contact_sheet", {}).get("image_sha256") != sheet_sha:
            return False
        single_render = manifest.get("single_render")
        if not isinstance(single_render, dict):
            return False
        single_render_sha = single_render.get("image_sha256")
        if not isinstance(single_render_sha, str) or len(single_render_sha) != 64:
            return False
        views_target = review_dir / _REVIEW_VIEWS_DIR
        for entry in manifest.get("views", []) or []:
            view_id = entry.get("view_id")
            expected_sha = entry.get("image_sha256")
            if not (isinstance(view_id, str) and isinstance(expected_sha, str)):
                return False
            destination = views_target / f"{view_id}.png"
            if not destination.is_file():
                return False
            actual_sha = hashlib.sha256(destination.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                return False
        return True

    @staticmethod
    def _failure_detail(output: str) -> str:
        """Extract the most relevant traceback frames and the final error message."""
        lines = [line.rstrip() for line in output.strip().splitlines()]
        frames: list[str] = []
        in_traceback = False
        for i, line in enumerate(lines):
            if line.startswith("Traceback "):
                in_traceback = True
                continue
            if in_traceback:
                if line.startswith("  File "):
                    if "runner.py" not in line and ".cad_runner" not in line:
                        frames.append(line)
                        if i + 1 < len(lines) and lines[i + 1].startswith("    "):
                            frames.append(lines[i + 1])
                elif not line.startswith("  ") and line.strip():
                    frames.append(line)
                    break
        if not frames:
            for i, line in enumerate(lines):
                if 'File "model.py"' in line:
                    frames.append(line)
                    if i + 1 < len(lines):
                        frames.append(lines[i + 1])
            final = next(
                (line for line in reversed(lines) if line.strip()),
                "Unknown CAD error.",
            )
            frames.append(final)
        return "\n".join(dict.fromkeys(frames))[-2000:]

    def run(self) -> dict[str, Any]:
        """Build once and return the geometry metrics."""
        payload = self._execute()
        return dict(payload.get("metrics") or {})

    @staticmethod
    def _summarize_payload(
        metrics: dict[str, Any] | None,
        feature_summary: dict[str, Any],
        render: bool,
        review_path: str | None,
    ) -> str:
        """Return a one-line human-readable summary for the agent's first glance.

        The full structured payload (metrics, feature_summary, review_manifest)
        is still included below — the agent only needs this summary to decide
        whether to read further or move on, which keeps the first 1-2 turns
        after a build cheap.
        """
        if not isinstance(metrics, dict):
            return "Build produced no metrics."
        dims = metrics.get("dimensions_mm") or {}
        solid_count = metrics.get("solid_count", 0)
        is_valid = metrics.get("is_valid", False)
        volume_mm3 = metrics.get("volume_mm3")
        try:
            x = float(dims.get("x", 0))
            y = float(dims.get("y", 0))
            z = float(dims.get("z", 0))
        except (TypeError, ValueError):
            x = y = z = 0.0
        feature_count = sum(
            int(feature_summary.get(key, 0) or 0)
            for key in (
                "through_hole_count",
                "blind_hole_count",
                "fillet_count",
                "chamfer_count",
            )
        )
        validity = "valid" if is_valid else "INVALID"
        try:
            volume_cm3 = float(volume_mm3 or 0.0) / 1000.0
            volume_text = f"{volume_cm3:.1f} cm³"
        except (TypeError, ValueError):
            volume_text = "unknown volume"
        render_state = (
            "with render" if render and review_path else ("metrics-only" if not render else "no review")
        )
        return (
            f"Solid {solid_count} ({validity}); "
            f"bbox {x:.1f}×{y:.1f}×{z:.1f} mm; "
            f"{volume_text}; {feature_count} features; {render_state}."
        )

    def build_and_verify(
        self,
        render: bool = False,
        parameter_checks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build once, validate metrics, and (optionally) produce preview + review.

        ``render=False`` (default) skips the legacy ``render.png`` artifact and
        the multi-view review sheet, returning only ``metrics`` + ``preview.stl``
        + ``model_sha256`` + ``preview_sha256``. This is the fast, cache-friendly
        path for early iteration — the agent should switch to ``render=True``
        only for the final verification before declaring the task ready.

        The metrics still include the canonical geometry sanity checks
        (bounding box, volume, solid count, is_valid) so a render-less build
        remains trustworthy for early iteration.
        """
        execute_args: dict[str, Any] = {"render": bool(render)}
        if parameter_checks:
            execute_args["parameter_checks"] = parameter_checks
        if self._call_id:
            execute_args["call_id"] = self._call_id
        payload = self._execute(**execute_args)
        metrics = payload.get("metrics") if isinstance(payload, dict) else None
        # Surface the model + preview SHAs once at the top level instead of
        # nesting them inside ``review_manifest``. The structured reviewer has
        # its own tool (``cad_review``); the agent does not need to re-parse a
        # full manifest just to correlate a build with its result.
        model_sha = (
            payload.get("model_sha256") if isinstance(payload, dict) else None
        )
        preview_sha = (
            payload.get("preview_sha256") if isinstance(payload, dict) else None
        )
        result: dict[str, Any] = {
            "metrics": metrics,
            "preview": "preview.stl",
            "render": "render.png" if render else None,
            "feature_summary": payload.get("feature_summary") or {},
            "validation_results": payload.get("validation_results") or [],
        }
        if model_sha:
            result["model_sha256"] = model_sha
        if preview_sha:
            result["preview_sha256"] = preview_sha
        if render:
            review_manifest = payload.get("review_manifest")
            review_path: str | None = None
            if review_manifest:
                promoted = self.promote_review(
                    review_manifest,
                    sandbox_views_dir=payload.get("review_views_dir") or "",
                    sandbox_sheet_path=payload.get("review_sheet_path") or "",
                )
                review_path = promoted.get("artifact_dir")
            if review_path:
                result["review"] = review_path
        result["summary"] = self._summarize_payload(
            metrics,
            result["feature_summary"],
            render,
            result.get("review"),
        )
        return result

    def stop(self) -> None:
        with self._lock:
            process = self._process
        if process is None:
            return
        _kill_process_group(process, force=False)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _kill_process_group(process, force=True)
