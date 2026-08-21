"""Host-side orchestrator for the ``cad_screenshot`` AI tool.

Re-rasterises a subset of the canonical eight views without re-running
build123d, sharing the artifact cache produced by ``CadTool.build_and_verify``.
The orchestrator reads ``model.py``, validates the AST, runs
``renderer.py + screenshot.py`` inside the bubblewrap sandbox, validates
the produced PNGs + manifest hashes, and promotes canonical review evidence
or an isolated screenshot variant into ``<project>/.cad-agent/``.

Cache strategy:

  cache_key = (model_sha256, canonical views, quality, contact_sheet)

- When ``cad_build_and_verify(render=true)`` has already populated the full
  canonical eight at ``standard`` quality, a matching ``cad_screenshot``
  call hits the cache without spawning the sandbox.
- Screenshot variants live outside the canonical review directory, so a
  subset or a different quality tier cannot destroy final-review evidence.
- A verified standard-quality review superset can seed a narrower screenshot
  variant without spawning the sandbox.

The orchestrator publishes ``preview_updated`` (with a ``source``
discriminator) and ``screenshot_updated`` events so the UI activity panel
can label the step and the history drawer can show a different icon than
the build-driven path.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from agent.io import utc_now_iso
from agent.sandbox import command as sandbox_command
from agent.tools.cad_scripts import screenshot as screenshot_script
from agent.tools.file_tool import FileTool
from agent.tools.process_runner import _stream_with_limit, _TimedOut
from agent.tools.tool_events import publish_tool_phase

_LOG = logging.getLogger(__name__)


def _kill_process_group(process: subprocess.Popen[str], *, force: bool) -> None:
    """Send a signal to the sandbox process group so descendant pythons exit."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass


class CadScreenshotTool:
    """Re-rasterise a subset of views from the latest model.py revision."""

    # All canonical view_ids the orchestrator accepts. Mirrors
    # ``screenshot.SCREENSHOT_VIEWS`` but kept as a class constant for tests
    # that want to validate the orchestrator's surface without importing the
    # sandbox-side module.
    SUBSET_VIEWS: tuple[str, ...] = screenshot_script.SUBSET_VIEWS
    QUALITY_TIERS: tuple[str, ...] = tuple(
        sorted(screenshot_script.QUALITY_TOLERANCES.keys())
    )

    # Sandbox timeout cap matches the schema's ``maximum: 120`` ceiling.
    _MAX_TIMEOUT = 120

    def __init__(
        self,
        project_dir: Path,
        publish: Any = None,
    ) -> None:
        """Wire the orchestrator to the active project and shared publish bus."""
        self.project_dir = project_dir.resolve()
        self._publish = publish
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._call_id = ""

    # ------------------------------------------------------------------ surface

    def with_call_id(self, call_id: str) -> CadScreenshotTool:
        self._call_id = call_id
        return self

    def stop(self) -> None:
        """Terminate any in-flight sandbox subprocess (used by AgentRunner)."""
        with self._lock:
            process = self._process
        if process is None:
            return
        _kill_process_group(process, force=False)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _kill_process_group(process, force=True)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _model_sha256(project_dir: Path) -> str | None:
        model_path = project_dir / "model.py"
        try:
            return (
                hashlib.sha256(model_path.read_bytes()).hexdigest()
                if model_path.is_file()
                else None
            )
        except OSError:
            return None

    def _review_dir(self, model_sha: str) -> Path:
        return self.project_dir / ".cad-agent" / "reviews" / model_sha

    def _cache_dir(
        self,
        model_sha: str,
        views: tuple[str, ...],
        quality: str,
        contact_sheet: bool,
    ) -> Path:
        """Return the exact variant directory without colliding with review data."""
        if quality == "standard" and views == self.SUBSET_VIEWS and contact_sheet:
            return self._review_dir(model_sha)
        key = json.dumps(
            {
                "views": [view for view in self.SUBSET_VIEWS if view in views],
                "quality": quality,
                "contact_sheet": contact_sheet,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        variant = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        return (
            self.project_dir
            / ".cad-agent"
            / "screenshots"
            / model_sha
            / quality
            / variant
        )

    def _preview_sha_for_model(self, model_sha: str) -> str:
        """Return the preview digest only when it was built from this model."""
        metrics_path = self.project_dir / ".cad_metrics.json"
        preview_path = self.project_dir / "preview.stl"
        if not preview_path.is_file() or not metrics_path.is_file():
            return ""
        try:
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        if (
            not isinstance(metrics_payload, dict)
            or metrics_payload.get("model_sha256") != model_sha
        ):
            return ""
        try:
            return hashlib.sha256(preview_path.read_bytes()).hexdigest()
        except OSError:
            return ""

    @staticmethod
    def _normalize_views(views: list[str] | None) -> tuple[str, ...]:
        """Resolve ``None``/missing to the canonical eight; validate ids."""
        canonical = screenshot_script.SUBSET_VIEWS
        if not views:
            return canonical
        requested = tuple(views)
        unknown = [v for v in requested if v not in canonical]
        if unknown:
            raise ValueError(
                f"Unknown view id(s): {unknown}. "
                f"Expected one of {list(canonical)}."
            )
        # De-duplicate while preserving user-provided order.
        seen: set[str] = set()
        ordered: tuple[str, ...] = tuple(
            v for v in requested if not (v in seen or seen.add(v))
        )
        return ordered

    @staticmethod
    def _normalize_quality(quality: str | None) -> str:
        if not quality:
            return "standard"
        if quality not in screenshot_script.QUALITY_TOLERANCES:
            raise ValueError(
                f"Unknown quality tier: {quality!r}. "
                f"Expected one of {list(screenshot_script.QUALITY_TOLERANCES)}."
            )
        return quality

    # ------------------------------------------------------------------ cache lookup

    def _cache_matches(
        self,
        review_dir: Path,
        model_sha: str,
        requested_views: tuple[str, ...],
        quality: str,
        contact_sheet: bool = True,
    ) -> bool:
        """True when the exact requested artifact set is valid on disk."""
        manifest_path = review_dir / "manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if manifest.get("model_sha256") != model_sha:
            return False
        manifest_quality = manifest.get("quality", "standard")
        if manifest_quality != quality:
            return False
        manifest_views = [
            entry.get("view_id")
            for entry in manifest.get("views") or []
            if isinstance(entry, dict) and isinstance(entry.get("view_id"), str)
        ]
        # A contact sheet represents one concrete selection of views. Serving
        # a larger/different set for a subset request would return a sheet
        # whose labels do not match the manifest advertised to the caller.
        cached_selection = manifest.get("requested_views", manifest_views)
        if (
            not isinstance(cached_selection, list)
            or any(not isinstance(view_id, str) for view_id in cached_selection)
            or set(cached_selection) != set(requested_views)
        ):
            return False
        # Verify each PNG actually exists with the manifest hash.
        for view_id in requested_views:
            view_path = review_dir / "views" / f"{view_id}.png"
            if not view_path.is_file():
                return False
            entry = next(
                (
                    e
                    for e in manifest.get("views", [])
                    if isinstance(e, dict) and e.get("view_id") == view_id
                ),
                None,
            )
            if not entry:
                return False
            expected_sha = entry.get("image_sha256")
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                return False
            try:
                actual_sha = hashlib.sha256(view_path.read_bytes()).hexdigest()
            except OSError:
                return False
            if actual_sha != expected_sha:
                return False
        if contact_sheet:
            sheet = manifest.get("contact_sheet")
            sheet_path = review_dir / "review-sheet.png"
            if not isinstance(sheet, dict) or not sheet_path.is_file():
                return False
            expected_sha = sheet.get("image_sha256")
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                return False
            try:
                actual_sheet_sha = hashlib.sha256(sheet_path.read_bytes()).hexdigest()
            except OSError:
                return False
            if actual_sheet_sha != expected_sha:
                return False
        return True

    def _read_manifest(self, review_dir: Path) -> dict[str, Any] | None:
        manifest_path = review_dir / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_has_views(
        self,
        cache_dir: Path,
        model_sha: str,
        requested_views: tuple[str, ...],
        quality: str,
    ) -> bool:
        """Validate a cached superset while ignoring its contact-sheet selection."""
        manifest = self._read_manifest(cache_dir)
        if not isinstance(manifest, dict):
            return False
        if manifest.get("model_sha256") != model_sha:
            return False
        if manifest.get("quality", "standard") != quality:
            return False
        entries = {
            entry.get("view_id"): entry
            for entry in manifest.get("views") or []
            if isinstance(entry, dict) and isinstance(entry.get("view_id"), str)
        }
        for view_id in requested_views:
            entry = entries.get(view_id)
            path = cache_dir / "views" / f"{view_id}.png"
            expected = entry.get("image_sha256") if isinstance(entry, dict) else None
            if not path.is_file() or not isinstance(expected, str) or len(expected) != 64:
                return False
            try:
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    return False
            except OSError:
                return False
        return True

    def _find_cached_superset(
        self,
        model_sha: str,
        requested_views: tuple[str, ...],
        quality: str,
        *,
        exclude: Path,
    ) -> Path | None:
        """Find verified cached views that can seed an exact variant."""
        candidates: list[Path] = []
        if quality == "standard":
            candidates.append(self._review_dir(model_sha))
        variants = (
            self.project_dir
            / ".cad-agent"
            / "screenshots"
            / model_sha
            / quality
        )
        if variants.is_dir():
            candidates.extend(sorted(path for path in variants.iterdir() if path.is_dir()))
        return next(
            (
                candidate
                for candidate in candidates
                if candidate != exclude
                and self._cache_has_views(
                    candidate, model_sha, requested_views, quality
                )
            ),
            None,
        )

    def _seed_variant_from_superset(
        self,
        source_dir: Path,
        target_dir: Path,
        model_sha: str,
        views: tuple[str, ...],
        quality: str,
        contact_sheet: bool,
    ) -> None:
        """Materialize an exact variant from already-verified cached PNGs."""
        source = self._read_manifest(source_dir)
        if not isinstance(source, dict):
            raise RuntimeError("Screenshot source manifest is missing.")
        entries = {
            entry.get("view_id"): entry
            for entry in source.get("views") or []
            if isinstance(entry, dict) and isinstance(entry.get("view_id"), str)
        }
        staging = Path(tempfile.mkdtemp(prefix="cad-screenshot-seed-"))
        try:
            staging_views = staging / "views"
            staging_views.mkdir()
            selected: list[dict[str, Any]] = []
            for view_id in views:
                entry = entries.get(view_id)
                if not isinstance(entry, dict):
                    raise RuntimeError(f"Cached view {view_id} is missing.")
                shutil.copyfile(
                    source_dir / "views" / f"{view_id}.png",
                    staging_views / f"{view_id}.png",
                )
                selected.append(dict(entry))
            sheet_info = None
            if contact_sheet:
                sheet_info = screenshot_script.build_contact_sheet_subset(
                    staging_views, staging / "review-sheet.png", views
                )
            single = next(
                (entry for entry in selected if entry.get("view_id") == "isometric_positive"),
                selected[0],
            )
            manifest = {
                "_staging_dir": str(staging),
                "model_sha256": model_sha,
                "preview_sha256": source.get("preview_sha256", ""),
                "requested_views": list(views),
                "views": selected,
                "view_count": len(selected),
                "quality": quality,
                "contact_sheet": sheet_info,
                "single_render": {
                    "path": f"views/{single['view_id']}.png",
                    "image_sha256": single.get("image_sha256"),
                },
                "rendered_at": source.get("rendered_at") or utc_now_iso(),
            }
            self._promote_to_cache(target_dir, manifest, model_sha=model_sha)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _emit_status(self, status: str, message: str) -> None:
        publish_tool_phase(
            self._publish,
            project=self.project_dir.name,
            tool="cad_screenshot",
            call_id=self._call_id,
            status=status,
            message=message,
        )

    # ------------------------------------------------------------------ main entry

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Public entry point invoked by ``AgentRunner._execute`` dispatch."""
        views = self._normalize_views(arguments.get("views"))
        quality = self._normalize_quality(arguments.get("quality", "standard"))
        contact_sheet = bool(arguments.get("contact_sheet", True))
        timeout = min(
            max(int(arguments.get("timeout_seconds") or 30), 1),
            self._MAX_TIMEOUT,
        )
        model_path = self.project_dir / "model.py"
        if not model_path.is_file():
            raise ValueError("model.py does not exist yet.")
        # AST preflight — same rule as CadTool so we never run an unsafe model.
        FileTool.validate_model(model_path.read_text(encoding="utf-8"))
        model_sha = self._model_sha256(self.project_dir)
        if not model_sha:
            raise RuntimeError("Could not hash model.py for cache lookup.")
        review_dir = self._cache_dir(model_sha, views, quality, contact_sheet)
        cache_hit = self._cache_matches(
            review_dir, model_sha, views, quality, contact_sheet
        )
        cached_superset = (
            None
            if cache_hit
            else self._find_cached_superset(
                model_sha, views, quality, exclude=review_dir
            )
        )
        if cached_superset is not None:
            try:
                self._seed_variant_from_superset(
                    cached_superset,
                    review_dir,
                    model_sha,
                    views,
                    quality,
                    contact_sheet,
                )
            except Exception:  # noqa: BLE001 - corrupt cache falls back to render.
                _LOG.warning(
                    "cached screenshot superset could not seed variant; re-rendering",
                    exc_info=_LOG.isEnabledFor(logging.DEBUG),
                )
            else:
                cache_hit = True
        if not cache_hit:
            self._emit_status(
                "rendering_subset",
                f"Re-rasterising {len(views)} view(s) at {quality} quality…",
            )
            manifest_payload = self._run_sandbox(
                model_path=model_path,
                views=views,
                quality=quality,
                contact_sheet=contact_sheet,
                timeout_seconds=timeout,
            )
            self._promote_to_cache(
                review_dir, manifest_payload, model_sha=model_sha
            )
        manifest = self._read_manifest(review_dir)
        if not isinstance(manifest, dict):
            raise RuntimeError("Screenshot manifest is missing or malformed.")
        views_subset = [
            entry
            for entry in manifest.get("views", [])
            if isinstance(entry, dict) and entry.get("view_id") in views
        ]
        manifest_sha = manifest.get("preview_sha256")
        if not isinstance(manifest_sha, str) or not manifest_sha:
            manifest_sha = self._preview_sha_for_model(model_sha)
        payload = {
            "manifest": {
                "model_sha256": manifest.get("model_sha256") or model_sha,
                "preview_sha256": manifest_sha,
                "views": views_subset,
                "contact_sheet": manifest.get("contact_sheet")
                if contact_sheet
                else None,
                "quality": manifest.get("quality", quality),
                "rendered_at": manifest.get("rendered_at"),
            },
            "cache_hit": cache_hit,
            "summary": (
                f"Rendered {len(views_subset)}/{len(self.SUBSET_VIEWS)} view(s) "
                f"at {quality} quality; "
                f"contact sheet {'included' if contact_sheet else 'skipped'}."
            ),
        }
        try:
            if callable(self._publish):
                self._publish(
                    "screenshot_updated",
                    {
                        "project": self.project_dir.name,
                        "cache_hit": cache_hit,
                        "model_sha256": model_sha,
                        "view_ids": list(views),
                        "quality": quality,
                    },
                )
                self._publish(
                    "preview_updated",
                    {
                        "project": self.project_dir.name,
                        "source": "screenshot",
                    },
                )
        except Exception:  # noqa: BLE001 - status events must never fail the tool.
            _LOG.warning(
                "screenshot updated event publish failed (project=%s): ignored",
                self.project_dir.name,
                exc_info=_LOG.isEnabledFor(logging.DEBUG),
            )
        return payload

    # ------------------------------------------------------------------ sandbox + promote

    def _run_sandbox(
        self,
        *,
        model_path: Path,
        views: tuple[str, ...],
        quality: str,
        contact_sheet: bool,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Spawn the sandbox subprocess, read the manifest, validate it."""
        # Copy ``renderer.py`` and ``screenshot.py`` as siblings into the
        # bubblewrap workspace so ``screenshot.py`` can ``import renderer``
        # directly. Pass the per-call settings as a JSON kwargs payload on
        # ``argv[1]``; the script's :func:`main` parses them and writes the
        # manifest. This replaces the previous source-string concatenation
        # (which stripped screenshot.py's ``__main__`` guard and rewrote it).
        script_dir = Path(screenshot_script.__file__).resolve().parent
        renderer_source = (script_dir / "renderer.py").read_text(encoding="utf-8")
        screenshot_source = (script_dir / "screenshot.py").read_text(
            encoding="utf-8"
        )
        with tempfile.TemporaryDirectory(prefix="cad-screenshot-") as tmp_root:
            workspace = Path(tmp_root)
            (workspace / "model.py").write_text(
                model_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            (workspace / "renderer.py").write_text(
                renderer_source, encoding="utf-8"
            )
            (workspace / "screenshot.py").write_text(
                screenshot_source, encoding="utf-8"
            )
            staging = workspace / ".screenshot-staging"
            staging.mkdir(parents=True, exist_ok=True)
            (staging / "views").mkdir(exist_ok=True)
            args_payload = json.dumps(
                {
                    "model_path": "model.py",
                    "requested_views": list(views),
                    # The sandbox mounts ``workspace`` at /workspace, so
                    # arguments consumed by the child must be workspace-
                    # relative; the host's TemporaryDirectory path is not
                    # visible inside Bubblewrap.
                    "output_dir": ".screenshot-staging",
                    "quality": quality,
                    "contact_sheet": contact_sheet,
                }
            )
            command, seccomp_fd = sandbox_command(
                workspace,
                ["screenshot.py", args_payload],
                writable=True,
                timeout_seconds=timeout_seconds,
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
                    stdout, stderr = _stream_with_limit(process, timeout=timeout_seconds)
                except _TimedOut as error:
                    raise RuntimeError(
                        f"cad_screenshot timed out after {timeout_seconds} seconds."
                    ) from error
                except RuntimeError as error:
                    raise RuntimeError(
                        f"cad_screenshot subprocess failed: {error}"
                    ) from error
                if process.returncode:
                    raise RuntimeError(
                        f"cad_screenshot execution failed:\n"
                        f"{stderr or stdout or 'unknown error'}"
                    )
            finally:
                os.close(seccomp_fd)
                with self._lock:
                    self._process = None
                if process is not None:
                    _kill_process_group(process, force=True)
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            manifest_path = staging / ".screenshot_manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(
                    "cad_screenshot did not produce a manifest."
                )
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"cad_screenshot manifest is malformed: {error}"
                ) from error
            if not isinstance(payload, dict):
                raise RuntimeError("cad_screenshot manifest is malformed.")
            # Validate view hashes against the produced PNG bytes.
            for entry in payload.get("views", []) or []:
                view_id = entry.get("view_id")
                if not isinstance(view_id, str):
                    raise RuntimeError("Manifest view entry is missing view_id.")
                png_path = staging / "views" / f"{view_id}.png"
                if not png_path.is_file() or png_path.stat().st_size == 0:
                    raise RuntimeError(
                        f"cad_screenshot view {view_id} is missing or empty."
                    )
                expected_sha = entry.get("image_sha256")
                actual_sha = hashlib.sha256(png_path.read_bytes()).hexdigest()
                if expected_sha and actual_sha != expected_sha:
                    raise RuntimeError(
                        f"cad_screenshot view {view_id} hash mismatch."
                    )
                entry["image_sha256"] = actual_sha
                entry["image_bytes"] = png_path.stat().st_size
            if payload.get("contact_sheet"):
                sheet_path = staging / "review-sheet.png"
                if not sheet_path.is_file() or sheet_path.stat().st_size == 0:
                    raise RuntimeError(
                        "cad_screenshot contact sheet is missing or empty."
                    )
                expected_sheet_sha = payload["contact_sheet"].get("image_sha256")
                actual_sheet_sha = hashlib.sha256(
                    sheet_path.read_bytes()
                ).hexdigest()
                if expected_sheet_sha and actual_sheet_sha != expected_sheet_sha:
                    raise RuntimeError(
                        "cad_screenshot contact sheet hash mismatch."
                    )
                payload["contact_sheet"]["image_sha256"] = actual_sheet_sha
                payload["contact_sheet"]["image_bytes"] = sheet_path.stat().st_size
            payload["quality"] = quality
            # ``workspace`` is removed when this function returns, while
            # promotion happens in ``execute`` afterwards. Copy the validated
            # output to a second temporary directory whose lifetime is owned
            # by ``_promote_to_cache``; returning ``staging`` directly would
            # otherwise leave the caller with a path that no longer exists.
            promoted_staging = Path(tempfile.mkdtemp(prefix="cad-screenshot-promote-"))
            shutil.rmtree(promoted_staging)
            try:
                shutil.copytree(staging, promoted_staging)
            except Exception:
                shutil.rmtree(promoted_staging, ignore_errors=True)
                raise
            payload["_staging_dir"] = str(promoted_staging)
            return payload

    def _promote_to_cache(
        self,
        review_dir: Path,
        manifest_payload: dict[str, Any],
        *,
        model_sha: str | None = None,
    ) -> None:
        """Move a staged screenshot/review variant into its cache atomically."""
        staging = Path(manifest_payload.pop("_staging_dir", ""))
        if not staging or not staging.is_dir():
            raise RuntimeError("cad_screenshot staging directory is missing.")
        review_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_views = staging / "views"
        staging_sheet = staging / "review-sheet.png"
        staging_tmp = review_dir.with_suffix(review_dir.suffix + ".tmp")
        if staging_tmp.exists():
            shutil.rmtree(staging_tmp, ignore_errors=True)
        staging_tmp.mkdir(parents=True, exist_ok=True)
        views_target = staging_tmp / "views"
        views_target.mkdir(exist_ok=True)
        # A manifest and its contact sheet describe one concrete render set.
        # Do not merge a new subset into an older set: that would leave the
        # manifest claiming views (or a quality) that the new sheet does not
        # contain. A later request can still use the exact-set cache above.
        try:
            for entry in manifest_payload.get("views", []) or []:
                view_id = entry.get("view_id")
                if not isinstance(view_id, str):
                    continue
                source = staging_views / f"{view_id}.png"
                if not source.is_file():
                    continue
                shutil.copyfile(source, views_target / f"{view_id}.png")
            if staging_sheet.is_file():
                shutil.copyfile(staging_sheet, staging_tmp / "review-sheet.png")
            merged = dict(manifest_payload)
            model_sha = model_sha or str(merged.get("model_sha256") or "")
            if not isinstance(merged.get("artifact_dir"), str):
                try:
                    merged["artifact_dir"] = str(
                        review_dir.relative_to(self.project_dir / ".cad-agent")
                    )
                except ValueError:
                    merged["artifact_dir"] = review_dir.name
            if not merged.get("preview_sha256"):
                merged["preview_sha256"] = self._preview_sha_for_model(
                    model_sha
                )
            merged["rendered_at"] = (
                merged.get("rendered_at") or utc_now_iso()
            )
            (staging_tmp / "manifest.json").write_text(
                json.dumps(merged, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            backup = review_dir.with_suffix(review_dir.suffix + ".previous")
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            if review_dir.exists():
                os.replace(review_dir, backup)
            try:
                os.replace(staging_tmp, review_dir)
            except OSError:
                if backup.exists() and not review_dir.exists():
                    os.replace(backup, review_dir)
                raise
            shutil.rmtree(backup, ignore_errors=True)
        finally:
            if staging_tmp.exists():
                shutil.rmtree(staging_tmp, ignore_errors=True)
            shutil.rmtree(staging, ignore_errors=True)
