"""Atomic, append-only, project-local persistence for CAD quality records.

Phase 1 of the CAD quality plan. Storage layout (all paths project-local):

    <project>/.cad-agent/quality/runs/<run_id>/
        run.json                  run manifest (atomic writes)
        events.jsonl              append-only audit trail (size-bounded)
        attempts/<attempt_id>.json  immutable attempt manifest
        artifacts/<attempt_id>/     copied small artifacts (metrics.json)

Rules enforced here:
- manifests are written atomically through same-directory temporary files;
- events are appended under a per-project lock and size-bounded;
- attempts are immutable once they reach a terminal status;
- malformed manifests are reported via ``QualityIntegrityError`` and are never
  silently overwritten;
- only relative artifact paths and digests are stored; artifact files are hashed
  at registration time.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable

from agent.quality.errors import (
    ACCEPTED_DECISION_TYPES,
    ATTEMPT_STATUSES,
    DECISION_TYPES,
    ISSUE_SEVERITIES,
    RUN_STATUSES,
    TERMINAL_ATTEMPT_STATUSES,
    TERMINAL_RUN_STATUSES,
)
from agent.quality.models import (
    SCHEMA_VERSION,
    Attempt,
    DesignSpec,
    EnvironmentInfo,
    Issue,
    ModelInfo,
    TaskRun,
    UserDecision,
    ValidationResult,
    _utc_now,
    new_id,
)

EVENTS_MAX_BYTES = 5 * 1024 * 1024
ARTIFACT_COPY_MAX_BYTES = 2 * 1024 * 1024


class QualityError(Exception):
    """Base class for quality-store failures."""


class QualityIntegrityError(QualityError):
    """Malformed data, immutability violations, or unknown records."""


class QualityLimitError(QualityError):
    """Size or pagination bounds exceeded."""


_guards_lock = threading.Lock()
_locks: dict[str, threading.RLock] = {}


def _lock_for(project_dir: Path) -> threading.RLock:
    key = str(Path(project_dir).resolve())
    with _guards_lock:
        return _locks.setdefault(key, threading.RLock())


class QualityStore:
    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.root = self.project_dir / ".cad-agent" / "quality"
        self._lock = _lock_for(self.project_dir)
        # Per-revision → run_id index (lazy-built on first access).
        self._revision_run_index: dict[str, str] | None = None
        # Per-attempt → run_id index (lazy-built on first access).
        self._attempt_run_index: dict[str, str] | None = None
        # Metrics cache: (runs_dir_mtime, metrics_dict).
        self._metrics_cache: tuple[float, dict[str, Any]] | None = None

    # ------------------------------------------------------------------ paths
    def _runs_dir(self) -> Path:
        return self.root / "runs"

    def _run_dir(self, run_id: str) -> Path:
        self._validate_id(run_id)
        return self._runs_dir() / run_id

    def _run_file(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "run.json"

    def _events_file(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "events.jsonl"

    def _attempt_file(self, run_id: str, attempt_id: str) -> Path:
        return self._run_dir(run_id) / "attempts" / f"{attempt_id}.json"

    def _artifacts_dir(self, run_id: str, attempt_id: str) -> Path:
        return self._run_dir(run_id) / "artifacts" / attempt_id

    def _decision_file(self, run_id: str, decision_id: str) -> Path:
        return self._run_dir(run_id) / "decisions" / f"{decision_id}.json"

    def _issue_file(self, run_id: str, issue_id: str) -> Path:
        return self._run_dir(run_id) / "issues" / f"{issue_id}.json"

    def _evidence_dir(self, run_id: str, attempt_id: str) -> Path:
        return self._artifacts_dir(run_id, attempt_id) / "evidence"

    def _spec_dir(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "specs"

    def _spec_file(self, run_id: str, version: int) -> Path:
        return self._spec_dir(run_id) / f"version-{int(version)}.json"

    def _validation_dir(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "validations"

    def _validation_file(self, run_id: str, validation_id: str) -> Path:
        return self._validation_dir(run_id) / f"{validation_id}.json"

    def _validation_index_file(self, run_id: str) -> Path:
        return self._validation_dir(run_id) / "index.jsonl"

    @staticmethod
    def _validate_id(value: str) -> None:
        if not isinstance(value, str) or len(value) != 32 or not value.isalnum():
            raise QualityIntegrityError(f"Invalid quality record id: {value!r}")

    # ------------------------------------------------------------------ indexes
    def _ensure_indexes(self) -> None:
        """Build revision→run and attempt→run indexes (once per instance)."""
        if self._revision_run_index is not None and self._attempt_run_index is not None:
            return
        rev_idx: dict[str, str] = {}
        att_idx: dict[str, str] = {}
        for run_id in self._all_run_ids():
            attempts_dir = self._run_dir(run_id) / "attempts"
            if attempts_dir.is_dir():
                for att_file in attempts_dir.glob("*.json"):
                    try:
                        data = self._read_json_required(att_file)
                        att_id = data.get("attempt_id", "")
                        rev_id = data.get("revision_id")
                        if att_id:
                            att_idx[att_id] = run_id
                        if isinstance(rev_id, str) and rev_id:
                            rev_idx.setdefault(rev_id, run_id)
                    except QualityIntegrityError:
                        continue
        self._revision_run_index = rev_idx
        self._attempt_run_index = att_idx

    def _revision_to_run(self, revision_id: str) -> str | None:
        """O(1) lookup of the run id that owns a revision (via in-memory index)."""
        if not revision_id:
            return None
        self._ensure_indexes()
        assert self._revision_run_index is not None
        return self._revision_run_index.get(revision_id)

    def _attempt_to_run(self, attempt_id: str) -> str | None:
        """O(1) lookup of the run id that owns an attempt (via in-memory index)."""
        if not attempt_id:
            return None
        self._ensure_indexes()
        assert self._attempt_run_index is not None
        return self._attempt_run_index.get(attempt_id)

    # ------------------------------------------------------------------ writes
    def start_run(
        self,
        *,
        project: str,
        request_message_id: str = "",
        request_sha256: str = "",
        system_prompt_sha256: str = "",
        model: ModelInfo | None = None,
        environment: EnvironmentInfo | None = None,
    ) -> TaskRun:
        run = TaskRun(
            run_id=new_id(),
            project=project,
            created_at=_utc_now(),
            status="running",
            request_message_id=request_message_id,
            request_sha256=request_sha256,
            system_prompt_sha256=system_prompt_sha256,
            model=model,
            environment=environment,
        )
        with self._lock:
            run_file = self._run_file(run.run_id)
            if run_file.exists():
                raise QualityIntegrityError(
                    f"Run {run.run_id} already exists; refusing to overwrite."
                )
            self._write_json_atomic(run_file, run.to_dict())
            self._append_event_locked(run.run_id, "run_started", {"run_id": run.run_id})
            self._invalidate_caches()
        return run

    def complete_run(
        self, run_id: str, *, status: str, error: dict[str, Any] | None = None
    ) -> TaskRun:
        if status not in TERMINAL_RUN_STATUSES:
            raise QualityError(f"Run status must be terminal, got {status!r}.")
        with self._lock:
            run = self.get_run(run_id)
            if run.status == status and run.ended_at is not None:
                return run  # Idempotent terminal transition.
            if run.status in {"completed", "failed", "stopped", "interrupted"}:
                raise QualityIntegrityError(
                    f"Run {run_id} is already terminal ({run.status}); "
                    f"cannot change it to {status}."
                )
            data = run.to_dict()
            data["status"] = status
            data["ended_at"] = _utc_now()
            if error is not None:
                data["error"] = error if isinstance(error, dict) else {"message": str(error)}
            self._write_json_atomic(self._run_file(run_id), data)
            self._append_event_locked(
                run_id,
                "run_completed",
                {"run_id": run_id, "status": status},
            )
            return self.get_run(run_id)

    def transition_run(self, run_id: str, *, status: str) -> TaskRun:
        """Transition a non-terminal run to another non-terminal status."""
        if status not in RUN_STATUSES:
            raise QualityError(f"Unknown run status: {status!r}")
        if status in TERMINAL_RUN_STATUSES:
            raise QualityError(
                f"Use complete_run() for terminal status {status!r}."
            )
        with self._lock:
            run = self.get_run(run_id)
            if run.status in TERMINAL_RUN_STATUSES:
                raise QualityIntegrityError(
                    f"Run {run_id} is already terminal ({run.status}); "
                    f"cannot transition to {status}."
                )
            if run.status == status:
                return run
            data = run.to_dict()
            data["status"] = status
            self._write_json_atomic(self._run_file(run_id), data)
            self._append_event_locked(
                run_id,
                "run_transitioned",
                {"run_id": run_id, "status": status},
            )
            return self.get_run(run_id)

    def start_attempt(
        self,
        run_id: str,
        *,
        revision_id: str | None = None,
        source_sha256: str | None = None,
        tool_call_id: str = "",
        phase: str = "execution",
    ) -> Attempt:
        attempt = Attempt(
            attempt_id=new_id(),
            run_id=run_id,
            revision_id=revision_id,
            tool_call_id=tool_call_id,
            source_sha256=source_sha256,
            started_at=_utc_now(),
            status="running",
            phase=phase,
        )
        with self._lock:
            self._require_run(run_id)
            attempt_file = self._attempt_file(run_id, attempt.attempt_id)
            if attempt_file.exists():
                raise QualityIntegrityError(
                    f"Attempt {attempt.attempt_id} already exists; refusing to overwrite."
                )
            self._write_json_atomic(attempt_file, attempt.to_dict())
            run_file = self._run_file(run_id)
            run_data = self._read_json_required(run_file)
            run_data["attempt_count"] = int(run_data.get("attempt_count", 0)) + 1
            self._write_json_atomic(run_file, run_data)
            self._append_event_locked(
                run_id,
                "attempt_started",
                {
                    "attempt_id": attempt.attempt_id,
                    "revision_id": attempt.revision_id,
                },
            )
            # Maintain in-memory indexes (audit_018).
            self._ensure_indexes()
            assert self._attempt_run_index is not None
            self._attempt_run_index[attempt.attempt_id] = run_id
            if revision_id is not None:
                assert self._revision_run_index is not None
                self._revision_run_index.setdefault(revision_id, run_id)
            self._metrics_cache = None
        return attempt

    def complete_attempt(
        self,
        run_id: str,
        attempt_id: str,
        *,
        status: str,
        phase: str,
        error: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        artifact_paths: dict[str, Path] | None = None,
    ) -> Attempt:
        if status not in ATTEMPT_STATUSES:
            raise QualityError(f"Unknown attempt status: {status!r}")
        artifact_paths = artifact_paths or {}
        with self._lock:
            attempt_file = self._attempt_file(run_id, attempt_id)
            data = self._read_json_required(attempt_file)
            if data.get("status") in TERMINAL_ATTEMPT_STATUSES:
                raise QualityIntegrityError(
                    f"Attempt {attempt_id} is already terminal; refusing to overwrite."
                )
            data["status"] = status
            data["phase"] = phase
            data["completed_at"] = _utc_now()
            if error is not None:
                data["error"] = error if isinstance(error, dict) else {"message": str(error)}
            artifacts: dict[str, dict[str, Any]] = {}
            for name, path in artifact_paths.items():
                digest, size = self._hash_file(path)
                artifact = {
                    "path": self._relative_posix(path),
                    "sha256": digest,
                    "size": size,
                }
                if size <= ARTIFACT_COPY_MAX_BYTES:
                    source = Path(path)
                    snapshot = self._artifacts_dir(run_id, attempt_id) / (
                        f"{name}{source.suffix}"
                    )
                    self._write_bytes_atomic(snapshot, source.read_bytes())
                    artifact["snapshot_path"] = self._relative_posix(snapshot)
                else:
                    artifact["snapshot_omitted"] = "size_limit"
                artifacts[str(name)] = artifact
            if metrics is not None:
                artifacts_dir = self._artifacts_dir(run_id, attempt_id)
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                payload_bytes = json.dumps(
                    metrics, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                if len(payload_bytes) > ARTIFACT_COPY_MAX_BYTES:
                    raise QualityLimitError(
                        "Attempt metrics artifact exceeds the size bound."
                    )
                digest = hashlib.sha256(payload_bytes).hexdigest()
                artifact_path = artifacts_dir / "metrics.json"
                self._write_bytes_atomic(artifact_path, payload_bytes)
                artifacts["metrics"] = {
                    "path": self._relative_posix(artifact_path),
                    "sha256": digest,
                    "size": len(payload_bytes),
                }
                data["metrics"] = metrics
            if artifacts:
                data["artifacts"] = artifacts
            self._write_json_atomic(attempt_file, data)
            self._append_event_locked(
                run_id,
                "attempt_completed",
                {
                    "attempt_id": attempt_id,
                    "status": status,
                    "phase": phase,
                    "error_code": data.get("error", {}).get("code")
                    if isinstance(data.get("error"), dict)
                    else None,
                },
            )
            # Return the in-memory state we just wrote instead of re-reading
            # the manifest (audit_021). The dataclass reflects the freshest fields.
            return Attempt.from_dict(data)

    def append_event(self, run_id: str, event_type: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._require_run(run_id)
            self._append_event_locked(run_id, event_type, data)

    def _append_event_locked(self, run_id: str, event_type: str, data: dict[str, Any]) -> None:
        event = {
            "schema_version": SCHEMA_VERSION,
            "event_type": event_type,
            "data": data,
            "timestamp": _utc_now(),
        }
        events_file = self._events_file(run_id)
        events_file.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        if events_file.is_file() and events_file.stat().st_size + len(line.encode("utf-8")) > EVENTS_MAX_BYTES:
            raise QualityLimitError("Run event log exceeds the size bound; refusing to append.")
        with events_file.open("a", encoding="utf-8") as stream:
            stream.write(line)

    # ------------------------------------------------------------------ reads
    def _all_run_ids(self) -> list[str]:
        """Return every run id on disk, sorted by created_at descending."""
        runs_dir = self._runs_dir()
        if not runs_dir.is_dir():
            return []
        ids: list[tuple[str, str]] = []
        for run_file in sorted(runs_dir.glob("*/run.json")):
            try:
                data = self._read_json_required(run_file)
                created_at = data.get("created_at", "")
                run_id = data.get("run_id", "")
                if run_id:
                    ids.append((created_at, run_id))
            except QualityIntegrityError:
                continue
        ids.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in ids]

    # --- Unbounded record iterators (audit_080) ---------------------------------
    # ``list_*`` methods cap at 500 for UI responsiveness; metrics aggregation
    # needs every record on disk so projects above the cap are not silently
    # undercounted. Each iterator reads its directory directly, skipping any
    # malformed file the same way the bounded variants do.

    def _iter_runs_all(self) -> Iterable[TaskRun]:
        runs_dir = self._runs_dir()
        if not runs_dir.is_dir():
            return iter(())
        ids = self._all_run_ids()
        for run_id in ids:
            try:
                yield self.get_run(run_id)
            except QualityIntegrityError:
                continue

    def _iter_attempts_all(self, run_id: str) -> Iterable[Attempt]:
        try:
            self._require_run(run_id)
        except QualityIntegrityError:
            return iter(())
        attempts_dir = self._run_dir(run_id) / "attempts"
        if not attempts_dir.is_dir():
            return iter(())
        results: list[Attempt] = []
        with self._lock:
            for attempt_file in sorted(attempts_dir.glob("*.json")):
                try:
                    results.append(
                        Attempt.from_dict(self._read_json_required(attempt_file))
                    )
                except QualityIntegrityError:
                    continue
        return results

    def _iter_decisions_all(self, run_id: str) -> Iterable[UserDecision]:
        try:
            self._require_run(run_id)
        except QualityIntegrityError:
            return iter(())
        decision_dir = self._run_dir(run_id) / "decisions"
        if not decision_dir.is_dir():
            return iter(())
        results: list[UserDecision] = []
        with self._lock:
            for decision_file in sorted(decision_dir.glob("*.json")):
                try:
                    results.append(
                        UserDecision.from_dict(self._read_json_required(decision_file))
                    )
                except QualityIntegrityError:
                    continue
        # Sort newest first to mirror ``list_decisions`` so callers that read
        # only the first hit per attempt see the most recent decision.
        results.sort(key=lambda item: item.created_at, reverse=True)
        return results

    def _iter_issues_all(self, run_id: str) -> Iterable[Issue]:
        try:
            self._require_run(run_id)
        except QualityIntegrityError:
            return iter(())
        issue_dir = self._run_dir(run_id) / "issues"
        if not issue_dir.is_dir():
            return iter(())
        results: list[Issue] = []
        with self._lock:
            for issue_file in sorted(issue_dir.glob("*.json")):
                try:
                    results.append(
                        Issue.from_dict(self._read_json_required(issue_file))
                    )
                except QualityIntegrityError:
                    continue
        return results

    def reconcile(self) -> int:
        """Mark runs left ``running`` by a dead process as ``interrupted``.

        Called before a new run starts; the runner is single-flight, so any
        leftover running record belongs to an aborted process.
        ``waiting_for_user`` runs are left alone — they are paused, not dead.
        """
        interrupted = 0
        for run_id in self._all_run_ids():
            try:
                run = self.get_run(run_id)
            except QualityIntegrityError:
                continue
            if run.status not in ("running",):
                continue
            with self._lock:
                try:
                    self.complete_run(run.run_id, status="interrupted")
                except QualityIntegrityError:
                    continue
            interrupted += 1
        return interrupted

    def get_run(self, run_id: str) -> TaskRun:
        self._validate_id(run_id)
        with self._lock:
            data = self._read_json_required(self._run_file(run_id))
            return TaskRun.from_dict(data)

    def list_runs(self, *, limit: int = 50, before: str | None = None) -> list[TaskRun]:
        try:
            limit = max(1, min(int(limit), 500))
        except (TypeError, ValueError) as error:
            raise QualityError("Run limit must be an integer.") from error
        runs_dir = self._runs_dir()
        if not runs_dir.is_dir():
            return []
        runs: list[TaskRun] = []
        with self._lock:
            for run_file in sorted(runs_dir.glob("*/run.json")):
                data = self._read_json_required(run_file)
                runs.append(TaskRun.from_dict(data))
            runs.sort(key=lambda run: run.created_at, reverse=True)
        if before is not None:
            cursor = next((run for run in runs if run.run_id == before), None)
            if cursor is None:
                raise QualityError("Unknown before cursor.")
            runs = [run for run in runs if run.created_at < cursor.created_at]
        return runs[:limit]

    def get_attempt(self, run_id: str, attempt_id: str) -> Attempt:
        self._validate_id(attempt_id)
        with self._lock:
            data = self._read_json_required(self._attempt_file(run_id, attempt_id))
            return Attempt.from_dict(data)

    def list_attempts(self, run_id: str, *, limit: int = 200) -> list[Attempt]:
        self._require_run(run_id)
        try:
            limit = max(1, min(int(limit), 500))
        except (TypeError, ValueError) as error:
            raise QualityError("Attempt limit must be an integer.") from error
        attempt_dir = self._run_dir(run_id) / "attempts"
        if not attempt_dir.is_dir():
            return []
        attempts: list[Attempt] = []
        with self._lock:
            for attempt_file in sorted(attempt_dir.glob("*.json")):
                attempts.append(
                    Attempt.from_dict(self._read_json_required(attempt_file))
                )
            attempts.sort(key=lambda attempt: attempt.started_at, reverse=True)
        return attempts[:limit]

    def get_events(self, run_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        self._require_run(run_id)
        try:
            limit = max(1, min(int(limit), 500))
        except (TypeError, ValueError) as error:
            raise QualityError("Event limit must be an integer.") from error
        events_file = self._events_file(run_id)
        if not events_file.is_file():
            return []
        events: list[dict[str, Any]] = []
        with events_file.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    raise QualityIntegrityError(
                        f"Malformed event line in {events_file}."
                    ) from None
                if isinstance(event, dict):
                    events.append(event)
        return events[-limit:]

    # ------------------------------------------------------- decisions & issues
    def record_decision(
        self,
        run_id: str,
        *,
        revision_id: str,
        attempt_id: str,
        decision: str,
        categories: list[str] | tuple[str, ...] | None = None,
        comment: str = "",
        camera: dict[str, Any] | None = None,
    ) -> UserDecision:
        if decision not in DECISION_TYPES:
            raise QualityError(f"Unknown decision type: {decision!r}")
        previous = next(
            (
                item
                for item in self.list_decisions(run_id, limit=500)
                if item.attempt_id == attempt_id
            ),
            None,
        )
        decision_record = UserDecision(
            decision_id=new_id(),
            run_id=run_id,
            revision_id=revision_id,
            attempt_id=attempt_id,
            decision=decision,
            categories=tuple(str(item) for item in (categories or ())),
            comment=str(comment)[:2000],
            camera=camera if isinstance(camera, dict) else None,
            supersedes_decision_id=previous.decision_id if previous else None,
            created_at=_utc_now(),
        )
        with self._lock:
            self._require_run(run_id)
            self.get_attempt(run_id, attempt_id)  # Decision must target a real attempt.
            decision_file = self._decision_file(run_id, decision_record.decision_id)
            if decision_file.exists():
                raise QualityIntegrityError(
                    f"Decision {decision_record.decision_id} already exists; refusing to overwrite."
                )
            self._write_json_atomic(decision_file, decision_record.to_dict())
            run_data = self._read_json_required(self._run_file(run_id))
            if decision in ACCEPTED_DECISION_TYPES:
                run_data["accepted_revision_id"] = revision_id
            elif run_data.get("accepted_revision_id") == revision_id:
                run_data["accepted_revision_id"] = None
            self._write_json_atomic(self._run_file(run_id), run_data)
            self._append_event_locked(
                run_id,
                "decision_recorded",
                {
                    "decision_id": decision_record.decision_id,
                    "attempt_id": attempt_id,
                    "revision_id": revision_id,
                    "decision": decision,
                },
            )
        return decision_record

    def list_decisions(self, run_id: str, *, limit: int = 200) -> list[UserDecision]:
        self._require_run(run_id)
        try:
            limit = max(1, min(int(limit), 500))
        except (TypeError, ValueError) as error:
            raise QualityError("Decision limit must be an integer.") from error
        decision_dir = self._run_dir(run_id) / "decisions"
        if not decision_dir.is_dir():
            return []
        decisions: list[UserDecision] = []
        with self._lock:
            for decision_file in sorted(decision_dir.glob("*.json")):
                decisions.append(
                    UserDecision.from_dict(self._read_json_required(decision_file))
                )
            decisions.sort(key=lambda item: item.created_at, reverse=True)
        return decisions[:limit]

    def latest_decision_for_revision(self, revision_id: str) -> UserDecision | None:
        """Latest explicit decision across all runs for one revision (or None)."""
        run_id = self._revision_to_run(revision_id)
        if run_id is None:
            return None
        try:
            decisions = self.list_decisions(run_id, limit=500)
        except QualityIntegrityError:
            return None
        latest: UserDecision | None = None
        for decision in decisions:
            if decision.revision_id != revision_id:
                continue
            if latest is None or decision.created_at > latest.created_at:
                latest = decision
        return latest

    def decisions_for_revision(self, revision_id: str) -> list[UserDecision]:
        run_id = self._revision_to_run(revision_id)
        if run_id is None:
            return []
        try:
            decisions = [
                decision
                for decision in self.list_decisions(run_id, limit=500)
                if decision.revision_id == revision_id
            ]
        except QualityIntegrityError:
            return []
        return sorted(decisions, key=lambda item: item.created_at)

    def create_issue(
        self,
        run_id: str,
        *,
        attempt_id: str,
        revision_id: str,
        category: str,
        severity: str = "blocking",
        message: str = "",
        requirement_ids: list[str] | tuple[str, ...] | None = None,
        evidence: list[str] | tuple[str, ...] | None = None,
        issue_id: str | None = None,
    ) -> Issue:
        if severity not in ISSUE_SEVERITIES:
            raise QualityError(f"Unknown issue severity: {severity!r}")
        if issue_id is not None:
            self._validate_id(issue_id)
        issue = Issue(
            issue_id=issue_id or new_id(),
            run_id=run_id,
            attempt_id=attempt_id,
            revision_id=revision_id,
            category=str(category),
            severity=severity,
            message=str(message)[:4000],
            requirement_ids=tuple(str(item) for item in (requirement_ids or ())),
            evidence=tuple(str(item) for item in (evidence or ())),
            created_at=_utc_now(),
        )
        with self._lock:
            self._require_run(run_id)
            self.get_attempt(run_id, attempt_id)
            issue_file = self._issue_file(run_id, issue.issue_id)
            if issue_file.exists():
                raise QualityIntegrityError(
                    f"Issue {issue.issue_id} already exists; refusing to overwrite."
                )
            self._write_json_atomic(issue_file, issue.to_dict())
            self._append_event_locked(
                run_id,
                "issue_created",
                {
                    "issue_id": issue.issue_id,
                    "attempt_id": attempt_id,
                    "revision_id": revision_id,
                    "category": issue.category,
                    "severity": issue.severity,
                },
            )
        return issue

    def list_issues(
        self,
        run_id: str | None = None,
        *,
        revision_id: str | None = None,
        open_only: bool = False,
        limit: int = 200,
    ) -> list[Issue]:
        try:
            limit = max(1, min(int(limit), 500))
        except (TypeError, ValueError) as error:
            raise QualityError("Issue limit must be an integer.") from error
        run_ids = [run_id] if run_id is not None else self._all_run_ids()
        issues: list[Issue] = []
        for current_run_id in run_ids:
            issue_dir = self._run_dir(current_run_id) / "issues"
            if not issue_dir.is_dir():
                continue
            with self._lock:
                for issue_file in sorted(issue_dir.glob("*.json")):
                    issue = Issue.from_dict(self._read_json_required(issue_file))
                    if revision_id is not None and issue.revision_id != revision_id:
                        continue
                    if open_only and issue.status != "open":
                        continue
                    issues.append(issue)
        return sorted(issues, key=lambda item: item.created_at, reverse=True)[:limit]

    def issues_for_revision(self, revision_id: str) -> list[Issue]:
        return self.list_issues(revision_id=revision_id, limit=500)

    def has_successful_attempt_for_revision(self, revision_id: str) -> bool:
        """Return whether a revision has at least one successful CAD attempt."""
        run_id = self._revision_to_run(revision_id)
        if run_id is None:
            return False
        try:
            attempts = self.list_attempts(run_id, limit=500)
        except QualityIntegrityError:
            return False
        return any(
            attempt.revision_id == revision_id and attempt.status == "succeeded"
            for attempt in attempts
        )

    def resolve_issue(
        self,
        run_id: str,
        issue_id: str,
        *,
        resolved_by_revision_id: str,
        confirmed_by: str,
    ) -> Issue:
        self._validate_id(issue_id)
        with self._lock:
            issue_file = self._issue_file(run_id, issue_id)
            data = self._read_json_required(issue_file)
            if data.get("status") == "resolved":
                raise QualityIntegrityError(
                    f"Issue {issue_id} is already resolved."
                )
            data["status"] = "resolved"
            data["resolved_by_revision_id"] = resolved_by_revision_id
            data["resolved_at"] = _utc_now()
            data["confirmed_by"] = confirmed_by
            self._write_json_atomic(issue_file, data)
            self._append_event_locked(
                run_id,
                "issue_resolved",
                {
                    "issue_id": issue_id,
                    "resolved_by_revision_id": resolved_by_revision_id,
                    "confirmed_by": confirmed_by,
                },
            )
            return Issue.from_dict(self._read_json_required(issue_file))

    def delete_issue(self, run_id: str, issue_id: str) -> bool:
        """Remove an issue and any persisted evidence, returning whether it existed.

        Intended for rollback when a downstream step (decision, evidence save)
        fails after the issue was created (audit_079). Resolved issues are
        left alone to preserve the audit trail.
        """
        self._validate_id(issue_id)
        removed = False
        with self._lock:
            issue_file = self._issue_file(run_id, issue_id)
            if issue_file.exists():
                try:
                    data = self._read_json_required(issue_file)
                except QualityIntegrityError:
                    data = None
                if not (isinstance(data, dict) and data.get("status") == "resolved"):
                    issue_file.unlink()
                    removed = True
        if removed:
            evidence_root = self.root / "runs" / run_id / "attempts"
            if evidence_root.is_dir():
                for artifact_dir in evidence_root.glob("*/artifacts/evidence"):
                    evidence_file = artifact_dir / f"issue-{issue_id}.png"
                    try:
                        evidence_file.unlink()
                    except FileNotFoundError:
                        pass
        return removed

    def save_issue_evidence(
        self,
        run_id: str,
        attempt_id: str,
        issue_id: str,
        png_bytes: bytes,
        max_bytes: int = 3 * 1024 * 1024,
    ) -> str:
        """Persist bounded PNG evidence and return its project-relative path."""
        if not png_bytes or len(png_bytes) > max_bytes:
            raise QualityLimitError("Issue evidence exceeds the size bound.")
        evidence_dir = self._evidence_dir(run_id, attempt_id)
        with self._lock:
            self._require_run(run_id)
            evidence_dir.mkdir(parents=True, exist_ok=True)
            path = evidence_dir / f"issue-{issue_id}.png"
            self._write_bytes_atomic(path, png_bytes)
        return self._relative_posix(path)

    def update_issue_evidence(
        self,
        run_id: str,
        issue_id: str,
        evidence: tuple[str, ...],
    ) -> Issue:
        """Replace the issue's evidence list, appending an event.

        Used by the issue-report endpoint after the screenshot has been written
        to disk so the JSON manifest stays in sync with the artifacts
        directory (audit_079).
        """
        self._validate_id(issue_id)
        with self._lock:
            issue_file = self._issue_file(run_id, issue_id)
            data = self._read_json_required(issue_file)
            data["evidence"] = [str(item) for item in evidence]
            self._write_json_atomic(issue_file, data)
            self._append_event_locked(
                run_id,
                "issue_updated",
                {"issue_id": issue_id, "evidence_count": len(evidence)},
            )
            return Issue.from_dict(self._read_json_required(issue_file))

    # -------------------------------------------------------------- specs & verifiers

    def save_spec(self, spec: DesignSpec) -> DesignSpec:
        """Persist a :class:`DesignSpec` for a run and return it unchanged.

        Each spec version is immutable on disk; saving an existing version is a
        :class:`QualityIntegrityError`.
        """
        with self._lock:
            self._require_run(spec.run_id)
            spec_dir = self._spec_dir(spec.run_id)
            spec_dir.mkdir(parents=True, exist_ok=True)
            spec_file = self._spec_file(spec.run_id, spec.version)
            if spec_file.exists():
                raise QualityIntegrityError(
                    f"Spec version {spec.version} for run {spec.run_id} already exists."
                )
            self._write_json_atomic(spec_file, spec.to_dict())
            self._append_event_locked(
                spec.run_id,
                "spec_saved",
                {"version": spec.version, "requirement_count": len(spec.requirements)},
            )
        return spec

    def latest_spec(self, run_id: str) -> DesignSpec | None:
        """Return the highest-version :class:`DesignSpec` for a run, or ``None``."""
        self._validate_id(run_id)
        spec_dir = self._spec_dir(run_id)
        if not spec_dir.is_dir():
            return None
        latest: DesignSpec | None = None
        with self._lock:
            for spec_file in sorted(spec_dir.glob("version-*.json")):
                try:
                    data = self._read_json_required(spec_file)
                    spec = DesignSpec.from_dict(data)
                except QualityIntegrityError:
                    continue
                if latest is None or spec.version > latest.version:
                    latest = spec
        return latest

    def get_spec(self, run_id: str, version: int) -> DesignSpec | None:
        """Return a specific :class:`DesignSpec` version, or ``None`` if missing."""
        self._validate_id(run_id)
        with self._lock:
            spec_file = self._spec_file(run_id, version)
            if not spec_file.is_file():
                return None
            try:
                return DesignSpec.from_dict(self._read_json_required(spec_file))
            except QualityIntegrityError:
                return None

    def save_validation(
        self,
        result: ValidationResult,
        *,
        run_id: str,
    ) -> ValidationResult:
        """Persist a :class:`ValidationResult` and index it for fast lookup.

        ``run_id`` is required (audit_038) — the caller always has it on hand
        when constructing the result, so a per-call disk scan was redundant.
        """
        with self._lock:
            self._validate_id(run_id)
            self._validate_id(result.attempt_id)
            self._require_run(run_id)
            validation_dir = self._validation_dir(run_id)
            validation_dir.mkdir(parents=True, exist_ok=True)
            validation_file = self._validation_file(run_id, result.validation_id)
            if validation_file.exists():
                raise QualityIntegrityError(
                    f"Validation {result.validation_id} already exists."
                )
            self._write_json_atomic(validation_file, result.to_dict())
            index_file = self._validation_index_file(run_id)
            entry = {
                "validation_id": result.validation_id,
                "attempt_id": result.attempt_id,
                "requirement_id": result.requirement_id,
                "verifier": result.verifier,
                "status": result.status,
                "severity": result.severity,
                "created_at": result.created_at,
            }
            with index_file.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        return result

    def _run_id_for_attempt(self, attempt_id: str) -> str:
        """Locate the run id that owns an attempt via the in-memory index (audit_018)."""
        self._validate_id(attempt_id)
        run_id = self._attempt_to_run(attempt_id)
        if run_id is None:
            raise QualityIntegrityError(f"Unknown attempt id: {attempt_id!r}")
        return run_id

    def validations_for_attempt(self, attempt_id: str) -> list[ValidationResult]:
        """Return every validation result for one attempt, newest first."""
        results: list[ValidationResult] = []
        with self._lock:
            try:
                run_id = self._run_id_for_attempt(attempt_id)
            except QualityIntegrityError:
                return []
            validation_dir = self._validation_dir(run_id)
            if not validation_dir.is_dir():
                return []
            for validation_file in sorted(validation_dir.glob("*.json")):
                if validation_file.name == "index.jsonl":
                    continue
                try:
                    results.append(
                        ValidationResult.from_dict(
                            self._read_json_required(validation_file)
                        )
                    )
                except QualityIntegrityError:
                    continue
        results.sort(key=lambda item: item.created_at, reverse=True)
        return results

    def get_metrics(self) -> dict[str, Any]:
        """Aggregate observability metrics for the project (mtime-cached).

        Reads every record on disk rather than sampling the first ``limit=500``
        of each list so projects that exceed the per-list cap are still
        aggregated accurately (audit_080).
        """
        sentinel = self._runs_dir().stat().st_mtime if self._runs_dir().exists() else 0.0
        if self._metrics_cache is not None:
            cached_mtime, cached_metrics = self._metrics_cache
            if cached_mtime == sentinel:
                return cached_metrics
        runs = list(self._iter_runs_all())
        attempts_total = 0
        attempts_succeeded = 0
        attempts_failed = 0
        decisions: dict[str, int] = {decision: 0 for decision in sorted(DECISION_TYPES)}
        decisions_on_succeeded = 0
        rejected_on_succeeded = 0
        issues_open = 0
        issues_resolved = 0
        accepted_revisions: set[str] = set()
        latest_by_revision: dict[str, UserDecision] = {}
        for run in runs:
            attempts = list(self._iter_attempts_all(run.run_id))
            attempts_total += len(attempts)
            for attempt in attempts:
                if attempt.status == "succeeded":
                    attempts_succeeded += 1
                elif attempt.status == "failed":
                    attempts_failed += 1
            attempts_by_id = {attempt.attempt_id: attempt for attempt in attempts}
            latest_by_attempt: dict[str, UserDecision] = {}
            for decision in self._iter_decisions_all(run.run_id):
                decisions[decision.decision] = decisions.get(decision.decision, 0) + 1
                latest_by_attempt.setdefault(decision.attempt_id, decision)
                current = latest_by_revision.get(decision.revision_id)
                if current is None or decision.created_at > current.created_at:
                    latest_by_revision[decision.revision_id] = decision
            for decision in latest_by_attempt.values():
                attempt = attempts_by_id.get(decision.attempt_id)
                if attempt is not None and attempt.status == "succeeded":
                    decisions_on_succeeded += 1
                    if decision.decision == "rejected":
                        rejected_on_succeeded += 1
            for issue in self._iter_issues_all(run.run_id):
                if issue.status == "open":
                    issues_open += 1
                elif issue.status == "resolved":
                    issues_resolved += 1
        accepted_revisions = {
            decision.revision_id
            for decision in latest_by_revision.values()
            if decision.decision in ACCEPTED_DECISION_TYPES
        }
        result = {
            "runs": len(runs),
            "attempts": {
                "total": attempts_total,
                "succeeded": attempts_succeeded,
                "failed": attempts_failed,
            },
            "decisions": decisions,
            "accepted_revisions": len(accepted_revisions),
            "issues": {"open": issues_open, "resolved": issues_resolved},
            "false_pass_rate": (
                round(rejected_on_succeeded / decisions_on_succeeded, 4)
                if decisions_on_succeeded
                else None
            ),
            "rejected_on_succeeded": rejected_on_succeeded,
            "decisions_on_succeeded": decisions_on_succeeded,
        }
        self._metrics_cache = (sentinel, result)
        return result

    # ---------------------------------------------------------------- helpers
    def _invalidate_caches(self) -> None:
        """Reset all in-memory caches after a mutation."""
        self._revision_run_index = None
        self._attempt_run_index = None
        self._metrics_cache = None

    def _require_run(self, run_id: str) -> TaskRun:
        self._validate_id(run_id)
        return TaskRun.from_dict(self._read_json_required(self._run_file(run_id)))

    def _read_json_required(self, path: Path) -> dict[str, Any]:
        data = _read_json_safe(path)
        if data is None:
            raise QualityIntegrityError(f"Quality record is missing or malformed: {path}")
        return data

    def _hash_file(self, path: Path) -> tuple[str, int]:
        path = Path(path).resolve()
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            size = path.stat().st_size
        except OSError as error:
            raise QualityError(f"Cannot hash artifact {path}: {error}") from error
        return digest, size

    def _relative_posix(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project_dir).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    @staticmethod
    def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
        _atomic_write_bytes(path, json.dumps(data, ensure_ascii=False).encode("utf-8"))

    @staticmethod
    def _write_bytes_atomic(path: Path, payload: bytes) -> None:
        _atomic_write_bytes(path, payload)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_safe(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
