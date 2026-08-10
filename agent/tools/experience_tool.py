"""Shared past-issues memory — search, store, and reuse verified solutions."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

_MEMORY_DIR = ".agent-memory"
_MEMORY_FILE = "past_issues.json"
_MEMORY_LOCKS: dict[Path, threading.RLock] = {}
_MEMORY_LOCKS_GUARD = threading.Lock()


def _lock_for_memory_file(path: Path) -> threading.RLock:
    """Return the process-local lock shared by users of one memory file."""
    path = path.resolve()
    with _MEMORY_LOCKS_GUARD:
        return _MEMORY_LOCKS.setdefault(path, threading.RLock())

_MAX_FIELD_LENGTHS = {
    "problem": 500,
    "solution": 2000,
}
_MAX_TAGS = 10
_MAX_TAG_LENGTH = 50
_SEARCH_RESULT_LIMIT = 5
_DUPLICATE_SIMILARITY_THRESHOLD = 0.6
_MIN_WORD_LENGTH = 3


class ExperienceTool:
    """Cross-project memory of encountered-and-verified problem/solution pairs."""

    __tool_schema__: ClassVar[dict[str, Any]] = {
        "type": "function",
        "function": {
            "name": "experience",
            "description": "Search or store verified problem-solution pairs shared across projects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["search", "add", "update"]},
                    "query": {"type": "string", "description": "Space-separated search terms for problems, solutions, and tags"},
                    "problem": {"type": "string", "description": "Short, generalized problem statement (max 500 chars)"},
                    "solution": {"type": "string", "description": "Verified solution (max 2000 chars)"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Up to 10 lowercase tags"},
                    "id": {"type": "string", "description": "Record id for update operation"},
                },
                "required": ["operation"],
            },
        },
    }

    def __init__(self, workspace_root: Path, project_name: str) -> None:
        self._workspace_root = workspace_root
        self._project_name = project_name
        self._memory_dir = workspace_root / _MEMORY_DIR
        self._memory_file = self._memory_dir / _MEMORY_FILE
        # The memory file is shared by every project in this workspace, so all
        # instances using it must share one read-modify-write lock.
        self._memory_lock = _lock_for_memory_file(self._memory_file)

    # ── public API ──────────────────────────────────────────────────────

    def execute(self, args: dict) -> tuple[str, bool]:
        """Dispatch experience operations. Returns (result_json, waiting=False)."""
        operation = args["operation"]
        if operation == "search":
            query = (args.get("query") or "").strip()
            if not query:
                return json.dumps({"error": "A search query is required."}), False
            result = self.search(query)
        elif operation == "add":
            problem = (args.get("problem") or "").strip()
            solution = (args.get("solution") or "").strip()
            if not problem or not solution:
                return json.dumps({"error": "Both problem and solution are required."}), False
            result = self.add(problem, solution, args.get("tags"))
        elif operation == "update":
            if not args.get("id"):
                return json.dumps({"error": "Record id is required for update."}), False
            result = self.update(
                args["id"],
                args.get("problem"),
                args.get("solution"),
                args.get("tags"),
            )
        else:
            raise ValueError("Unsupported experience operation.")
        return json.dumps(result), False

    def search(self, query: str) -> dict[str, Any]:
        """Return matching records for a case-insensitive word query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")
        records = self._read()
        words = self._query_words(query)
        if not words:
            return {"matches": []}
        matches = []
        for issue in records:
            score = self._match_score(issue, words)
            if score > 0:
                matches.append({**issue, "_score": score})
        matches.sort(key=lambda item: item["_score"], reverse=True)
        return {"matches": matches[:_SEARCH_RESULT_LIMIT]}

    def add(self, problem: str, solution: str, tags: list[str] | None = None) -> dict[str, Any]:
        """Store a verified problem and solution. Updates existing similar record."""
        tags = self._normalize_tags(tags or [])
        problem = self._validate_field("problem", problem)
        solution = self._validate_field("solution", solution)
        with self._memory_lock:
            records = self._read_unlocked()
            duplicate = self._find_duplicate(records, problem)
            now = _utc_now()
            if duplicate:
                duplicate["solution"] = solution
                duplicate["tags"] = self._merge_tags(duplicate.get("tags", []), tags)
                duplicate["last_seen_at"] = now
                duplicate["reuse_count"] = duplicate.get("reuse_count", 0) + 1
                if self._project_name not in duplicate["projects"]:
                    duplicate["projects"].append(self._project_name)
                self._write(records)
                return {"id": duplicate["id"], "updated": True}
            new_id = uuid.uuid4().hex
            record = {
                "id": new_id,
                "problem": problem,
                "solution": solution,
                "tags": tags,
                "projects": [self._project_name],
                "created_at": now,
                "last_seen_at": now,
                "reuse_count": 0,
            }
            records.append(record)
            self._write(records)
            return {"id": new_id, "updated": False}

    def update(
        self,
        record_id: str,
        problem: str | None = None,
        solution: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Correct an existing solution or its tags, bumping usage metadata."""
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("Record id is required.")
        with self._memory_lock:
            records = self._read_unlocked()
            target = self._find_by_id(records, record_id)
            if target is None:
                raise ValueError(f"No record found with id: {record_id}")
            if problem is not None:
                target["problem"] = self._validate_field("problem", problem)
            if solution is not None:
                target["solution"] = self._validate_field("solution", solution)
            if tags is not None:
                target["tags"] = self._normalize_tags(tags)
            now = _utc_now()
            target["last_seen_at"] = now
            target["reuse_count"] = target.get("reuse_count", 0) + 1
            if self._project_name not in target["projects"]:
                target["projects"].append(self._project_name)
            self._write(records)
            return {"id": target["id"], "updated": True}

    # ── persistence ─────────────────────────────────────────────────────

    def _read(self) -> list[dict[str, Any]]:
        # Reads still take the shared lock so the JSON file cannot be torn
        # mid-write by ``_write`` (audit_034).
        with self._memory_lock:
            return self._read_unlocked()

    def _read_unlocked(self) -> list[dict[str, Any]]:
        if not self._memory_file.is_file():
            return []
        try:
            raw = self._memory_file.read_text(encoding="utf-8")
        except OSError as error:
            raise RuntimeError(f"Failed to read memory file: {error}") from error
        if not raw.strip():
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise RuntimeError(
                "Memory file is malformed and has been preserved. "
                "The file was not overwritten for safety."
            )
        if not isinstance(data, dict):
            raise RuntimeError("Memory file root must be a JSON object.")
        issues = data.get("issues")
        if not isinstance(issues, list):
            raise RuntimeError("Memory file 'issues' must be an array.")
        return issues

    def _write(self, records: list[dict[str, Any]]) -> None:
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "issues": records}
        tmp_path = self._memory_file.with_suffix(self._memory_file.suffix + ".tmp")
        with self._memory_lock:
            try:
                tmp_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                tmp_path.replace(self._memory_file)
            except OSError as error:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"Failed to write memory file: {error}") from error

    # ── search helpers ──────────────────────────────────────────────────

    @staticmethod
    def _query_words(query: str) -> list[str]:
        return [w.lower() for w in query.split() if len(w) >= _MIN_WORD_LENGTH]

    @classmethod
    def _match_score(cls, issue: dict, words: list[str]) -> float:
        fields = " ".join(
            [issue.get("problem", "") or "", issue.get("solution", "") or ""]
            + (issue.get("tags") or [])
        ).lower()
        hits = sum(1 for w in words if w in fields)
        return hits / len(words) if words else 0

    # ── duplicate detection ─────────────────────────────────────────────

    @classmethod
    def _find_duplicate(
        cls, records: list[dict], problem: str
    ) -> dict[str, Any] | None:
        """Quick exact-match lookup before falling back to Jaccard similarity."""
        norm = problem.strip().lower()
        # Fast path: exact normalized match.
        for record in records:
            existing = (record.get("problem") or "").strip().lower()
            if existing == norm:
                return record
        # Slow path: Jaccard similarity for near-duplicates.
        for record in records:
            existing = (record.get("problem") or "").strip().lower()
            if not existing:
                continue
            if cls._similarity(norm, existing) >= _DUPLICATE_SIMILARITY_THRESHOLD:
                return record
        return None

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Jaccard similarity on word sets."""
        set_a = set(a.split())
        set_b = set(b.split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    # ── identity ────────────────────────────────────────────────────────

    @staticmethod
    def _find_by_id(
        records: list[dict[str, Any]], record_id: str
    ) -> dict[str, Any] | None:
        for record in records:
            if record.get("id") == record_id:
                return record
        return None

    # ── validation ─────────────────────────────────────────────────────

    def _validate_field(self, field: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} cannot be empty.")
        limit = _MAX_FIELD_LENGTHS.get(field)
        if limit is not None and len(value) > limit:
            raise ValueError(
                f"{field} exceeds {limit} characters ({len(value)} provided)."
            )
        return value.strip()

    def _normalize_tags(self, tags: list[str]) -> list[str]:
        if not isinstance(tags, list):
            raise ValueError("tags must be an array of strings.")
        normalized: list[str] = []
        for tag in tags:
            if not isinstance(tag, str) or not tag.strip():
                raise ValueError("Each tag must be a non-empty string.")
            tag = tag.strip().lower()
            if len(tag) > _MAX_TAG_LENGTH:
                raise ValueError(
                    f"Tag exceeds {_MAX_TAG_LENGTH} characters: {tag[:50]}…"
                )
            if tag not in normalized:
                normalized.append(tag)
        if len(normalized) > _MAX_TAGS:
            raise ValueError(f"Maximum {_MAX_TAGS} unique tags allowed.")
        return sorted(normalized)

    @staticmethod
    def _merge_tags(
        existing: list[str], incoming: list[str]
    ) -> list[str]:
        seen = set(existing)
        for tag in incoming:
            seen.add(tag)
        return sorted(seen)[:_MAX_TAGS]


# ── module helpers ────────────────────────────────────────────────────


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
