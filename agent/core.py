"""Background OpenRouter tool-calling loop for one local CAD task at a time."""

from __future__ import annotations

import hashlib
import json
import platform
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from agent.constraints import ConstraintStore, ModelConstraintValidator
from agent.build_feedback import sanitize_build_result
from agent.images import as_chat_image
from agent.llm_base import create_llm_client, provider_label, sanitize_assistant_message
from agent.prompt import get_system_prompt
from agent.quality.errors import normalize_error
from agent.quality.models import Attempt, EnvironmentInfo, ModelInfo
from agent.quality.store import QualityError, QualityStore
from agent.revisions import RevisionStore
from agent.settings import Settings
from agent.tool_results import failure as tool_failure
from agent.tool_results import success as tool_success
from agent.tool_schemas import TOOL_SCHEMAS
from agent.tools.cad_tool import CadTool
from agent.tools.experience_tool import ExperienceTool
from agent.tools.file_tool import FileTool
from agent.tools.question_tool import QuestionTool, normalize_questions
from agent.tools.question_validator import QuestionValidator
from agent.tools.terminal_tool import TerminalTool


class ProjectTools:
    def __init__(
        self,
        project_dir: Path,
        publish: Callable[[str, dict], None],
        settings: Settings | None = None,
    ) -> None:
        self.project_dir = project_dir
        self.revisions = RevisionStore(
            project_dir,
            retention_count=settings.revision_retention_count if settings else 0,
        )
        self.constraints = ConstraintStore(project_dir)
        # Reconcile on project load: import existing model or recover from crash.
        self.revisions.reconcile()
        self.file = FileTool(project_dir, self.revisions, constraints=self.constraints)
        self.terminal = TerminalTool(project_dir)
        self.cad = CadTool(project_dir, publish, self.revisions)
        self.question = QuestionTool(publish)
        # Experience tool lives at workspace scope, so we need the workspace root.
        # project_dir is <workspace>/<project>; parent yields the workspace.
        self.experience = ExperienceTool(project_dir.parent, project_dir.name)

    def stop(self) -> None:
        self.terminal.stop()
        self.cad.stop()


# Module-level handle to the SSE EventBus history lock so the conversation
# mirror in ``_append_api_message`` serializes with the events the bus
# writes. Wired from ``app.create_app`` after the bus is built; falls back
# to a private per-process lock so tests and offline callers stay safe.
_history_lock_slot: list[threading.Lock] = [threading.Lock()]


def _shared_history_lock() -> threading.Lock:
    """Return the active history lock (configurable by the Flask app)."""
    return _history_lock_slot[0]


class AgentRunner:
    # Per-project cache of the stripped/truncated API history (audit_041).
    _api_history_cache: dict[str, list[dict]] = {}

    def __init__(
        self,
        settings: Settings,
        publish: Callable[[str, dict], None],
        *,
        history_lock: threading.Lock | None = None,
    ) -> None:
        self.settings = settings
        self.publish = publish
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_tools: ProjectTools | None = None
        self._active_project: str | None = None
        self._lock = threading.Lock()
        # Shared with the SSE EventBus so the readable conversation.jsonl mirror
        # serializes with the events it writes; falls back to a private lock
        # when the runner is constructed without a Flask app.
        self._history_lock: threading.Lock = history_lock or threading.Lock()
        self._waiting_questions: dict[str, dict[str, object]] = {}
        self._preview_attempts: dict[tuple[str, str], dict[str, str]] = {}
        self._pending_completions: dict[str, dict[str, str]] = {}
        # Active quality run bookkeeping so preview-time finalization can reach
        # the run record after the agent thread has returned. We only persist
        # the run id; the QualityStore instance is rebuildable from project_dir.
        self._run_ids: dict[str, str] = {}
        self._project_dirs: dict[str, Path] = {}

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def has_active_state_for(self, project: str) -> bool:
        """Return whether the runner has any in-flight activity for ``project``.

        Covers running threads, awaiting-preview gates, and pending questions.
        """
        if self.is_running() and self.active_project() == project:
            return True
        if self.is_awaiting_preview(project):
            return True
        if self.waiting_question(project) is not None:
            return True
        return False

    def active_project(self) -> str | None:
        with self._lock:
            return self._active_project

    def is_awaiting_preview(self, project: str | None = None) -> bool:
        with self._lock:
            return bool(
                self._pending_completions
                if project is None
                else project in self._pending_completions
            )

    def pending_preview_id(self, project: str) -> str | None:
        with self._lock:
            pending = self._pending_completions.get(project)
            return pending["preview_id"] if pending else None

    def confirm_preview(self, project: str, preview_id: str) -> bool:
        completion = None
        with self._lock:
            attempt = self._preview_attempts.get((project, preview_id))
            if not attempt or not self._preview_matches(project, attempt):
                return False
            attempt["status"] = "displayed"
            pending = self._pending_completions.get(project)
            if pending and pending["preview_id"] == preview_id:
                completion = self._pending_completions.pop(project)
        if completion:
            self._complete(project, completion["message"])
        return True

    def fail_preview(self, project: str, preview_id: str, message: str) -> bool:
        error_message = (
            f"Drawing could not be displayed: {message or 'the preview is invalid.'}"
        )
        should_publish = False
        with self._lock:
            attempt = self._preview_attempts.get((project, preview_id))
            if not attempt:
                return False
            attempt.update({"status": "error", "error": error_message})
            pending = self._pending_completions.get(project)
            if pending and pending["preview_id"] == preview_id:
                self._pending_completions.pop(project)
                should_publish = True
        if should_publish:
            self.publish("agent_error", {"project": project, "message": error_message})
            self._finalize_run(
                project,
                "failed",
                {
                    "code": "PREVIEW_LOAD_FAILED",
                    "category": "Delivery",
                    "phase": "delivery",
                    "message": error_message,
                },
            )
        return True

    def waiting_question(self, project: str) -> dict[str, object] | None:
        with self._lock:
            question = self._waiting_questions.get(project)
        if question:
            return question.copy()
        state_path = self.settings.workspace_root / project / ".agent_state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        question = state.get("waiting_question") if isinstance(state, dict) else None
        if not isinstance(question, dict):
            return None
        with self._lock:
            self._waiting_questions[project] = question
        return question.copy()

    def start(
        self,
        project: str,
        message: str,
        image_paths: list[Path] | None = None,
    ) -> bool:
        if self.waiting_question(project):
            return False
        with self._lock:
            return self._start_locked(project, message, image_paths or [])

    def _start_locked(
        self, project: str, message: str, image_paths: list[Path]
    ) -> bool:
        if (self._thread and self._thread.is_alive()) or self._pending_completions:
            return False
        self._stop_event.clear()
        self._active_project = project
        self._thread = threading.Thread(
            target=self._run, args=(project, message, image_paths), daemon=True
        )
        self._thread.start()
        return True

    def answer(self, project: str, answer: str) -> bool:
        question = self.waiting_question(project)
        if not question or not self._validate_answer(question, answer):
            return False
        with self._lock:
            previous_thread = self._thread
        if previous_thread and previous_thread.is_alive():
            previous_thread.join(timeout=2)
        project_dir = self.settings.workspace_root / project
        formatted = self._format_answer(question, answer)
        with self._lock:
            if (self._thread and self._thread.is_alive()) or self._pending_completions:
                return False
            self._log(project_dir, "user", formatted)
            if not self._start_locked(project, formatted, []):
                return False
            self._waiting_questions.pop(project, None)
            (project_dir / ".agent_state.json").unlink(missing_ok=True)
            return True

    def stop(self, project: str | None = None) -> list[str]:
        """Stop agent work and clear pending state.

        A project-scoped stop clears that project's waiting question, pending
        completion, and preview attempts. A projectless stop clears that state
        for every project. Returns the list of affected projects.
        """
        affected: list[str] = []
        with self._lock:
            target_project = project or self._active_project
            stop_active_task = (
                target_project is None or target_project == self._active_project
            )
            if project is None:
                affected = list(
                    dict.fromkeys(
                        list(self._waiting_questions)
                        + list(self._pending_completions)
                        + [key[0] for key in self._preview_attempts]
                    )
                )
                self._waiting_questions.clear()
                self._pending_completions.clear()
                self._preview_attempts.clear()
            else:
                self._waiting_questions.pop(project, None)
                self._pending_completions.pop(project, None)
                for key in [
                    key for key in self._preview_attempts if key[0] == project
                ]:
                    self._preview_attempts.pop(key, None)
                affected = [project]
            if stop_active_task:
                self._stop_event.set()
            if stop_active_task and self._active_tools:
                self._active_tools.stop()
        for cleared in affected:
            (
                self.settings.workspace_root / cleared / ".agent_state.json"
            ).unlink(missing_ok=True)
            self._finalize_run(cleared, "stopped")
        return affected

    def _run(
        self,
        project: str,
        message: str,
        image_paths: list[Path] | None = None,
    ) -> None:
        self.publish(
            "agent_status",
            {
                "project": project,
                "status": "started",
                "message": "Planning CAD task...",
            },
        )
        project_dir = self.settings.workspace_root / project
        tools = ProjectTools(project_dir, self.publish, self.settings)
        with self._lock:
            self._active_tools = tools
        quality: QualityStore | None = None
        run_id: str | None = None
        run_outcome: str | None = None
        run_error: dict | None = None
        try:
            if self.settings.quality_enabled:
                quality = QualityStore(project_dir)
                quality.reconcile()
                run = quality.start_run(
                    project=project,
                    request_message_id=uuid.uuid4().hex,
                    request_sha256=hashlib.sha256(
                        message.encode("utf-8")
                    ).hexdigest(),
                    system_prompt_sha256=hashlib.sha256(
                        get_system_prompt().encode("utf-8")
                    ).hexdigest(),
                    model=ModelInfo(
                        provider=self.settings.llm_provider,
                        name=self.settings.llm_model,
                        reasoning_effort=(
                            self.settings.openai_reasoning_effort
                            if self.settings.llm_provider == "openai"
                            else self.settings.openrouter_reasoning_effort
                        ),
                    ),
                    environment=EnvironmentInfo(
                        python_version=platform.python_version(),
                        build123d_version=self._build123d_version(),
                    ),
                )
                run_id = run.run_id
                with self._lock:
                    self._run_ids[project] = run_id
                    self._project_dirs[project] = tools.project_dir
                self.publish(
                    "quality_run_started",
                    {"project": project, "run_id": run_id},
                )
                # Write a fresh parsed spec for the new run so the next
                # ``cad_build_and_verify`` can pick it up without re-parsing
                # the conversation. Persist the same spec on disk for the
                # runner and in the QualityStore for the API.
                self._write_active_spec(
                    quality, run_id, project_dir, message
                )
            messages = self._context(project_dir, message, image_paths or [])
            preview_id: str | None = None
            cad_error: str | None = None
            cad_fix_required = not self._model_is_built(project_dir)
            any_tool_used = False
            nudged_cad = False
            # Keep one-argument construction compatible with test doubles and older integrations.
            client = create_llm_client(self.settings)
            client.stop_event = self._stop_event
            client.session_id = f"{self.settings.llm_provider}:{project}"
            for _ in range(self.settings.agent_tool_call_limit):
                if self._stop_event.is_set():
                    run_outcome = "stopped"
                    self.publish(
                        "agent_status",
                        {
                            "project": project,
                            "status": "stopped",
                            "message": "Task stopped.",
                        },
                    )
                    return
                message_id = uuid.uuid4().hex
                self.publish(
                    "agent_stream_start", {"project": project, "message_id": message_id}
                )

                def publish_stream(
                    event: dict,
                    current_message_id: str = message_id,
                ) -> None:
                    event_type = event.pop("type")
                    self.publish(
                        f"agent_{event_type}_delta",
                        {"project": project, "message_id": current_message_id, **event},
                    )

                client.stream_callback = publish_stream
                response = client.chat(messages, TOOL_SCHEMAS)
                self._publish_usage(project, getattr(client, "last_usage", None))
                assistant_message = sanitize_assistant_message(
                    response["choices"][0]["message"]
                )
                tool_calls = self._normalize_tool_calls(
                    assistant_message.get("tool_calls")
                )
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                else:
                    assistant_message.pop("tool_calls", None)
                self.publish(
                    "agent_stream_end",
                    {
                        "project": project,
                        "message_id": message_id,
                        "message": assistant_message.get("content") or "",
                    },
                )
                messages.append(assistant_message)
                self._append_api_message(project_dir, assistant_message)
                if not tool_calls:
                    content = assistant_message.get("content") or "Task completed."
                    if not preview_id:
                        if cad_error:
                            run_outcome = "failed"
                            run_error = {
                                "code": "CAD_BUILD_FAILED",
                                "category": "Execution",
                                "phase": "execution",
                                "message": cad_error,
                            }
                            self.publish(
                                "agent_error",
                                {
                                    "project": project,
                                    "message": f"Drawing was not created: {cad_error}",
                                },
                            )
                        elif any_tool_used:
                            if not nudged_cad and (project_dir / "model.py").is_file():
                                nudged_cad = True
                                reminder = {
                                    "role": "user",
                                    "content": (
                                        "model.py exists but it has not been verified. "
                                        "Call cad_build_and_verify now."
                                    ),
                                }
                                messages.append(reminder)
                                self._append_api_message(project_dir, reminder)
                                continue
                            run_outcome = "failed"
                            run_error = {
                                "code": "CAD_BUILD_FAILED",
                                "category": "Execution",
                                "phase": "execution",
                                "message": "the task did not produce a new CAD preview",
                            }
                            self.publish(
                                "agent_error",
                                {
                                    "project": project,
                                    "message": "Drawing was not created: the task did not produce a new CAD preview.",
                                },
                            )
                        else:
                            # No tools were used at all — just a conversation; complete silently.
                            self._complete(project, content)
                        return
                    self._await_preview(project, preview_id, content)
                    return
                any_tool_used = True
                needs_visual_review = False
                processed_call_ids: set[str] = set()
                for call in tool_calls:
                    call_id = call.get("id", "")
                    if self._stop_event.is_set():
                        self._cancel_remaining_tool_calls(
                            project_dir, tool_calls, processed_call_ids, messages
                        )
                        return
                    (
                        preview_id,
                        cad_error,
                        cad_fix_required,
                        needs_critique,
                        waiting,
                    ) = self._process_tool_call(
                        tools,
                        project,
                        project_dir,
                        call,
                        cad_fix_required,
                        preview_id,
                        cad_error,
                        messages,
                        run_id=run_id,
                        quality=quality,
                    )
                    processed_call_ids.add(call_id)
                    if needs_critique:
                        needs_visual_review = True
                    if waiting:
                        try:
                            q_args = json.loads(call["function"]["arguments"] or "{}")
                        except json.JSONDecodeError:
                            q_args = {}
                        questions_list = q_args.get("questions", [])
                        if isinstance(questions_list, list) and questions_list:
                            question_text = "; ".join(
                                q.get("question", "")
                                for q in questions_list
                                if isinstance(q, dict)
                            )
                        else:
                            question_text = q_args.get(
                                "question", "Clarification requested"
                            )
                        self._log(
                            project_dir, "assistant", f"Question: {question_text}"
                        )
                        self.publish(
                            "agent_status",
                            {
                                "project": project,
                                "status": "waiting_for_user",
                                "message": "Waiting for user input.",
                            },
                        )
                        self._cancel_remaining_tool_calls(
                            project_dir, tool_calls, processed_call_ids, messages
                        )
                        if quality is not None and run_id is not None:
                            try:
                                quality.transition_run(run_id, status="waiting_for_user")
                            except QualityError:
                                pass
                        return
                if needs_visual_review and not cad_fix_required:
                    critique = self._verification_context(project, project_dir)
                    messages.append(critique)
                    self._append_api_message(project_dir, critique)
            run_outcome = "failed"
            run_error = {
                "code": "TOOL_CALL_LIMIT",
                "category": "System",
                "phase": "execution",
                "message": (
                    f"Tool-call limit ({self.settings.agent_tool_call_limit}) reached; "
                    "increase agent.tool_call_limit or continue with a narrower request."
                ),
            }
            self.publish(
                "agent_error",
                {
                    "project": project,
                    "message": run_error["message"],
                },
            )
        except Exception as error:  # noqa: BLE001 - Surface all agent failures to the local UI.
            import traceback

            detail = str(error)
            err_type = type(error).__name__
            # Log the full traceback server-side.
            traceback.print_exc()
            if self._stop_event.is_set():
                run_outcome = "stopped"
                self.publish(
                    "agent_status",
                    {
                        "project": project,
                        "status": "stopped",
                        "message": "Task stopped.",
                    },
                )
            else:
                run_outcome = "failed"
                run_error = {
                    "code": "UNKNOWN_ERROR",
                    "category": "System",
                    "phase": "execution",
                    "message": detail,
                }
                self.publish(
                    "agent_error",
                    {
                        "project": project,
                        "message": self._user_error_message(detail, err_type, provider=self.settings.llm_provider),
                    },
                )
        finally:
            with self._lock:
                self._active_tools = None
                if self._active_project == project:
                    self._active_project = None
            if run_id is not None and quality is not None:
                try:
                    current = quality.get_run(run_id)
                except QualityError:
                    current = None
                if current is not None and current.status == "running":
                    # A running run left behind by this thread is either
                    # stopped/failed here, or awaiting preview confirmation —
                    # in which case `_complete`/`confirm_preview` finalizes it.
                    if run_outcome == "stopped" or self._stop_event.is_set():
                        self._finalize_run(project, "stopped")
                    elif run_outcome == "failed":
                        self._finalize_run(project, "failed", run_error)

    def _context(
        self, project_dir: Path, message: str, image_paths: list[Path]
    ) -> list[dict]:
        history = self._load_api_history(project_dir)
        self._append_constraint_context(project_dir, history)
        user_message: dict[str, object] = {"role": "user", "content": message}
        if image_paths:
            user_message = {
                "role": "user",
                "content": [{"type": "text", "text": message}]
                + [as_chat_image(path) for path in image_paths],
            }
        if not history or history[-1] != user_message:
            history.append(user_message)
            self._append_api_message(project_dir, user_message)
        return [{"role": "system", "content": get_system_prompt()}] + history

    @classmethod
    def _append_constraint_context(cls, project_dir: Path, history: list[dict]) -> None:
        """Record constraint changes without mutating the cacheable system prefix."""
        tag = "<runtime_active_constraints>"
        summary = ModelConstraintValidator(ConstraintStore(project_dir)).constraint_summary()
        content = f"{tag}\n{summary or 'None.'}\n</runtime_active_constraints>"
        previous = next(
            (
                item.get("content")
                for item in reversed(history)
                if item.get("role") == "user"
                and isinstance(item.get("content"), str)
                and item["content"].startswith(tag)
            ),
            None,
        )
        if previous == content or (previous is None and not summary):
            return
        context = {"role": "user", "content": content}
        history.append(context)
        cls._append_api_message(project_dir, context)

    @staticmethod
    def _api_history_path(project_dir: Path) -> Path:
        return project_dir / "api_messages.jsonl"

    # Cap the API message history; older turns are truncated on load.
    MAX_API_HISTORY = 100

    @staticmethod
    def _strip_image_parts(item: dict) -> dict:
        """Replace inline image parts with placeholders when loading history.

        Base64 image payloads are only sent for the turn that introduces them;
        on later turns they are replaced with a readable placeholder so the
        full image is not re-sent. Other message metadata is preserved.
        """
        content = item.get("content")
        if item.get("role") != "user" or not isinstance(content, list):
            return item
        parts: list[object] = []
        replaced = False
        image_index = 0
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                replaced = True
                image_index += 1
                name = part.get("filename") or f"reference-{image_index}"
                parts.append({"type": "text", "text": f"[image: {name}]"})
            else:
                parts.append(part)
        if not replaced:
            return item
        cleaned = dict(item)
        cleaned["content"] = parts
        return cleaned

    @staticmethod
    def _is_constraint_message(item: dict) -> bool:
        content = item.get("content")
        return (
            item.get("role") == "user"
            and isinstance(content, str)
            and content.startswith("<runtime_active_constraints>")
        )

    @classmethod
    def _truncate_history(cls, history: list[dict]) -> list[dict]:
        """Keep at most MAX_API_HISTORY messages, preserving the newest constraint context."""
        if len(history) <= cls.MAX_API_HISTORY:
            return history
        tail_start = len(history) - cls.MAX_API_HISTORY
        newest_constraint = next(
            (
                index
                for index in range(len(history) - 1, -1, -1)
                if cls._is_constraint_message(history[index])
            ),
            None,
        )
        indices = list(range(tail_start, len(history)))
        if newest_constraint is not None and newest_constraint < tail_start:
            indices[0] = newest_constraint
        return [history[index] for index in sorted(indices)]

    @classmethod
    def _load_api_history(cls, project_dir: Path) -> list[dict]:
        """Load the append-only, protocol-level history for cache-stable turns.

        Inline image payloads are stripped to placeholders and the history is
        truncated to MAX_API_HISTORY messages so old turns are not re-sent.

        Result is cached in a class-level dict keyed by project_dir, invalidated
        whenever :meth:`_append_api_message` is called (audit_041).
        """
        cache_key = str(project_dir)
        cached = cls._api_history_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        history: list[dict] = []
        history_path = cls._api_history_path(project_dir)
        if history_path.exists():
            for line in history_path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and item.get("role") in {
                    "user",
                    "assistant",
                    "tool",
                }:
                    history.append(item)
        else:
            # Projects created by earlier releases retain their readable transcript.
            # Convert it only once into the protocol history; never rewrite it.
            log_path = project_dir / "conversation.jsonl"
            if log_path.exists():
                for line in log_path.read_text(encoding="utf-8").splitlines():
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if item.get("role") in {"user", "assistant"} and isinstance(
                        item.get("content"), str
                    ):
                        history.append({"role": item["role"], "content": item["content"]})
            for item in history:
                with cls._api_history_path(project_dir).open("a", encoding="utf-8") as log:
                    log.write(
                        json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
        history = [cls._strip_image_parts(item) for item in history]
        history = cls._truncate_history(history)
        cls._api_history_cache[cache_key] = list(history)
        return history

    @classmethod
    def _append_api_message(cls, project_dir: Path, message: dict) -> None:
        """Append a message to both the protocol history and the conversation log.

        Single sink for both writes (audit_040); also invalidates the in-memory
        history cache (audit_041) so the next load re-reads the appended turn.
        """
        cls._api_history_cache.pop(str(project_dir), None)
        with cls._api_history_path(project_dir).open("a", encoding="utf-8") as log:
            log.write(
                json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        # Mirror to the readable conversation log so the SSE transcript stays
        # in sync without a second call site. Acquire the same lock the SSE
        # EventBus uses for its event writes so the two streams cannot
        # interleave on the same project (audit_078 follow-up).
        role = message.get("role", "assistant")
        readable = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "content": message.get("content"),
        }
        with _shared_history_lock():
            with (project_dir / "conversation.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(readable, ensure_ascii=False) + "\n")

    def _publish_usage(self, project: str, usage: dict | None) -> None:
        if not usage:
            return
        details = usage.get("prompt_tokens_details")
        cached_tokens = (
            details.get("cached_tokens", 0) if isinstance(details, dict) else 0
        )
        self.publish(
            "agent_usage",
            {
                "project": project,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cached_tokens": cached_tokens,
                "cache_write_tokens": details.get("cache_write_tokens", 0)
                if isinstance(details, dict)
                else 0,
            },
        )

    def _execute(
        self,
        tools: ProjectTools,
        project: str,
        name: str,
        args: dict,
        call_id: str = "",
    ) -> tuple[object, bool]:
        # Tool argument validation now lives in each tool's execute()
        # method (audit_028). The JSON schemas emitted to the model still
        # declare the required keys for the LLM's benefit.

        if name == "cad_build_and_verify":
            payload = tools.cad.build_and_verify()
            # Persist the complete payload to disk; only the compact envelope
            # goes back to the LLM so it cannot rediscover the design from raw
            # metrics, evidence, or duplicated requirements.
            return payload, False
        if name.startswith("file_"):
            tool = tools.file.with_call_id(call_id) if call_id else tools.file
            operation = name.removeprefix("file_")
            if operation == "read":
                return tool.read(args["filename"]), False
            if operation == "write":
                return tool.write(args["filename"], args["content"]), False
            if operation == "replace":
                return tool.replace(args["filename"], args["old"], args["new"]), False
            return (
                tool.regex_replace(
                    args["filename"],
                    args["pattern"],
                    args["replacement"],
                    args.get("count", 1),
                ),
                False,
            )
        if name == "terminal_run":
            return tools.terminal.run(
                args["arguments"], args.get("timeout_seconds", 30)
            ), False
        if name == "terminal_check":
            return tools.terminal.check(
                args["arguments"], args.get("timeout_seconds", 15)
            ), False
        if name == "experience_search":
            return tools.experience.search(args["query"]), False
        if name == "experience_add":
            return tools.experience.add(
                args["problem"], args["solution"], args.get("tags")
            ), False
        if name == "experience_update":
            return tools.experience.update(
                args["id"],
                args.get("problem"),
                args.get("solution"),
                args.get("tags"),
            ), False

        # Legacy names remain dispatchable for append-only protocol histories and
        # older integrations, but are intentionally absent from TOOL_SCHEMAS.
        if name == "screenshot":
            view = args.get("view", "isometric")
            proximity = args.get("proximity", 1.0)
            return json.dumps(tools.cad.screenshot(view, proximity)), False
        if name == "question":
            # execute() validates, normalizes, and publishes the questions;
            # the normalized list is also needed for the persisted state.
            result, _waiting = tools.question.execute(args, project=project)
            questions = normalize_questions(args)
            title = args.get("title", "")
            question_state = {
                "title": title.strip() if isinstance(title, str) else "",
                "questions": questions,
            }
            state_path = tools.project_dir / ".agent_state.json"
            temporary_state = tools.project_dir / ".agent_state.json.tmp"
            temporary_state.write_text(
                json.dumps(
                    {"status": "WAITING_FOR_USER", "waiting_question": question_state}
                ),
                encoding="utf-8",
            )
            temporary_state.replace(state_path)
            with self._lock:
                self._waiting_questions[project] = question_state
            return result, True
        if name == "file" and call_id:
            # Bind the tool-call ID so revision manifests can correlate with
            # the protocol log without FileTool reading conversation logs.
            file_tool = tools.file.with_call_id(call_id)
            return file_tool.execute(args)
        tool = getattr(tools, name)
        return tool.execute(args)

    @staticmethod
    def _normalize_tool_calls(raw_calls: object) -> list[dict]:
        """Ensure persisted tool calls remain valid protocol messages."""
        if not isinstance(raw_calls, list):
            return []
        normalized: list[dict] = []
        for raw_call in raw_calls:
            call = raw_call if isinstance(raw_call, dict) else {}
            function = call.get("function")
            function = function if isinstance(function, dict) else {}
            name = function.get("name")
            arguments = function.get("arguments")
            normalized.append(
                {
                    "id": str(call.get("id") or f"invalid-{uuid.uuid4().hex}"),
                    "type": "function",
                    "function": {
                        "name": name
                        if isinstance(name, str) and name
                        else "unknown_tool",
                        "arguments": arguments if isinstance(arguments, str) else "{}",
                    },
                }
            )
        return normalized

    @staticmethod
    def _is_model_mutation(name: str, arguments: dict) -> bool:
        if name in {"file_write", "file_replace", "file_regex_replace"}:
            return arguments.get("filename") == "model.py"
        return (
            name == "file"
            and arguments.get("filename") == "model.py"
            and arguments.get("operation") in {"write", "replace", "regex_replace"}
        )

    @staticmethod
    def _is_cad_build(name: str, arguments: dict) -> bool:
        return name == "cad_build_and_verify" or (
            name == "cad" and arguments.get("operation") == "run"
        )

    def _process_tool_call(
        self,
        tools: ProjectTools,
        project: str,
        project_dir: Path,
        call: dict,
        cad_fix_required: bool,
        prev_preview_id: str | None,
        cad_error: str | None,
        messages: list[dict],
        run_id: str | None = None,
        quality: QualityStore | None = None,
    ) -> tuple[str | None, str | None, bool, bool, bool]:
        """Execute a single tool call, update state, and feed visual results back.

        When a quality run is active, every CAD build creates one immutable
        attempt record: started before execution and completed with the tool
        result envelope (success or classified failure).

        Returns:
            (preview_id, cad_error, cad_fix_required, needs_visual_review, waiting)
        """
        call = call if isinstance(call, dict) else {}
        call_id = str(call.get("id") or f"invalid-{uuid.uuid4().hex}")
        function = call.get("function")
        name = function.get("name") if isinstance(function, dict) else ""
        if not isinstance(name, str) or not name:
            name = "unknown_tool"
        tool_succeeded = False
        preview_id = prev_preview_id
        waiting = False
        arguments: dict = {}
        attempt: Attempt | None = None
        if run_id is not None and quality is not None and name in (
            "cad",
            "cad_build_and_verify",
        ):
            try:
                probe = json.loads(function.get("arguments") or "{}")
                probe = probe if isinstance(probe, dict) else {}
            except json.JSONDecodeError:
                probe = {}
            if self._is_cad_build(name, probe):
                try:
                    head = tools.revisions.head()
                    revision_id = head.id if head is not None else None
                except Exception:  # noqa: BLE001 - Revision linkage is best-effort.
                    revision_id = None
                attempt = quality.start_attempt(
                    run_id,
                    revision_id=revision_id,
                    source_sha256=self._model_digest(project_dir),
                    tool_call_id=call_id,
                    phase="execution",
                )
        try:
            argument_text = (
                function.get("arguments") if isinstance(function, dict) else ""
            )
            arguments = json.loads(argument_text or "{}")
            if not isinstance(arguments, dict):
                raise TypeError("Tool arguments must be a JSON object.")
            tool_event = {
                "project": project,
                "call_id": call_id,
                "tool": name,
                "arguments": arguments,
            }
            self.publish("tool_status", {**tool_event, "status": "running"})
            if (
                name == "cad"
                and arguments.get("operation") == "inspect"
                and cad_fix_required
            ):
                raise RuntimeError(
                    "cad.inspect was skipped because model.py has not built successfully. "
                    "Run cad.run after fixing model.py."
                )
            if self._is_cad_build(name, arguments):
                preview_id = None
            raw_result, waiting = self._execute(
                tools, project, name, arguments, call_id
            )
            result = tool_success(name, raw_result)
            tool_succeeded = True
            if self._is_model_mutation(name, arguments):
                preview_id = None
                cad_error = None
                cad_fix_required = True
                self.publish("revision_updated", {"project": project})
            if self._is_cad_build(name, arguments):
                preview_id = self._register_preview(project, project_dir)
                cad_error = None
                cad_fix_required = False
                if attempt is not None:
                    metrics = raw_result if isinstance(raw_result, dict) else None
                    if isinstance(metrics, dict) and isinstance(
                        metrics.get("metrics"), dict
                    ):
                        metrics = metrics["metrics"]
                    attempt = quality.complete_attempt(
                        attempt.run_id,
                        attempt.attempt_id,
                        status="succeeded",
                        phase="kernel",
                        metrics=metrics,
                        artifact_paths=self._attempt_artifact_paths(project_dir),
                    )
                    # Persist validation results from the build envelope next
                    # to the attempt so the API and UI can show per-requirement
                    # outcomes. The LLM never sees this raw payload.
                    if isinstance(raw_result, dict):
                        validation_items = raw_result.get("validation") or []
                        if validation_items and quality is not None:
                            persisted = 0
                            for item in validation_items:
                                if not isinstance(item, dict):
                                    continue
                                vr = self._coerce_validation_result(item, attempt.attempt_id)
                                if vr is None:
                                    continue
                                try:
                                    quality.save_validation(vr, run_id=attempt.run_id)
                                    persisted += 1
                                except QualityError:
                                    pass
                            if persisted:
                                quality.append_event(
                                    attempt.run_id,
                                    "validation_recorded",
                                    {
                                        "attempt_id": attempt.attempt_id,
                                        "count": persisted,
                                    },
                                )
                    self.publish(
                        "quality_attempt_completed",
                        {
                            "project": project,
                            "run_id": attempt.run_id,
                            "attempt_id": attempt.attempt_id,
                            "status": "succeeded",
                            "phase": attempt.phase,
                        },
                    )
                # Replace the LLM-facing tool result with the compact envelope
                # defined in agent/build_feedback.py so the model does not
                # rediscover the design from raw metrics, evidence, or
                # duplicated requirement dumps.
                envelope = sanitize_build_result(raw_result)
                result = json.dumps(
                    {"ok": True, "tool": name, "data": envelope},
                    ensure_ascii=False,
                )
                self.publish(
                    "preview_updated",
                    {"project": project, "preview_id": preview_id},
                )
            self.publish(
                "tool_status",
                {**tool_event, "status": "completed", "result": result},
            )
        except Exception as error:  # noqa: BLE001 - Tool errors are useful LLM context.
            result, waiting = tool_failure(name, error), False
            if self._is_cad_build(name, arguments):
                cad_error = str(error)
                cad_fix_required = True
                preview_id = None
            if attempt is not None:
                attempt_error = normalize_error(
                    json.loads(result).get("error")
                )
                attempt = quality.complete_attempt(
                    attempt.run_id,
                    attempt.attempt_id,
                    status="failed",
                    phase=attempt_error["phase"],
                    error=attempt_error,
                )
                self.publish(
                    "quality_attempt_completed",
                    {
                        "project": project,
                        "run_id": attempt.run_id,
                        "attempt_id": attempt.attempt_id,
                        "status": "failed",
                        "phase": attempt.phase,
                        "error": attempt.error,
                    },
                )
            self.publish(
                "tool_status",
                {
                    "project": project,
                    "call_id": call_id,
                    "tool": name,
                    "arguments": arguments,
                    "status": "error",
                    "result": result,
                },
            )
        messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
        self._append_api_message(project_dir, messages[-1])
        self._log(
            project_dir,
            "tool",
            {
                "call_id": call_id,
                "name": name,
                "operation": arguments.get("operation"),
                "result": result,
            },
        )
        # Post-processing hooks: visual feed and verification trigger.
        needs_visual_review = self._post_process_tool_result(
            project_dir,
            name,
            arguments,
            tool_succeeded,
            waiting,
            cad_fix_required,
            messages,
        )
        return preview_id, cad_error, cad_fix_required, needs_visual_review, waiting

    @classmethod
    def _cancel_remaining_tool_calls(
        cls,
        project_dir: Path,
        tool_calls: list[dict],
        processed_call_ids: set[str],
        messages: list[dict],
    ) -> None:
        """Write cancelled tool-result entries for unprocessed calls.

        Required by the OpenAI chat-completions protocol: every tool call in
        an assistant message must have a matching tool result before the next
        assistant turn.  Without these entries the API will reject the
        conversation.
        """
        cancelled = json.dumps({"error": "Tool call cancelled (question or stop)."})
        for tc in tool_calls:
            cid = tc.get("id", "")
            if cid and cid not in processed_call_ids:
                entry = {"role": "tool", "tool_call_id": cid, "content": cancelled}
                messages.append(entry)
                cls._append_api_message(project_dir, entry)

    @staticmethod
    def _attempt_artifact_paths(project_dir: Path) -> dict[str, Path]:
        """Collect the project artifacts produced by a successful build."""
        paths: dict[str, Path] = {}
        for name, relative in (
            ("source", "model.py"),
            ("preview", "preview.stl"),
        ):
            path = project_dir / relative
            if path.is_file() and path.stat().st_size > 0:
                paths[name] = path
        render = project_dir / "render.png"
        if render.is_file() and render.stat().st_size > 0:
            paths["render"] = render
        return paths

    @staticmethod
    def _post_process_tool_result(
        project_dir: Path,
        name: str,
        arguments: dict,
        tool_succeeded: bool,
        waiting: bool,
        cad_fix_required: bool,
        messages: list[dict],
    ) -> bool:
        """Run post-tool-call hooks: screenshot visual feed and self-critique trigger.

        Returns True if a self-critique should follow this tool call.
        """
        # Feed screenshot result back as a user message with visual content.
        if tool_succeeded and name == "screenshot":
            ss_path = project_dir / "screenshot.png"
            if ss_path.is_file():
                ss_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Screenshot of the 3D preview from the {arguments.get('view', 'current')} view.",
                        },
                        as_chat_image(ss_path),
                    ],
                }
                messages.append(ss_msg)
                AgentRunner._append_api_message(project_dir, ss_msg)
        # The unified build already produced metrics and a render. Append its
        # visual verification context only after every parallel tool result.
        return (
            tool_succeeded
            and AgentRunner._is_cad_build(name, arguments)
            and not waiting
            and not cad_fix_required
        )

    def deliver_screenshot(
        self,
        project: str,
        request_id: str,
        image_base64: str = "",
        error: str = "",
    ) -> bool:
        with self._lock:
            if self._active_project != project or not self._active_tools:
                return False
            try:
                return self._active_tools.cad.receive_screenshot(
                    request_id, image_base64, error
                )
            except (TypeError, ValueError):
                return False

    @staticmethod
    def _user_error_message(detail: str, err_type: str, *, provider: str = "openrouter") -> str:
        """Map technical error messages to user-friendly messages."""
        lower = detail.lower()
        provider_name = provider_label(provider)
        key_url = (
            "https://platform.openai.com/api-keys"
            if provider == "openai"
            else "https://openrouter.ai/keys"
        )

        if "401" in detail or "unauthorized" in lower or "invalid api key" in lower:
            return f"Invalid {provider_name} API key. Check your key at {key_url}."
        if "429" in detail or "rate limit" in lower:
            return f"{provider_name} rate limit reached. Wait a moment and try again."
        if "model" in lower and (
            "not found" in lower or "invalid" in lower or "not available" in lower
        ):
            return f"The configured model is not available: {detail}"
        if (
            "bubblewrap" in lower
            or "bwrap" in lower
            or "sandbox" in lower
            or "seccomp" in lower
        ):
            return "CAD sandbox failed. Ensure bubblewrap and libseccomp2 are installed (sudo apt install bubblewrap libseccomp2)."
        if ("openrouter" in lower or "openai" in lower) and (
            "timeout" in lower or "timed out" in lower or "connection" in lower
        ):
            return f"Connection to {provider_name} timed out. Check your internet connection."
        if "timeout" in lower or "timed out" in lower:
            return "CAD code execution timed out. Try simplifying the design or increasing the timeout."
        if "export" in lower and ("step" in lower or "stl" in lower):
            return f"Export failed: {detail}"
        if "permission" in lower or "access denied" in lower or "not writable" in lower:
            return f"Workspace permission error: {detail}"
        if "cancelled" in lower or "stop" in lower:
            return "Task was cancelled."

        # Fallback: return the original detail (it will be logged fully server-side).
        return detail


    def _register_preview(self, project: str, project_dir: Path) -> str:
        preview_path = project_dir / "preview.stl"
        if not preview_path.is_file() or preview_path.stat().st_size == 0:
            raise RuntimeError("CAD execution did not save a usable preview.")
        preview_id = uuid.uuid4().hex
        digest = hashlib.sha256(preview_path.read_bytes()).hexdigest()
        with self._lock:
            self._preview_attempts[(project, preview_id)] = {
                "digest": digest,
                "model_digest": self._model_digest(project_dir) or "",
                "status": "loading",
            }
        return preview_id

    def _preview_matches(self, project: str, attempt: dict[str, str]) -> bool:
        preview_path = self.settings.workspace_root / project / "preview.stl"
        project_dir = preview_path.parent
        try:
            return (
                preview_path.is_file()
                and preview_path.stat().st_size > 0
                and hashlib.sha256(preview_path.read_bytes()).hexdigest()
                == attempt["digest"]
                and (self._model_digest(project_dir) or "") == attempt["model_digest"]
            )
        except (KeyError, OSError):
            return False

    @staticmethod
    def _model_digest(project_dir: Path) -> str | None:
        model_path = project_dir / "model.py"
        try:
            return (
                hashlib.sha256(model_path.read_bytes()).hexdigest()
                if model_path.is_file()
                else None
            )
        except OSError:
            return None

    @classmethod
    def _model_is_built(cls, project_dir: Path) -> bool:
        model_digest = cls._model_digest(project_dir)
        if model_digest is None:
            return True
        try:
            cached = json.loads(
                (project_dir / ".cad_metrics.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return False
        return isinstance(cached, dict) and cached.get("model_sha256") == model_digest

    def _await_preview(self, project: str, preview_id: str, message: str) -> None:
        completion = None
        error_message = None
        with self._lock:
            attempt = self._preview_attempts.get((project, preview_id))
            if not attempt or not self._preview_matches(project, attempt):
                error_message = "Drawing was not saved correctly; the preview file is missing or changed."
            elif attempt["status"] == "displayed":
                completion = message
            elif attempt["status"] == "error":
                error_message = attempt["error"]
            else:
                self._pending_completions[project] = {
                    "preview_id": preview_id,
                    "message": message,
                }
        if completion:
            self._complete(project, completion)
        elif error_message:
            self.publish("agent_error", {"project": project, "message": error_message})
        else:
            self.publish(
                "agent_status",
                {
                    "project": project,
                    "status": "rendering",
                    "message": "Waiting for the drawing to become visible...",
                },
            )

    def _complete(self, project: str, message: str) -> None:
        project_dir = self.settings.workspace_root / project
        self._log(project_dir, "assistant", message)
        self.publish("agent_message", {"project": project, "message": message})
        self._finalize_run(project, "completed")

    def _finalize_run(
        self, project: str, status: str, error: dict | None = None
    ) -> None:
        """Mark the active run terminal and publish the completion event."""
        with self._lock:
            run_id = self._run_ids.get(project)
            project_dir = self._project_dirs.get(project)
        if run_id is None or project_dir is None:
            return
        # QualityStore is reconstructable from project_dir; rebuild lazily
        # so a stop signal arriving after thread exit can still finalize.
        quality = QualityStore(project_dir)
        try:
            run = quality.complete_run(run_id, status=status, error=error)
        except QualityError:
            run = None  # Already terminal or corrupted; never block task progress.
        if run is not None and run.status == status:
            self.publish(
                "quality_run_completed",
                {"project": project, "run_id": run_id, "status": run.status},
            )
        with self._lock:
            self._run_ids.pop(project, None)
            self._project_dirs.pop(project, None)

    @staticmethod
    def _build123d_version() -> str | None:
        try:
            from importlib.metadata import version

            return version("build123d")
        except Exception:  # noqa: BLE001 - Version lookup is best-effort.
            return None

    @staticmethod
    def _validate_answer(question: dict[str, object], answer: str) -> bool:
        return QuestionValidator.validate(question, answer)

    @staticmethod
    def _format_answer(question: dict[str, object], answer: str) -> str:
        """Format the user's answer as readable context for the LLM."""
        questions = question.get("questions")
        if isinstance(questions, list) and questions:
            try:
                answers = json.loads(answer)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(answers, dict):
                    lines = ["User answers:"]
                    for q in questions:
                        qid = q.get("id", "")
                        qtext = q.get("question", qid)
                        value = answers.get(qid, "")
                        if value:
                            lines.append(f"- {qtext}: {value}")
                    return "\n".join(lines)
        return answer

    @staticmethod
    def _log(project_dir: Path, role: str, content: object) -> None:
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "content": content,
        }
        with (project_dir / "conversation.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_active_spec(quality, run_id, project_dir, message):
        """Parse the request and persist the spec for both runner and API."""
        from agent.quality.specification import parse_request

        try:
            spec = parse_request(run_id=run_id, request_text=message or "")
        except Exception:  # noqa: BLE001 - spec failures must never block the run
            return None
        # Persist on disk for the runner.
        try:
            (project_dir / ".cad_spec.json").write_text(
                json.dumps(spec.to_dict(), ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass
        # Persist in the QualityStore for the API.
        try:
            quality.save_spec(spec)
        except QualityError:
            pass
        return spec

    @staticmethod
    def _coerce_validation_result(item: dict, attempt_id: str):
        """Turn a verifier dict (from the runner artifact) into a ValidationResult."""
        from agent.quality.models import ValidationResult, _utc_now, new_validation_id

        try:
            tolerance = item.get("tolerance")
            evidence = tuple(
                str(value)
                for value in (item.get("evidence") or ())
                if isinstance(value, str)
            )
            expected = item.get("expected")
            observed = item.get("observed")
            return ValidationResult(
                validation_id=new_validation_id(),
                attempt_id=attempt_id,
                requirement_id=str(item.get("requirement_id") or ""),
                verifier=str(item.get("verifier") or ""),
                status=str(item.get("status") or "unclear"),
                severity=str(item.get("severity") or "blocking"),
                confidence=float(item.get("confidence", 1.0) or 1.0),
                expected=dict(expected) if isinstance(expected, dict) else {},
                observed=dict(observed) if isinstance(observed, dict) else {},
                tolerance=float(tolerance)
                if isinstance(tolerance, (int, float))
                else None,
                evidence=evidence,
                message=str(item.get("message") or "")[:4000],
                created_at=str(item.get("created_at") or _utc_now()),
            )
        except Exception:  # noqa: BLE001 - malformed verifier rows must never break the build
            return None

    def _verification_context(self, project: str, project_dir: Path) -> dict:
        self.publish(
            "agent_status",
            {
                "project": project,
                "status": "reviewing",
                "message": "Reviewing CAD preview...",
            },
        )
        render_path = project_dir / "render.png"
        # The spec-driven envelope tells the agent exactly what to fix; the
        # image is only for sanity, not for re-deriving the design.
        envelope_path = project_dir / ".cad_validation.json"
        base_text = (
            "Review the image and verified geometry from cad_build_and_verify against "
            "the user's request. Fix model.py if anything is missing, misaligned, or implausible."
        )
        if envelope_path.is_file():
            try:
                import json as _json

                payload = _json.loads(envelope_path.read_text(encoding="utf-8"))
                actionable = [
                    result
                    for result in (payload.get("results") or [])
                    if isinstance(result, dict)
                    and result.get("status") in {"failed", "unclear", "not_implemented"}
                ]
                if actionable:
                    summary = "\n".join(
                        "- [{rid}] {status}: {msg}".format(
                            rid=item.get("requirement_id", ""),
                            status=item.get("status", ""),
                            msg=(item.get("message") or "")[:160],
                        )
                        for item in actionable[:5]
                    )
                    text = base_text + "\n\nParsed spec findings:\n" + summary
                    if render_path.is_file():
                        return {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                                as_chat_image(render_path),
                            ],
                        }
                    return {"role": "user", "content": text}
            except (OSError, ValueError):
                pass
        text = base_text
        if render_path.is_file():
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    as_chat_image(render_path),
                ],
            }
        return {"role": "user", "content": text}
