"""Background tool-calling loop for local CAD tasks.

The runner allows one active request per project (and a bounded number of
projects concurrently).  Each run's worker thread streams the model's
chat-completions response, dispatches tool calls, and re-prompts the model
with the tool results until it returns a final message or the configured
tool-call budget is exhausted.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

from agent.activity_log import ActivityLogger, get_logger
from agent.activity_log import is_enabled as activity_logging_enabled
from agent.context_info import resolve_context_limit
from agent.conversation import ConversationStore
from agent.converter import export_project_model
from agent.dispatcher import (
    cancel_remaining_tool_calls,
    group_tool_calls_into_waves,
    is_cad_build,
    is_model_mutation,
    normalize_tool_calls,
    process_tool_call,
)
from agent.images import as_chat_image
from agent.io import atomic_write_json
from agent.llm_base import (
    CompletionTruncatedError,
    RequestCancelled,
    create_llm_client,
    ensure_closed_code_fences,
    extract_text_tool_calls,
    extract_usage_summary,
    provider_label,
    sanitize_assistant_message,
    strip_encrypted_reasoning,
)
from agent.locks import project_history_lock
from agent.print_advisor import (
    THREE_MF_COMMAND_MARKER,
    extract_json_block,
    parse_nozzle_spec,
    sanitize_print_profile,
    validate_profile_universal,
)
from agent.prompt import (
    NUDGE_CAD_FIX_REQUIRED,
    NUDGE_CAD_WARNING_FIX,
    NUDGE_REASONING_ONLY_PENDING_CAD,
    NUDGE_REASONING_ONLY_READY,
    NUDGE_TRUNCATED_REASONING_PENDING_CAD,
    NUDGE_TRUNCATED_REASONING_READY,
    format_budget_warning_nudge,
    format_calculation_limit_nudge,
    format_project_state,
    format_truncated_tool_call_nudge,
    format_unverified_model_nudge,
    get_system_prompt,
)
from agent.revisions import (
    MODEL_FILENAME,
    RevisionStore,
    compute_model_sha256,
    model_is_built,
)
from agent.settings import (
    Settings,
    with_model_selection,
    with_reasoning_effort,
    with_user_preferences,
)
from agent.tool_schemas import TOOL_SCHEMAS
from agent.tools.cad_tool import CadTool
from agent.tools.file_tool import FileTool
from agent.tools.image_tool import ImageTool
from agent.tools.python_tool import PythonTool
from agent.tools.question_tool import QuestionTool
from agent.tools.question_validator import QuestionValidator

# Circuit-breaker thresholds for repeated ``cad_build`` failures.
# ``_BUILD_FAILURE_TOTAL_MAX`` stops the agent after a broad burst of mixed
# errors; ``_BUILD_FAILURE_PER_SIGNATURE_MAX`` stops it earlier when the
# same logical error keeps repeating (useful for catching repair loops).
# Both values flow through :meth:`AgentRunner._build_failure_exhausted` so
# tests can target the predicate directly without re-rolling the literals.
_BUILD_FAILURE_TOTAL_MAX = 7
_BUILD_FAILURE_PER_SIGNATURE_MAX = 4
_MAX_TOTAL_TRUNCATION_CONTINUATIONS = 8

# Run-scoped artifact directories. ``AgentRunner.start`` takes ownership of
# uploaded inputs by moving them into a per-run directory under the agent
# state directory (``<project>/.cad-agent/``), keyed by ``run_id``. The
# directory is removed in ``_run``'s ``finally`` block so the runner is
# the sole owner of its artifacts throughout the run lifecycle.
_CAD_AGENT_DIRNAME = ".cad-agent"
_RUNS_DIRNAME = "runs"
_RUN_INPUTS_DIRNAME = "inputs"

_LOG = logging.getLogger(__name__)


class ProjectTools:
    """Per-request bundle of tool instances for one project."""

    def __init__(
        self,
        project_dir: Path,
        publish: Callable[[str, dict], None],
        settings: Settings | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.project_dir = project_dir
        self.revisions = RevisionStore(
            project_dir,
            retention_count=settings.revision_retention_count if settings else 0,
        )
        # Reconcile on project load: import existing model or recover from crash.
        self.revisions.reconcile()
        self.file = FileTool(project_dir, self.revisions)
        self.cad = CadTool(
            project_dir,
            publish,
            self.revisions,
            review_render_workers=(settings.review_render_workers if settings else 4),
            review_required_views=(settings.review_required_views if settings else 8),
            user_printer=(settings.user_printer if settings else ""),
            user_nozzle=(settings.user_nozzle if settings else ""),
            user_filament=(settings.user_filament if settings else "PLA"),
            stop_event=stop_event,
        )
        self.image = ImageTool(project_dir, publish)
        self.question = QuestionTool(publish)
        self.python = PythonTool(project_dir, publish, stop_event=stop_event)

    def stop(self) -> None:
        self.cad.stop()
        self.python.stop()


def _synthetic_user(content: str) -> dict[str, object]:
    """Build a ``role: user`` message flagged as agent-generated.

    The agent loop occasionally appends a ``role: user`` reminder to nudge the
    LLM (e.g. "Call cad_build now.") without losing the
    user-role framing the provider expects. The ``synthetic`` flag is the
    durable marker for that provenance:

    - Persisted in ``conversation.jsonl`` alongside every other entry so the
      prompt prefix — including the cache-stable seed of provider-side
      request caches — remains stable across turns.
    - Filtered out by ``app.py:project_history`` (drops ``synthetic: true``
      items) so the History view never surfaces internal nudges to the user;
      ``ConversationStore.load`` independently filters by role for prompt
      construction.
    """
    return {"role": "user", "content": content, "synthetic": True}


def _trailing_tool_batch(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the newest contiguous ``role: tool`` results, newest first.

    Trailing synthetic nudges are skipped so a parallel batch such as
    ``[cad_build, python]`` is inspected as a whole even after a reminder
    was appended behind it.
    """
    batch: list[dict[str, Any]] = []
    for msg in reversed(messages):
        role = msg.get("role")
        if role == "tool":
            batch.append(msg)
        elif batch or not (role == "user" and msg.get("synthetic")):
            break
    return batch


def _tool_payload(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Parse the JSON envelope of a tool result (plain or multimodal content)."""
    content = msg.get("content")
    text: Any = content
    if isinstance(content, list):
        text = next(
            (
                part.get("text")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ),
            None,
        )
    if not isinstance(text, str) or not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _set_tool_payload(msg: dict[str, Any], payload: dict[str, Any]) -> None:
    """Write ``payload`` back into the text slot read by :func:`_tool_payload`."""
    text = json.dumps(payload, ensure_ascii=False)
    content = msg.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = text
                return
    msg["content"] = text


# ---------------------------------------------------------------------------
# Agent-loop decomposition: per-iteration helpers and lifecycle state.
# ---------------------------------------------------------------------------


class _TurnOutcome(Enum):
    """Result of one iteration of the agent tool-loop.

    Variants map to the control flow originally inlined in ``_run``:

    * ``CONTINUE`` — the model returned a synthetic reminder; loop back.
    * ``STOPPED`` / ``PARKED_QUESTION`` — user pressed stop or a tool
      parked the run; ``_run`` publishes the diagnostic and exits.
    * ``COMPLETED`` — terminal assistant turn.
    * ``BUILD_FAILURE_EXHAUSTED`` / ``DRAWING_NOT_CREATED`` /
      ``FINAL_VERIFICATION_MISSING`` — failed gates; terminal.
    """

    CONTINUE = "continue"
    STOPPED = "stopped"
    COMPLETED = "completed"
    PARKED_QUESTION = "parked_question"
    BUILD_FAILURE_EXHAUSTED = "build_failure_exhausted"
    DRAWING_NOT_CREATED = "drawing_not_created"
    FINAL_VERIFICATION_MISSING = "final_verification_missing"


from agent.tools.cad_verifier import PHYSICAL_WARNING_CODES


@dataclass
class _RunState:
    """Mutable per-run state shared between ``_run`` and ``_run_turn``."""

    project_dir: Path
    messages: list[dict[str, Any]]
    tools: ProjectTools
    client: Any
    activity_logger: ActivityLogger | None
    run_id: str
    project: str
    preview_id: str | None = None
    cad_error: str | None = None
    cad_fix_required: bool = True
    cad_warning_fix_required: bool = False
    warning_nudge_count: int = 0
    any_tool_used: bool = False
    model_mutated: bool = False
    nudged_cad: bool = False
    nudged_final_verification: bool = False
    cad_nudge_count: int = 0
    verification_nudge_count: int = 0
    build_failure_count: int = 0
    build_failure_signatures: dict[str, int] = field(default_factory=dict)
    current_message_id: str | None = None
    total_turns: int = 0
    tool_call_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost: float = 0.0
    run_reasonings: list[str] = field(default_factory=list)
    current_content: str = ""
    current_reasoning: str = ""
    start_perf_time: float = 0.0
    truncation_continuations: int = 0
    total_truncation_continuations: int = 0
    budget_warning_sent: bool = False
    consecutive_readonly_turns: int = 0
    calculation_warning_sent: bool = False
    last_turn_was_continuation: bool = False
    tool_call_turn: int = 0
    settings: Settings | None = None
    export_3mf: bool = False
    export_3mf_retries: int = 0
    stop_event: threading.Event = field(default_factory=threading.Event)


def _extract_reasoning(message: dict[str, Any]) -> str:
    """Extract plain text reasoning from reasoning or reasoning_details."""
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        return strip_encrypted_reasoning(reasoning)
    details = message.get("reasoning_details")
    if isinstance(details, list):
        parts: list[str] = []
        for d in details:
            if not isinstance(d, dict) or d.get("type") == "reasoning.encrypted":
                continue
            for field in ("text", "summary"):
                val = d.get(field)
                if isinstance(val, str) and val:
                    parts.append(val)
                    break
        return strip_encrypted_reasoning("".join(parts))
    return ""


@dataclass
class _ProjectRunSession:
    """Live handles of one project's in-flight run (owned by ``AgentRunner._lock``)."""

    project: str
    run_id: str
    thread: threading.Thread
    stop_event: threading.Event
    run_complete: threading.Event
    active_tools: ProjectTools | None = None
    active_client: Any | None = None


class AgentRunner:
    """One-thread-per-task chat-completions driver with tool dispatch."""

    def __init__(
        self,
        settings: Settings,
        publish: Callable[..., None],
    ) -> None:
        self.settings = settings
        self.publish = publish
        # At most one live session per project; ``_finalize_run`` removes
        # its own entry (guarded by ``run_id``) and ``_reap_locked`` drops
        # entries whose thread died without reaching ``finally``.
        self._sessions: dict[str, _ProjectRunSession] = {}
        self._lock = threading.Lock()
        self._waiting_questions: dict[str, dict[str, object]] = {}

    def _reap_locked(self) -> None:
        """Drop sessions whose run finished. Caller holds ``self._lock``."""
        for proj, sess in list(self._sessions.items()):
            if not sess.thread.is_alive() or sess.run_complete.is_set():
                sess.run_complete.set()
                del self._sessions[proj]

    def _at_capacity_locked(self) -> bool:
        """Caller holds ``self._lock`` (a plain ``Lock``; never re-acquire it here)."""
        self._reap_locked()
        return len(self._sessions) >= self.settings.agent_max_concurrent_runs

    # ------------------------------------------------------------------ state

    def is_running(self, project: str | None = None) -> bool:
        with self._lock:
            self._reap_locked()
            if project is not None:
                return project in self._sessions
            return bool(self._sessions)

    def has_active_state_for(self, project: str) -> bool:
        """Return whether the project has an agent run that must not be deleted."""
        # A persisted waiting question is recoverable UI state, not an active
        # worker.  It must not prevent deleting the project.
        return self.is_running(project)

    def active_projects(self) -> list[str]:
        with self._lock:
            self._reap_locked()
            return sorted(self._sessions)

    def active_run_id(self, project: str) -> str | None:
        with self._lock:
            self._reap_locked()
            sess = self._sessions.get(project)
            return sess.run_id if sess else None

    def at_capacity(self) -> bool:
        with self._lock:
            return self._at_capacity_locked()

    def _agent_state(self, project: str) -> dict[str, Any]:
        """Return the persisted ``.agent_state.json`` (``{}`` when absent or unreadable)."""
        state_path = self.settings.workspace_root / project / ".agent_state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return state if isinstance(state, dict) else {}

    def waiting_question(self, project: str) -> dict[str, object] | None:
        with self._lock:
            question = self._waiting_questions.get(project)
        if question:
            return question.copy()
        question = self._agent_state(project).get("waiting_question")
        if not isinstance(question, dict):
            return None
        with self._lock:
            self._waiting_questions[project] = question
        return question.copy()

    def set_reasoning_effort(self, effort: str | None) -> None:
        """Update reasoning effort dynamically for future runs."""
        with self._lock:
            self.settings = with_reasoning_effort(self.settings, effort)

    def set_model_selection(self, model_key: str) -> None:
        """Update the active ``<provider>:<model>`` dynamically for future runs."""
        with self._lock:
            self.settings = with_model_selection(self.settings, model_key)

    def set_user_preferences(
        self,
        *,
        printer: str | None = None,
        nozzle: str | None = None,
        filament: str | None = None,
        additional_info: str | None = None,
    ) -> None:
        """Update hardware preferences dynamically for future runs."""
        with self._lock:
            self.settings = with_user_preferences(
                self.settings,
                printer=printer,
                nozzle=nozzle,
                filament=filament,
                additional_info=additional_info,
            )

    # ------------------------------------------------------------------ lifecycle

    def start(
        self,
        project: str,
        message: str,
        image_paths: list[Path] | None = None,
        *,
        reasoning_effort: str | None = None,
        model_key: str | None = None,
    ) -> bool:
        """Validate inputs, take ownership, then spawn the worker thread.

        The capacity/ownership checks run *before* ``_take_run_inputs`` so a
        rejection never leaves a half-populated run directory behind and the
        caller's originals in ``<project>/inputs/`` are untouched.
        """
        if self.waiting_question(project):
            return False
        with self._lock:
            if project in self._sessions or self._at_capacity_locked():
                return False
            run_id = uuid.uuid4().hex
            prepared_paths = self._take_run_inputs(
                project, image_paths or [], run_id
            )
            return self._start_locked(
                project,
                message,
                prepared_paths,
                run_id,
                reasoning_effort=reasoning_effort,
                model_key=model_key,
            )

    def start_retry(
        self,
        project: str,
        message: str = "",
        *,
        reasoning_effort: str | None = None,
        model_key: str | None = None,
    ) -> bool:
        """Retry the last turn for a project without re-appending the user prompt."""
        if self.waiting_question(project):
            return False
        with self._lock:
            if project in self._sessions or self._at_capacity_locked():
                return False
            run_id = uuid.uuid4().hex
            return self._start_locked(
                project,
                message,
                [],
                run_id,
                is_retry=True,
                reasoning_effort=reasoning_effort,
                model_key=model_key,
            )

    def _start_locked(
        self,
        project: str,
        message: str,
        image_paths: list[Path],
        run_id: str,
        *,
        is_retry: bool = False,
        reasoning_effort: str | None = None,
        model_key: str | None = None,
    ) -> bool:
        if project in self._sessions:
            return False
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._run,
            args=(project, message, image_paths, run_id, is_retry, reasoning_effort, model_key, stop_event),
            daemon=True,
        )
        self._sessions[project] = _ProjectRunSession(
            project=project,
            run_id=run_id,
            thread=thread,
            stop_event=stop_event,
            run_complete=threading.Event(),
        )
        thread.start()
        return True

    def _take_run_inputs(
        self,
        project: str,
        image_paths: list[Path],
        run_id: str,
    ) -> list[Path]:
        """Move uploaded inputs into ``<project>/.cad-agent/runs/<run_id>/``.

        Once ``start()`` returns, the runner owns the artifacts: the
        active ``_run`` thread reads them via :func:`as_chat_image`, and
        ``_run``'s ``finally`` block removes the run directory on every
        terminal state.

        Returns the new paths. Any mid-move failure rolls back the
        partial copy and returns ``[]`` so the workspace is never left
        in a half-initialized state.
        """
        if not image_paths:
            return []
        # Reject upfront: ``shutil.move`` deletes the source before the
        # destination is committed, so a missing file mid-batch would
        # otherwise lose every successfully moved sibling.
        for source in image_paths:
            if not source.is_file():
                return []
        project_dir = self.settings.workspace_root / project
        run_dir = project_dir / _CAD_AGENT_DIRNAME / _RUNS_DIRNAME / run_id
        run_inputs_dir = run_dir / _RUN_INPUTS_DIRNAME
        try:
            run_inputs_dir.mkdir(parents=True, exist_ok=False)
        except OSError:
            # UUID collision is improbable but possible; roll the whole
            # run directory back so the retry starts clean.
            shutil.rmtree(run_dir, ignore_errors=True)
            try:
                run_inputs_dir.mkdir(parents=True, exist_ok=False)
            except OSError:
                return []
        moved: list[Path] = []
        try:
            for source in image_paths:
                target = run_inputs_dir / source.name
                shutil.copy2(str(source), str(target))
                moved.append(target)
        except OSError:
            shutil.rmtree(run_dir, ignore_errors=True)
            return []
        return moved

    def _discard_run_directory(self, project: str, run_id: str) -> None:
        """Best-effort removal of the run-scoped inputs directory.

        ``_run`` always invokes this in its ``finally`` block. Cleanup
        is incidental to the run, not part of its result, so a failure
        here must never mask the agent's terminal state.
        """
        project_dir = self.settings.workspace_root / project
        run_dir = project_dir / _CAD_AGENT_DIRNAME / _RUNS_DIRNAME / run_id
        shutil.rmtree(run_dir, ignore_errors=True)

    def answer(
        self,
        project: str,
        answer: str,
        *,
        model_key: str | None = None,
        reasoning_effort: str | None = None,
    ) -> str | None:
        """Resume a parked run with the user's answer.

        Returns the formatted answer persisted as the new user turn, or
        ``None`` when no valid question is pending. Without an explicit
        ``model_key`` / ``reasoning_effort`` the run resumes with the
        selection the parked run used (persisted by ``_park_for_question``).
        """
        state = self._agent_state(project)
        question = state.get("waiting_question")
        if not isinstance(question, dict) or not self._validate_answer(question, answer):
            return None
        if not model_key:
            model_key = state.get("model_key") or None
        if reasoning_effort is None:
            reasoning_effort = state.get("reasoning_effort")
        # Wait for the parked run's ``finally`` so we never observe a
        # half-completed run while still accepting a follow-up message.
        with self._lock:
            self._reap_locked()
            parked = self._sessions.get(project)
        if parked is not None:
            parked.run_complete.wait(timeout=10.0)
        project_dir = self.settings.workspace_root / project
        formatted = self._format_answer(question, answer)
        with self._lock:
            self._reap_locked()
            if self._at_capacity_locked() or project in self._sessions:
                return None
            # Persist the user answer BEFORE starting the new thread so
            # ``_context()`` reads it from the canonical log instead of
            # racing against this append.
            self._append_message(project_dir, {"role": "user", "content": formatted})
            if not self._start_locked(
                project,
                formatted,
                [],
                uuid.uuid4().hex,
                reasoning_effort=reasoning_effort,
                model_key=model_key,
            ):
                return None
            self._waiting_questions.pop(project, None)
            (project_dir / ".agent_state.json").unlink(missing_ok=True)
            return formatted

    def stop(self, project: str | None = None) -> list[str]:
        """Stop agent work and clear pending state for one or all projects."""
        with self._lock:
            self._reap_locked()
            if project is not None:
                session = self._sessions.get(project)
                target_sessions = [session] if session else []
                # Always report the project so an on-disk waiting question
                # (not yet mirrored into ``_waiting_questions``) is cleared too.
                affected = [project]
                self._waiting_questions.pop(project, None)
            else:
                target_sessions = list(self._sessions.values())
                affected = list(self._sessions)
                for q_proj in self._waiting_questions:
                    if q_proj not in affected:
                        affected.append(q_proj)
                if self.settings.workspace_root.is_dir():
                    for item in self.settings.workspace_root.iterdir():
                        if (
                            item.is_dir()
                            and (item / ".agent_state.json").is_file()
                            and item.name not in affected
                        ):
                            affected.append(item.name)
                self._waiting_questions.clear()

            for sess in target_sessions:
                sess.stop_event.set()
                if sess.active_tools:
                    sess.active_tools.stop()
                if sess.active_client:
                    try:
                        sess.active_client.abort()
                    except Exception:  # noqa: BLE001, S110
                        pass

        for sess in target_sessions:
            if sess.thread.is_alive():
                sess.thread.join(timeout=3.5)

        with self._lock:
            self._reap_locked()

        for cleared in affected:
            (self.settings.workspace_root / cleared / ".agent_state.json").unlink(missing_ok=True)
        return affected

    # ------------------------------------------------------------------ main loop

    def _run(
        self,
        project: str,
        message: str,
        image_paths: list[Path] | None = None,
        run_id: str = "",
        is_retry: bool = False,
        reasoning_effort: str | None = None,
        model_key: str | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Lifecycle wrapper: init, dispatch turns, terminal cleanup.

        Per-iteration logic lives in :meth:`_run_turn`; helpers handle the
        specific branches (no tool calls, parked question, etc.).
        """
        if not run_id:
            # ``start()`` always passes a run_id; the default keeps the
            # parameter optional for direct tests that bypass ``start()``.
            run_id = uuid.uuid4().hex
        with self._lock:
            workspace_root = self.settings.workspace_root
        project_dir = workspace_root / project
        state: _RunState | None = None
        activity_logger: ActivityLogger | None = None
        try:
            state = self._init_run(
                project,
                message,
                image_paths or [],
                project_dir,
                run_id,
                is_retry=is_retry,
                reasoning_effort=reasoning_effort,
                model_key=model_key,
                stop_event=stop_event,
            )
            activity_logger = state.activity_logger
            # Bound once: the publisher reads ``current_message_id`` lazily
            # so the per-iteration update is just an attribute assignment.
            state.client.stream_callback = self._build_stream_publisher(state)
            run_settings = state.settings or self.settings
            limit = run_settings.agent_tool_call_limit
            max_iterations = limit + _MAX_TOTAL_TRUNCATION_CONTINUATIONS + 8
            tool_turn = 0
            for _ in range(max_iterations):
                if state.stop_event.is_set():
                    self._handle_stop_pre_turn(state)
                    return
                state.current_message_id = uuid.uuid4().hex
                state.current_content = ""
                state.current_reasoning = ""
                state.last_turn_was_continuation = False
                outcome = self._run_turn(state)
                if outcome is _TurnOutcome.CONTINUE:
                    if not state.last_turn_was_continuation:
                        tool_turn += 1
                        state.tool_call_turn = tool_turn
                        if tool_turn >= limit:
                            break
                        self._report_step_budget(state, tool_turn)
                    continue
                # Non-CONTINUE is terminal; the helper already published
                # the diagnostic and finalised ``state.messages``.
                return
            # Outer loop exhausted without a terminal outcome.
            self._handle_tool_limit_reached(state)
        except RequestCancelled:
            self._handle_request_cancelled(state)
        except Exception as error:  # noqa: BLE001 - Surface all failures to the local UI.
            self._handle_unexpected_error(error, project, state, stop_event)
        finally:
            self._finalize_run(project, state, activity_logger, run_id)

    # ------------------------------------------------------------------ step 2.1 helpers

    def _init_run(
        self,
        project: str,
        message: str,
        image_paths: list[Path],
        project_dir: Path,
        run_id: str,
        *,
        is_retry: bool = False,
        reasoning_effort: str | None = None,
        model_key: str | None = None,
        stop_event: threading.Event | None = None,
    ) -> _RunState:
        """Build tools/client, log ``run_start``, return the run state.

        Pulled out of ``_run`` so the lifecycle wrapper stays focused on
        dispatch. If this raises, ``_run``'s exception handlers take
        over and ``state`` is ``None`` for the ``finally`` block.
        """
        eff_stop = stop_event or threading.Event()
        with self._lock:
            effective_settings = self.settings
        if model_key:
            effective_settings = with_model_selection(effective_settings, model_key)
        if reasoning_effort is not None:
            effective_settings = with_reasoning_effort(effective_settings, reasoning_effort)

        self.publish(
            "agent_status",
            {
                "project": project,
                "run_id": run_id,
                "status": "started",
                "message": "Planning CAD task...",
                "model_key": effective_settings.active_model_key,
                "reasoning_effort": effective_settings.reasoning_effort,
            },
        )
        tools = ProjectTools(
            project_dir, self.publish, effective_settings, eff_stop
        )
        client = create_llm_client(effective_settings)
        client.stop_event = eff_stop
        session_prefix = (
            effective_settings.openrouter_session_prefix
            if effective_settings.llm_provider == "openrouter"
            else effective_settings.llm_provider
        )
        client.session_id = f"{session_prefix}:{project}"
        client.error_callback = lambda phase, err, att: self._debug_tool_error(
            project_dir,
            f"llm_{att}",
            "llm",
            err,
            "",
            phase=f"llm_{phase}",
            run_id=run_id,
            settings=effective_settings,
        )
        with self._lock:
            session = self._sessions.get(project)
            if session is not None and session.run_id == run_id:
                session.active_tools = tools
                session.active_client = client
        messages = self._context(
            project_dir,
            message,
            image_paths,
            is_retry=is_retry,
            settings=effective_settings,
        )
        activity_logger: ActivityLogger | None = None
        if activity_logging_enabled(effective_settings):
            activity_logger = get_logger(
                project_dir,
                log_mode=getattr(effective_settings, "agent_log_mode", "debug"),
            )
            client.activity_logger = activity_logger
            client.run_id = run_id
            activity_logger.log(
                "run_start",
                {
                    "project": project,
                    "model": effective_settings.llm_model,
                    "provider": effective_settings.llm_provider,
                    "reasoning_effort": effective_settings.reasoning_effort,
                },
                run_id=run_id,
            )
        export_3mf = False
        if messages:
            last_user_msg = None
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "user" and not m.get("synthetic"):
                    last_user_msg = m
                    break
            if last_user_msg:
                c = last_user_msg.get("content")
                if isinstance(c, str) and c.startswith(THREE_MF_COMMAND_MARKER):
                    export_3mf = True
                elif isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and str(part.get("text", "")).startswith(THREE_MF_COMMAND_MARKER):
                            export_3mf = True
                            break
        return _RunState(
            project_dir=project_dir,
            messages=messages,
            tools=tools,
            client=client,
            activity_logger=activity_logger,
            run_id=run_id,
            cad_fix_required=not model_is_built(project_dir),
            project=project,
            start_perf_time=time.perf_counter(),
            settings=effective_settings,
            export_3mf=export_3mf,
            stop_event=eff_stop,
        )

    def _build_stream_publisher(
        self, state: _RunState
    ) -> Callable[[dict[str, Any]], None]:
        """Bind the ``stream_callback`` closure once at method entry.

        The publisher reads ``state.current_message_id`` lazily so each
        iteration only updates that field — the closure is never
        reconstructed inside the loop.
        """
        publish = self.publish
        log = (
            state.activity_logger.log
            if state.activity_logger is not None
            else None
        )
        project = state.project
        run_id = state.run_id

        def publish_stream(event: dict[str, Any]) -> None:
            event_type = event.get("type")
            if not event_type:
                return
            event_payload = {k: v for k, v in event.items() if k != "type"}
            msg_id = state.current_message_id
            delta = event_payload.get("delta") or ""
            if event_type == "content":
                state.current_content += delta
                sse_name = "agent_content_delta"
            elif event_type == "reasoning":
                state.current_reasoning += delta
                sse_name = "agent_reasoning_delta"
            elif event_type == "tool_call":
                sse_name = "agent_tool_delta"
            elif event_type == "status":
                status_msg = event_payload.get("message") or "Thinking..."
                publish(
                    "agent_status",
                    {
                        "project": project,
                        "run_id": run_id,
                        "status": "started",
                        "message": status_msg,
                    },
                )
                if log is not None:
                    log(
                        "agent_status",
                        {"run_id": run_id, "message": status_msg},
                        run_id=run_id,
                    )
                return
            elif event_type == "reset":
                state.current_content = ""
                state.current_reasoning = ""
                state.current_message_id = uuid.uuid4().hex
                publish(
                    "agent_stream_reset",
                    {
                        "project": project,
                        "run_id": run_id,
                        "message_id": state.current_message_id,
                    },
                )
                if log is not None:
                    log(
                        "agent_stream_reset",
                        {"run_id": run_id, "message_id": state.current_message_id},
                        run_id=run_id,
                    )
                return
            else:
                sse_name = f"agent_{event_type}_delta"
            publish(
                sse_name,
                {"project": project, "run_id": run_id, "message_id": msg_id, **event_payload},
            )
            if log is not None:
                # Mirror the SSE stream into the activity log so the
                # operator has the same content the UI consumed.
                log(
                    sse_name,
                    {"run_id": run_id, "message_id": msg_id, **event_payload},
                    run_id=run_id,
                )

        return publish_stream

    def _run_turn(self, state: _RunState) -> _TurnOutcome:
        """One iteration of the agent tool-loop.

        Drives ``state.client.chat``, normalises the response, persists
        the assistant turn, dispatches any tool calls, and applies the
        circuit-breaker / question-parking rules.
        """
        awaiting_tool_render = self._last_message_has_tool_image(state.messages)
        if state.total_turns > 0:
            self.publish(
                "agent_status",
                {
                    "project": state.project,
                    "run_id": state.run_id,
                    "status": "started",
                    "message": "Thinking...",
                },
            )
        active_tools = None if state.export_3mf else TOOL_SCHEMAS
        try:
            response = state.client.chat(state.messages, active_tools)
        except CompletionTruncatedError as trunc_err:
            has_partial = bool(
                trunc_err.reasoning_text
                or trunc_err.partial_content
                or trunc_err.partial_tool_calls
            )
            if (
                state.truncation_continuations < 3
                and state.total_truncation_continuations < _MAX_TOTAL_TRUNCATION_CONTINUATIONS
                and has_partial
                and not state.stop_event.is_set()
            ):
                state.total_turns += 1
                state.truncation_continuations += 1
                state.total_truncation_continuations += 1
                state.last_turn_was_continuation = True
                usage = trunc_err.usage or getattr(state.client, "last_usage", None)
                self._accumulate_state_usage(state, usage)
                self._publish_usage(state.project, usage, state=state)
                raw_partial = trunc_err.partial_content or ""
                cleaned_partial, extra_tcs = extract_text_tool_calls(raw_partial)
                raw_text = cleaned_partial.strip()
                has_real_prose = bool(raw_text) and not state.export_3mf
                published_text = (
                    ensure_closed_code_fences(raw_text) if has_real_prose else ""
                )
                partial_tcs = list(trunc_err.partial_tool_calls or [])
                if not partial_tcs and extra_tcs:
                    partial_tcs = extra_tcs
                truncated_tools: list[str] = []
                partial_arg_snippets: list[str] = []
                for tc in partial_tcs:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    t_name = fn.get("name") or "tool"
                    truncated_tools.append(t_name)
                    raw_args = str(fn.get("arguments") or "").strip()
                    if raw_args:
                        tail = raw_args[-300:] if len(raw_args) > 300 else raw_args
                        partial_arg_snippets.append(f"{t_name} tail: ...{tail}")
                stored_content = published_text if has_real_prose else raw_text
                if not stored_content and truncated_tools:
                    tail_info = (
                        f" ({'; '.join(partial_arg_snippets)})"
                        if partial_arg_snippets
                        else ""
                    )
                    stored_content = f"[Truncated tool call: {', '.join(truncated_tools)}{tail_info}]"
                partial_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": stored_content,
                }
                turn_reasoning = (trunc_err.reasoning_text or "").strip()
                if turn_reasoning and (
                    not state.run_reasonings
                    or state.run_reasonings[-1] != turn_reasoning
                ):
                    state.run_reasonings.append(turn_reasoning)
                published_reasoning = turn_reasoning
                if has_real_prose and state.run_reasonings:
                    combined_reasoning = "\n\n".join(state.run_reasonings)
                    if combined_reasoning and combined_reasoning.strip() != published_text:
                        published_reasoning = combined_reasoning
                stored_reasoning = (
                    published_reasoning if has_real_prose else turn_reasoning
                )
                if stored_reasoning:
                    if len(stored_reasoning) > 4000:
                        partial_msg["reasoning"] = (
                            "...[earlier reasoning truncated]...\n"
                            + stored_reasoning[-4000:]
                        )
                    else:
                        partial_msg["reasoning"] = stored_reasoning
                self.publish(
                    "agent_stream_end",
                    {
                        "project": state.project,
                        "run_id": state.run_id,
                        "message_id": state.current_message_id,
                        "message": published_text,
                        "reasoning": published_reasoning,
                        "has_tools": False,
                        "intermediate": has_real_prose,
                        "continuing": not has_real_prose,
                    },
                )
                if has_real_prose:
                    state.run_reasonings.clear()
                else:
                    partial_msg["synthetic"] = True
                state.messages.append(partial_msg)
                ConversationStore.append(state.project_dir, partial_msg)
                cad_pending = self._should_nudge_incomplete_turn(state)
                if truncated_tools:
                    nudge_text = format_truncated_tool_call_nudge(truncated_tools)
                else:
                    nudge_text = (
                        NUDGE_TRUNCATED_REASONING_PENDING_CAD
                        if cad_pending
                        else NUDGE_TRUNCATED_REASONING_READY
                    )
                continuation_nudge = _synthetic_user(nudge_text)
                state.messages.append(continuation_nudge)
                ConversationStore.append(state.project_dir, continuation_nudge)
                self.publish(
                    "agent_status",
                    {
                        "project": state.project,
                        "run_id": state.run_id,
                        "status": "started",
                        "message": "Continuing generation after token limit...",
                    },
                )
                if state.activity_logger is not None:
                    state.activity_logger.log(
                        "assistant_turn",
                        {
                            "project": state.project,
                            "turn": state.total_turns,
                            "message": partial_msg,
                            "truncated": True,
                        },
                        run_id=state.run_id,
                    )
                return _TurnOutcome.CONTINUE
            raise
        if state.stop_event.is_set():
            raise RequestCancelled("Task stopped by the user.")
        if (
            awaiting_tool_render
            and getattr(state.client, "last_image_fallback_used", False)
        ):
            # The provider rejected the trailing inline render — keep
            # ``cad_fix_required`` true so the final-verification gate
            # treats the model as unverified, and feed the same string to
            # the build-failure tracker.
            state.cad_fix_required = True
            state.preview_id = None
            state.cad_error = (
                "The model provider rejected the required final render; "
                "visual verification could not be completed."
            )
        state.total_turns += 1
        usage = getattr(state.client, "last_usage", None)
        self._accumulate_state_usage(state, usage)
        self._publish_usage(state.project, usage, state=state)
        assistant_message = sanitize_assistant_message(
            response["choices"][0]["message"],
            preserve_reasoning=getattr(state.client, "preserve_reasoning", False),
            for_storage=True,
        )
        tool_calls = normalize_tool_calls(assistant_message.get("tool_calls"))
        raw_content = (assistant_message.get("content") or "").strip()
        turn_reasoning = _extract_reasoning(assistant_message)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
            state.tool_call_count += len(tool_calls)
            state.truncation_continuations = 0
            invalid_final = False
        else:
            assistant_message.pop("tool_calls", None)
            if not raw_content and turn_reasoning:
                can_continue = (
                    state.truncation_continuations < 3
                    and state.total_truncation_continuations < _MAX_TOTAL_TRUNCATION_CONTINUATIONS
                    and not state.stop_event.is_set()
                )
                self.publish(
                    "agent_stream_end",
                    {
                        "project": state.project,
                        "run_id": state.run_id,
                        "message_id": state.current_message_id,
                        "message": "",
                        "reasoning": turn_reasoning,
                        "has_tools": False,
                        "continuing": can_continue,
                    },
                )
                if can_continue:
                    state.truncation_continuations += 1
                    state.total_truncation_continuations += 1
                    state.last_turn_was_continuation = True
                    if (
                        not state.run_reasonings
                        or state.run_reasonings[-1] != turn_reasoning
                    ):
                        state.run_reasonings.append(turn_reasoning)
                    assistant_message["content"] = ""
                    assistant_message["synthetic"] = True
                    self._append_message(
                        state.project_dir,
                        assistant_message,
                    )
                    state.messages.append(assistant_message)
                    cad_pending = self._should_nudge_incomplete_turn(state)
                    continuation_nudge = _synthetic_user(
                        NUDGE_REASONING_ONLY_PENDING_CAD
                        if cad_pending
                        else NUDGE_REASONING_ONLY_READY
                    )
                    state.messages.append(continuation_nudge)
                    self._append_message(state.project_dir, continuation_nudge)
                    self.publish(
                        "agent_status",
                        {
                            "project": state.project,
                            "run_id": state.run_id,
                            "status": "started",
                            "message": "Continuing after reasoning...",
                        },
                    )
                    if state.activity_logger is not None:
                        state.activity_logger.log(
                            "assistant_turn",
                            {
                                "project": state.project,
                                "turn": state.total_turns,
                                "message": assistant_message,
                                "reasoning_only": True,
                            },
                            run_id=state.run_id,
                        )
                    return _TurnOutcome.CONTINUE
                self.publish(
                    "agent_error",
                    {
                        "project": state.project,
                        "run_id": state.run_id,
                        "message": (
                            "Model completed its reasoning process but did not produce "
                            "an action or final response."
                        ),
                    },
                )
                self._publish_terminal_failure(state.project, run_id=state.run_id)
                return _TurnOutcome.DRAWING_NOT_CREATED
            invalid_final = self._should_nudge_incomplete_turn(
                state, assistant_message
            )
            if not raw_content and not invalid_final:
                assistant_message["content"] = "Task completed."
        self._publish_assistant_turn(state, assistant_message, invalid_final)
        if not tool_calls:
            return self._handle_no_tool_calls(state, assistant_message)
        state.any_tool_used = True
        return self._handle_tool_calls(state, tool_calls)

    def _should_nudge_incomplete_turn(
        self, state: _RunState, assistant_message: dict[str, Any] | None = None
    ) -> bool:
        """Return True when a text-only turn should be intercepted and nudged."""
        if state.export_3mf:
            if state.export_3mf_retries >= 1:
                return False
            content = (assistant_message.get("content") if assistant_message else "") or ""
            json_block = extract_json_block(content)
            if not json_block or not isinstance(json_block, dict):
                return True
            eff_settings = state.settings or self.settings
            user_printer = (getattr(eff_settings, "user_printer", "") or "").strip()
            user_nozzle = (getattr(eff_settings, "user_nozzle", "") or "").strip()
            nozzle_dia = parse_nozzle_spec(user_nozzle)["diameter_mm"]
            profile = sanitize_print_profile(
                json_block,
                nozzle_diameter_mm=nozzle_dia,
                user_printer=user_printer,
            )
            val_err = validate_profile_universal(profile, state.project_dir)
            return bool(val_err)
        if not state.model_mutated:
            return False
        if not state.preview_id:
            return not state.cad_error and state.cad_nudge_count < 2
        if state.cad_fix_required and state.verification_nudge_count < 2:
            return True
        run_settings = state.settings or self.settings
        limit = run_settings.agent_tool_call_limit
        remaining_budget = max(0, limit - state.tool_call_turn)
        return bool(
            state.cad_warning_fix_required
            and state.warning_nudge_count < 1
            and remaining_budget >= 2
        )

    def _publish_assistant_turn(
        self,
        state: _RunState,
        assistant_message: dict[str, Any],
        invalid_final: bool,
    ) -> None:
        """Persist and broadcast the assistant turn; handle invalid-final.

        An *invalid final* is a text-only turn after the agent invoked
        tools but visual verification is still missing. We persist it with
        ``"synthetic": True`` so ``ConversationStore.load`` preserves the
        prompt prefix across turns while UI history reloads skip the
        unverified ghost turn.
        """
        reasoning = _extract_reasoning(assistant_message)
        has_tools = bool(assistant_message.get("tool_calls"))
        if reasoning and (
            not state.run_reasonings or state.run_reasonings[-1] != reasoning
        ):
            state.run_reasonings.append(reasoning)

        if invalid_final:
            status_msg = (
                "Validating 3MF slicer profile..."
                if state.export_3mf
                else "Model verification required before finalizing."
            )
            self.publish(
                "agent_status",
                {
                    "project": state.project,
                    "run_id": state.run_id,
                    "status": "verifying",
                    "message": status_msg,
                },
            )
            raw_text = (assistant_message.get("content") or "").strip()
            preserve_prose = bool(raw_text) and not state.export_3mf
            if preserve_prose:
                published_reasoning = reasoning
                if state.run_reasonings:
                    combined_reasoning = "\n\n".join(state.run_reasonings)
                    if combined_reasoning and combined_reasoning.strip() != raw_text:
                        published_reasoning = combined_reasoning
                self.publish(
                    "agent_stream_end",
                    {
                        "project": state.project,
                        "run_id": state.run_id,
                        "message_id": state.current_message_id,
                        "message": raw_text,
                        "reasoning": published_reasoning,
                        "has_tools": False,
                        "intermediate": True,
                        "continuing": False,
                    },
                )
                stored_invalid = (
                    {**assistant_message, "reasoning": published_reasoning}
                    if published_reasoning and assistant_message.get("reasoning") != published_reasoning
                    else assistant_message
                )
                self._append_message(state.project_dir, stored_invalid)
                state.run_reasonings.clear()
            else:
                self.publish(
                    "agent_stream_end",
                    {
                        "project": state.project,
                        "run_id": state.run_id,
                        "message_id": state.current_message_id,
                        "message": "",
                        "reasoning": reasoning,
                        "has_tools": False,
                        "continuing": True,
                    },
                )
                self._append_message(
                    state.project_dir,
                    {**assistant_message, "synthetic": True},
                )
        else:
            published_reasoning = reasoning
            if state.run_reasonings:
                combined_reasoning = "\n\n".join(state.run_reasonings)
                content_text = (assistant_message.get("content") or "").strip()
                if combined_reasoning and combined_reasoning.strip() != content_text:
                    published_reasoning = combined_reasoning
                    if not has_tools:
                        assistant_message["reasoning"] = combined_reasoning
            self.publish(
                "agent_stream_end",
                {
                    "project": state.project,
                    "run_id": state.run_id,
                    "message_id": state.current_message_id,
                    "message": assistant_message.get("content") or "",
                    "reasoning": published_reasoning,
                    "has_tools": has_tools,
                },
            )
            stored_message = (
                {**assistant_message, "reasoning": published_reasoning}
                if published_reasoning and assistant_message.get("reasoning") != published_reasoning
                else assistant_message
            )
            self._append_message(state.project_dir, stored_message)
            if has_tools:
                state.run_reasonings.clear()
        state.messages.append(assistant_message)
        if state.activity_logger is not None:
            state.activity_logger.log(
                "assistant_turn",
                {
                    "project": state.project,
                    "turn": state.total_turns,
                    "message": assistant_message,
                    "invalid_final": invalid_final,
                },
                run_id=state.run_id,
            )

    def _handle_no_tool_calls(
        self, state: _RunState, assistant_message: dict[str, Any]
    ) -> _TurnOutcome:
        """Decide the outcome of a text-only turn.

        Mirrors the original control flow: missing preview + cad_error
        or two nudges without a build are terminal; otherwise the model
        gets a synthetic reminder (``CONTINUE``) or the run completes.
        """
        content = assistant_message.get("content") or "Task completed."
        full_reasoning = "\n\n".join(state.run_reasonings)
        if state.export_3mf:
            outcome = self._finalize_3mf_export(state, content, full_reasoning)
            if outcome is not None:
                return outcome
        if not state.preview_id:
            if state.cad_error:
                self.publish(
                    "agent_error",
                    {
                        "project": state.project,
                        "run_id": state.run_id,
                        "message": f"Drawing was not created: {state.cad_error}",
                    },
                )
                self._publish_terminal_failure(state.project, run_id=state.run_id)
                return _TurnOutcome.DRAWING_NOT_CREATED
            if state.model_mutated:
                if state.cad_nudge_count < 2:
                    state.nudged_cad = True
                    state.cad_nudge_count += 1
                    state.last_turn_was_continuation = True
                    reminder = _synthetic_user(
                        format_unverified_model_nudge(MODEL_FILENAME)
                    )
                    state.messages.append(reminder)
                    self._append_message(state.project_dir, reminder)
                    return _TurnOutcome.CONTINUE
                self.publish(
                    "agent_error",
                    {
                        "project": state.project,
                        "run_id": state.run_id,
                        "message": (
                            "Drawing was not created: the task did not "
                            "produce a new CAD preview."
                        ),
                    },
                )
                self._publish_terminal_failure(state.project, run_id=state.run_id)
                return _TurnOutcome.DRAWING_NOT_CREATED
            self._complete(
                state.project,
                content,
                reasoning=full_reasoning,
                run_id=state.run_id,
                message_id=state.current_message_id,
            )
            return _TurnOutcome.COMPLETED
        if state.model_mutated and state.cad_fix_required:
            if state.verification_nudge_count < 2:
                state.nudged_final_verification = True
                state.verification_nudge_count += 1
                state.last_turn_was_continuation = True
                reminder = _synthetic_user(NUDGE_CAD_FIX_REQUIRED)
                state.messages.append(reminder)
                self._append_message(state.project_dir, reminder)
                return _TurnOutcome.CONTINUE
            self.publish(
                "agent_error",
                {
                    "project": state.project,
                    "run_id": state.run_id,
                    "message": (
                        "Task stopped: CAD build succeeded but critical verification "
                        "issues were not resolved."
                    ),
                },
            )
            self._publish_terminal_failure(state.project, run_id=state.run_id)
            return _TurnOutcome.FINAL_VERIFICATION_MISSING
        run_settings = state.settings or self.settings
        limit = run_settings.agent_tool_call_limit
        remaining_budget = max(0, limit - state.tool_call_turn)
        if (
            state.model_mutated
            and state.cad_warning_fix_required
            and state.warning_nudge_count < 1
            and remaining_budget >= 2
        ):
            state.warning_nudge_count += 1
            state.last_turn_was_continuation = True
            reminder = _synthetic_user(NUDGE_CAD_WARNING_FIX)
            state.messages.append(reminder)
            self._append_message(state.project_dir, reminder)
            return _TurnOutcome.CONTINUE
        self._complete(
            state.project,
            content,
            reasoning=full_reasoning,
            run_id=state.run_id,
            message_id=state.current_message_id,
        )
        return _TurnOutcome.COMPLETED

    def _finalize_3mf_export(
        self, state: _RunState, content: str, full_reasoning: str
    ) -> _TurnOutcome:
        """Process and validate 3MF slicer profile from the model's text response."""
        json_block = extract_json_block(content)
        if not json_block or not isinstance(json_block, dict):
            if state.export_3mf_retries < 1:
                state.export_3mf_retries += 1
                state.last_turn_was_continuation = True
                reminder = _synthetic_user(
                    "The response did not contain a valid JSON print profile block. "
                    "Please reply with ONLY the complete JSON print profile block enclosed in ```json ... ``` as requested."
                )
                state.messages.append(reminder)
                self._append_message(state.project_dir, reminder)
                return _TurnOutcome.CONTINUE
            self.publish(
                "agent_error",
                {
                    "project": state.project,
                    "run_id": state.run_id,
                    "message": "3MF export failed: No valid JSON print profile found in the model response.",
                },
            )
            self._publish_terminal_failure(state.project, run_id=state.run_id)
            return _TurnOutcome.DRAWING_NOT_CREATED

        eff_settings = state.settings or self.settings
        user_printer = (getattr(eff_settings, "user_printer", "") or "").strip()
        user_nozzle = (getattr(eff_settings, "user_nozzle", "") or "").strip()
        nozzle_dia = parse_nozzle_spec(user_nozzle)["diameter_mm"]

        profile = sanitize_print_profile(
            json_block,
            nozzle_diameter_mm=nozzle_dia,
            user_printer=user_printer,
        )
        val_err = validate_profile_universal(profile, state.project_dir)
        if val_err:
            if state.export_3mf_retries < 1:
                state.export_3mf_retries += 1
                state.last_turn_was_continuation = True
                reminder = _synthetic_user(
                    f"3MF slicer validation failed: {val_err}\n"
                    "Please fix the offending parameters and output the complete corrected JSON object enclosed in ```json ... ```."
                )
                state.messages.append(reminder)
                self._append_message(state.project_dir, reminder)
                return _TurnOutcome.CONTINUE
            self.publish(
                "agent_error",
                {
                    "project": state.project,
                    "run_id": state.run_id,
                    "message": f"3MF slicer validation failed: {val_err}",
                },
            )
            self._publish_terminal_failure(state.project, run_id=state.run_id)
            return _TurnOutcome.DRAWING_NOT_CREATED

        # Save profile to .cad-agent/print_profile.json
        current_sha = compute_model_sha256(state.project_dir)
        profile_store = {
            "model_sha256": current_sha,
            "profile": profile,
        }
        profile_path = state.project_dir / ".cad-agent" / "print_profile.json"
        try:
            atomic_write_json(profile_path, profile_store)
        except OSError as exc:
            logging.getLogger(__name__).warning("Failed to save 3MF print profile: %s", exc)

        try:
            export_project_model(state.project_dir, "3mf", print_profile=profile)
            self.publish(
                "export_ready",
                {
                    "project": state.project,
                    "run_id": state.run_id,
                    "format": "3mf",
                },
            )
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("Failed to package 3MF: %s", exc)
            self.publish(
                "agent_error",
                {
                    "project": state.project,
                    "run_id": state.run_id,
                    "message": f"Failed to package 3MF: {exc}",
                },
            )
            self._publish_terminal_failure(state.project, run_id=state.run_id)
            return _TurnOutcome.DRAWING_NOT_CREATED

        self._complete(
            state.project,
            content,
            reasoning=full_reasoning,
            run_id=state.run_id,
            message_id=state.current_message_id,
        )
        return _TurnOutcome.COMPLETED

    def _handle_tool_calls(
        self, state: _RunState, tool_calls: list[dict[str, Any]]
    ) -> _TurnOutcome:
        """Process every tool call in this iteration, executing compatible waves in parallel.

        Stops early on user-cancellation, repeated CAD build failures,
        or a tool that parked the agent waiting for user input. Returns
        :class:`_TurnOutcome.CONTINUE` once the batch completes.
        """
        processed_call_ids: set[str] = set()
        run_settings = state.settings or self.settings
        budget_limit = run_settings.agent_tool_call_limit
        turn_step = state.tool_call_turn + 1
        turn_budget = {
            "step": turn_step,
            "limit": budget_limit,
            "remaining": max(0, budget_limit - turn_step),
        }
        debug_log = partial(self._debug_tool_error, run_id=state.run_id, settings=run_settings)
        waves = group_tool_calls_into_waves(tool_calls)
        for wave_idx, wave in enumerate(waves):
            if state.stop_event.is_set():
                self._cancel_remaining_tool_calls(
                    state.project_dir,
                    tool_calls,
                    processed_call_ids,
                    state.messages,
                )
                _close_dangling_tool_tail(
                    state.project_dir,
                    state.messages,
                    "Task stopped by the user mid-batch.",
                )
                return _TurnOutcome.STOPPED

            if len(wave) == 1:
                call = wave[0]
                local_msgs: list[dict] = []
                res_preview_id, res_cad_error, res_cad_fix_required, waiting = (
                    process_tool_call(
                        state.tools,
                        state.project,
                        state.project_dir,
                        call,
                        state.cad_fix_required,
                        state.preview_id,
                        state.cad_error,
                        local_msgs,
                        publish=self.publish,
                        register_preview=self._register_preview,
                        append_message=lambda _p, _m: None,
                        debug_log=debug_log,
                        activity_logger=state.activity_logger,
                        run_id=state.run_id,
                    )
                )
                wave_outcomes: list[
                    tuple[
                        dict[str, Any],
                        str | None,
                        str | None,
                        bool,
                        bool,
                        list[dict],
                    ]
                ] = [
                    (
                        call,
                        res_preview_id,
                        res_cad_error,
                        res_cad_fix_required,
                        waiting,
                        local_msgs,
                    )
                ]
            else:
                base_fix_required = state.cad_fix_required
                base_preview_id = state.preview_id
                base_cad_error = state.cad_error

                def _exec_parallel_call(
                    call_item: dict[str, Any],
                    _fix: bool = base_fix_required,
                    _prev: str | None = base_preview_id,
                    _err: str | None = base_cad_error,
                ) -> tuple[str | None, str | None, bool, bool, list[dict]]:
                    local_msgs: list[dict] = []
                    p_id, c_err, c_fix, wait = process_tool_call(
                        state.tools,
                        state.project,
                        state.project_dir,
                        call_item,
                        _fix,
                        _prev,
                        _err,
                        local_msgs,
                        publish=self.publish,
                        register_preview=self._register_preview,
                        append_message=lambda _p, _m: None,
                        debug_log=debug_log,
                        activity_logger=state.activity_logger,
                        run_id=state.run_id,
                    )
                    return p_id, c_err, c_fix, wait, local_msgs

                with ThreadPoolExecutor(max_workers=min(len(wave), 4)) as pool:
                    futures = [pool.submit(_exec_parallel_call, c) for c in wave]
                    wave_outcomes = [
                        (c, *fut.result()) for c, fut in zip(wave, futures)
                    ]

            # If this is the final wave of the turn, attach step_budget to the
            # trailing tool message before persisting to conversation.jsonl.
            if wave_idx == len(waves) - 1:
                last_msg = None
                for _c, _p, _e, _f, _w, d_msgs in reversed(wave_outcomes):
                    if d_msgs:
                        last_msg = d_msgs[-1]
                        break
                if last_msg is not None:
                    payload = _tool_payload(last_msg)
                    if payload is not None:
                        payload["step_budget"] = turn_budget
                        _set_tool_payload(last_msg, payload)

            for (
                call,
                _res_preview_id,
                _res_cad_error,
                _res_cad_fix_required,
                _waiting,
                deferred_msgs,
            ) in wave_outcomes:
                if deferred_msgs is not None:
                    for msg in deferred_msgs:
                        state.messages.append(msg)
                        self._append_message(state.project_dir, msg)
                processed_call_ids.add(call.get("id", ""))

            mutation_failed_in_wave = False
            for (
                call,
                res_preview_id,
                res_cad_error,
                res_cad_fix_required,
                waiting,
                _deferred_msgs,
            ) in wave_outcomes:
                tool_name = call.get("function", {}).get("name", "")
                if (
                    len(wave) == 1
                    or is_model_mutation(tool_name)
                    or is_cad_build(tool_name)
                ):
                    state.preview_id = res_preview_id
                    state.cad_error = res_cad_error
                    state.cad_fix_required = res_cad_fix_required
                if is_model_mutation(tool_name):
                    if self._tool_call_succeeded(call.get("id", ""), state.messages):
                        state.model_mutated = True
                        state.cad_warning_fix_required = False
                        state.warning_nudge_count = 0
                        if state.cad_nudge_count < 2:
                            state.nudged_cad = False
                        if state.verification_nudge_count < 2:
                            state.nudged_final_verification = False
                    else:
                        mutation_failed_in_wave = True
                elif is_cad_build(tool_name):
                    state.nudged_cad = False
                    tool_msg = next(
                        (
                            m
                            for m in reversed(state.messages)
                            if m.get("tool_call_id") == call.get("id")
                        ),
                        None,
                    )
                    if tool_msg:
                        payload = _tool_payload(tool_msg)
                        if isinstance(payload, dict):
                            verif = payload.get("verification")
                            if isinstance(verif, dict):
                                status = verif.get("status")
                                findings = verif.get("findings") or []
                                has_physical_warning = (
                                    status == "WARNING"
                                    and any(
                                        isinstance(f, dict)
                                        and f.get("category") == "WARNING"
                                        and f.get("code") in PHYSICAL_WARNING_CODES
                                        for f in findings
                                    )
                                )
                                state.cad_warning_fix_required = has_physical_warning
                                if not has_physical_warning:
                                    state.warning_nudge_count = 0
                    # Circuit-breaker for repeated CAD build or verification failures.
                    # Consecutive failure count resets on a clean, verified build,
                    # while per-signature counts persist across the turn to catch
                    # alternating break/revert/break loops.
                    if not state.cad_error and not state.cad_fix_required:
                        state.build_failure_count = 0
                        state.nudged_final_verification = False
                    else:
                        state.build_failure_count += 1
                        signature = (
                            self._failure_signature(state.cad_error)
                            if state.cad_error
                            else self._verification_failure_signature(state.messages)
                        )
                        state.build_failure_signatures[signature] = (
                            state.build_failure_signatures.get(signature, 0) + 1
                        )
                        if self._build_failure_exhausted(
                            state.build_failure_count,
                            state.build_failure_signatures,
                            signature,
                        ):
                            self._cancel_remaining_tool_calls(
                                state.project_dir,
                                tool_calls,
                                processed_call_ids,
                                state.messages,
                            )
                            self.publish(
                                "agent_error",
                                {
                                    "project": state.project,
                                    "run_id": state.run_id,
                                    "message": (
                                        "Task stopped after repeated CAD build failures. "
                                        "Review the latest error or continue with a narrower repair."
                                    ),
                                },
                            )
                            self._publish_terminal_failure(state.project, run_id=state.run_id)
                            _close_dangling_tool_tail(
                                state.project_dir,
                                state.messages,
                                "Task stopped after repeated CAD build failures.",
                            )
                            return _TurnOutcome.BUILD_FAILURE_EXHAUSTED
                if waiting:
                    return self._park_for_question(
                        state, tool_calls, processed_call_ids
                    )
            if mutation_failed_in_wave and len(processed_call_ids) < len(tool_calls):
                # Later waves (typically ``cad_build``) would run against a
                # stale ``model.scad``; skip them so the model fixes the edit.
                self._cancel_remaining_tool_calls(
                    state.project_dir,
                    tool_calls,
                    processed_call_ids,
                    state.messages,
                    reason="Tool call skipped: an earlier write_file/edit_file in this turn failed.",
                )
                break
        # A turn counts once regardless of how many parallel calls it held.
        if any(
            is_model_mutation(name) or is_cad_build(name)
            for name in (c.get("function", {}).get("name", "") for c in tool_calls)
        ):
            state.consecutive_readonly_turns = 0
            state.calculation_warning_sent = False
        else:
            state.consecutive_readonly_turns += 1
        return _TurnOutcome.CONTINUE

    def _park_for_question(
        self,
        state: _RunState,
        tool_calls: list[dict[str, Any]],
        processed_call_ids: set[str],
    ) -> _TurnOutcome:
        """Park the agent waiting for a user answer.

        The model's tool result already captured the question text; the
        UI surfaces it via ``.agent_state.json`` and the
        ``agent_status:waiting_for_user`` SSE event. This helper just
        finalises the transcript and emits the status event. The run's
        model/effort selection is persisted next to the question *before*
        the status event so ``answer()`` can resume with the same choice.
        """
        run_settings = state.settings or self.settings
        agent_state = self._agent_state(state.project)
        agent_state["model_key"] = run_settings.active_model_key
        agent_state["reasoning_effort"] = run_settings.reasoning_effort or ""
        try:
            atomic_write_json(state.project_dir / ".agent_state.json", agent_state)
        except OSError:
            pass
        self.publish(
            "agent_status",
            {
                "project": state.project,
                "run_id": state.run_id,
                "status": "waiting_for_user",
                "message": "Waiting for user input.",
            },
        )
        self._cancel_remaining_tool_calls(
            state.project_dir, tool_calls, processed_call_ids, state.messages
        )
        _close_dangling_tool_tail(
            state.project_dir,
            state.messages,
            "Question sent to the user; this turn is parked until a reply arrives.",
        )
        return _TurnOutcome.PARKED_QUESTION

    def _handle_stop_pre_turn(self, state: _RunState) -> None:
        """Stop signal observed before the next turn started."""
        self.publish(
            "agent_status",
            {
                "project": state.project,
                "run_id": state.run_id,
                "status": "stopped",
                "message": "Task stopped.",
            },
        )
        if (
            not _close_dangling_tool_tail(
                state.project_dir,
                state.messages,
                "Task stopped by the user before the next turn started.",
            )
            and state.messages
            and state.messages[-1].get("role") == "user"
        ):
            item = {"role": "assistant", "content": "Task stopped."}
            state.messages.append(item)
            ConversationStore.append(state.project_dir, item)

    @staticmethod
    def _tool_call_succeeded(call_id: str, messages: list[dict[str, Any]]) -> bool:
        """Return False only when the result for ``call_id`` reports ``ok: false``."""
        for msg in _trailing_tool_batch(messages):
            if msg.get("tool_call_id") == call_id:
                payload = _tool_payload(msg)
                return payload is None or payload.get("ok") is not False
        return True

    def _report_step_budget(self, state: _RunState, used_steps: int) -> None:
        """Tell the model its remaining step budget without touching the cached prefix.

        ``step_budget`` is written into the newest tool result of the turn
        that just finished, before that message is ever sent, and one-shot
        synthetic nudges are appended behind it when the budget runs low or
        the model stalls in read-only turns. Messages already sent are never
        mutated, so the provider prompt-cache prefix stays byte-identical.
        One turn (including all of its parallel calls) costs one step.
        """
        run_settings = state.settings or self.settings
        limit = run_settings.agent_tool_call_limit
        remaining = limit - used_steps
        budget = {"step": used_steps, "limit": limit, "remaining": remaining}
        for msg in _trailing_tool_batch(state.messages):
            payload = _tool_payload(msg)
            if payload is not None:
                if payload.get("step_budget") != budget:
                    payload["step_budget"] = budget
                    _set_tool_payload(msg, payload)
                    ConversationStore.replace_tool_result(state.project_dir, msg)
                break

        if state.messages and state.messages[-1].get("role") == "user":
            return
        warn_threshold = 5 if limit >= 15 else max(1, limit // 3)
        if 1 <= remaining <= warn_threshold and not state.budget_warning_sent:
            state.budget_warning_sent = True
            text = format_budget_warning_nudge(remaining, MODEL_FILENAME)
        elif state.consecutive_readonly_turns >= 2 and not state.calculation_warning_sent:
            state.calculation_warning_sent = True
            text = format_calculation_limit_nudge(
                state.consecutive_readonly_turns, remaining, MODEL_FILENAME
            )
        else:
            return
        nudge = _synthetic_user(text)
        state.messages.append(nudge)
        self._append_message(state.project_dir, nudge)

    def _handle_tool_limit_reached(self, state: _RunState) -> None:
        """The ``agent.tool_call_limit`` budget was exhausted."""
        run_settings = state.settings or self.settings
        limit = run_settings.agent_tool_call_limit
        self.publish(
            "agent_error",
            {
                "project": state.project,
                "run_id": state.run_id,
                "message": (
                    f"Tool-call limit ({limit}) reached; "
                    "increase agent.tool_call_limit or continue with a narrower request."
                ),
            },
        )
        self._publish_terminal_failure(state.project, run_id=state.run_id)
        # If the final iteration processed at least one tool call, the
        # transcript still ends with ``role: tool``; close the tail so
        # the next user turn has a well-formed prompt prefix.
        if (
            not _close_dangling_tool_tail(
                state.project_dir,
                state.messages,
                (
                    f"Tool-call limit ({limit}) "
                    "reached without resolving the task."
                ),
            )
            and state.messages
            and state.messages[-1].get("role") == "user"
        ):
            item = {
                "role": "assistant",
                "content": f"Tool-call limit ({limit}) reached.",
                "synthetic": True,
            }
            state.messages.append(item)
            ConversationStore.append(state.project_dir, item)

    def _handle_request_cancelled(self, state: _RunState | None) -> None:
        """``RequestCancelled`` raised mid-iteration: expected control flow.

        Stopping is not a provider error, so the user sees the same
        ``agent_status: stopped`` event as a stop between iterations.
        When interrupted, any content or reasoning streamed so far is preserved
        into the conversation history as an assistant turn so the output is not lost.
        """
        if state is None:
            return
        self.publish(
            "agent_status",
            {
                "project": state.project,
                "run_id": state.run_id,
                "status": "stopped",
                "message": "Task stopped.",
            },
        )
        partial_content = (getattr(state, "current_content", "") or "").strip()
        partial_reasoning = (getattr(state, "current_reasoning", "") or "").strip()

        if partial_content or partial_reasoning:
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": partial_content or "Task stopped.",
            }
            if partial_reasoning:
                assistant_message["reasoning"] = partial_reasoning
            ConversationStore.append(state.project_dir, assistant_message)
            state.messages.append(assistant_message)
            self.publish(
                "agent_stream_end",
                {
                    "project": state.project,
                    "run_id": state.run_id,
                    "message_id": state.current_message_id,
                    "message": assistant_message["content"],
                    "reasoning": partial_reasoning,
                    "has_tools": False,
                },
            )
        else:
            if _close_dangling_tool_tail(
                state.project_dir,
                state.messages,
                "Task stopped by the user while the model was responding.",
            ):
                self.publish(
                    "agent_stream_end",
                    {
                        "project": state.project,
                        "run_id": state.run_id,
                        "message_id": state.current_message_id,
                        "message": "Task stopped.",
                        "reasoning": "",
                        "has_tools": False,
                    },
                )
            elif state.messages and state.messages[-1].get("role") == "user":
                item = {"role": "assistant", "content": "Task stopped."}
                state.messages.append(item)
                ConversationStore.append(state.project_dir, item)
                self.publish(
                    "agent_stream_end",
                    {
                        "project": state.project,
                        "run_id": state.run_id,
                        "message_id": state.current_message_id,
                        "message": item["content"],
                        "reasoning": "",
                        "has_tools": False,
                    },
                )

    def _handle_unexpected_error(
        self,
        error: BaseException,
        project: str,
        state: _RunState | None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Surface any unhandled exception to the local UI.

        ``state`` may be ``None`` when the failure happened during
        ``_init_run``; in that case we fall back to
        ``settings.workspace_root / project`` so the operator audit
        trail still lands in the right place, and to the run's
        ``stop_event`` to tell a user stop from a genuine failure.
        """
        detail = str(error)
        err_type = type(error).__name__
        tb_text = traceback.format_exc()
        traceback.print_exc()
        project_dir = (
            state.project_dir
            if state is not None
            else self.settings.workspace_root / project
        )
        run_id = state.run_id if state is not None else ""
        run_settings = (state.settings if state is not None else None) or self.settings
        # Route through ``debug-errors.jsonl`` so agent-loop faults
        # share the same audit trail as tool errors instead of
        # vanishing into stderr.
        self._debug_tool_error(
            project_dir,
            call_id="",
            tool="agent_loop",
            error=error,
            result="",
            phase="agent_loop",
            traceback_text=tb_text,
            run_id=run_id,
            settings=run_settings,
        )
        current_message_id = (
            state.current_message_id if state is not None else None
        )
        if current_message_id is not None:
            stream_end_payload: dict[str, Any] = {
                "project": project,
                "message_id": current_message_id,
                "message": "",
            }
            if run_id:
                stream_end_payload["run_id"] = run_id
            self.publish("agent_stream_end", stream_end_payload)
        stop_event = state.stop_event if state is not None else stop_event
        if stop_event is not None and stop_event.is_set():
            stop_payload: dict[str, Any] = {
                "project": project,
                "status": "stopped",
                "message": "Task stopped.",
            }
            if run_id:
                stop_payload["run_id"] = run_id
            self.publish("agent_status", stop_payload)
            return
        err_payload: dict[str, Any] = {
            "project": project,
            "error_type": err_type,
            "message": self._user_error_message(
                detail,
                err_type,
                provider=run_settings.llm_provider,
            ),
        }
        if run_id:
            err_payload["run_id"] = run_id
        self.publish("agent_error", err_payload)
        self._publish_terminal_failure(project, run_id=run_id)

    def _finalize_run(
        self,
        project: str,
        state: _RunState | None,
        activity_logger: ActivityLogger | None,
        run_id: str,
    ) -> None:
        """Terminal ``finally`` block — runs on every exit path.

        Session cleanup is guarded by ``run_id`` so an exiting thread
        from a stopped run never clobbers a newly started run's state.
        """
        with self._lock:
            session = self._sessions.get(project)
            if session is not None and session.run_id == run_id:
                session.run_complete.set()
                del self._sessions[project]
        if activity_logger is not None:
            try:
                if state is not None:
                    was_stopped = state.stop_event.is_set()
                    if was_stopped:
                        outcome = "USER_STOP"
                    elif state.cad_error:
                        outcome = "CAD_ERROR"
                    elif not state.cad_fix_required and state.preview_id:
                        outcome = "SUCCESS"
                    elif state.model_mutated and not state.preview_id:
                        outcome = "DRAWING_NOT_CREATED"
                    else:
                        outcome = "INCOMPLETE"

                    total_duration_ms = (
                        round((time.perf_counter() - state.start_perf_time) * 1000, 2)
                        if state.start_perf_time
                        else 0.0
                    )

                    activity_logger.log(
                        "run_end",
                        {
                            "project": project,
                            "outcome": outcome,
                            "cad_valid": not state.cad_fix_required and bool(state.preview_id),
                            "preview_id": state.preview_id,
                            "total_turns": state.total_turns,
                            "tool_calls": state.tool_call_count,
                            "duration_ms": total_duration_ms,
                            "cancelled": was_stopped,
                            "usage": {
                                "prompt_tokens": state.total_prompt_tokens,
                                "completion_tokens": state.total_completion_tokens,
                                "reasoning_tokens": state.total_reasoning_tokens,
                            },
                        },
                        run_id=run_id,
                    )
            except Exception:  # noqa: BLE001, S110
                pass
            finally:
                activity_logger.close()
        # Run-scoped inputs are owned by the runner from the moment
        # ``start()`` moves them in; HTTP handlers must never
        # speculatively unlink these files. ``ignore_errors`` because
        # cleanup is incidental — a partial removal here must not mask
        # the agent's terminal state.
        self._discard_run_directory(project, run_id)

    # ------------------------------------------------------------------ context

    def _context(
        self,
        project_dir: Path,
        message: str,
        image_paths: list[Path],
        *,
        is_retry: bool = False,
        settings: Settings | None = None,
    ) -> list[dict]:
        eff_settings = settings or self.settings
        history = self._load_history(project_dir)
        if not is_retry:
            user_message: dict[str, object] = {"role": "user", "content": message}
            if image_paths:
                user_message = {
                    "role": "user",
                    "content": [{"type": "text", "text": message}]
                    + [as_chat_image(path) for path in image_paths],
                }
            if not history or history[-1] != user_message:
                history.append(user_message)
                self._append_message(project_dir, user_message)
        return [
            {
                "role": "system",
                "content": get_system_prompt(
                    bosl2_enabled=eff_settings.openscad_bosl2_enabled
                ),
            },
            self._project_state_message(
                project_dir,
                history,
                is_retry=is_retry,
                user_printer=eff_settings.user_printer,
                user_nozzle=eff_settings.user_nozzle,
                user_filament=eff_settings.user_filament,
                user_additional_info=eff_settings.user_additional_info,
            ),
            *history,
        ]

    @classmethod
    def _project_state_message(
        cls,
        project_dir: Path,
        history: list[dict] | None = None,
        *,
        is_retry: bool = False,
        persist: bool = True,
        user_printer: str = "",
        user_nozzle: str = "",
        user_filament: str = "",
        user_additional_info: str = "",
    ) -> dict[str, str]:
        """Provide initial workspace state after the cacheable system prefix.

        The state reflects whether model.scad existed at the start of the conversation,
        frozen across turns so provider prompt caching remains valid.
        See :data:`agent.prompt._OPERATIONAL_RULES` for the instruction the
        model receives.
        """
        existed, code = cls._initial_model_state(
            project_dir, history, is_retry=is_retry, persist=persist
        )
        content = format_project_state(
            existed,
            MODEL_FILENAME,
            code,
            user_printer=user_printer,
            user_nozzle=user_nozzle,
            user_filament=user_filament,
            user_additional_info=user_additional_info,
        )
        # Gemini normalizes all system messages into an immutable instruction.
        # Keeping workspace state in a later user message lets its
        # explicit system-message cache breakpoint remain reusable.
        return {"role": "user", "content": content}

    @classmethod
    def _initial_model_existed(
        cls, project_dir: Path, history: list[dict] | None = None
    ) -> bool:
        """Determine whether model.scad existed at the start of this conversation."""
        existed, _ = cls._initial_model_state(project_dir, history)
        return existed

    @classmethod
    def _initial_model_state(
        cls,
        project_dir: Path,
        history: list[dict] | None = None,
        *,
        is_retry: bool = False,
        persist: bool = True,
    ) -> tuple[bool, str]:
        """Determine whether model.scad existed and its content at the start of this conversation.

        To preserve byte-identical prompt prefixes across all turns (enabling
        provider-side prompt caching), the initial project state must remain
        frozen for the duration of the conversation.
        """
        state_file = project_dir / ".agent_initial_state.json"
        if state_file.is_file():
            try:
                data = json.loads(state_file.read_text("utf-8"))
                if isinstance(data, dict) and "existed" in data:
                    existed = bool(data["existed"])
                    content = str(data.get("content", ""))
                    should_refresh = (
                        not history
                        if not persist
                        else (not is_retry and (not history or len(history) <= 1))
                    )
                    if should_refresh:
                        model_file = project_dir / MODEL_FILENAME
                        real_exists = model_file.is_file()
                        current_content = (
                            model_file.read_text("utf-8", errors="replace")
                            if (real_exists and model_file.stat().st_size <= 100_000)
                            else ""
                        )
                        if real_exists != existed or current_content != content:
                            existed = real_exists
                            content = current_content
                            if persist:
                                try:
                                    atomic_write_json(
                                        state_file,
                                        {"existed": existed, "content": content},
                                    )
                                except OSError:
                                    pass
                    return existed, content
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                pass

        model_file = project_dir / MODEL_FILENAME
        existed = False
        content = ""
        has_mutation = False
        if history and len(history) > 1:
            first_file_tool = None
            for m in history:
                if isinstance(m, dict) and m.get("role") == "assistant":
                    tool_calls = m.get("tool_calls") or []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn_name = tc.get("function", {}).get("name")
                            if fn_name in {"write_file", "edit_file"}:
                                has_mutation = True
                            if first_file_tool is None and fn_name in {
                                "write_file",
                                "edit_file",
                                "read_file",
                            }:
                                first_file_tool = fn_name
            if first_file_tool is not None:
                existed = first_file_tool in {"edit_file", "read_file"}
            else:
                existed = model_file.is_file()
        else:
            existed = model_file.is_file()

        if existed and not has_mutation and model_file.is_file():
            try:
                if model_file.stat().st_size <= 100_000:
                    content = model_file.read_text("utf-8", errors="replace")
            except OSError:
                pass

        if persist:
            try:
                atomic_write_json(
                    state_file,
                    {"existed": existed, "content": content},
                )
            except OSError:
                pass
        return existed, content

    @staticmethod
    def _last_message_has_tool_image(messages: list[dict]) -> bool:
        return any(
            isinstance(msg.get("content"), list)
            and any(
                isinstance(part, dict) and part.get("type") == "image_url"
                for part in msg["content"]
            )
            for msg in _trailing_tool_batch(messages)
        )

    @staticmethod
    def _failure_signature(message: str) -> str:
        """Collapse volatile line numbers so equivalent failures match.

        Returns a short hash of the *full* normalized text. Hashing the entire
        message (instead of slicing the trailing 1000 chars) prevents
        distinct long errors from accidentally colliding when the tail of the
        text happens to line up — e.g. an OOM traceback that ends in a shared
        subprocess stderr tail.
        """
        import hashlib

        normalized = re.sub(r"\bline \d+\b", "line #", message.lower())
        normalized = re.sub(r"\b0x[0-9a-f]+\b", "0x#", normalized)
        # The sandbox stages the build under a fresh ``/tmp/<random>/...``
        # directory every call, so two occurrences of the same logical error
        # carry different absolute paths. Without this rule the signature
        # drifts on every retry and
        # :meth:`AgentRunner._build_failure_exhausted`'s
        # ``signatures[signature] >= 3`` loop-breaker never fires. Match
        # either the immediate subfolder (``/tmp/tmp_xyz123/model.scad``) or
        # the staging namespace (``/tmp/tmp_*/.staging/...``).
        normalized = re.sub(r"/tmp/[^/\s]+/", "/tmp/<scratch>/", normalized)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def _verification_failure_signature(cls, messages: list[dict]) -> str:
        """Build a stable signature from critical CAD verification findings in the latest tool batch."""
        for msg in _trailing_tool_batch(messages):
            payload = _tool_payload(msg) or {}
            data = payload.get("data")
            verif = data.get("verification") if isinstance(data, dict) else None
            findings = verif.get("findings") if isinstance(verif, dict) else None
            codes = sorted(
                str(f.get("code"))
                for f in (findings if isinstance(findings, list) else [])
                if isinstance(f, dict) and f.get("category") == "CRITICAL" and f.get("code")
            )
            if codes:
                return cls._failure_signature("VERIFY_FAIL:" + ",".join(codes))
        return cls._failure_signature("verification_failed")

    @staticmethod
    def _build_failure_exhausted(
        total_count: int,
        signatures: dict[str, int],
        signature: str,
    ) -> bool:
        """Return True when the build-failure circuit breaker should fire.

        Stops the agent when *either*:

        - ``total_count`` (the overall failure count for this run) hits
          :data:`_BUILD_FAILURE_TOTAL_MAX`, or
        - the per-signature count for ``signature`` hits
          :data:`_BUILD_FAILURE_PER_SIGNATURE_MAX` — catching repair loops
          where the agent keeps producing the same logical error.

        Extracted from :meth:`_run` so tests can target the predicate
        directly instead of re-implementing ``signatures[signature] >= 3``
        in user space (which silently drifts if the constants change).
        """
        return (
            total_count >= _BUILD_FAILURE_TOTAL_MAX
            or signatures.get(signature, 0) >= _BUILD_FAILURE_PER_SIGNATURE_MAX
        )

    @classmethod
    def _load_history(cls, project_dir: Path) -> list[dict]:
        """Load the canonical conversation.jsonl for ``project_dir``."""
        return ConversationStore.load(project_dir)

    @classmethod
    def _append_message(cls, project_dir: Path, message: dict) -> None:
        """Append a message to the canonical conversation log."""
        ConversationStore.append(project_dir, message)

    @classmethod
    def clear_history(cls, project_dir: Path) -> bool:
        """Reset the agent's memory for ``project_dir``.

        Truncates the canonical ``conversation.jsonl`` log, evicts the
        in-memory history cache entry, and removes the per-project agent
        state files (``agent_state.json`` and ``.agent_initial_state.json``).
        Returns ``True`` when a log file was removed, ``False`` when the
        project had no recorded conversation. The model, preview, renders,
        and revision blobs are left untouched.
        """
        (project_dir / ".agent_state.json").unlink(missing_ok=True)
        # Clear the initial state cache so the next conversation run resets cleanly.
        (project_dir / ".agent_initial_state.json").unlink(missing_ok=True)
        return ConversationStore.clear(project_dir)

    @staticmethod
    def _accumulate_state_usage(state: _RunState, usage: dict | None) -> None:
        summary = extract_usage_summary(usage)
        state.total_prompt_tokens += summary["prompt_tokens"]
        state.total_completion_tokens += summary["completion_tokens"]
        state.total_reasoning_tokens += summary["reasoning_tokens"]
        state.total_cached_tokens += summary["cached_tokens"]
        if summary["cost"] is not None:
            state.total_cost += summary["cost"]

    def _publish_usage(
        self,
        project: str,
        usage: dict | None,
        state: _RunState | None = None,
    ) -> None:
        if not usage:
            return
        summary = extract_usage_summary(usage)
        cached_tokens = summary["cached_tokens"]
        p_val = summary["prompt_tokens"]
        c_val = summary["completion_tokens"]
        r_val = summary["reasoning_tokens"]
        cost_val = summary["cost"]
        total_cost = round(state.total_cost, 6) if state and state.total_cost > 0 else cost_val
        preserve_r = bool(state and getattr(state.client, "preserve_reasoning", False))
        persisted_out = c_val if preserve_r else max(0, c_val - r_val)
        eff_settings = state.settings if state and state.settings else self.settings
        context_used = (p_val + persisted_out) if p_val > 0 else None
        context_limit = resolve_context_limit(eff_settings, allow_network=False)
        usage_payload: dict[str, Any] = {
            "project": project,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": cached_tokens,
            "cache_write_tokens": summary["cache_write_tokens"],
            "cache_hit_rate": summary["cache_hit_rate"],
            "cost": cost_val,
            "total_cost": total_cost,
            "total_prompt_tokens": state.total_prompt_tokens if state else p_val,
            "total_completion_tokens": state.total_completion_tokens if state else c_val,
            "total_cached_tokens": state.total_cached_tokens if state else cached_tokens,
            "context_used_tokens": context_used,
            "context_limit_tokens": context_limit,
            "model_key": eff_settings.active_model_key,
        }
        if state and state.run_id:
            usage_payload["run_id"] = state.run_id
        self.publish("agent_usage", usage_payload)

    # ------------------------------------------------------------------ dispatch

    @classmethod
    def _cancel_remaining_tool_calls(
        cls,
        project_dir: Path,
        tool_calls: list[dict],
        processed_call_ids: set[str],
        messages: list[dict],
        reason: str | None = None,
    ) -> None:
        """Per-call wrapper that injects the runner's append callback into the dispatcher."""
        cancel_remaining_tool_calls(
            project_dir,
            tool_calls,
            processed_call_ids,
            messages,
            append_message=cls._append_message,
            reason=reason,
        )

    @staticmethod
    def _user_error_message(detail: str, err_type: str, *, provider: str = "openrouter") -> str:
        """Map technical error messages to user-friendly messages."""
        lower = detail.lower()
        provider_name = provider_label(provider)
        if provider == "openai":
            key_url = "https://platform.openai.com/api-keys"
        elif provider in ("gemini", "google"):
            key_url = "https://aistudio.google.com/apikey"
        elif provider in ("mimo", "xiaomi", "xiaomi_mimo", "xiaomi-mimo", "xiaomi mimo"):
            key_url = "https://platform.xiaomimimo.com"
        else:
            key_url = "https://openrouter.ai/keys"

        # Cancellation takes priority: \"cancelled" / "stop" substrings
        # are common in unrelated error messages, so check ``err_type``
        # first and only fall back to the substring match for callers
        # that did not pass the type.
        if err_type == "RequestCancelled" or "task was cancelled" in lower:
            return "Task was cancelled."
        if (
            err_type == "UpstreamModerationError"
            or "blocked the response via content filter" in lower
            or "canned moderation refusal" in lower
        ):
            return (
                f"{provider_name} content filter flagged the response (likely false-positive on mechanical CAD terminology). "
                "Retry the request or switch to a different model."
            )
        if (
            err_type == "CompletionTruncatedError"
            or "completion was truncated" in lower
            or "finish_reason='length'" in lower
            or "finish_reason='max_tokens'" in lower
        ):
            return (
                f"{provider_name} model output was truncated by token limits before completing tools. "
                "Consider reducing 'reasoning_effort' (e.g. to 'medium' or 'low') or setting "
                "'reasoning_max_tokens' in config.yaml."
            )
        if "401" in detail or "unauthorized" in lower or "invalid api key" in lower:
            if provider == "ollama":
                return "Authentication failed for Ollama. Check your OLLAMA_API_KEY environment variable."
            if provider in ("vertex", "gcloud"):
                return "Google Cloud authentication failed. Run 'gcloud auth login' in your terminal."
            return f"Invalid {provider_name} API key. Check your key at {key_url}."
        if provider in ("vertex", "gcloud") and "project" in lower and ("not configured" in lower or "not found" in lower):
            return (
                f"Google Cloud project error: {detail}. "
                "Set 'vertex.project_id' in config.yaml or run 'gcloud config set project <PROJECT_ID>'."
            )
        if "429" in detail or "rate limit" in lower:
            return f"{provider_name} rate limit reached. Wait a moment and try again."
        if provider == "ollama":
            if (
                "connection refused" in lower
                or "failed to establish a new connection" in lower
                or "could not connect" in lower
                or "max retries exceeded" in lower
            ):
                return (
                    "Could not connect to Ollama server at http://localhost:11434. "
                    "Ensure Ollama is running (`ollama serve`)."
                )
            if "not found" in lower and ("model" in lower or "try pulling" in lower):
                return (
                    f"Ollama model not found: {detail}. "
                    "Make sure the model is pulled (`ollama pull <model>`) or use `./setup_ollama.sh`."
                )
        if (
            "model not found" in lower
            or "model not available" in lower
            or "model is invalid" in lower
            or "invalid model" in lower
        ):
            return f"The configured model is not available: {detail}"
        if (
            "bubblewrap" in lower
            or "bwrap" in lower
            or "sandbox" in lower
            or "seccomp" in lower
        ):
            return "CAD sandbox failed. Ensure bubblewrap and libseccomp2 are installed (sudo apt install bubblewrap libseccomp2)."
        if (
            "ssl" in lower
            or "decryption_failed" in lower
            or "bad record mac" in lower
            or "handshake" in lower
            or "tlsv" in lower
        ):
            return (
                f"Network SSL/TLS connection error with {provider_name} ({detail}). "
                "This is usually a transient network, router, or proxy interruption. Please retry."
            )
        if (
            "openrouter" in lower
            or "openai" in lower
            or "ollama" in lower
            or "gemini" in lower
            or "google" in lower
            or "vertex" in lower
            or "gcloud" in lower
            or "mimo" in lower
            or "xiaomi" in lower
            or provider in (
                "openrouter",
                "openai",
                "ollama",
                "gemini",
                "google",
                "vertex",
                "gcloud",
                "mimo",
                "xiaomi",
                "xiaomi_mimo",
                "xiaomi-mimo",
                "xiaomi mimo",
            )
        ) and (
            "timeout" in lower or "timed out" in lower or "connection" in lower
        ):
            if provider == "ollama":
                return "Connection to Ollama server timed out. Check if Ollama is responsive."
            return f"Connection to {provider_name} timed out. Check your internet connection."
        if "timeout" in lower or "timed out" in lower:
            return "CAD code execution timed out. Try simplifying the design or increasing the timeout."
        if "permission" in lower or "access denied" in lower or "not writable" in lower:
            return f"Workspace permission error: {detail}"

        return detail

    # ------------------------------------------------------------------ preview tracking

    def _register_preview(self, project: str, project_dir: Path) -> str:
        """Return a fresh ``preview_id`` correlation token.

        The token is published alongside ``preview_updated`` so the UI can
        decide whether to fetch the new STL. The host's render-failure path
        surfaces any consumer error through ``agent_error`` independently.
        """
        preview_path = project_dir / "preview.stl"
        if not preview_path.is_file() or preview_path.stat().st_size == 0:
            # The CAD tool already produces a structured RuntimeError with
            # the actual sandbox/OpenSCAD cause (``_failure_detail``). When
            # that signal is missing we still surface the workspace path and
            # filesystem state so the operator can see whether the file was
            # never produced or removed after the build returned.
            exists = preview_path.is_file()
            size = preview_path.stat().st_size if exists else 0
            raise RuntimeError(
                "CAD execution did not save a usable preview: "
                f"preview.stl missing or empty at {preview_path} "
                f"(exists={exists}, size={size})."
            )
        return uuid.uuid4().hex

    def _publish_terminal_failure(self, project: str, run_id: str = "") -> None:
        # Mirror ``_complete``'s success-side ``agent_status`` so the UI
        # clears the thinking indicator on every error path; ``stopped``
        # is already published for user-initiated stops. Transient so
        # the ``agent_error`` event that preceded this remains the
        # canonical terminal record.
        payload: dict[str, Any] = {
            "project": project,
            "status": "failed",
            "message": "Task failed.",
        }
        if run_id:
            payload["run_id"] = run_id
        self.publish(
            "agent_status",
            payload,
            transient=True,
        )

    def _complete(
        self,
        project: str,
        message: str,
        reasoning: str = "",
        run_id: str = "",
        message_id: str = "",
    ) -> None:
        """Persist the final assistant turn and publish it.

        Ensures the final user-facing response is recorded even when
        the loop reaches ``_complete`` without persisting the message
        (e.g. a no-tool-calls path that branched earlier).
        """
        cleaned_msg, _ = extract_text_tool_calls(message)
        message = cleaned_msg if cleaned_msg is not None else message
        if reasoning and message.strip() == reasoning.strip():
            reasoning = ""

        project_dir = self.settings.workspace_root / project
        history = self._load_history(project_dir)
        if not history or not (
            isinstance(history[-1], dict)
            and history[-1].get("role") == "assistant"
            and history[-1].get("content") == message
        ):
            record: dict[str, Any] = {"role": "assistant", "content": message}
            if reasoning:
                record["reasoning"] = reasoning
            self._append_message(project_dir, record)
        msg_payload: dict[str, Any] = {"project": project, "message": message}
        if run_id:
            msg_payload["run_id"] = run_id
        if message_id:
            msg_payload["message_id"] = message_id
        if reasoning:
            msg_payload["reasoning"] = reasoning
        self.publish("agent_message", msg_payload)
        # Transient so the ``agent_message`` above remains the canonical
        # terminal entry in the conversation log.
        status_payload: dict[str, Any] = {
            "project": project,
            "status": "completed",
            "message": "Task completed.",
        }
        if run_id:
            status_payload["run_id"] = run_id
        self.publish(
            "agent_status",
            status_payload,
            transient=True,
        )

    def _debug_tool_error(
        self,
        project_dir: Path,
        call_id: str,
        tool: str,
        error: BaseException,
        result: str,
        *,
        phase: str = "tool_call",
        traceback_text: str | None = None,
        run_id: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Append recoverable tool failures to a project-local debug log.

        ``phase`` distinguishes the dispatcher ``tool_call`` records (which
        carry the ``tool_results.failure`` classification envelope) from
        internal ``agent_loop`` faults (which carry an optional
        ``traceback_text``). The shared file format keeps the on-disk
        footprint stable for downstream tools.
        """
        eff_settings = settings or self.settings
        if not eff_settings.agent_debug_log_tool_errors:
            return
        try:
            entry: dict[str, object] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id or "",
                "call_id": call_id,
                "tool": tool,
                "phase": phase,
                "error_type": type(error).__name__,
                "message": str(error),
            }
            if phase == "tool_call":
                # Best-effort: decode the existing tool_results.failure
                # envelope for downstream reconciliation. Skip silently when
                # the result is not the expected JSON shape.
                try:
                    payload = json.loads(result) if isinstance(result, str) else None
                except (TypeError, ValueError):
                    payload = None
                error_detail = (
                    payload.get("error", {}) if isinstance(payload, dict) else {}
                )
                entry["classification"] = error_detail
            if traceback_text:
                # Cap traceback size so a runaway loop does not grow the log
                # without bound; the cap mirrors the dispatcher's
                # ``_failure_detail`` truncation.
                entry["traceback"] = traceback_text[-4000:]
            with (
                project_history_lock(project_dir),
                (project_dir / "debug-errors.jsonl").open(
                    "a", encoding="utf-8"
                ) as log,
            ):
                log.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError):
            # Debug logging must not affect the agent's recovery path.
            return

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
                    if len(questions) == 1 and isinstance(questions[0], dict):
                        qid = str(questions[0].get("id", ""))
                        if qid and qid not in answers:
                            if "answer" in answers:
                                answers = {qid: answers["answer"]}
                            elif len(answers) == 1:
                                answers = {qid: next(iter(answers.values()))}
                    lines = ["User answers:"]
                    for q in questions:
                        qid = q.get("id", "")
                        qtext = q.get("question", qid)
                        value = answers.get(qid)
                        if value is not None and value != "":
                            if isinstance(value, list):
                                formatted_val = ", ".join(str(v) for v in value)
                            else:
                                formatted_val = str(value)
                            lines.append(f"- {qtext}: {formatted_val}")
                    return "\n".join(lines)
        else:
            try:
                parsed = json.loads(answer)
            except json.JSONDecodeError:
                return answer
            if isinstance(parsed, dict) and ("answer" in parsed or len(parsed) == 1):
                parsed = parsed["answer"] if "answer" in parsed else next(iter(parsed.values()))
                if not isinstance(parsed, list):
                    return str(parsed)
            if isinstance(parsed, list):
                return ", ".join(str(v) for v in parsed)
        return answer

    @staticmethod
    def _log(project_dir: Path, role: str, content: object) -> None:
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "content": content,
        }
        # Route through ``ConversationStore.append`` so the canonical write
        # path holds ``project_history_lock`` and updates the cache. The
        # previous direct ``open(..., "a")`` shortcut interleaved bytes with
        # concurrent canonical writes and left the in-memory cache stale,
        # which then dropped the question-log line on the next history load.
        ConversationStore.append(project_dir, item)


def _close_dangling_tool_tail(
    project_dir: Path,
    messages: list[dict],
    content: str,
) -> bool:
    """Append a terminal ``role: assistant`` message if the tail dangles.

    Several abrupt termination paths (user stop, build-failure threshold,
    question waiting, ``RequestCancelled`` inside the LLM call, and the
    tool-call limit guard) leave the in-memory transcript and the on-disk
    JSONL log appended with one or more ``role: tool`` records and no
    following assistant turn. Anthropic and OpenAI-compatible providers
    reject the next ``role: user`` request when tool results have not been
    closed by an assistant turn (``400 Bad Request: roles must alternate /
    tool results must be followed by assistant turn``).

    Persisting the same terminal message via :func:`ConversationStore.append`
    keeps the in-memory ``messages`` list and the on-disk ``conversation.jsonl``
    transcript in lockstep, so the next user turn sees a clean
    ``[..., assistant]`` closing turn.
    """
    if not messages or messages[-1].get("role") != "tool":
        return False
    item = {"role": "assistant", "content": content, "synthetic": True}
    messages.append(item)
    ConversationStore.append(project_dir, item)
    return True
