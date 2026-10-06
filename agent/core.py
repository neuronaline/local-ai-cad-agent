"""Background tool-calling loop for one local CAD task at a time.

The agent runs one user request at a time per workspace.  A worker thread
streams the model's chat-completions response, dispatches tool calls, and
re-prompts the model with the tool results until it returns a final message
or the configured tool-call budget is exhausted.
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from agent.activity_log import ActivityLogger, get_logger
from agent.activity_log import is_enabled as activity_logging_enabled
from agent.conversation import (
    ConversationStore,
    shared_history_lock,
)
from agent.dispatcher import (
    cancel_remaining_tool_calls,
    is_cad_build,
    normalize_tool_calls,
    process_tool_call,
)
from agent.images import as_chat_image
from agent.llm_base import (
    RequestCancelled,
    create_llm_client,
    extract_text_tool_calls,
    provider_label,
    sanitize_assistant_message,
)
from agent.prompt import (
    NUDGE_FINAL_VERIFICATION,
    format_project_state,
    format_unverified_model_nudge,
    get_system_prompt,
)
from agent.revisions import (
    MODEL_FILENAME,
    RevisionStore,
    model_is_built,
)
from agent.settings import Settings
from agent.tool_schemas import TOOL_SCHEMAS
from agent.tools.cad_tool import CadTool
from agent.tools.file_tool import FileTool
from agent.tools.image_tool import ImageTool
from agent.tools.question_tool import QuestionTool
from agent.tools.question_validator import QuestionValidator

# Circuit-breaker thresholds for repeated ``cad_build`` failures.
# ``_BUILD_FAILURE_TOTAL_MAX`` stops the agent after a broad burst of mixed
# errors; ``_BUILD_FAILURE_PER_SIGNATURE_MAX`` stops it earlier when the
# same logical error keeps repeating (useful for catching repair loops).
# Both values flow through :meth:`AgentRunner._build_failure_exhausted` so
# tests can target the predicate directly without re-rolling the literals.
_BUILD_FAILURE_TOTAL_MAX = 6
_BUILD_FAILURE_PER_SIGNATURE_MAX = 3

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
        )
        self.image = ImageTool(project_dir, publish)
        self.question = QuestionTool(publish)

    def stop(self) -> None:
        self.cad.stop()


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


def _shared_history_lock() -> threading.Lock:
    """Return the active history lock (configurable by the Flask app)."""
    return shared_history_lock()


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
    any_tool_used: bool = False
    nudged_cad: bool = False
    nudged_final_verification: bool = False
    build_failure_count: int = 0
    build_failure_signatures: dict[str, int] = field(default_factory=dict)
    current_message_id: str | None = None
    total_turns: int = 0
    tool_call_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_reasoning_tokens: int = 0
    run_reasonings: list[str] = field(default_factory=list)
    start_perf_time: float = 0.0


def _extract_reasoning(message: dict[str, Any]) -> str:
    """Extract plain text reasoning from reasoning or reasoning_details."""
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    details = message.get("reasoning_details")
    if isinstance(details, list):
        return "".join(
            d.get("text", "") for d in details if isinstance(d, dict) and d.get("text")
        )
    return ""


class AgentRunner:
    """One-thread-per-task chat-completions driver with tool dispatch."""

    def __init__(
        self,
        settings: Settings,
        publish: Callable[..., None],
    ) -> None:
        self.settings = settings
        self.publish = publish
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_tools: ProjectTools | None = None
        self._active_client: Any | None = None
        self._active_project: str | None = None
        # Per-run activity-log handles; ``_run()`` populates them and
        # ``finally`` clears them so a stale logger never leaks across
        # runs.
        self._active_activity_logger: ActivityLogger | None = None
        self._active_run_id: str | None = None
        self._lock = threading.Lock()
        self._waiting_questions: dict[str, dict[str, object]] = {}
        # Set by ``_run``'s finally so callers can observe completion
        # without polling ``thread.is_alive()``.
        self._run_complete: threading.Event = threading.Event()
        self._run_complete.set()

    # ------------------------------------------------------------------ state

    def is_running(self) -> bool:
        with self._lock:
            thread = self._thread
            if thread is None:
                return False
            if not thread.is_alive():
                # Thread has terminated. Clean up stale references if finally was bypassed.
                self._thread = None
                self._run_complete.set()
                self._active_project = None
                self._active_run_id = None
                self._active_tools = None
                self._active_client = None
                return False
            return not self._run_complete.is_set()

    def has_active_state_for(self, project: str) -> bool:
        """Return whether the project has an agent run that must not be deleted."""
        # A persisted waiting question is recoverable UI state, not an active
        # worker.  It must not prevent deleting the project.
        return self.is_running() and self.active_project() == project

    def active_project(self) -> str | None:
        with self._lock:
            return self._active_project

    def active_run_id(self) -> str | None:
        with self._lock:
            return self._active_run_id

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

    # ------------------------------------------------------------------ lifecycle

    def start(
        self,
        project: str,
        message: str,
        image_paths: list[Path] | None = None,
    ) -> bool:
        """Validate inputs, take ownership, then spawn the worker thread.

        Validation runs *before* ``_take_run_inputs`` so a rejection
        never leaves a half-populated run directory behind and the
        caller's originals in ``<project>/inputs/`` are untouched.
        """
        if self.waiting_question(project):
            return False
        with self._lock:
            if self._thread is not None and not self._run_complete.is_set():
                return False
            run_id = uuid.uuid4().hex
            prepared_paths = self._take_run_inputs(
                project, image_paths or [], run_id
            )
            return self._start_locked(project, message, prepared_paths, run_id)

    def _start_locked(
        self,
        project: str,
        message: str,
        image_paths: list[Path],
        run_id: str,
    ) -> bool:
        if self._thread is not None and not self._run_complete.is_set():
            return False
        self._stop_event.clear()
        self._run_complete.clear()
        self._active_project = project
        self._active_run_id = run_id
        self._thread = threading.Thread(
            target=self._run,
            args=(project, message, image_paths, run_id),
            daemon=True,
        )
        self._thread.start()
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
                shutil.move(str(source), str(target))
                moved.append(target)
        except OSError:
            # ``shutil.move`` may leave the source in place when the
            # target already exists; ``rmtree`` scrubs whatever made it
            # into the run dir.
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

    def answer(self, project: str, answer: str) -> bool:
        question = self.waiting_question(project)
        if not question or not self._validate_answer(question, answer):
            return False
        # Wait for the previous run's ``finally`` so we never observe a
        # half-completed run while still accepting a follow-up message.
        with self._lock:
            previous_event = self._run_complete
        previous_event.wait(timeout=10.0)
        project_dir = self.settings.workspace_root / project
        formatted = self._format_answer(question, answer)
        with self._lock:
            if self._thread is not None and not self._run_complete.is_set():
                return False
            # Persist the user answer BEFORE starting the new thread so
            # ``_context()`` reads it from the canonical log instead of
            # racing against this append.
            self._append_message(project_dir, {"role": "user", "content": formatted})
            if not self._start_locked(project, formatted, [], uuid.uuid4().hex):
                return False
            self._waiting_questions.pop(project, None)
            (project_dir / ".agent_state.json").unlink(missing_ok=True)
            return True
    def stop(self, project: str | None = None) -> list[str]:
        """Stop agent work and clear pending state for one or all projects."""
        affected: list[str] = []
        with self._lock:
            target_project = project or self._active_project
            stop_active_task = (
                target_project is None or target_project == self._active_project
            )
            if project is None:
                affected = list(dict.fromkeys(list(self._waiting_questions)))
                if self._active_project and self._active_project not in affected:
                    affected.append(self._active_project)
                if self.settings.workspace_root.is_dir():
                    for item in self.settings.workspace_root.iterdir():
                        if (
                            item.is_dir()
                            and (item / ".agent_state.json").is_file()
                            and item.name not in affected
                        ):
                            affected.append(item.name)
                self._waiting_questions.clear()
            else:
                self._waiting_questions.pop(project, None)
                affected = [project]
            if stop_active_task:
                self._stop_event.set()
                if self._active_tools:
                    self._active_tools.stop()
                if self._active_client:
                    try:
                        self._active_client.abort()
                    except Exception:  # noqa: BLE001, S110
                        pass
            thread_to_join = self._thread if stop_active_task else None

        if thread_to_join is not None and thread_to_join.is_alive():
            thread_to_join.join(timeout=5.0)

        with self._lock:
            if stop_active_task and (self._thread is None or not self._thread.is_alive()):
                self._active_project = None
                self._active_run_id = None
                self._thread = None
                self._run_complete.set()
                self._active_tools = None
                self._active_client = None

        for cleared in affected:
            (
                self.settings.workspace_root / cleared / ".agent_state.json"
            ).unlink(missing_ok=True)
        return affected

    # ------------------------------------------------------------------ main loop

    def _run(
        self,
        project: str,
        message: str,
        image_paths: list[Path] | None = None,
        run_id: str = "",
    ) -> None:
        """Lifecycle wrapper: init, dispatch turns, terminal cleanup.

        Per-iteration logic lives in :meth:`_run_turn`; helpers handle the
        specific branches (no tool calls, parked question, etc.).
        """
        if not run_id:
            # ``start()`` always passes a run_id; the default keeps the
            # parameter optional for direct tests that bypass ``start()``.
            run_id = uuid.uuid4().hex
        project_dir = self.settings.workspace_root / project
        state: _RunState | None = None
        activity_logger: ActivityLogger | None = None
        try:
            state = self._init_run(
                project, message, image_paths or [], project_dir, run_id
            )
            activity_logger = state.activity_logger
            # Bound once: the publisher reads ``current_message_id`` lazily
            # so the per-iteration update is just an attribute assignment.
            state.client.stream_callback = self._build_stream_publisher(state)
            for _ in range(self.settings.agent_tool_call_limit):
                if self._stop_event.is_set():
                    self._handle_stop_pre_turn(state)
                    return
                state.current_message_id = uuid.uuid4().hex
                outcome = self._run_turn(state)
                if outcome is _TurnOutcome.CONTINUE:
                    continue
                # Non-CONTINUE is terminal; the helper already published
                # the diagnostic and finalised ``state.messages``.
                return
            # Outer loop exhausted without a terminal outcome.
            self._handle_tool_limit_reached(state)
        except RequestCancelled:
            self._handle_request_cancelled(state)
        except Exception as error:  # noqa: BLE001 - Surface all failures to the local UI.
            self._handle_unexpected_error(error, project, state)
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
    ) -> _RunState:
        """Build tools/client, log ``run_start``, return the run state.

        Pulled out of ``_run`` so the lifecycle wrapper stays focused on
        dispatch. If this raises, ``_run``'s exception handlers take
        over and ``state`` is ``None`` for the ``finally`` block.
        """
        self.publish(
            "agent_status",
            {
                "project": project,
                "run_id": run_id,
                "status": "started",
                "message": "Planning CAD task...",
            },
        )
        tools = ProjectTools(
            project_dir, self.publish, self.settings, self._stop_event
        )
        client = create_llm_client(self.settings)
        client.stop_event = self._stop_event
        session_prefix = (
            self.settings.openrouter_session_prefix
            if self.settings.llm_provider == "openrouter"
            else self.settings.llm_provider
        )
        client.session_id = f"{session_prefix}:{project}"
        with self._lock:
            self._active_tools = tools
            self._active_client = client
        messages = self._context(project_dir, message, image_paths)
        activity_logger: ActivityLogger | None = None
        if activity_logging_enabled(self.settings):
            activity_logger = get_logger(
                project_dir,
                log_mode=getattr(self.settings, "agent_log_mode", "debug"),
            )
            client.activity_logger = activity_logger
            client.run_id = run_id
            # Stash on the runner so the per-call dispatcher wrapper can
            # read the logger without re-deriving it. ``finally`` clears
            # these to keep a stale logger from leaking across runs.
            self._active_activity_logger = activity_logger
            self._active_run_id = run_id
            activity_logger.log(
                "run_start",
                {
                    "project": project,
                    "model": self.settings.llm_model,
                    "provider": self.settings.llm_provider,
                },
                run_id=run_id,
            )
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
            event_type = event.pop("type")
            msg_id = state.current_message_id
            if event_type in ("content", "reasoning"):
                sse_name = f"agent_{event_type}_delta"
            elif event_type == "tool_call":
                sse_name = "agent_tool_delta"
            else:
                sse_name = f"agent_{event_type}_delta"
            publish(
                sse_name,
                {"project": project, "run_id": run_id, "message_id": msg_id, **event},
            )
            if log is not None:
                # Mirror the SSE stream into the activity log so the
                # operator has the same content the UI consumed.
                log(
                    sse_name,
                    {"run_id": run_id, "message_id": msg_id, **event},
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
        response = state.client.chat(state.messages, TOOL_SCHEMAS)
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
        if isinstance(usage, dict):
            state.total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
            state.total_completion_tokens += int(usage.get("completion_tokens") or 0)
            details = usage.get("completion_tokens_details") or {}
            reasoning_tokens = details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0
            state.total_reasoning_tokens += int(reasoning_tokens)
        self._publish_usage(state.project, usage)
        assistant_message = sanitize_assistant_message(
            response["choices"][0]["message"],
            preserve_reasoning=getattr(state.client, "preserve_reasoning", False),
            for_storage=True,
        )
        tool_calls = normalize_tool_calls(assistant_message.get("tool_calls"))
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
            state.tool_call_count += len(tool_calls)
        else:
            assistant_message.pop("tool_calls", None)
            if not assistant_message.get("content"):
                assistant_message["content"] = "Task completed."
        invalid_final = (
            not tool_calls
            and state.any_tool_used
            and (not state.preview_id or state.cad_fix_required)
        )
        self._publish_assistant_turn(state, assistant_message, invalid_final)
        if not tool_calls:
            return self._handle_no_tool_calls(state, assistant_message)
        state.any_tool_used = True
        return self._handle_tool_calls(state, tool_calls)

    def _publish_assistant_turn(
        self,
        state: _RunState,
        assistant_message: dict[str, Any],
        invalid_final: bool,
    ) -> None:
        """Persist and broadcast the assistant turn; handle invalid-final.

        An *invalid final* is a text-only turn after the agent invoked
        tools but visual verification is still missing. We skip the JSONL
        append (a History reload would surface a ghost turn) but keep
        the message in ``state.messages`` for prompt-prefix stability.
        """
        reasoning = _extract_reasoning(assistant_message)
        if reasoning:
            state.run_reasonings.append(reasoning)
        if invalid_final:
            self.publish(
                "agent_status",
                {
                    "project": state.project,
                    "run_id": state.run_id,
                    "status": "verifying",
                    "message": "Model verification required before finalizing.",
                },
            )
            self.publish(
                "agent_stream_end",
                {
                    "project": state.project,
                    "run_id": state.run_id,
                    "message_id": state.current_message_id,
                    "message": "",
                    "reasoning": reasoning,
                    "has_tools": False,
                },
            )
        else:
            self.publish(
                "agent_stream_end",
                {
                    "project": state.project,
                    "run_id": state.run_id,
                    "message_id": state.current_message_id,
                    "message": assistant_message.get("content") or "",
                    "reasoning": reasoning,
                    "has_tools": bool(assistant_message.get("tool_calls")),
                },
            )
            self._append_message(state.project_dir, assistant_message)
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
            if state.any_tool_used:
                if not state.nudged_cad and (
                    state.project_dir / MODEL_FILENAME
                ).is_file():
                    state.nudged_cad = True
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
            self._complete(state.project, content, reasoning=full_reasoning, run_id=state.run_id)
            return _TurnOutcome.COMPLETED
        if state.cad_fix_required:
            if not state.nudged_final_verification:
                state.nudged_final_verification = True
                reminder = _synthetic_user(NUDGE_FINAL_VERIFICATION)
                state.messages.append(reminder)
                self._append_message(state.project_dir, reminder)
                return _TurnOutcome.CONTINUE
            self.publish(
                "agent_error",
                {
                    "project": state.project,
                    "run_id": state.run_id,
                    "message": (
                        "Task stopped: final CAD verification is still missing."
                    ),
                },
            )
            self._publish_terminal_failure(state.project, run_id=state.run_id)
            return _TurnOutcome.FINAL_VERIFICATION_MISSING
        self._complete(state.project, content, reasoning=full_reasoning, run_id=state.run_id)
        return _TurnOutcome.COMPLETED

    def _handle_tool_calls(
        self, state: _RunState, tool_calls: list[dict[str, Any]]
    ) -> _TurnOutcome:
        """Process every tool call in this iteration.

        Stops early on user-cancellation, repeated CAD build failures,
        or a tool that parked the agent waiting for user input. Returns
        :class:`_TurnOutcome.CONTINUE` once the batch completes.
        """
        processed_call_ids: set[str] = set()
        for call in tool_calls:
            call_id = call.get("id", "")
            if self._stop_event.is_set():
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
            (
                state.preview_id,
                state.cad_error,
                state.cad_fix_required,
                waiting,
            ) = self._process_tool_call(
                state.tools,
                state.project,
                state.project_dir,
                call,
                state.cad_fix_required,
                state.preview_id,
                state.cad_error,
                state.messages,
            )
            processed_call_ids.add(call_id)
            if is_cad_build(call.get("function", {}).get("name", "")):
                # Circuit-breaker for repeated CAD build failures; reset
                # on every successful build so a stale history cannot
                # trip the breaker on a future invocation.
                if not state.cad_error:
                    state.build_failure_count = 0
                    state.build_failure_signatures.clear()
                else:
                    state.build_failure_count += 1
                    signature = self._failure_signature(state.cad_error)
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
                                "message": (
                                    "Task stopped after repeated CAD build failures. "
                                    "Review the latest error or continue with a narrower repair."
                                ),
                            },
                        )
                        self._publish_terminal_failure(state.project)
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
        finalises the transcript and emits the status event.
        """
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
        _close_dangling_tool_tail(
            state.project_dir,
            state.messages,
            "Task stopped by the user before the next turn started.",
        )

    def _handle_tool_limit_reached(self, state: _RunState) -> None:
        """The ``agent.tool_call_limit`` budget was exhausted."""
        self.publish(
            "agent_error",
            {
                "project": state.project,
                "run_id": state.run_id,
                "message": (
                    f"Tool-call limit ({self.settings.agent_tool_call_limit}) reached; "
                    "increase agent.tool_call_limit or continue with a narrower request."
                ),
            },
        )
        self._publish_terminal_failure(state.project, run_id=state.run_id)
        # If the final iteration processed at least one tool call, the
        # transcript still ends with ``role: tool``; close the tail so
        # the next user turn has a well-formed prompt prefix.
        _close_dangling_tool_tail(
            state.project_dir,
            state.messages,
            (
                f"Tool-call limit ({self.settings.agent_tool_call_limit}) "
                "reached without resolving the task."
            ),
        )

    def _handle_request_cancelled(self, state: _RunState | None) -> None:
        """``RequestCancelled`` raised mid-iteration: expected control flow.

        Stopping is not a provider error, so the user sees the same
        ``agent_status: stopped`` event as a stop between iterations.
        ``chat()`` was interrupted before the new assistant turn could
        be appended, so the prior iteration's tool results are still
        the tail of ``state.messages``; close them so the transcript
        is well-formed for the next user request.
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
        _close_dangling_tool_tail(
            state.project_dir,
            state.messages,
            "Task stopped by the user while the model was responding.",
        )

    def _handle_unexpected_error(
        self,
        error: BaseException,
        project: str,
        state: _RunState | None,
    ) -> None:
        """Surface any unhandled exception to the local UI.

        ``state`` may be ``None`` when the failure happened during
        ``_init_run``; in that case we fall back to
        ``settings.workspace_root / project`` so the operator audit
        trail still lands in the right place.
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
        )
        current_message_id = (
            state.current_message_id if state is not None else None
        )
        run_id = state.run_id if state is not None else ""
        if current_message_id is not None:
            stream_end_payload: dict[str, Any] = {
                "project": project,
                "message_id": current_message_id,
                "message": "",
            }
            if run_id:
                stream_end_payload["run_id"] = run_id
            self.publish("agent_stream_end", stream_end_payload)
        if self._stop_event.is_set():
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
                provider=self.settings.llm_provider,
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

        Active-handle cleanup is unconditional; activity-log
        finalisation and run-directory removal are guarded so a partial
        init does not crash the cleanup pass.
        """
        with self._lock:
            self._active_tools = None
            self._active_client = None
            if self._active_project == project:
                self._active_project = None
                self._active_run_id = None
            # Mark the run finished and drop the thread reference so a
            # subsequent ``start()`` / ``answer()`` never observes a
            # stale ``_thread``.
            self._thread = None
            self._run_complete.set()
        self._active_activity_logger = None
        if activity_logger is not None:
            try:
                if state is not None:
                    if self._stop_event.is_set():
                        outcome = "USER_STOP"
                    elif state.cad_error:
                        outcome = "CAD_ERROR"
                    elif not state.cad_fix_required and state.preview_id:
                        outcome = "SUCCESS"
                    elif state.any_tool_used and not state.preview_id:
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
                            "cancelled": self._stop_event.is_set(),
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
        self, project_dir: Path, message: str, image_paths: list[Path]
    ) -> list[dict]:
        history = self._load_history(project_dir)
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
            {"role": "system", "content": get_system_prompt()},
            self._project_state_message(project_dir, history),
            *history,
        ]

    @classmethod
    def _project_state_message(
        cls, project_dir: Path, history: list[dict] | None = None
    ) -> dict[str, str]:
        """Provide initial workspace state after the cacheable system prefix.

        The state reflects whether model.scad existed at the start of the conversation,
        frozen across turns so provider prompt caching remains valid.
        See :data:`agent.prompt._OPERATIONAL_RULES` for the instruction the
        model receives.
        """
        existed = cls._initial_model_existed(project_dir, history)
        content = format_project_state(existed, MODEL_FILENAME)
        # Gemini normalizes all system messages into an immutable instruction.
        # Keeping workspace state in a later user message lets its
        # explicit system-message cache breakpoint remain reusable.
        return {"role": "user", "content": content}

    @classmethod
    def _initial_model_existed(
        cls, project_dir: Path, history: list[dict] | None = None
    ) -> bool:
        """Determine whether model.scad existed at the start of this conversation.

        To preserve byte-identical prompt prefixes across all turns (enabling
        provider-side prompt caching), the initial project state must remain
        frozen for the duration of the conversation.
        """
        state_file = project_dir / ".agent_initial_state.json"
        if state_file.is_file():
            try:
                data = json.loads(state_file.read_text("utf-8"))
                if isinstance(data, dict) and "existed" in data:
                    return bool(data["existed"])
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                pass

        if history and len(history) > 1:
            first_turn_wrote_model = False
            for m in history:
                if isinstance(m, dict) and m.get("role") == "assistant":
                    tool_calls = m.get("tool_calls") or []
                    if any(
                        isinstance(tc, dict)
                        and tc.get("function", {}).get("name") == "write_file"
                        for tc in tool_calls
                    ):
                        first_turn_wrote_model = True
                    break
            existed = not first_turn_wrote_model
        else:
            existed = (project_dir / MODEL_FILENAME).is_file()

        try:
            state_file.write_text(json.dumps({"existed": existed}), encoding="utf-8")
        except OSError:
            pass
        return existed

    @staticmethod
    def _last_message_has_tool_image(messages: list[dict]) -> bool:
        if not messages or messages[-1].get("role") != "tool":
            return False
        content = messages[-1].get("content")
        return isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in content
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

    # ------------------------------------------------------------------ dispatch

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
    ) -> tuple[str | None, str | None, bool, bool]:
        """Per-call wrapper that injects runner callbacks into ``dispatcher.process_tool_call``.

        Centralised here so the agent loop stays focused on lifecycle state
        and the dispatcher stays unaware of the ``AgentRunner`` instance.
        """
        result = process_tool_call(
            tools,
            project,
            project_dir,
            call,
            cad_fix_required,
            prev_preview_id,
            cad_error,
            messages,
            publish=self.publish,
            register_preview=self._register_preview,
            append_message=self._append_message,
            debug_log=self._debug_tool_error,
            activity_logger=self._active_activity_logger,
            run_id=self._active_run_id,
        )
        return result

    @classmethod
    def _cancel_remaining_tool_calls(
        cls,
        project_dir: Path,
        tool_calls: list[dict],
        processed_call_ids: set[str],
        messages: list[dict],
    ) -> None:
        """Per-call wrapper that injects the runner's append callback into the dispatcher."""
        cancel_remaining_tool_calls(
            project_dir,
            tool_calls,
            processed_call_ids,
            messages,
            append_message=cls._append_message,
        )

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

        # Cancellation takes priority: \"cancelled" / "stop" substrings
        # are common in unrelated error messages, so check ``err_type``
        # first and only fall back to the substring match for callers
        # that did not pass the type.
        if err_type == "RequestCancelled" or "task was cancelled" in lower:
            return "Task was cancelled."
        if "401" in detail or "unauthorized" in lower or "invalid api key" in lower:
            if provider == "ollama":
                return "Authentication failed for Ollama. Check your OLLAMA_API_KEY environment variable."
            return f"Invalid {provider_name} API key. Check your key at {key_url}."
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
            "openrouter" in lower
            or "openai" in lower
            or "ollama" in lower
            or provider in ("openrouter", "openai", "ollama")
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
        error: Exception,
        result: str,
        *,
        phase: str = "tool_call",
        traceback_text: str | None = None,
    ) -> None:
        """Append recoverable tool failures to a project-local debug log.

        ``phase`` distinguishes the dispatcher ``tool_call`` records (which
        carry the ``tool_results.failure`` classification envelope) from
        internal ``agent_loop`` faults (which carry an optional
        ``traceback_text``). The shared file format keeps the on-disk
        footprint stable for downstream tools.
        """
        if not self.settings.agent_debug_log_tool_errors:
            return
        try:
            entry: dict[str, object] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
                _shared_history_lock(),
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
        return answer

    @staticmethod
    def _log(project_dir: Path, role: str, content: object) -> None:
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "content": content,
        }
        # Route through ``ConversationStore.append`` so the canonical write
        # path holds ``shared_history_lock`` and invalidates the cache. The
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
