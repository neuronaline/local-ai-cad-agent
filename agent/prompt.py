"""The core system prompt used as the cacheable request prefix."""

import hashlib
import os
from pathlib import Path
from string import Template

_PLAYBOOK_PATH = (
    Path(__file__).resolve().parent / "resources" / "openscad_cli_playbook.md"
)

_SUFFIX_FORMAT = """\n<openscad_cli_playbook>\n{playbook}\n</openscad_cli_playbook>"""

# Set ``CAD_AGENT_PROMPT_HOT_RELOAD=1`` to make ``PromptCache.get_playbook``
# re-stat and re-read the playbook on every prompt request. Off by default
# because the playbook is a static markdown file shipped with the agent;
# stat'ing it on every LLM call adds a syscall per request for no payoff.
_HOT_RELOAD_ENV = "CAD_AGENT_PROMPT_HOT_RELOAD"

_DESIGN_PRINCIPLES = """\
Satisfy explicit dimensions and functional requirements first. Prefer compact,
material-efficient geometry and the fewest features needed for the job. Do not
invent decorative details or enlarge the part for appearance. When a reference
image is provided, use it for shape and proportion while treating stated
dimensions as authoritative. State any important assumption in the final reply."""

# Single source of truth for parameter formatting and units.
# OpenSCAD syntax, EPS cutter conventions, manifold union, and difference rules
# are fully defined in the attached Playbook.
_GOLDEN_RULES = """\
- Units: millimeters for linear dimensions, degrees for angles.
- Parameters: declare every numeric dimension, angle, clearance, and count at the
  top of model.scad as an UPPER_CASE parameter with a unit-and-purpose comment
  (e.g. ``PLATE_LENGTH = 120.0; // mm, X span of the base plate``).
- Geometry & syntax: adhere to EPS cutter clearance, manifold union, and difference
  rules defined in the attached Playbook."""

_OPERATIONAL_RULES = """\
- Edit only the active project's model.scad.
- Multi-variant & multi-part models: When multiple components or variants are requested,
  declare a top-level Customizer parameter: ``PART_TYPE = "ALL"; // [ALL, 50, 15, 5]``.
  Arrange components side-by-side with clear spacing (dx >= (width_1 + width_2)/2 + 10 mm)
  and support isolating single components (e.g. ``PART_TYPE = "50";``).
- Fine detail verification: Use ``crop=[ymin, xmin, ymax, xmax]`` and ``resolution=1024``
  in ``cad_build`` or ``get_view_images`` to inspect fine features and debossed text.
- Tool workflow:
  * Create new models directly with ``write_file``.
  * Modify existing code incrementally with ``edit_file`` (using ``old_string`` and ``new_string``, or ``edits`` for batch changes).
  * Build, inspect physical metrics, and verify geometry in one step with ``cad_build(views=['isometric'])``.
  * Inspect additional camera angles without rebuilding via ``get_view_images``.
  * Clarify blocking physical contradictions with a single batched ``question``.
- Dimensions: Preserve explicit user dimensions as authoritative. Infer unstated dimensions
  from standard hardware practices (fastener clearances, wall thicknesses) and disclose all
  assumptions in the final reply.
- Verification requirement: A task is ready only after model.scad compiles and passes
  ``cad_build`` without critical verification issues.
- Response style: Keep final answers concise and factual (1-3 sentences or clear bullet points),
  stating what was created, confirmed dimensions, and notable assumptions."""

_OPENSCAD_RULES = """\
- On build failure, inspect the compiler error location, edit model.scad to resolve it, and rebuild.
- Validate dimensions, volume, solid count, and manifoldness against requirements.
- Follow the attached Playbook (<openscad_cli_playbook>) as the authoritative reference for
  variable scope, loop semantics, hull()/offset() replacements, and standard helpers."""

# Ordered list of (section_tag, body) pairs. Adding or reordering a section is a
# one-line change here; the render loop below produces the final prompt.
_PROMPT_SECTIONS: list[tuple[str, str]] = [
    (
        "identity",
        (
            "You are a pragmatic local CAD assistant that creates and repairs "
            "OpenSCAD models. Be concise and direct with the user, and precise with tools."
        ),
    ),
    ("design_principles", _DESIGN_PRINCIPLES),
    ("golden_rules", _GOLDEN_RULES),
    ("openscad_rules", _OPENSCAD_RULES),
    ("operational_rules", _OPERATIONAL_RULES),
]

_STATIC_BUNDLE_TAG = "<!-- StaticBundle:v5.3 -->"

# Template-driven render keeps section markers, the bundle tag, and the optional
# playbook suffix in one consistent style — no f-string brace escaping is needed
# when adding a new section.
_BASE_PROMPT_TEMPLATE = Template(
    "$bundle_tag\n" + "\n".join(f"<{tag}>\n${tag}\n</{tag}>" for tag, _ in _PROMPT_SECTIONS)
)


def _render_base_prompt() -> str:
    return _BASE_PROMPT_TEMPLATE.substitute(
        bundle_tag=_STATIC_BUNDLE_TAG,
        **{tag: body.strip() for tag, body in _PROMPT_SECTIONS},
    )


_BASE_PROMPT = _render_base_prompt()


def _read_playbook_once() -> str:
    """Read the playbook from disk at import time.

    Reading on every prompt call added a syscall per LLM request for a
    static markdown file; instead we read it once at module import and
    hand the bytes to the cache. The ``CAD_AGENT_PROMPT_HOT_RELOAD``
    env var opts back into the per-call re-read for development.
    """
    try:
        return _PLAYBOOK_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


_BOOTSTRAP_PLAYBOOK = _read_playbook_once()


class PromptCache:
    """Lazy, mtime-aware prompt with embedded playbook content.

    The playbook is loaded once at module import (``_BOOTSTRAP_PLAYBOOK``)
    so the hot path does not stat or read the file on every request. The
    ``hot_reload`` constructor flag (or the
    ``CAD_AGENT_PROMPT_HOT_RELOAD`` env var) opts back into the original
    mtime-aware behaviour for local development.
    """

    def __init__(self, *, hot_reload: bool | None = None) -> None:
        self._content: str | None = None
        self._playbook_content: str | None = None
        self._playbook_mtime: float | None = None
        self._last_playbook: str | None = None
        if hot_reload is None:
            hot_reload = os.environ.get(_HOT_RELOAD_ENV, "") == "1"
        self._hot_reload: bool = hot_reload

    def get(self) -> str:
        playbook = self.get_playbook()
        if self._content is not None and playbook == self._last_playbook:
            return self._content
        self._content = _BASE_PROMPT + _SUFFIX_FORMAT.format(playbook=playbook)
        self._last_playbook = playbook
        return self._content

    def get_playbook(self) -> str:
        """Return the playbook content.

        Default: return the module-import snapshot, no filesystem calls.
        With ``hot_reload=True``: re-stat the file and re-read only when
        the mtime has changed (the original behaviour, useful when
        iterating on the playbook content during development).
        """
        if not self._hot_reload:
            if self._playbook_content is None:
                self._playbook_content = _BOOTSTRAP_PLAYBOOK
            return self._playbook_content
        try:
            stat = _PLAYBOOK_PATH.stat()
            mtime = stat.st_mtime
        except OSError:
            mtime = -1.0
        if self._playbook_content is not None and self._playbook_mtime == mtime:
            return self._playbook_content
        self._playbook_content = (
            _PLAYBOOK_PATH.read_text(encoding="utf-8").strip() if mtime >= 0 else ""
        )
        self._playbook_mtime = mtime
        return self._playbook_content

    def hash(self) -> str:
        """Stable hash for cache-busting / session-reset detection."""
        return hashlib.sha256(self.get().encode("utf-8")).hexdigest()


_PROMPT_CACHE = PromptCache()


def get_system_prompt() -> str:
    """Return the latest system prompt, using the cached playbook by default."""
    return _PROMPT_CACHE.get()


def get_prompt_cache_key(namespace: str | None = None) -> str:
    """Return a stable, bounded routing key for one prompt/session pair."""
    key = f"local-ai-cad-agent:{_PROMPT_CACHE.hash()[:16]}"
    if namespace:
        session_hash = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16]
        key = f"{key}:{session_hash}"
    return key


# ---------------------------------------------------------------------------
# Centralized Prompt Catalog: Dynamic state, nudges, hints & tool schemas
# ---------------------------------------------------------------------------

# Dynamic Workspace State Templates (injected as role: user after system prompt)
PROJECT_STATE_EXISTS_WITH_CONTENT_TEMPLATE = (
    "<project_state>\n"
    "{filename} exists with initial content:\n"
    "```scad\n"
    "{content}\n"
    "```\n"
    "Inspect it directly. Do not call read_file unless needed.\n"
    "</project_state>"
)

PROJECT_STATE_EXISTS_TEMPLATE = (
    "<project_state>\n"
    "{filename} exists. Read it before making a targeted edit.\n"
    "</project_state>"
)

PROJECT_STATE_MISSING_TEMPLATE = (
    "<project_state>\n"
    "{filename} does not exist. Create it directly with write_file; do not "
    "call read_file, edit_file, or cad_build first.\n"
    "</project_state>"
)


def format_project_state(
    exists: bool, filename: str = "model.scad", content: str = ""
) -> str:
    """Format the workspace state message appended after the cacheable system prefix."""
    if not exists:
        return PROJECT_STATE_MISSING_TEMPLATE.format(filename=filename)
    if content.strip():
        return PROJECT_STATE_EXISTS_WITH_CONTENT_TEMPLATE.format(
            filename=filename, content=content.rstrip()
        )
    return PROJECT_STATE_EXISTS_TEMPLATE.format(filename=filename)


# Synthetic Nudges & Reminders (role: user nudges during agent loop)
NUDGE_UNVERIFIED_MODEL_TEMPLATE = (
    "{filename} exists but has not been built. Call cad_build now."
)
NUDGE_FINAL_VERIFICATION = (
    "Build and verification required before finalizing. Call cad_build."
)
NUDGE_CAD_FIX_REQUIRED = (
    "CAD build succeeded, but geometry verification reported critical issues. "
    "Inspect the verification findings, edit model.scad to fix them, and call cad_build again."
)


def format_unverified_model_nudge(filename: str = "model.scad") -> str:
    """Format synthetic reminder when model file exists but cad_build was not run."""
    return NUDGE_UNVERIFIED_MODEL_TEMPLATE.format(filename=filename)


# Multimodal Visual Inspection Prompt (wrapped around relocated tool images)
TOOL_IMAGE_PROMPT = (
    "The attached image is the visual artifact returned by the latest tool "
    "call. Inspect it and continue the task."
)


# Model-facing Tool Execution and Recovery Hints
TOOL_HINTS = {
    "INVALID_TOOL_ARGUMENTS": "Send valid JSON arguments conforming strictly to the tool schema.",
    "REVISION_INTEGRITY": "Inspect the latest file content before retrying edits.",
    "TIMEOUT": "Simplify geometry or reduce $fn to accelerate compilation.",
    "MODEL_MISSING": "Create {filename} with write_file before calling other tools.",
    "CAD_BUILD_FAILED": "Inspect the compiler error output, fix syntax or geometry in {filename}, then rebuild.",
    "CAD_CGAL_ERROR": (
        "CGAL boolean assertion violation in {filename}. "
        "Common fixes: (1) Ensure 2D profiles in rotate_extrude() or linear_extrude() do not self-intersect, "
        "(2) Check for non-manifold or inverted faces in custom polyhedron(), "
        "(3) Prevent coplanar faces in difference() by overshooting cutters with EPS."
    ),
    "VALIDATION_ERROR": "Correct the parameter values named in the message before retrying.",
    "FILE_NOT_FOUND": "Create the required file using write_file first.",
    "TOOL_EXECUTION_FAILED": "Review the error details to adjust the tool arguments before retrying.",
    "IMAGE_RUN_CAD_BUILD_FIRST": "Run cad_build(views=[...]) to compile and render view images first.",
    "IMAGE_RUN_CAD_BUILD_AGAIN": "Run cad_build(views=[...]) to refresh stale render artifacts.",
    "IMAGE_REFRESH_REVIEW": "Run cad_build(views=[...]) to update render views for the latest geometry.",
}


# Tool Schemas Prompts and Parameter Descriptions
TOOL_DESCRIPTIONS = {
    "read_file": {
        "description": (
            "Read model.scad. Returns the code content and line count. "
            "Optional offset and limit for line ranges."
        ),
        "offset": "1-indexed starting line number (defaults to 1).",
        "limit": "Maximum number of lines to return (1-2000).",
    },
    "write_file": {
        "description": (
            "Write or replace the complete content of model.scad. "
            "Use for initial model creation or complete rewrites."
        ),
        "content": "Complete file contents.",
    },
    "edit_file": {
        "description": (
            "Search and replace exact code in model.scad. Provide old_string and new_string "
            "for a single replacement, or edits for multiple atomic replacements."
        ),
        "old_string": "Exact text in model.scad to replace.",
        "new_string": "Replacement text (use empty string to delete).",
        "edits": "Optional list of {old_string, new_string} replacements to apply atomically.",
    },
    "cad_build": {
        "description": (
            "Compile model.scad into 3D preview (STL), render images, and verify geometry in the sandbox. "
            "Produces physical dimensions, volume, and deterministic quality validation (manifoldness, "
            "watertightness, $fn, EPS rules, risk score). Pass optional 'views' (e.g. ['isometric'], ['all']) "
            "to inspect renders in the same turn. Supports 'resolution' (512 or 1024) and 'crop' [ymin, xmin, ymax, xmax] "
            "to zoom into fine details."
        ),
        "views": (
            "Optional view list to return immediately (e.g. ['isometric'], ['all']). "
            "If omitted, returns only geometric metrics."
        ),
        "resolution": "Optional render resolution: 512 (default) or 1024 for high detail.",
        "crop": "Optional normalized crop box [ymin, xmin, ymax, xmax] (0.0 to 1.0) to zoom into a sub-region.",
    },
    "get_view_images": {
        "description": (
            "Retrieve visual render image(s) from a previously completed cad_build. Options: 'all' (composite "
            "contact sheet showing all 8 views, default), 'isometric', 'top', 'bottom', "
            "'front', 'back', 'left', 'right'. Pass optional 'crop' [ymin, xmin, ymax, xmax] to zoom in. "
            "Requires a successful cad_build; to compile or re-render, call cad_build(views=[...])."
        ),
        "views": "List of views to inspect. Defaults to ['all'].",
        "crop": "Optional normalized crop box [ymin, xmin, ymax, xmax] (0.0 to 1.0) to zoom into a sub-region.",
    },
    "question": {
        "description": (
            "Ask all blocking clarification questions together, then stop and wait. "
            "``input_type`` defaults to ``text``; pass ``select`` or ``multiselect`` "
            "for choice questions. ``required`` defaults to ``true``."
        ),
        "title": "Optional short heading.",
        "questions": "Blocking questions to present in one form (max 3 per batch).",
        "id": "Short key, such as hole_diameter.",
        "question": "Direct user-facing question.",
        "input_type": "Answer control; defaults to text.",
        "options": "Required for select and multiselect.",
        "required": "Whether an answer is mandatory; defaults to true.",
    },
}


# UI Example Prompts (presented on landing page)
UI_EXAMPLE_PROMPTS = [
    {
        "title": "Mounting plate with four holes",
        "prompt": "Create a 60 × 30 × 4 mm mounting plate with four 3 mm corner holes.",
    },
    {
        "title": "Wheel for a 6 mm D shaft",
        "prompt": "Create a 50 mm diameter wheel, 12 mm thick, for a 6 mm D-shaft motor.",
    },
    {
        "title": "Cube with a centered through-hole",
        "prompt": "Create a 40 mm cube with a centered 20 mm through-hole.",
    },
    {
        "title": "U-bracket with mounting holes",
        "prompt": "Create a 60 × 40 × 30 mm U-shaped mounting bracket with two holes.",
    },
    {
        "title": "Spoked servo horn",
        "prompt": "Create a 50 mm servo horn with six radial spokes and a center bore.",
    },
    {
        "title": "Cable management tray",
        "prompt": "Create a 100 × 50 mm cable tray with side walls and cable slots.",
    },
    {
        "title": "Pipe saddle clamp",
        "prompt": "Create a pipe saddle clamp for a 50 mm tube with two mounting ears.",
    },
    {
        "title": "Quick-release camera plate",
        "prompt": "Create a 70 × 40 × 8 mm quick-release camera plate with a center slot.",
    },
    {
        "title": "Snap-fit test coupon",
        "prompt": "Create a snap-fit test coupon with two flexible cantilever arms.",
    },
]

