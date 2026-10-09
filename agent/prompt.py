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
- Units: millimetres for linear dimensions, degrees for angles. Never mix.
- Parameters: every numeric dimension, angle, clearance, and count goes at the
  top of model.scad as an UPPER_CASE parameter with a unit-and-purpose comment
  (e.g. ``PLATE_LENGTH = 120.0; // mm, X span of the base plate``). No bare
  magic numbers inside the geometry body.
- Geometry & syntax rules: follow all syntax traps, EPS cutter rules, manifold
  union, and difference rules defined in the attached Playbook."""

_OPERATIONAL_RULES = """\
- Edit only the active project's model.scad. Everything else is read-only.
- Multi-variant & multi-part models: This app previews a single compiled STL at a time.
  When the user requests multiple components, different sizes/variants (e.g. 3 scoops), or
  an assembly (e.g. box & lid): declare a top-level parameter using standard OpenSCAD Customizer
  combo-box syntax on the same line:
  ``PART_TYPE = "ALL"; // [ALL, 50, 15, 5]`` or ``PART_TYPE = "ALL"; // [ALL:All Parts, 50:50ml, 15:15ml, 5:5ml]``.
  By default, set ``PART_TYPE = "ALL";`` and arrange all requested components side-by-side
  with collision-free clearance (minimum 10 mm gap between outer bounding surfaces:
  ``dx >= (width_1 + width_2)/2 + 10``) so the complete set previews cleanly without overlapping.
  Also support isolating any single component (e.g. ``PART_TYPE = "50";``). Mention in your final
  reply that all components are displayed together and can be isolated via ``PART_TYPE``.
  CadVerifier recognizes multi-part layouts and verifies them cleanly.
- Cooperative assistance: Never refuse a user's layout, orientation, or arrangement request
  with rigid refusals (e.g. "I won't combine them"). Provide the requested arrangement using
  clean OpenSCAD translations (e.g. ``PART_TYPE = "ALL"``) while documenting individual and
  overall bounding dimensions.
- Resolve blocking ambiguity first (ask one batched ``question`` if needed),
  then iterate: edit model.scad → ``cad_build(views=['isometric'])`` (or
  ``views=['all']``) to inspect metrics and renders in a single turn → fix or finish.
  Use ``get_view_images`` only when inspecting other camera angles without rebuilding.
- model.scad layout: parameter block first (see Golden Rules), geometry in
  named modules below, every major block marked with a module or short header
  comment. Comments must stay in sync with the code.
- The initial ``<project_state>`` user message indicates whether ``model.scad``
  exists on disk at the start of the task. If missing, create it directly
  with ``write_file``. Once created, your own tool execution results are the
  definitive source of truth; never repeat ``write_file`` unless explicitly
  intending a full rewrite. Proceed directly to ``cad_build`` after creating
  the model.
- Use the right tool for the job. ``read_file`` is for content you don't
  already know; skip it when your own previous ``write_file``/``edit_file``
  already returned the post-state. Use ``write_file`` only for the initial
  model.scad or a deliberate full rewrite. Use ``edit_file`` for all incremental
  modifications (pass ``old_string`` and ``new_string`` for targeted replacements).
- ``cad_build`` compiles model.scad in the sandbox into a 3D preview STL,
  renders canonical camera views, and runs deterministic code and mesh quality
  verification (checking manifoldness, watertightness, volume, solid count, $fn,
  EPS usage, and risk score). Pass optional ``views`` (e.g. ``views=['isometric']``
  or ``views=['all']``) to inspect the rendered design in the same turn.
- ``get_view_images`` retrieves additional render views after a build
  if further visual inspection from other angles is required without rebuilding.
- Geometric conflict (slot clipping a fastener hole, wall-thickness violation,
  etc.): STOP and call ``question`` with the trade-off. Never silently mutate a
  user-stated dimension to "make it fit" — ask once, then proceed.
- Assume-then-disclose: dimensions the user did not explicitly state are
  inferred from functional standards (typical clearances, standard fastener
  sizes, reasonable wall thicknesses, common gauge thicknesses) and every
  such assumption is summarised in the final reply. Ask a ``question`` at
  most once per task, and only when the missing dimension creates a
  physically impossible contradiction — never for preferences.
- A geometry-changing task is ready only after the latest model.scad revision
  passes ``cad_build`` without critical verification errors and satisfies all
  dimensions and functional requirements. Use ``views`` in ``cad_build`` or
  ``get_view_images`` whenever visual confirmation of alignment, proportions,
  or complex contours is required.
- Conciseness by default: Keep user-facing responses brief, direct, and factual (1-3 sentences or a few concise bullet points). State what was created or changed, key confirmed dimensions, and any critical assumptions. Do not output conversational filler, polite pleasantries, design philosophy essays, or unsolicited tutorials on OpenSCAD basics unless the user explicitly asks for detailed explanations.
- Final reply: a concise description of the produced part, its confirmed
  dimensions, and any notable assumptions. No separate summary file."""

_OPENSCAD_RULES = """\
- On failure, read the tool's code, phase, message, and hint; change model.scad
  before retrying. Do not repeat an identical failed call.
- Check returned dimensions, volume, solid count, validity, and render against
  the request. A successful process exit alone is not sufficient.
- The Playbook (embedded below as ``<openscad_cli_playbook>``) is the single
  reference for OpenSCAD specifics: variable immutability and the spatial
  semantics of ``for`` loops, the ``minkowski()`` ban and its
  ``hull()`` / 2D ``offset()`` replacements, extrusion restrictions, and the
  compact ``rounded_box`` / ``hex_nut_pocket`` helpers. Do not restate those
  rules here — reading the Playbook once at the bottom of this prompt is
  enough."""

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
    "INVALID_TOOL_ARGUMENTS": "Send one valid JSON object matching the tool schema.",
    "REVISION_INTEGRITY": "Do not retry the same edit; revision history needs user attention.",
    "TIMEOUT": "Simplify the operation before retrying.",
    "MODEL_MISSING": "Create {filename} first.",
    "CAD_BUILD_FAILED": "Fix {filename} using the reported location and cause, then rebuild.",
    "VALIDATION_ERROR": "Correct the arguments or source named in the message.",
    "FILE_NOT_FOUND": "Create the required project file first.",
    "TOOL_EXECUTION_FAILED": "Use the message to correct the request before retrying.",
    "IMAGE_RUN_CAD_BUILD_FIRST": "Run cad_build first.",
    "IMAGE_RUN_CAD_BUILD_AGAIN": "Run cad_build again.",
    "IMAGE_REFRESH_REVIEW": "Run cad_build to refresh the review.",
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
            "for a replacement."
        ),
        "old_string": "Exact text in model.scad to replace.",
        "new_string": "Replacement text (use empty string to delete).",
    },
    "cad_build": {
        "description": (
            "Compile model.scad into 3D preview (STL), render images, and verify geometry in the sandbox. "
            "Produces physical dimensions, volume, and deterministic quality validation (manifoldness, "
            "watertightness, $fn, EPS rules, risk score). Pass optional 'views' (e.g. ['isometric'], ['all']) "
            "to inspect renders in the same turn."
        ),
        "views": (
            "Optional view list to return immediately (e.g. ['isometric'], ['all']). "
            "If omitted, returns only geometric metrics."
        ),
    },
    "get_view_images": {
        "description": (
            "Retrieve visual render image(s) of the model. Options: 'all' (composite "
            "contact sheet showing all 8 views, default), 'isometric', 'top', 'bottom', "
            "'front', 'back', 'left', 'right'. Requires a successful cad_build."
        ),
        "views": "List of views to inspect. Defaults to ['all'].",
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

