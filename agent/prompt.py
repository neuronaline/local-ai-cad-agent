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

# Single source of truth for the recurring OpenSCAD requirements. These six
# items are the only place ``$fn``, ``EPS``, the UPPER_CASE parameter rule,
# the EPS cutter rule, the ``difference()`` trap, and the semicolon rule
# live. ``_OPERATIONAL_RULES`` and ``_OPENSCAD_RULES`` refer back here instead
# of restating them, which keeps the cacheable prefix short and stops the
# model from receiving three slightly-different copies of the same rule.
_GOLDEN_RULES = """\
- Units: millimetres for linear dimensions, degrees for angles. Never mix.
- Parameters: every numeric dimension, angle, clearance, and count goes at the
  top of model.scad as an UPPER_CASE parameter with a unit-and-purpose comment
  (e.g. ``PLATE_LENGTH = 120.0; // mm, X span of the base plate``). No bare
  magic numbers inside the geometry body.
- Declare ``$fn = 60;`` and ``EPS = 0.01;`` once near the top of model.scad.
  ``$fn = 60`` smooths every circle, cylinder, and fillet; small fastener
  holes can drop to ``$fn = 32`` locally if needed.
- The EPS cutter rule: in every ``difference()``, extend the cutter ``EPS``
  past the boundary on both ends (e.g. ``translate([x, y, -EPS]) cylinder(
  h = H + 2*EPS, d = D);``). Missing overshoot causes non-manifold coincident
  faces and Z-fighting. In ``union()``, overlap joined parts by at least
  ``EPS`` for a watertight manifold.
- The ``difference()`` trap: ``difference()`` subtracts every 2nd+ child from
  the first child only. When the base has multiple bodies, wrap them in
  ``union()`` as the first child — otherwise stray bodies get subtracted
  along with the cutter.
- Semicolons: every statement, every assignment, every module call ends with
  ``;``. Missing semicolons are the single most common parse error."""

_OPERATIONAL_RULES = """\
- Edit only the active project's model.scad. Everything else is read-only.
- This app previews a single STL at a time. For multi-variant requests (Front /
  Rear, Left / Right, cap / body, etc.) build and ship each variant in its own
  iteration: declare a top-level ``PART_TYPE = "FRONT";`` (or similar) at the
  top of model.scad as a self-documenting marker, run ``cad_build``
  on that variant, then start a fresh build for the next one. Do not stack
  variants into one scene — the bounding box and contact sheet all assume a
  single part.
- Resolve blocking ambiguity first (ask one batched ``question`` if needed),
  then iterate: edit model.scad → ``cad_build`` → inspect metrics (call
  ``get_view_images`` if visual inspection needed) → fix or finish.
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
  modifications (pass ``old_string`` and ``new_string`` for a single edit, or
  an ``edits`` array for multiple related changes).
- ``cad_build`` compiles model.scad in the sandbox, validates geometry
  (manifold, volume, bounding box), and extracts parameters. Pass optional
  ``views`` (e.g. ``views=['isometric']`` or ``views=['all']``) to inspect
  the rendered design in the same turn. If ``views`` is omitted, it returns
  geometric metrics without images.
- ``get_view_images`` retrieves additional or zoomed render views after a build
  if further visual inspection is required (defaults to the 8-view sheet).
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
  passes ``cad_build`` without errors and satisfies all dimensions and functional
  requirements. Use ``views`` in ``cad_build`` or ``get_view_images`` whenever
  visual confirmation of alignment, proportions, or complex contours is required.
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
            "OpenSCAD models. Be concise with the user and precise with tools."
        ),
    ),
    ("design_principles", _DESIGN_PRINCIPLES),
    ("golden_rules", _GOLDEN_RULES),
    ("openscad_rules", _OPENSCAD_RULES),
    ("operational_rules", _OPERATIONAL_RULES),
]

_STATIC_BUNDLE_TAG = "<!-- StaticBundle:v4.9 -->"

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


def format_project_state(exists: bool, filename: str = "model.scad") -> str:
    """Format the workspace state message appended after the cacheable system prefix."""
    template = PROJECT_STATE_EXISTS_TEMPLATE if exists else PROJECT_STATE_MISSING_TEMPLATE
    return template.format(filename=filename)


# Synthetic Nudges & Reminders (role: user nudges during agent loop)
NUDGE_UNVERIFIED_MODEL_TEMPLATE = (
    "{filename} exists but it has not been verified. Call cad_build now."
)
NUDGE_FINAL_VERIFICATION = (
    "Verification required before finalizing. Call cad_build."
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
            "for a single replacement, or an edits array for multiple simultaneous replacements."
        ),
        "old_string": "Exact text in model.scad to replace.",
        "new_string": "Replacement text (use empty string to delete).",
        "edits": "Optional batch of multiple replacements to apply atomically.",
        "edit_old_string": (
            "Exact text to find, copied verbatim from read_file. Must occur exactly once across the file."
        ),
        "edit_new_string": "Replacement text; may be empty to delete the block.",
    },
    "cad_build": {
        "description": (
            "Build and validate model.scad in the sandbox. Checks manifold validity, "
            "solid count, bounding box dimensions, and volume. Extracts declared "
            "UPPER_CASE parameters. Pass optional 'views' to inspect renders in the same turn."
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

