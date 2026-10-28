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
Satisfy explicit user dimensions (authoritative) and functional requirements first.
Infer unstated dimensions from standard hardware and mechanical practices (fastener clearances,
wall thicknesses, mating tolerances). Always include essential functional and mechanical
features implied by the object's real-world purpose (e.g. mating screw threads or snap/bayonet
joints on caps and jars, >= 45 deg conical transition shoulders on necked vessels so internal
cavities are continuous and self-supporting without flat internal roofs, lead-in chamfers, and
mating clearances). Avoid unrequested purely decorative ornaments, but never omit functional
mechanics. When a reference image is provided, use it for shape and proportion while keeping
stated dimensions exact."""

# Single source of truth for parameter formatting and units.
# OpenSCAD syntax, EPS cutter conventions, manifold union, and difference rules
# are fully defined in the attached Playbook.
_GOLDEN_RULES = """\
- Units: millimeters for linear dimensions, degrees for angles.
- Code language: always write OpenSCAD code (parameter names, module/function identifiers,
  Customizer labels, and comments in model.scad) in English by default, regardless of the user's chat language.
- Parameters: declare every numeric dimension, angle, clearance, and count at the
  top of model.scad as an UPPER_CASE parameter with a unit-and-purpose comment
  (e.g. ``PLATE_LENGTH = 120.0; // mm, X span of the base plate``).
- Geometry & Playbook: use the attached Playbook (<openscad_cli_playbook>) as a practical
  reference for OpenSCAD syntax, EPS cutter clearance, manifold rules, and helpers while
  adapting designs creatively to the task."""

_HARDWARE_AND_MATERIAL_RULES = """\
- Material-aware tolerances & walls (explicit user prompt material always overrides <user_preferences>):
  * PLA: radial mating/thread clearance 0.20-0.25 mm (0.40-0.50 mm diametral), min wall 1.2-1.6 mm, max unsupported overhang 45 deg.
  * PETG: higher friction and layer adhesion; use radial thread/sliding clearance 0.30-0.40 mm (0.60-0.80 mm diametral), min wall 1.6-2.0 mm, generous lead-in chamfers, and >= 45 deg transition cones to prevent internal bridging/fusion.
  * ABS / ASA: account for thermal shrinkage; use radial clearance 0.25-0.35 mm, min wall 1.6-2.0 mm, and filleted internal corners to prevent warping/delamination.
  * TPU / Flex: use 0.10-0.20 mm press-fit clearance and >= 2.0 mm walls unless designing a thin living hinge or gasket.
- Nozzle-aware wall & feature sizing: scale minimum wall thicknesses and fine features to multiples of the nozzle diameter / line width when specified in <user_preferences> (e.g. for a 0.6 mm nozzle with ~0.62-0.65 mm line width, design load-bearing walls >= 1.8 mm (3 perimeters) or thin walls >= 1.25 mm (2 perimeters), avoid thin features < 0.65 mm, and account for coarser Z layer steps of 0.20-0.36 mm on vertical clearances and shallow slopes).
- Printer build volume: keep single parts and plate_layout arrangements within the target printer's build envelope when specified."""

_OPERATIONAL_RULES = """\
- Edit only the active project's model.scad.
- Aggressive reasoning economy (CRITICAL for reasoning models):
  * Strictly budget your internal thinking/reasoning. Never generate lengthy monologues, explore speculative alternatives, or overthink obvious geometry.
  * Offload coordinate arithmetic, Z-stacks, clearances, overhang angles, and physical stability checks from internal thinking tokens to the ``python`` tool, and immediately emit the required tool calls.
  * Reserve your output token budget for the OpenSCAD code and tool arguments rather than thinking.
- Mandatory Python engineering, geometry & physics verification:
  * Do not rely on mental arithmetic. Almost always verify geometry, mechanical tolerances, and physical stability with ``python`` (using ``np``, ``sp``, and ``math``):
    1. **Mode A — Parallel Verify + Write (Default for Standard & Parametric Models):** Emit ``python`` and ``write_file`` (or ``edit_file``) **in parallel in the same turn**. In the ``python`` script, define the model's parameters and programmatically ``assert`` + ``print`` key invariants:
       - *Z-Stack & Cutter Continuity:* Every vertical stage meets without gaps (``z_next == z_prev``) and subtractive cutters overshoot target faces by ``EPS``.
       - *Wall & Hole Clearances:* Minimum wall thickness across bores, radial/linear cuts, and centered arrays (``-(N - 1) * PITCH / 2``) satisfies nozzle perimeter multiples and material clearances.
       - *Overhang & Transition Slopes:* Conical shoulders, chamfers, and unsupported transitions satisfy ``math.degrees(math.atan2(dr, dz)) <= 45``.
       - *Center of Mass & Static Stability:* For asymmetric, cantilevered, or tall freestanding parts, verify that the composite XY Center of Mass lies well inside the Z=0 base support footprint and the slenderness ratio is stable.
       - *Multi-Part Plate Layout:* Bounding boxes do not overlap, maintain >= 10 mm clearance, and fit inside the printer build volume.
    2. **Mode B — Compute-First then Write (For Unknowns & Symbolic Constraints):** When coordinates require solving non-trivial equations first (e.g. ``SymPy`` tangent intersections, gear involutes/pitch radii, custom thread profiles, linkage kinematics, or multi-constraint layouts), call ``python`` first (multiple independent ``python`` calls may be emitted in parallel), inspect the exact numeric output, and then emit ``write_file`` or ``edit_file``.
- Multi-variant & multi-part models: When multiple components or variants are requested,
  declare a top-level Customizer selector (e.g. ``PART_TYPE = "ALL";``) and arrange parts
  side-by-side with >= 10 mm clearance following the Playbook's ``plate_layout`` pattern, with
  every independent part resting flat on the build plate (``min_z = 0``).
- Tool workflow:
  * Create new models by emitting ``python`` (geometric/physical invariant verification) + ``write_file`` in parallel, or call ``python`` first when symbolic/open-form coordinate solving is needed before writing code. For complex or multi-part assemblies, scaffold iteratively: write and build the core geometry and layout first, then add secondary features with ``edit_file``.
  * Modify existing code incrementally with ``edit_file`` (paired in parallel with ``python`` when modifying dimensions, coordinates, or clearances). When multiple edits have accumulated across turns, call ``read_file`` to inspect the current file directly.
  * Independent tool calls (e.g. multiple ``python`` scripts, ``python`` + ``write_file``/``edit_file``, or ``python`` + ``read_file`` + ``get_view_images``) execute concurrently in parallel—batch them in the same turn whenever possible.
  * Build, inspect metrics, and verify geometry in one step with ``cad_build(views=['isometric', 'front', 'top'])``. On build or verification failure, fix model.scad and rebuild.
  * When the user reports a visual or spatial defect (e.g. a floating part, missing/off-center hole, sealed opening, or misaligned feature), call ``get_view_images(views=['front', 'top', 'isometric'])`` (or ``views=['all']``) to inspect the rendered geometry before editing model.scad.
  * Inspect additional camera angles without rebuilding via ``get_view_images(views=[...], crop=[ymin, xmin, ymax, xmax])``, or rebuild at high detail with ``cad_build(views=[...], resolution=1024, crop=[ymin, xmin, ymax, xmax])``.
  * Clarify blocking physical contradictions with a single batched ``question``.
- Verification requirement: A task is ready only after model.scad compiles and passes
  ``cad_build`` without critical verification issues.
- Response style: Keep final answers concise and factual (1-3 sentences or clear bullet points),
  stating what was created, confirmed dimensions, and any notable assumptions."""

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
    ("hardware_and_material_rules", _HARDWARE_AND_MATERIAL_RULES),
    ("operational_rules", _OPERATIONAL_RULES),
]

# Template-driven render keeps section markers and the optional playbook suffix
# in one consistent style — no f-string brace escaping is needed when adding a section.
_BASE_PROMPT_TEMPLATE = Template(
    "\n".join(f"<{tag}>\n${tag}\n</{tag}>" for tag, _ in _PROMPT_SECTIONS)
)


def _render_base_prompt() -> str:
    return _BASE_PROMPT_TEMPLATE.substitute(
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
    "Inspect it directly; call read_file only after multiple edits have diverged the file from this initial snapshot. "
    "Keep reasoning concise and offload dimensional, clearance, or coordinate math to python (emitted in parallel with edit_file, or first for complex equations). "
    "If the user reports a visual or spatial defect, call get_view_images(views=['front', 'top', 'isometric']) (can be batched in parallel with python) to inspect the rendered views alongside the code before editing.\n"
    "</project_state>"
)

PROJECT_STATE_EXISTS_TEMPLATE = (
    "<project_state>\n"
    "{filename} exists. Read it before making a targeted edit, and verify dimensional or spatial changes with python (in parallel with edit_file or prior to it).\n"
    "</project_state>"
)

PROJECT_STATE_MISSING_TEMPLATE = (
    "<project_state>\n"
    "{filename} does not exist. Keep internal reasoning to an aggressively concise structural outline and offload math to python: "
    "either emit python (asserting Z-stack continuity, wall/hole clearances, overhang angles <= 45 deg, and Center of Mass stability) + write_file in parallel in this turn, "
    "or call python first if symbolic/non-trivial coordinate equations must be solved before writing {filename} (do not call read_file, edit_file, or cad_build first).\n"
    "</project_state>"
)


def clean_preference_string(val: str | None, max_len: int = 80) -> str:
    """Sanitize user hardware/filament preference strings for prompt injection."""
    if not val or not isinstance(val, str):
        return ""
    cleaned = "".join(ch if ch.isprintable() and ch not in "<>" else " " for ch in val)
    return " ".join(cleaned.split())[:max_len]


def format_project_state(
    exists: bool,
    filename: str = "model.scad",
    content: str = "",
    *,
    user_printer: str = "",
    user_nozzle: str = "",
    user_filament: str = "",
) -> str:
    """Format the workspace state message appended after the cacheable system prefix."""
    if not exists:
        state_block = PROJECT_STATE_MISSING_TEMPLATE.format(filename=filename)
    elif content.strip():
        state_block = PROJECT_STATE_EXISTS_WITH_CONTENT_TEMPLATE.format(
            filename=filename, content=content.rstrip()
        )
    else:
        state_block = PROJECT_STATE_EXISTS_TEMPLATE.format(filename=filename)

    printer_clean = clean_preference_string(user_printer)
    nozzle_clean = clean_preference_string(user_nozzle)
    filament_clean = clean_preference_string(user_filament)
    if printer_clean or nozzle_clean or filament_clean:
        pref_lines = ["<user_preferences>"]
        if printer_clean:
            pref_lines.append(f"Target 3D Printer: {printer_clean}")
        if nozzle_clean:
            pref_lines.append(f"Nozzle Configuration: {nozzle_clean}")
        if filament_clean:
            pref_lines.append(f"Default Filament Material: {filament_clean}")
        pref_lines.append(
            "Apply the matching printer build volume, nozzle line-width/wall multiples, and material clearances "
            "from <hardware_and_material_rules> unless the user's message explicitly requests a different material or configuration."
        )
        pref_lines.append("</user_preferences>")
        return state_block + "\n" + "\n".join(pref_lines)
    return state_block


# Synthetic Nudges & Reminders (role: user nudges during agent loop)
NUDGE_UNVERIFIED_MODEL_TEMPLATE = (
    "{filename} exists but has not been built. Call cad_build(views=['isometric', 'front', 'top']) now."
)
NUDGE_CAD_FIX_REQUIRED = (
    "CAD build succeeded, but geometry verification reported critical issues. "
    "Inspect the verification findings, use python alongside edit_file (in parallel) to calculate and verify the exact corrective coordinates/clearances, and call cad_build again."
)


def format_unverified_model_nudge(filename: str = "model.scad") -> str:
    """Format synthetic reminder when model file exists but cad_build was not run."""
    return NUDGE_UNVERIFIED_MODEL_TEMPLATE.format(filename=filename)


NUDGE_BUDGET_WARNING_TEMPLATE = (
    "You are taking too long and approaching the tool-call limit ({remaining_text} remaining). "
    "Stop making repetitive micro-edits, over-complicating geometry, or running unnecessary secondary checks. "
    "Finalize the core {filename} geometry now, verify it with cad_build, and provide your final response."
)


def format_budget_warning_nudge(remaining: int, filename: str = "model.scad") -> str:
    """Format synthetic reminder when approaching the tool-call limit."""
    remaining_text = "1 turn" if remaining == 1 else f"{remaining} turns"
    return NUDGE_BUDGET_WARNING_TEMPLATE.format(remaining_text=remaining_text, filename=filename)


# Multimodal Visual Inspection Prompt (wrapped around relocated tool images)
TOOL_IMAGE_PROMPT = (
    "Rendered view(s) from the preceding tool call. Visually verify: "
    "(1) no component floats in mid-air above Z=0, "
    "(2) hollow vessels have an unobstructed through-opening from neck to body, and "
    "(3) hole patterns and mating features are centered and symmetric:"
)


# Model-facing Tool Execution and Recovery Hints
TOOL_HINTS = {
    "INVALID_TOOL_ARGUMENTS": "Send valid JSON arguments conforming strictly to the tool schema.",
    "REVISION_INTEGRITY": "Inspect the latest file content before retrying edits.",
    "TIMEOUT": (
        "OpenSCAD compilation timed out. Avoid 3D minkowski(), hull() loops over spheres, "
        "or high-slice linear_extrude(twist=...). Keep $fn <= 64 (<= 36 for threads) and slices <= turns*18."
    ),
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
    "PYTHON_TIMEOUT": "Python calculation timed out after 5 seconds. Simplify the algorithm, vectorize loops with NumPy, or reduce symbolic complexity in SymPy.",
    "PYTHON_EXECUTION_FAILED": "Python calculation failed. Review the error details, check variable names and types, and retry with corrected code.",
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
        "new_string": "Replacement text (empty string to delete).",
        "edits": "Batch list of {old_string, new_string} replacements applied atomically.",
    },
    "cad_build": {
        "description": (
            "Compile model.scad into 3D preview (STL), render views, and verify geometry "
            "(dimensions, volume, solid count, manifoldness, $fn, EPS rules). "
            "Pass views=['isometric', 'front', 'top'] for standard one-step build and visual verification."
        ),
        "views": (
            "Views to return as images in this turn (e.g. ['isometric'], ['all']). "
            "If omitted, returns only geometric metrics."
        ),
        "resolution": "Render resolution: 512 (default) or 1024 for fine detail.",
        "crop": "Normalized crop box [ymin, xmin, ymax, xmax] (0.0 to 1.0) to zoom into a sub-region.",
    },
    "get_view_images": {
        "description": (
            "Retrieve rendered view image(s) from the latest cad_build without recompiling. "
            "Valid views: 'all' (8-view contact sheet, default), 'isometric', 'isometric_negative', "
            "'top', 'bottom', 'front', 'back', 'left', 'right'."
        ),
        "views": "List of views to inspect (defaults to ['all']).",
        "crop": "Normalized crop box [ymin, xmin, ymax, xmax] (0.0 to 1.0) to zoom into a sub-region.",
    },
    "question": {
        "description": (
            "Ask blocking clarification questions in a single batch, then stop and wait. "
            "Use ONLY when conflicting or missing requirements make it impossible to proceed; "
            "otherwise infer standard dimensions."
        ),
        "title": "Optional short heading.",
        "questions": "Blocking questions to present in one form (max 3).",
        "id": "Short key, such as hole_diameter.",
        "question": "Direct user-facing question.",
        "input_type": "Control type: 'text' (default), 'number', 'select', or 'multiselect'.",
        "options": "Choices list (required for 'select' and 'multiselect').",
        "required": "Whether an answer is mandatory (defaults to true).",
    },
    "python": {
        "description": (
            "Execute stateless Python code for exact mathematical, geometric, and physical engineering verification in an isolated sandbox (5s timeout). "
            "Supports parallel execution: call in parallel alongside write_file or edit_file to programmatically assert Z-stack continuity, wall/hole clearances, overhang angles (<= 45 deg), and Center of Mass stability, or call first (single or parallel batch) when solving unknown coordinates. "
            "Use NumPy (`import numpy as np` or pre-imported `np`) as the primary backbone for coordinate arrays, vectors, matrices, polygon vertices, and trigonometry. "
            "Use SymPy (`import sympy as sp` or pre-imported `sp`) as the engineering constraint solver for symbolic equations (sp.solve, sp.Eq, sp.symbols), tangents, gear/thread geometry, and tolerances. "
            "Use standard `math` for simple scalar operations. Multi-line scripts with print() and automatic return of the last expression are supported."
        ),
        "code": "Python code or expression to execute. Pre-imported: np (numpy), sp (sympy), math, cmath, Fraction, Decimal, statistics, itertools, functools, collections, json.",
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

