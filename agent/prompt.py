"""The core system prompt used as the cacheable request prefix."""

import hashlib
import os
from pathlib import Path
from string import Template

_PLAYBOOK_PATH = (
    Path(__file__).resolve().parent / "resources" / "openscad_cli_playbook.md"
)
_BOSL2_PLAYBOOK_PATH = (
    Path(__file__).resolve().parent / "resources" / "bosl2_playbook.md"
)

_SUFFIX_FORMAT = """\n<openscad_cli_playbook>\n{playbook}\n</openscad_cli_playbook>"""
_BOSL2_SUFFIX_FORMAT = """\n<bosl2_playbook>\n{bosl2_playbook}\n</bosl2_playbook>"""

# Set ``CAD_AGENT_PROMPT_HOT_RELOAD=1`` to make ``PromptCache`` re-stat and
# re-read the playbooks on every prompt request. Off by default because the
# playbooks are static markdown files shipped with the agent;
# stat'ing them on every LLM call adds a syscall per request for no payoff.
_HOT_RELOAD_ENV = "CAD_AGENT_PROMPT_HOT_RELOAD"

_DESIGN_PRINCIPLES = """\
Satisfy explicit user dimensions (authoritative) and functional requirements first.
Infer unstated dimensions from standard hardware and mechanical practices (fastener clearances,
wall thicknesses, mating tolerances). Always include essential functional and mechanical
features implied by the object's real-world purpose (e.g. mating screw threads or snap/bayonet
joints on caps and jars, >= 45 deg conical transition shoulders on necked vessels, lead-in
chamfers, and mating clearances). Design every part for support-free FDM 3D printing: every
component must rest squarely on a flat Z = 0 base face, caps/lids must rest flat outer-face-down
at Z = 0 with cavities opening upward (+Z), horizontal holes/bores (d > 3 mm) and internal
captive pockets at Z > 0 must use self-supporting <= 45 deg teardrop or chamfered roofs (never
flat horizontal roofs or uncompensated horizontal cylinders that produce steep > 60 deg crown
overhangs), and remaining walls behind grooves/bores—especially inside tapered or angled bodies—must
preserve >= 2 extrusion perimeters at the narrowest cross-section. Avoid unrequested decorative ornaments,
but never omit functional mechanics. When a reference image is provided, use it for shape and
proportion while keeping stated dimensions exact."""

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
- Printer build volume: keep single parts and plate_layout arrangements within the target printer's build envelope when specified.
- Additional user instructions: follow any custom instructions, tolerances, or printer setup details specified under <user_preferences>."""

_OPERATIONAL_RULES = """\
- Edit only the active project's model.scad.
- Reasoning economy:
  * Budget internal thinking concisely; keep internal monologues brief and focused strictly on high-level strategy and immediate tool choices.
  * NEVER derive complex trigonometry, teardrop offsets, rotation matrices, or tapered wall clearances via mental monologues—immediately invoke ``python`` (`np`, `sp`, `math`) standalone or in parallel with `write_file`/`edit_file` to calculate exact coordinates.
- Python engineering & geometry verification:
  1. **Compute-First (`python` then `write_file`/`edit_file`):** For multi-part assemblies, mating threads/rails, or symbolic coordinate solving, consolidate the upfront math into one ``python`` script (two turns at most), inspect the solved output, then immediately write or edit `model.scad`. Never chain 3+ consecutive `python`-only turns without modifying `model.scad`.
  2. **Parallel Verify + Write (`python` + `write_file`/`edit_file`):** For single-part models or localized edits where dimensions are straightforward, you may emit ``python`` and ``write_file``/``edit_file`` in parallel in the same turn to assert Z-stack continuity (`min_z == 0`), wall clearances (`>= 2 * nozzle_dia`), overhang slopes (`<= 45 deg`), and Center of Mass stability.
- Step budget discipline: The newest tool result of each turn carries a `step_budget` object (`step`, `limit`, `remaining` turns). All tool calls emitted in one turn cost **1 step** total, so batch compatible calls together (`read_file` + `python`, `get_view_images` + `python`, `write_file`/`edit_file` + `python`, or `write_file`/`edit_file` + `cad_build`) and pace your workflow so `model.scad` is built and verified well before `remaining` reaches 0.
- Multi-variant & multi-part models: Declare a top-level Customizer selector (e.g. ``PART_TYPE = "ALL";``) and arrange parts side-by-side with `>= 10 mm` gap on `Z = 0` following the Playbook's ``plate_layout`` pattern.
- Tool workflow:
  * Use ``write_file`` for initial creation or full rewrites, and ``edit_file`` (`edits=[{old_string, new_string}]`) for targeted updates. Call ``read_file`` after multiple edits have diverged from the initial snapshot.
  * Batch independent tool calls in the same turn whenever possible.
  * Compile and visually verify geometry in one step with ``cad_build(views=['isometric', 'front', 'top'])``. On build or verification failure, fix `model.scad` and rebuild.
  * When verification findings report specific coordinates (e.g. steep overhangs at `Z = ...`, thin local walls at `XY = ...`, or unstable base), use `python` to solve exact geometry adjustments at those coordinates, apply fixes with `edit_file` or `write_file`, and rebuild automatically.
  * When a visual or spatial defect is reported, call ``get_view_images(views=['front', 'top', 'isometric'])`` (or ``views=['all']``) to inspect rendered views before editing.
  * Clarify blocking physical contradictions with a single batched ``question``.
- Verification requirement: A task is ready only after `model.scad` compiles and passes ``cad_build`` with zero critical issues and—while step budget remains (remaining >= 2)—zero unresolved physical printability WARNINGs (e.g. SEVERE_OVERHANG, SINGLE_PERIMETER_LOCAL_WALL, UNSTABLE_CENTER_OF_MASS).
- Response style: Keep final answers concise and factual (1-3 sentences or bullet points) stating what was created, confirmed dimensions, and key assumptions."""

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


def _read_playbook_once(path: Path) -> str:
    """Read a playbook file from disk at import time.

    Reading on every prompt call added a syscall per LLM request for a
    static markdown file; instead we read it once at module import and
    hand the bytes to the cache. The ``CAD_AGENT_PROMPT_HOT_RELOAD``
    env var opts back into the per-call re-read for development.
    """
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


_BOOTSTRAP_PLAYBOOK = _read_playbook_once(_PLAYBOOK_PATH)
_BOOTSTRAP_BOSL2_PLAYBOOK = _read_playbook_once(_BOSL2_PLAYBOOK_PATH)


class PromptCache:
    """Lazy, mtime-aware prompt with embedded playbook content.

    The playbooks are loaded once at module import (``_BOOTSTRAP_PLAYBOOK``,
    ``_BOOTSTRAP_BOSL2_PLAYBOOK``) so the hot path does not stat or read the
    files on every request. The ``hot_reload`` constructor flag (or the
    ``CAD_AGENT_PROMPT_HOT_RELOAD`` env var) opts back into the original
    mtime-aware behaviour for local development. ``bosl2_enabled`` controls
    whether the BOSL2 playbook is appended after the OpenSCAD CLI playbook.
    """

    def __init__(
        self,
        *,
        hot_reload: bool | None = None,
        bosl2_enabled: bool = True,
    ) -> None:
        self._content: str | None = None
        self._last_playbooks: tuple[str, str] | None = None
        self._hot_entries: dict[Path, tuple[float, str]] = {}
        self._bosl2_enabled: bool = bool(bosl2_enabled)
        if hot_reload is None:
            hot_reload = os.environ.get(_HOT_RELOAD_ENV, "") == "1"
        self._hot_reload: bool = hot_reload

    def get(self) -> str:
        playbooks = (
            self.get_playbook(),
            self.get_bosl2_playbook() if self._bosl2_enabled else "",
        )
        if self._content is not None and playbooks == self._last_playbooks:
            return self._content
        playbook, bosl2_playbook = playbooks
        content = _BASE_PROMPT + _SUFFIX_FORMAT.format(playbook=playbook)
        if bosl2_playbook:
            content += _BOSL2_SUFFIX_FORMAT.format(bosl2_playbook=bosl2_playbook)
        self._content = content
        self._last_playbooks = playbooks
        return content

    def get_playbook(self) -> str:
        """Return the OpenSCAD CLI playbook content."""
        return self._read_playbook(_PLAYBOOK_PATH, _BOOTSTRAP_PLAYBOOK)

    def get_bosl2_playbook(self) -> str:
        """Return the BOSL2 playbook content."""
        return self._read_playbook(_BOSL2_PLAYBOOK_PATH, _BOOTSTRAP_BOSL2_PLAYBOOK)

    def _read_playbook(self, path: Path, bootstrap: str) -> str:
        """Return one playbook's content.

        Default: return the module-import snapshot, no filesystem calls.
        With ``hot_reload=True``: re-stat the file and re-read only when
        the mtime has changed (the original behaviour, useful when
        iterating on the playbook content during development).
        """
        if not self._hot_reload:
            return bootstrap
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = -1.0
        cached = self._hot_entries.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        content = path.read_text(encoding="utf-8").strip() if mtime >= 0 else ""
        self._hot_entries[path] = (mtime, content)
        return content

    def hash(self) -> str:
        """Stable hash for cache-busting / session-reset detection."""
        return hashlib.sha256(self.get().encode("utf-8")).hexdigest()


# One cache per ``openscad.bosl2.enabled`` value, keyed by that flag.
_PROMPT_CACHES: dict[bool, PromptCache] = {
    True: PromptCache(bosl2_enabled=True),
    False: PromptCache(bosl2_enabled=False),
}


def get_system_prompt(*, bosl2_enabled: bool = True) -> str:
    """Return the latest system prompt, using the cached playbooks by default."""
    return _PROMPT_CACHES[bool(bosl2_enabled)].get()


def get_prompt_cache_key(
    namespace: str | None = None, *, bosl2_enabled: bool = True
) -> str:
    """Return a stable, bounded routing key for one prompt/session pair."""
    key = f"local-ai-cad-agent:{_PROMPT_CACHES[bool(bosl2_enabled)].hash()[:16]}"
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
    "If the user reports a visual or spatial defect, call get_view_images(views=['front', 'top', 'isometric']) before editing.\n"
    "</project_state>"
)

PROJECT_STATE_EXISTS_TEMPLATE = (
    "<project_state>\n"
    "{filename} exists. Call read_file before making targeted edits.\n"
    "</project_state>"
)

PROJECT_STATE_MISSING_TEMPLATE = (
    "<project_state>\n"
    "{filename} does not exist. Create it with write_file (calling python first if coordinate/tolerance equations must be solved, or in parallel to verify geometric invariants). Do not call read_file, edit_file, or cad_build before write_file.\n"
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
    user_additional_info: str = "",
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
    additional_clean = clean_preference_string(user_additional_info, max_len=1000)
    if printer_clean or nozzle_clean or filament_clean or additional_clean:
        pref_lines = ["<user_preferences>"]
        if printer_clean:
            pref_lines.append(f"Target 3D Printer: {printer_clean}")
        if nozzle_clean:
            pref_lines.append(f"Nozzle Configuration: {nozzle_clean}")
        if filament_clean:
            pref_lines.append(f"Default Filament Material: {filament_clean}")
        if additional_clean:
            pref_lines.append(f"Additional Instructions: {additional_clean}")
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


NUDGE_CALCULATION_LIMIT_TEMPLATE = (
    "You have run {consecutive} consecutive calculation/inspection turns without updating {filename} "
    "({remaining_text} remaining). Stop calling python, read_file, or get_view_images repeatedly. "
    "Consolidate your findings and call write_file or edit_file NOW (or provide your final response if no code changes are needed)."
)


def format_calculation_limit_nudge(
    consecutive: int, remaining: int, filename: str = "model.scad"
) -> str:
    """Format synthetic reminder when consecutive read-only/calculation turns stall progress."""
    remaining_text = "1 turn" if remaining == 1 else f"{remaining} turns"
    return NUDGE_CALCULATION_LIMIT_TEMPLATE.format(
        consecutive=consecutive,
        remaining_text=remaining_text,
        filename=filename,
    )


NUDGE_TRUNCATED_TOOL_CALL_TEMPLATE = (
    "CRITICAL: Your previous tool call ({tools_text}) was cut off by token limits before arguments finished. "
    "Cease all internal reasoning immediately. Do NOT explain, repeat your reasoning, or apologize. "
    "Emit the complete tool call NOW with concise, compact arguments "
    "(for write_file/edit_file: keep OpenSCAD code compact, omit verbose comments, scaffold core geometry first)."
)


def format_truncated_tool_call_nudge(tool_names: list[str]) -> str:
    """Format synthetic reminder when a tool call was truncated mid-argument stream."""
    tools_text = ", ".join(tool_names) if tool_names else "tool"
    return NUDGE_TRUNCATED_TOOL_CALL_TEMPLATE.format(tools_text=tools_text)


NUDGE_TRUNCATED_REASONING_PENDING_CAD = (
    "CRITICAL: Output token limit reached. Cease all internal reasoning immediately. "
    "Do NOT repeat reasoning, explain, or apologize. "
    "CAD operations or verification fixes are pending: emit your tool call "
    "(e.g. python, write_file, edit_file, cad_build) directly NOW."
)

NUDGE_TRUNCATED_REASONING_READY = (
    "CRITICAL: Output token limit reached. Cease all internal reasoning immediately. "
    "Do NOT repeat reasoning or output commentary. Emit your tool call "
    "(e.g. write_file, edit_file, cad_build) or final answer directly NOW."
)

NUDGE_REASONING_ONLY_PENDING_CAD = (
    "CRITICAL: Reasoning finished but no tool call was emitted. "
    "Do NOT reason further or repeat your analysis. "
    "CAD operations or verification fixes are pending: output your tool call "
    "(e.g. python, write_file, edit_file, cad_build) directly NOW."
)

NUDGE_REASONING_ONLY_READY = (
    "CRITICAL: Reasoning finished but no tool call was emitted. "
    "Do NOT reason further or repeat your analysis. "
    "Output your tool call (e.g. write_file, cad_build) or final answer directly NOW."
)

NUDGE_CAD_WARNING_FIX = (
    "CRITICAL: cad_build reported physical printability WARNING(s) with specific coordinates "
    "(e.g. steep unsupported overhangs, thin local walls, or unstable base). "
    "Do NOT finalize or explain the warning away. Use python to compute exact clearances or "
    "teardrop geometry at the reported coordinates if needed, apply the fix to model.scad using "
    "edit_file or write_file, and re-run cad_build NOW."
)


# Multimodal Visual Inspection Prompt (wrapped around relocated tool images)
TOOL_IMAGE_PROMPT = (
    "Rendered view(s) from the preceding tool call. Visually verify: "
    "(1) no component floats in mid-air above Z=0, "
    "(2) hollow vessels have an unobstructed through-opening from neck to body, "
    "(3) hole patterns and mating features are centered and symmetric, and "
    "(4) if the preceding cad_build verification reported any CRITICAL issues or physical printability WARNINGs "
    "(e.g. SEVERE_OVERHANG, SINGLE_PERIMETER_LOCAL_WALL, UNSTABLE_BASE) and step budget remains, fix model.scad and rebuild before finalizing:"
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
    "EDIT_MATCH_FAILED": (
        "Call read_file to inspect the exact current lines in model.scad, then retry edit_file "
        "with a unique matching old_string block (or use write_file for a complete rewrite)."
    ),
    "OPENSCAD_PREFLIGHT_FAILED": (
        "Fix the OpenSCAD syntax error, delimiter mismatch, or blocked construct reported in "
        "message before retrying write_file or edit_file."
    ),
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
            "Search and replace exact code in model.scad using an atomic list of "
            "{old_string, new_string} replacements."
        ),
        "old_string": "Exact text in model.scad to replace.",
        "new_string": "Replacement text (empty string to delete).",
        "edits": "Non-empty list of {old_string, new_string} replacements applied atomically.",
    },
    "cad_build": {
        "description": (
            "Compile model.scad into 3D preview (STL), render views, and verify geometry "
            "(dimensions, volume, solid count, manifoldness, $fn, EPS rules). "
            "Defaults to returning ['isometric', 'front', 'top'] view images."
        ),
        "views": (
            "Views to return as images in this turn (defaults to ['isometric', 'front', 'top']; "
            "pass ['all'] for 8-view contact sheet or [] for metrics only)."
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

