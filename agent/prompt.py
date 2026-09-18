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
  top of model.scad as a self-documenting marker, run ``cad_build_and_verify``
  on that variant, then start a fresh build for the next one. Do not stack
  variants into one scene — the bounding box and contact sheet all assume a
  single part.
- Resolve blocking ambiguity first (ask one batched ``question`` if needed),
  then iterate: edit model.scad → ``cad_build_and_verify`` → inspect metrics and
  inline render → fix or finish.
- model.scad layout: parameter block first (see Golden Rules), geometry in
  named modules below, every major block marked with a module or short header
  comment. Comments must stay in sync with the code.
- Fresh-project workspace state is delivered as a ``<project_state>`` user
  message after the cacheable system prefix; follow whatever it says about
  whether ``model.scad`` already exists. Do not re-derive the answer from the
  current directory listing.
- Use the right tool for the job. ``read_file`` is for content you don't
  already know; skip it when your own previous ``write_file``/``edit_file``
  already returned the post-state. Use ``write_file`` only for the initial
  model.scad or a deliberate full rewrite. Use ``edit_file`` for all incremental
  modifications (pass ``old_string`` and ``new_string`` for a single edit, or
  an ``edits`` array for multiple related changes).
- ``cad_build_and_verify`` validates geometry, extracts numeric UPPER_CASE
  parameters from the initial model.scad parameter block, and produces the
  canonical eight-view visual evidence + contact sheet in one call. Inspect
  the inline evidence and either accept or iterate.
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
  passes the default rendered ``cad_build_and_verify`` AND the inline contact sheet
  confirms the design. Do not claim success from source inspection alone.
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
