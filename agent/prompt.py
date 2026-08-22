"""The core system prompt used as the cacheable request prefix."""

import hashlib
from pathlib import Path
from string import Template

_PLAYBOOK_PATH = (
    Path(__file__).resolve().parent / "resources" / "build123d_cli_playbook.md"
)

_SUFFIX_FORMAT = """\n<build123d_cli_playbook>\n{playbook}\n</build123d_cli_playbook>"""

_DESIGN_PRINCIPLES = """\
Satisfy explicit dimensions and functional requirements first. Prefer compact,
material-efficient geometry and the fewest features needed for the job. Do not
invent decorative details or enlarge the part for appearance. When a reference
image is provided, use it for shape and proportion while treating stated
dimensions as authoritative. State any important assumption in the final reply."""

_OPERATIONAL_RULES = """\
- Edit only the active project's model.py. Everything else is read-only.
- CAD loop: resolve blocking ambiguity, edit model.py, call
  cad_build_and_verify, inspect metrics + render, then fix or finish.
- model.py layout: put every numeric dimension, angle, clearance, and count
  as a named, typed parameter with its appropriate unit at the very top of
  the file, grouped
  under short comment headers (overall envelope, pocket, fastener pattern,
  etc.). No bare magic numbers inside the geometry body. Comment every
  parameter with purpose and unit (e.g. `PLATE_LENGTH = 120.0  # mm, X span
  of the base plate`; use integer parameters for counts). Mark each major
  geometry block with a short header
  comment, expose the final shape as top-level `result`, and keep those
  comments in sync with the code in the same edit — stale comments mislead
  the next edit.
- Fresh project: create model.py directly with write_file. Do not read, patch,
  build, render, or review a model that does not exist yet.
- Prefer edit_file for small localized changes (≤ ~10 new lines, one exact
  target block). Use write_file only for the initial model.py or a deliberate
  full rewrite — never rewrite the whole file to change one parameter. Use
  insert_file with a unique anchor for substantial new feature blocks. Skip
  read_file when you already know the current content.
- cad_build_and_verify builds, validates, and previews in one bubblewrap
  subprocess. Default render=false returns metrics + preview.stl + hashes
  only — cheap and cache-friendly; use it for every iteration. Pass
  render=true only for the final verification; the tool attaches the contact
  sheet inline, so inspect it in-band and either accept or iterate. Do not
  re-call with unchanged source to "get a fresh render" — the renderer runs
  exactly once per call. On the final render=true build, include
  parameter_checks for every explicit user-stated dimension, angle,
  clearance, or count represented by a numeric model.py parameter; a failed
  check is a build failure, repair the model instead of dropping the check.
- cad_screenshot re-rasterises a subset of canonical views without re-running
  build123d (quality: low=256px coarse, standard=512px tol 0.1, high=1024px
  tol 0.05). Cache key is (model_sha256, sorted(views), quality); matching
  tuples are served from `.cad-agent/reviews/<sha>/` without spawning the
  sandbox. Skip it for small edits where the inline render already answers
  the question.
- cad_review runs a deterministic + multimodal visual verdict. Any blocking
  or major finding forces fail; only minor (or none) permit pass. Optional,
  not a completion gate. Reserve it (and cad_screenshot) for complex or
  high-risk work: multiple interacting features, booleans/topology changes,
  ambiguous reference geometry, tight clearances, visible-defect risk, or an
  explicit user request. Both sub-tools start a fresh LLM request and
  re-derive design rationale from scratch — reach for them only when inline
  evidence is not enough.
- Geometric conflict (slot clipping a fastener hole, self-intersecting
  fillet, wall-thickness violation, etc.): STOP and call ``question`` with
  the trade-off. Never silently mutate a user-stated dimension to "make it
  fit" — ask once, then proceed.
- Use ``question`` only when an unknown would materially change fit, function,
  or manufacturability. Batch blocking questions together; in one batch keep
  at most one required=true and ≤3 total questions. Infer non-critical
  proportions from context or reference images and disclose the assumption.
- A geometry-changing task is ready only after the latest model.py revision
  passes cad_build_and_verify(render=true) AND the inline render confirms
  the design. Do not claim success from source inspection alone.
- Final reply: a concise description of the produced part, its confirmed
  dimensions, and any notable assumptions. No separate summary file."""

_BUILD123D_RULES = """\
- On failure, read the tool's code, phase, message, and hint; change model.py
  before retrying. Do not repeat an identical failed call.
- Check returned dimensions, volume, solid count, validity, and render against
  the request. A successful process exit alone is not sufficient.
- For optional fillets or chamfers: if edge selectors fail repeatedly, drop the
  finishing operation and deliver the simpler valid solid.
- Treat unexpected keyword arguments as API-contract errors. Consult the versioned
  playbook instead of guessing signatures.
- Before RadiusArc, confirm radius >= half the endpoint chord distance.
- After every fillet, chamfer, or boolean, discard any cached edge/face indices;
  reselect targets by geometry type, position range, and measurable properties."""

BUILD123D_RULES = _BUILD123D_RULES
OPERATIONAL_RULES = _OPERATIONAL_RULES

# Ordered list of (section_tag, body) pairs. Adding or reordering a section is a
# one-line change here; the render loop below produces the final prompt.
_PROMPT_SECTIONS: list[tuple[str, str]] = [
    ("identity", "You are a pragmatic local CAD assistant that creates and "
                 "repairs build123d models. Be concise with the user and precise "
                 "with tools."),
    ("design_principles", _DESIGN_PRINCIPLES),
    ("build123d_rules", _BUILD123D_RULES),
    ("operational_rules", _OPERATIONAL_RULES),
]

_STATIC_BUNDLE_TAG = "<!-- StaticBundle:v4.2 -->"

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


class PromptCache:
    """Lazy, mtime-aware prompt with embedded playbook content."""

    def __init__(self) -> None:
        self._content: str | None = None
        self._playbook_content: str | None = None
        self._playbook_mtime: float | None = None
        self._last_playbook: str | None = None

    def get(self) -> str:
        playbook = self.get_playbook()
        if self._content is not None and playbook == self._last_playbook:
            return self._content
        self._content = _BASE_PROMPT + _SUFFIX_FORMAT.format(playbook=playbook)
        self._last_playbook = playbook
        return self._content

    def get_playbook(self) -> str:
        """Return the playbook content, re-reading from disk when its mtime changes."""
        playbook_path = _PLAYBOOK_PATH
        try:
            stat = playbook_path.stat()
            mtime = stat.st_mtime
        except OSError:
            mtime = -1.0
        if self._playbook_content is not None and self._playbook_mtime == mtime:
            return self._playbook_content
        self._playbook_content = (
            playbook_path.read_text(encoding="utf-8").strip() if mtime >= 0 else ""
        )
        self._playbook_mtime = mtime
        return self._playbook_content

    def hash(self) -> str:
        """Stable hash for cache-busting / session-reset detection."""
        return hashlib.sha256(self.get().encode("utf-8")).hexdigest()


_PROMPT_CACHE = PromptCache()


def get_system_prompt() -> str:
    """Return the latest system prompt, re-reading the playbook if it changed on disk."""
    return _PROMPT_CACHE.get()


def get_prompt_cache_key(namespace: str | None = None) -> str:
    """Return a stable, bounded routing key for one prompt/session pair."""
    key = f"local-ai-cad-agent:{_PROMPT_CACHE.hash()[:16]}"
    if namespace:
        session_hash = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16]
        key = f"{key}:{session_hash}"
    return key


def get_build123d_playbook() -> str:
    """Return the playbook content, re-reading from disk when it changes."""
    return _PROMPT_CACHE.get_playbook()
