"""Deterministic verifier registry and runner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable

from agent.quality.errors import VALIDATION_STATUSES
from agent.quality.models import (
    Requirement,
    ValidationResult,
    _utc_now,
    new_validation_id,
)

# A verifier takes (requirement, shape) and returns a ValidationResult.
Verifier = Callable[[Requirement, Any], ValidationResult]


@dataclass(frozen=True)
class _RegisteredVerifier:
    kind: str
    name: str
    callable: Verifier
    version: str = "v1"


_REGISTRY: dict[str, _RegisteredVerifier] = {}


def register(kind: str, name: str, version: str = "v1") -> Callable[[Verifier], Verifier]:
    """Decorator that registers a verifier callable for a requirement kind."""

    def decorator(func: Verifier) -> Verifier:
        if kind in _REGISTRY:
            raise RuntimeError(
                f"Verifier for kind {kind!r} is already registered as "
                f"{_REGISTRY[kind].name}."
            )
        _REGISTRY[kind] = _RegisteredVerifier(kind, name, func, version)
        return func

    return decorator


def _ensure_builtins() -> None:
    """Import bundled verifier modules so they self-register on first use."""
    from agent.quality.verifiers import (  # noqa: F401 - import side effects
        dimension,
        hole,
        solid_count,
    )


def registered_kinds() -> tuple[str, ...]:
    """Return every registered requirement kind (after importing builtins)."""
    _ensure_builtins()
    return tuple(_REGISTRY.keys())


def get_verifier(kind: str) -> _RegisteredVerifier | None:
    """Return the registered verifier for ``kind`` (or ``None``)."""
    _ensure_builtins()
    return _REGISTRY.get(kind)


def run_verifiers(
    requirements: Iterable[Requirement],
    shape: Any,
    *,
    attempt_id: str,
) -> list[ValidationResult]:
    """Run every registered verifier against the live shape.

    Requirements whose kind has no verifier are emitted as ``not_implemented``
    so the UI/API still records an inspectable artifact for them.
    """
    _ensure_builtins()
    results: list[ValidationResult] = []
    for requirement in requirements:
        entry = _REGISTRY.get(requirement.kind)
        if entry is None:
            results.append(
                replace(
                    ValidationResult(
                        validation_id=new_validation_id(),
                        attempt_id=attempt_id,
                        requirement_id=requirement.id,
                        verifier=f"unknown.{requirement.kind}",
                        status="not_implemented",
                        expected={},
                        observed={},
                        severity="blocking" if requirement.required else "minor",
                        message=(
                            f"No verifier is registered for kind "
                            f"{requirement.kind!r} in this build."
                        ),
                        created_at=_utc_now(),
                    )
                )
            )
            continue
        try:
            result = entry.callable(requirement, shape)
        except Exception as error:  # noqa: BLE001 - verifier errors are findings, not crashes
            results.append(
                ValidationResult(
                    validation_id=new_validation_id(),
                    attempt_id=attempt_id,
                    requirement_id=requirement.id,
                    verifier=entry.name,
                    status="unclear",
                    expected={},
                    observed={},
                    severity="blocking" if requirement.required else "minor",
                    message=f"Verifier raised {type(error).__name__}: {error}",
                    created_at=_utc_now(),
                )
            )
            continue
        # Verifier return values use empty attempt_id; stamp the real one here.
        result = replace(result, attempt_id=attempt_id)
        if result.status not in VALIDATION_STATUSES:
            results.append(
                ValidationResult(
                    validation_id=new_validation_id(),
                    attempt_id=attempt_id,
                    requirement_id=requirement.id,
                    verifier=entry.name,
                    status="unclear",
                    expected={},
                    observed={},
                    severity="blocking" if requirement.required else "minor",
                    message=(
                        f"Verifier {entry.name} returned unknown status "
                        f"{result.status!r}; treating as unclear."
                    ),
                    created_at=_utc_now(),
                )
            )
            continue
        results.append(result)
    return results
