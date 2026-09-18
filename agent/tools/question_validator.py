"""Validate user answers against question schemas."""

from __future__ import annotations

import json
import math
import re


class QuestionValidator:
    """Validates user answers for multi-question, single-question, and legacy formats."""

    @staticmethod
    def validate(question: dict[str, object], answer: str) -> bool:
        """Dispatch to the correct validator based on the question format."""
        questions = question.get("questions")
        if isinstance(questions, list) and questions:
            try:
                answers = json.loads(answer)
            except json.JSONDecodeError:
                # Not JSON — fall through to single/legacy validation.
                pass
            else:
                if isinstance(answers, dict):
                    return QuestionValidator._validate_multi(questions, answers)
            # JSON parse failed — try single-question format with raw text answer.
            if len(questions) == 1:
                return QuestionValidator._validate_single(questions[0], answer)
            return False
        # Legacy single-question format.
        return QuestionValidator._validate_legacy(question, answer)

    @staticmethod
    def _validate_multi(questions: list[dict[str, object]], answers: dict[str, object]) -> bool:
        """Validate a JSON dict of answers against a multi-question schema."""
        for q in questions:
            if not isinstance(q, dict):
                return False
            qid = q.get("id")
            if not isinstance(qid, str):
                return False
            required = q.get("required", True)
            input_type = q.get("input_type", "text")
            value = answers.get(qid, "")

            # ``multiselect`` answers are JSON arrays (preferred) or legacy
            # comma-separated strings. Treat both as legitimate "non-text"
            # values and validate them below — the previous implementation
            # bailed on ``not isinstance(value, str)`` before the multiselect
            # branch could ever run, which made every JSON-array answer fail.
            if input_type == "multiselect":
                options = q.get("options", [])
                if not isinstance(options, list):
                    return False
                if isinstance(value, list):
                    selected = [str(v).strip() for v in value if str(v).strip()]
                elif isinstance(value, str):
                    selected = [v.strip() for v in value.split(",") if v.strip()]
                else:
                    # Anything else (number, null, …) is not a valid multiselect
                    # payload. Fall through to the required-check below.
                    selected = []
                if not selected:
                    # Empty selection is only valid for optional multiselects.
                    if required:
                        return False
                    continue
                if not all(v in options for v in selected):
                    return False
                continue

            # Numeric questions may arrive as JSON numbers (e.g. ``25`` or
            # ``3.14``) when the web UI or API client serialises the
            # payload. Normalise those to strings so the unit-aware regex
            # below can process them; ``bool`` is rejected because Python
            # treats it as an ``int`` subclass and ``True`` / ``False`` are
            # never legitimate numeric answers here.
            if (
                input_type == "number"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                value = str(value)

            # All other input types expect a string. Reject non-string payloads
            # (e.g. a stray list, None, or number) outright.
            if not isinstance(value, str):
                if required:
                    return False
                continue
            if not value.strip():
                if required:
                    return False
                continue

            if input_type == "select":
                options = q.get("options", [])
                if not isinstance(options, list) or value not in options:
                    return False
            elif input_type == "number":
                if not QuestionValidator._is_valid_number_with_unit(value):
                    return False
            # Plain "text" answers only need to be non-empty strings, which the
            # strip() check above already enforced.
        return True

    @staticmethod
    def _validate_single(question: dict[str, object], answer: str) -> bool:
        """Validate a raw text answer against a single question's schema."""
        input_type = question.get("input_type", "text")
        if input_type == "select":
            options = question.get("options", [])
            return isinstance(options, list) and answer in options
        if input_type == "multiselect":
            options = question.get("options", [])
            if not isinstance(options, list):
                return False
            selected = [value.strip() for value in answer.split(",") if value.strip()]
            return bool(selected) and all(value in options for value in selected)
        if input_type == "number":
            return QuestionValidator._is_valid_number_with_unit(answer)
        return bool(answer.strip())

    @staticmethod
    def _validate_legacy(question: dict[str, object], answer: str) -> bool:
        """Validate against the deprecated flat question format."""
        return QuestionValidator._validate_single(question, answer)

    _NUMBER_UNIT_RE = re.compile(
        r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z°]+)?\s*$"
    )
    _VALID_UNITS = frozenset(
        {"mm", "cm", "m", "in", "inch", "inches", "deg", "degree", "degrees", "°", "rad"}
    )

    @classmethod
    def _is_valid_number_with_unit(cls, value: str) -> bool:
        """Accept a finite number, optionally suffixed with a length or angle unit."""
        if not isinstance(value, str):
            return False
        match = cls._NUMBER_UNIT_RE.match(value)
        if not match:
            return False
        try:
            number = float(match.group(1))
        except ValueError:
            return False
        if not math.isfinite(number):
            return False
        unit = match.group(2)
        if unit:
            return unit.lower() in cls._VALID_UNITS
        return True
