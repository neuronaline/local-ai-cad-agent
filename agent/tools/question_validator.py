"""Validate user answers against question schemas."""

from __future__ import annotations

import json
import math


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
            value = answers.get(qid, "")
            if required and (not isinstance(value, str) or not value.strip()):
                return False
            if not isinstance(value, str) or not value.strip():
                continue
            input_type = q.get("input_type", "text")
            if input_type == "select":
                options = q.get("options", [])
                if not isinstance(options, list) or value not in options:
                    return False
            elif input_type == "multiselect":
                options = q.get("options", [])
                if not isinstance(options, list):
                    return False
                # Accept JSON arrays (preferred) or legacy comma-separated strings.
                if isinstance(value, list):
                    selected = [str(v).strip() for v in value if str(v).strip()]
                else:
                    selected = [v.strip() for v in str(value).split(",") if v.strip()]
                if not selected:
                    # All-optional: treat empty selection as valid.
                    if required is False:
                        continue
                    return False
                if not all(v in options for v in selected):
                    return False
            elif input_type == "number":
                if not QuestionValidator._is_valid_number_with_unit(value):
                    return False
        return True

    @staticmethod
    def _validate_single(question: dict[str, object], answer: str) -> bool:
        """Validate a raw text answer against a single question's schema."""
        input_type = question.get("input_type", "text")
        if input_type == "select":
            options = question.get("options", [])
            return isinstance(options, list) and answer in options
        if input_type == "number":
            return QuestionValidator._is_valid_number_with_unit(answer)
        return bool(answer.strip())

    @staticmethod
    def _validate_legacy(question: dict[str, object], answer: str) -> bool:
        """Validate against the deprecated flat question format."""
        input_type = question.get("input_type", "text")
        if input_type == "select":
            options = question.get("options", [])
            return isinstance(options, list) and answer in options
        if input_type == "number":
            return QuestionValidator._is_valid_number_with_unit(answer)
        return bool(answer.strip())

    @staticmethod
    def _is_valid_number_with_unit(value: str) -> bool:
        """Accept a finite number, optionally suffixed with a length unit."""
        parts = value.split()
        if not 1 <= len(parts) <= 2:
            return False
        try:
            number = float(parts[0])
        except ValueError:
            return False
        return math.isfinite(number) and (
            len(parts) == 1 or parts[1].lower() in {"mm", "in", "inch", "inches"}
        )
