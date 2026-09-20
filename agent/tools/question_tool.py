"""Ask the user one or more clarifying questions, then stop and wait for answers."""

from __future__ import annotations

from collections.abc import Callable

INPUT_TYPES = frozenset({"text", "select", "number", "multiselect"})
SINGLE_CHOICE_TYPES = frozenset({"select", "multiselect"})


def normalize_questions(args: dict) -> list[dict]:
    """Normalize the legacy flat question format into the list format."""
    questions = args.get("questions")
    if "questions" in args:
        if isinstance(questions, dict):
            return [questions]
        if not isinstance(questions, list):
            raise ValueError("'questions' must be a list.")
        return questions
    input_type = args.get("input_type")
    if not input_type:
        raw_opts = args.get("options")
        input_type = "select" if isinstance(raw_opts, list) and len(raw_opts) >= 2 else "text"
    options = args.get("options") if input_type in {"select", "multiselect"} else []
    question_text = args.get("question", "")
    if not isinstance(question_text, str) or not question_text.strip():
        raise ValueError(
            "question descriptor is malformed: 'question' missing or empty."
        )
    return [
        {
            "id": "q1",
            "question": question_text,
            "input_type": input_type,
            "options": options if isinstance(options, list) else [],
        }
    ]


class QuestionTool:
    def __init__(self, publish: Callable[[str, dict], None]) -> None:
        self.publish = publish

    @staticmethod
    def validate_questions(
        questions: list[dict],
    ) -> None:
        """Validate a list of question descriptors. Raises ValueError on the first failure."""
        if not isinstance(questions, list) or not questions:
            raise ValueError("At least one question is required.")
        if len(questions) > 3:
            questions[:] = questions[:3]
        seen: set[str] = set()
        for i, item in enumerate(questions):
            if not isinstance(item, dict):
                raise ValueError(f"Question {i} must be an object.")
            qid = item.get("id")
            if qid is None or (isinstance(qid, str) and not qid.strip()):
                qid = f"q{i+1}"
            elif isinstance(qid, (int, float)):
                qid = str(qid)
            elif not isinstance(qid, str):
                qid = f"q{i+1}"
            qid = qid.strip()
            if qid in seen:
                qid = f"{qid}_{i+1}"
            seen.add(qid)
            item["id"] = qid

            text = item.get("question", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Question '{qid}': text cannot be empty.")
            input_type = item.get("input_type")
            if not input_type:
                raw_opts = item.get("options")
                input_type = "select" if isinstance(raw_opts, list) and len(raw_opts) >= 2 else "text"
                item["input_type"] = input_type
            if input_type not in INPUT_TYPES:
                raise ValueError(
                    f"Question '{qid}': input_type must be one of {sorted(INPUT_TYPES)}."
                )
            if input_type in SINGLE_CHOICE_TYPES:
                options = item.get("options", [])
                if isinstance(options, list):
                    item["options"] = [
                        str(o).strip()
                        for o in options
                        if o is not None and str(o).strip()
                    ]
                else:
                    item["options"] = []
                if len(item["options"]) < 2:
                    input_type = "text"
                    item["input_type"] = "text"
                    item["options"] = []
            else:
                item.pop("options", None)

    def execute(self, args: dict, project: str = "") -> tuple[str, bool, list[dict]]:
        """Validate, normalize, and publish questions.

        Returns ``(result, waiting, normalized_questions)`` so callers
        (notably :mod:`agent.dispatcher`) can reuse the normalized list for
        persisted state without re-running ``normalize_questions``.
        """
        questions = normalize_questions(args)
        self.validate_questions(questions)
        title = args.get("title", "")
        result = self.ask(project, questions, title if isinstance(title, str) else "")
        return result, True, questions

    def ask(
        self,
        project: str,
        questions: list[dict],
        title: str = "",
    ) -> str:
        """Publish an SSE question event and return a stop instruction.

        Validation already happened in :meth:`execute`; re-validating here
        would double-validate the same payload for every dispatch.
        """
        self.publish(
            "question",
            {
                "project": project,
                "title": title.strip() if title else "",
                "questions": [
                    {
                        "id": q["id"].strip(),
                        "question": q["question"].strip(),
                        "input_type": q.get("input_type", "text"),
                        "options": q.get("options", [])
                        if q.get("input_type") in SINGLE_CHOICE_TYPES
                        else [],
                        "required": q.get("required", True),
                    }
                    for q in questions
                ],
            },
        )
        lines = ["Questions sent — wait for the user's answers:"]
        if title.strip():
            lines.insert(0, f"## {title.strip()}")
        for q in questions:
            qid = q["id"].strip()
            qtype = q.get("input_type", "text")
            hint = ""
            if qtype == "select":
                hint = f" (choose: {', '.join(q.get('options', []))})"
            elif qtype == "multiselect":
                hint = f" (choose any: {', '.join(q.get('options', []))})"
            elif qtype == "number":
                hint = " (number)"
            lines.append(f"- [{qid}] {q['question'].strip()}{hint}")
        return "\n".join(lines)
