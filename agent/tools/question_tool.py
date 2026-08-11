"""Ask the user one or more clarifying questions, then stop and wait for answers."""

from __future__ import annotations

from collections.abc import Callable

INPUT_TYPES = frozenset({"text", "select", "number", "multiselect"})
SINGLE_CHOICE_TYPES = frozenset({"select", "multiselect"})


def normalize_questions(args: dict) -> list[dict]:
    """Normalize the legacy flat question format into the list format."""
    questions = args.get("questions")
    if "questions" in args:
        if not isinstance(questions, list):
            raise ValueError("'questions' must be a list.")
        return questions
    input_type = args.get("input_type", "text")
    options = args.get("options") if input_type in {"select", "multiselect"} else []
    return [
        {
            "id": "q1",
            "question": args["question"],
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
        seen: set[str] = set()
        for i, item in enumerate(questions):
            if not isinstance(item, dict):
                raise ValueError(f"Question {i} must be an object.")
            qid = item.get("id")
            if not isinstance(qid, str) or not qid.strip():
                raise ValueError(f"Question {i}: a non-empty string 'id' is required.")
            qid = qid.strip()
            if qid in seen:
                raise ValueError(f"Duplicate question id: {qid}")
            seen.add(qid)
            text = item.get("question", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Question '{qid}': text cannot be empty.")
            input_type = item.get("input_type", "text")
            if input_type not in INPUT_TYPES:
                raise ValueError(
                    f"Question '{qid}': input_type must be one of {sorted(INPUT_TYPES)}."
                )
            if input_type in SINGLE_CHOICE_TYPES:
                options = item.get("options", [])
                if (
                    not isinstance(options, list)
                    or len(options) < 2
                    or not all(isinstance(o, str) and o.strip() for o in options)
                ):
                    raise ValueError(
                        f"Question '{qid}': {input_type} requires at least two non-empty string options."
                    )

    def execute(self, args: dict, project: str = "") -> tuple[str, bool]:
        """Validate, normalize, and publish questions. Returns (result_msg, waiting=True)."""
        questions = normalize_questions(args)
        self.validate_questions(questions)
        title = args.get("title", "")
        result = self.ask(project, questions, title if isinstance(title, str) else "")
        return result, True

    def ask(
        self,
        project: str,
        questions: list[dict],
        title: str = "",
    ) -> str:
        """Validate, publish an SSE question event, and return a stop instruction."""
        self.validate_questions(questions)
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
