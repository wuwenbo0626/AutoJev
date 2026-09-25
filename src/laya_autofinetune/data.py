from __future__ import annotations

import csv
import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

QUESTION_TYPES = {"choice", "score", "noul"}


@dataclass
class DecisionRecord:
    group_id: str
    question_id: str
    state: Any
    question_type: str
    instructions: str
    criteria: Any
    target: list[float]


def _jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text[:1] in {"{", "[", '"'} or text in {"true", "false", "null"}:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return value


def _options(question_type: str, criteria: Any) -> list[str]:
    if question_type == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice criteria must be an object with at least two options")
        return list(criteria)
    if question_type == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score criteria must be a list with at least two levels")
        return [str(i) for i in range(len(criteria))]
    return ["false", "true"]


def _normalise_target(question_type: str, criteria: Any, answer: Any) -> list[float]:
    option_keys = _options(question_type, criteria)
    if isinstance(answer, dict) and "probabilities" in answer:
        answer = answer["probabilities"]
    if isinstance(answer, dict):
        target = [float(answer.get(key, answer.get(str(key), 0.0))) for key in option_keys]
    else:
        if isinstance(answer, dict):
            answer = answer.get("label")
        if question_type == "choice":
            label = str(answer)
            if label not in option_keys:
                raise ValueError(f"choice label {label!r} is not in criteria")
            target = [1.0 if key == label else 0.0 for key in option_keys]
        elif question_type == "score":
            index = round(float(answer))
            if not 0 <= index < len(option_keys):
                raise ValueError(f"score label {index} is outside 0..{len(option_keys) - 1}")
            target = [1.0 if i == index else 0.0 for i in range(len(option_keys))]
        else:
            if isinstance(answer, str):
                value = answer.strip().lower() in {"1", "true", "yes", "y"}
            else:
                value = bool(answer)
            target = [0.0, 1.0] if value else [1.0, 0.0]
    total = sum(target)
    if total <= 0:
        raise ValueError("target probabilities must contain positive mass")
    return [value / total for value in target]


def _answer_value(question_type: str, answer: Any) -> Any:
    if not isinstance(answer, dict) or "probabilities" in answer:
        return answer
    for key in ("label", "choice", "score", "noul"):
        if key in answer:
            value = answer[key]
            if key == "noul" and isinstance(value, (int, float)) and not isinstance(value, bool):
                return {"false": 1.0 - float(value), "true": float(value)}
            return value
    return answer


def _expand_systemone(row: dict[str, Any], row_number: int) -> Iterable[DecisionRecord]:
    state = _jsonish(row.get("state"))
    questions = _jsonish(row.get("questions"))
    answers = _jsonish(row.get("answers", row.get("gold")))
    if not isinstance(questions, dict) or not isinstance(answers, dict):
        raise TypeError("systemone rows require object-valued questions and answers/gold")
    group_id = str(row.get("group_id", row.get("id", f"row-{row_number}")))
    for question_id, question in questions.items():
        if question_id not in answers:
            raise ValueError(f"missing answer for question {question_id!r}")
        qtype = question.get("type")
        criteria = question.get("criteria")
        if qtype == "noul" and criteria is None:
            criteria = {"false": "statement does not hold", "true": "statement holds"}
        yield DecisionRecord(
            group_id=group_id,
            question_id=str(question_id),
            state=state,
            question_type=qtype,
            instructions=str(question.get("instructions", "")),
            criteria=criteria,
            target=_normalise_target(qtype, criteria, _answer_value(qtype, answers[question_id])),
        )


def _expand_flat(row: dict[str, Any], row_number: int) -> Iterable[DecisionRecord]:
    qtype = row.get("question_type", row.get("type"))
    criteria = _jsonish(row.get("criteria"))
    if qtype == "noul" and criteria in (None, ""):
        criteria = {"false": "statement does not hold", "true": "statement holds"}
    answer = _jsonish(row.get("probabilities", row.get("label", row.get("answer"))))
    yield DecisionRecord(
        group_id=str(row.get("group_id", row.get("id", f"row-{row_number}"))),
        question_id=str(row.get("question_id", "decision")),
        state=_jsonish(row.get("state", row.get("text"))),
        question_type=qtype,
        instructions=str(row.get("instructions", row.get("question", ""))),
        criteria=criteria,
        target=_normalise_target(qtype, criteria, answer),
    )


def load_records(path: str | Path) -> list[DecisionRecord]:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        with source.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        with source.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    records: list[DecisionRecord] = []
    errors: list[str] = []
    for index, row in enumerate(rows, 1):
        try:
            expanded = _expand_systemone(row, index) if "questions" in row else _expand_flat(row, index)
            for record in expanded:
                if record.question_type not in QUESTION_TYPES:
                    raise ValueError(f"unsupported question type {record.question_type!r}")
                if not record.instructions.strip():
                    raise ValueError("instructions cannot be empty")
                records.append(record)
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"row {index}: {exc}")
    if errors:
        preview = "\n".join(errors[:20])
        suffix = f"\n... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"dataset validation failed:\n{preview}{suffix}")
    if not records:
        raise ValueError("dataset contains no usable records")
    return records


def split_records(
    records: list[DecisionRecord], train_ratio: float, calibration_ratio: float, seed: int
) -> dict[str, list[DecisionRecord]]:
    groups = sorted({record.group_id for record in records})
    if len(groups) < 3:
        raise ValueError("at least three distinct group_id values are required for leakage-safe splits")
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_train = max(1, min(n - 2, round(n * train_ratio)))
    n_calibration = max(1, min(n - n_train - 1, round(n * calibration_ratio)))
    assignment = {group: "train" for group in groups[:n_train]}
    assignment.update({group: "calibration" for group in groups[n_train : n_train + n_calibration]})
    assignment.update({group: "test" for group in groups[n_train + n_calibration :]})
    result = {"train": [], "calibration": [], "test": []}
    for record in records:
        result[assignment[record.group_id]].append(record)
    return result


def fingerprint(records: list[DecisionRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: (item.group_id, item.question_id)):
        digest.update(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False).encode())
    return digest.hexdigest()


def write_prepared(splits: dict[str, list[DecisionRecord]], output_dir: str | Path) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"schema_version": 1, "splits": {}}
    for name, records in splits.items():
        path = destination / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        manifest["splits"][name] = {
            "records": len(records),
            "groups": len({record.group_id for record in records}),
            "fingerprint": fingerprint(records),
            "path": path.name,
        }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def read_prepared(path: str | Path) -> list[DecisionRecord]:
    with Path(path).open(encoding="utf-8") as handle:
        return [DecisionRecord(**json.loads(line)) for line in handle if line.strip()]
