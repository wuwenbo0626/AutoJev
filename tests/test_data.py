import json

import pytest

from laya_autofinetune.data import load_records, split_records


def test_systemone_expands_questions_and_keeps_groups(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = []
    for index in range(6):
        rows.append({
            "id": f"case-{index}",
            "state": f"ticket {index}",
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "route it",
                    "criteria": {"billing": "money", "support": "technical"},
                },
                "urgent": {"type": "noul", "instructions": "is it urgent?"},
            },
            "answers": {"intent": {"choice": "billing"}, "urgent": {"noul": index % 2 == 0}},
        })
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    records = load_records(path)
    assert len(records) == 12
    splits = split_records(records, 0.67, 0.16, seed=7)
    group_sets = [{record.group_id for record in split} for split in splits.values()]
    assert all(group_sets)
    assert not (group_sets[0] & group_sets[1])
    assert not (group_sets[0] & group_sets[2])
    assert not (group_sets[1] & group_sets[2])


def test_flat_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        'id,state,type,instructions,criteria,label\n'
        '1,refund please,choice,route,"{""refund"":""money back"",""other"":""anything else""}",refund\n',
        encoding="utf-8",
    )
    record = load_records(path)[0]
    assert record.target == [1.0, 0.0]


def test_invalid_label_is_reported(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({
        "state": "x", "type": "choice", "instructions": "route",
        "criteria": {"a": "A", "b": "B"}, "label": "c",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="not in criteria"):
        load_records(path)

