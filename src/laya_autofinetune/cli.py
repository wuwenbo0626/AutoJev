from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import TrainConfig
from .data import load_records, split_records, write_prepared


def _prepare(input_path: str, output_dir: str, cfg: TrainConfig) -> dict:
    records = load_records(input_path)
    splits = split_records(records, cfg.train_ratio, cfg.calibration_ratio, cfg.seed)
    manifest = write_prepared(splits, output_dir)
    manifest["source"] = str(Path(input_path).resolve())
    manifest["config"] = cfg.to_dict()
    Path(output_dir, "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="laya-autofinetune")
    subcommands = root.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate", help="validate and summarize a JSONL/CSV dataset")
    validate.add_argument("--input", required=True)
    prepare = subcommands.add_parser("prepare", help="validate, normalize and split a dataset")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--config")
    train = subcommands.add_parser("train", help="fine-tune from a prepared dataset directory")
    train.add_argument("--prepared", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--config")
    run = subcommands.add_parser("run", help="prepare, train, calibrate, evaluate and export")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--config")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "validate":
            records = load_records(args.input)
            summary = {
                "records": len(records),
                "groups": len({record.group_id for record in records}),
                "question_types": {
                    name: sum(record.question_type == name for record in records)
                    for name in ("choice", "score", "noul")
                },
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        cfg = TrainConfig.from_json(args.config)
        if args.command == "prepare":
            print(json.dumps(_prepare(args.input, args.output, cfg), ensure_ascii=False, indent=2))
            return 0
        from .trainer import train
        if args.command == "train":
            report = train(args.prepared, args.output, cfg)
        else:
            prepared = Path(args.output) / "prepared"
            _prepare(args.input, str(prepared), cfg)
            report = train(prepared, Path(args.output) / "model", cfg)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

