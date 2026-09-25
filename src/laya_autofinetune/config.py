from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    model: str = "convaiinnovations/laya-multilingual"
    revision: str | None = None
    seed: int = 42
    train_ratio: float = 0.8
    calibration_ratio: float = 0.1
    test_ratio: float = 0.1
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 4
    encoder_lr: float = 2.5e-5
    head_lr: float = 1.0e-4
    weight_decay: float = 0.01
    max_len: int = 1024
    head_max_len: int = 256
    objective: str = "supervised"
    rlcd_weight: float = 1.0
    group_size: int = 4
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    train_scope: str = "full"
    device: str = "auto"
    num_workers: int = 0
    max_grad_norm: float = 1.0
    temperature_min: float = 0.5
    temperature_max: float = 5.0

    def validate(self) -> None:
        ratios = self.train_ratio + self.calibration_ratio + self.test_ratio
        if abs(ratios - 1.0) > 1e-6:
            raise ValueError("train_ratio + calibration_ratio + test_ratio must equal 1")
        if self.objective not in {"supervised", "rlcd"}:
            raise ValueError("objective must be 'supervised' or 'rlcd'")
        if self.train_scope not in {"full", "head"}:
            raise ValueError("train_scope must be 'full' or 'head'")
        if min(self.epochs, self.batch_size, self.grad_accum, self.group_size) < 1:
            raise ValueError("epochs, batch_size, grad_accum and group_size must be positive")

    @classmethod
    def from_json(cls, path: str | Path | None) -> TrainConfig:
        if path is None:
            cfg = cls()
        else:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            allowed = {f.name for f in fields(cls)}
            unknown = sorted(set(raw) - allowed)
            if unknown:
                raise ValueError(f"unknown configuration keys: {', '.join(unknown)}")
            cfg = cls(**raw)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
