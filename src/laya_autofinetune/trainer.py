from __future__ import annotations

import json
import math
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import TrainConfig
from .data import DecisionRecord, read_prepared
from .metrics import classification_metrics


def _imports():
    try:
        import torch
        from huggingface_hub import snapshot_download
        from laya.common import QTYPES, build_model, build_sequence, proper_reward, render_options
        from safetensors.torch import load_file, save_file
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("training dependencies are missing; run `pip install -e .`") from exc
    return torch, snapshot_download, QTYPES, build_model, build_sequence, proper_reward, render_options, load_file, save_file, AutoTokenizer


def _device(torch, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_path(snapshot_download, model: str, revision: str | None) -> str:
    candidate = Path(model)
    if candidate.exists():
        return str(candidate.resolve())
    return snapshot_download(repo_id=model, revision=revision)


def _prepare_items(records, tokenizer, model_cfg, max_len, head_max_len, build_sequence, render_options, qtypes):
    items = []
    skipped = []
    for record in records:
        q = {"t": record.question_type, "ins": record.instructions, "crit": record.criteria}
        sequence, markers = build_sequence(tokenizer, record.state, q, max_len, head_max_len)
        expected = len(render_options(q))
        if len(markers) != expected:
            skipped.append({"group_id": record.group_id, "question_id": record.question_id, "reason": "options exceed token budget"})
            continue
        if len(record.target) != expected:
            raise ValueError(f"{record.group_id}/{record.question_id}: target width does not match options")
        items.append({
            "ids": sequence,
            "markers": markers,
            "qtype": qtypes[record.question_type],
            "target": record.target,
            "label": int(np.argmax(record.target)),
        })
    if not items:
        raise ValueError("all records were rejected during tokenization")
    return items, skipped


def _collate(torch, items, pad_id):
    count = len(items)
    length = max(len(item["ids"]) for item in items)
    max_options = max(len(item["markers"]) for item in items)
    input_ids = torch.full((count, length), pad_id, dtype=torch.long)
    attention = torch.zeros((count, length), dtype=torch.long)
    marker_pos = torch.zeros((count, max_options), dtype=torch.long)
    marker_mask = torch.zeros((count, max_options), dtype=torch.bool)
    target = torch.zeros((count, max_options), dtype=torch.float32)
    for index, item in enumerate(items):
        input_ids[index, : len(item["ids"])] = torch.tensor(item["ids"])
        attention[index, : len(item["ids"])] = 1
        option_count = len(item["markers"])
        marker_pos[index, :option_count] = torch.tensor(item["markers"])
        marker_mask[index, :option_count] = True
        target[index, :option_count] = torch.tensor(item["target"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "target": target,
        "qtype": torch.tensor([item["qtype"] for item in items]),
    }


def _fit_temperature(torch, rows, low: float, high: float) -> float:
    if len(rows) < 10:
        return 1.0
    width = max(len(logits) for logits, _ in rows)
    logits_tensor = torch.full((len(rows), width), -1e4)
    targets = torch.zeros((len(rows), width))
    for index, (logits, target) in enumerate(rows):
        logits_tensor[index, : len(logits)] = torch.tensor(logits)
        targets[index, : len(target)] = torch.tensor(target)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = -(targets * torch.log_softmax(logits_tensor / log_temperature.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(log_temperature.exp(), low, high).item())


def _predict(torch, model, items, pad_id, device, batch_size):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            batch = _collate(torch, chunk, pad_id)
            logits, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
            values = logits.float().cpu().numpy()
            for row, item in zip(values, chunk):
                outputs.append((item["qtype"], row[: len(item["target"])], item["target"]))
    return outputs


def _evaluate(rows, temperatures):
    probs, targets = [], []
    by_type: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    names = {0: "choice", 1: "score", 2: "noul"}
    for qtype, logits, target in rows:
        scaled = np.asarray(logits) / temperatures[qtype]
        probability = np.exp(scaled - scaled.max())
        probability /= probability.sum()
        target_array = np.asarray(target)
        probs.append(probability)
        targets.append(target_array)
        bucket = by_type.setdefault(names[qtype], ([], []))
        bucket[0].append(probability)
        bucket[1].append(target_array)
    return {
        "overall": classification_metrics(probs, targets),
        "by_type": {name: classification_metrics(*values) for name, values in by_type.items()},
    }


def train(prepared_dir: str | Path, output_dir: str | Path, cfg: TrainConfig) -> dict[str, Any]:
    (torch, snapshot_download, qtypes, build_model, build_sequence, proper_reward,
     render_options, load_file, save_file, AutoTokenizer) = _imports()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = _device(torch, cfg.device)
    model_dir = _model_path(snapshot_download, cfg.model, cfg.revision)
    model_dir_path = Path(model_dir)
    base_cfg = json.loads((model_dir_path / "rl_agent_config.json").read_text())
    base_cfg["max_len"] = cfg.max_len
    base_cfg["head_max_len"] = cfg.head_max_len
    tokenizer = AutoTokenizer.from_pretrained(model_dir_path / "tokenizer")
    model = build_model(base_cfg, encoder_dir=str(model_dir_path / "encoder"))
    model.load_state_dict(load_file(model_dir_path / "model.safetensors"), strict=True)
    if cfg.train_scope == "head":
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    model.to(device)

    split_items = {}
    skipped = {}
    for split in ("train", "calibration", "test"):
        records = read_prepared(Path(prepared_dir) / f"{split}.jsonl")
        split_items[split], skipped[split] = _prepare_items(
            records, tokenizer, base_cfg, cfg.max_len, cfg.head_max_len,
            build_sequence, render_options, qtypes,
        )

    encoder_params = [parameter for name, parameter in model.named_parameters() if name.startswith("encoder.") and parameter.requires_grad]
    head_params = [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.") and parameter.requires_grad]
    parameter_groups = []
    if encoder_params:
        parameter_groups.append({"params": encoder_params, "lr": cfg.encoder_lr})
    parameter_groups.append({"params": head_params, "lr": cfg.head_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=cfg.weight_decay)
    updates_per_epoch = max(1, math.ceil(len(split_items["train"]) / (cfg.batch_size * cfg.grad_accum)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates_per_epoch * cfg.epochs, eta_min=1e-6
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []
    started = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        random.Random(cfg.seed + epoch).shuffle(split_items["train"])
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batches = 0
        progress = epoch / max(1, cfg.epochs - 1)
        sigma = cfg.sigma_start + (cfg.sigma_end - cfg.sigma_start) * progress
        for start in range(0, len(split_items["train"]), cfg.batch_size):
            chunk = split_items["train"][start : start + cfg.batch_size]
            batch = _collate(torch, chunk, tokenizer.pad_token_id)
            autocast = torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled)
            with autocast:
                logits, act = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = loss_ce
            if cfg.objective == "rlcd":
                option_count = mask.sum(-1, keepdim=True).float()
                noise = torch.randn((cfg.group_size,) + logits.shape, device=device) * sigma * mask
                noise = (noise - noise.sum(-1, keepdim=True) / option_count) * mask
                sampled = logits.detach().unsqueeze(0) + noise
                distributions = torch.softmax(sampled.masked_fill(~mask, -1e4), -1)
                with torch.no_grad():
                    reward = proper_reward(
                        distributions, target.unsqueeze(0), batch["qtype"].to(device), mask,
                        w_sph=0.75, w_rps=1.0,
                    )
                    advantage = reward - reward.mean(0, keepdim=True)
                    advantage /= advantage.std().clamp_min(1e-6)
                log_probability = -(((sampled - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
                loss_rlcd = -(advantage * log_probability).mean()
                loss = loss_ce + cfg.rlcd_weight * loss_rlcd
            loss = loss / cfg.grad_accum + 0.0 * act.sum()
            scaler.scale(loss).backward()
            batches += 1
            total_loss += float(loss.detach().cpu()) * cfg.grad_accum
            final_batch = start + cfg.batch_size >= len(split_items["train"])
            if batches % cfg.grad_accum == 0 or final_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        history.append({"epoch": epoch + 1, "loss": total_loss / max(1, batches), "sigma": sigma})

    calibration_rows = _predict(
        torch, model, split_items["calibration"], tokenizer.pad_token_id, device, cfg.batch_size
    )
    temperatures = []
    for qtype in range(3):
        selected = [(logits, target) for current, logits, target in calibration_rows if current == qtype]
        temperatures.append(_fit_temperature(torch, selected, cfg.temperature_min, cfg.temperature_max))
    test_rows = _predict(torch, model, split_items["test"], tokenizer.pad_token_id, device, cfg.batch_size)
    metrics = _evaluate(test_rows, temperatures)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().half().contiguous().cpu() for name, value in model.state_dict().items()}
    save_file(state, destination / "model.safetensors")
    model.encoder.config.save_pretrained(destination / "encoder")
    tokenizer.save_pretrained(destination / "tokenizer")
    base_cfg.update({
        "fine_tuned": True,
        "model_name": destination.name,
        "temperature": temperatures,
    })
    base_cfg.pop("temperature_by_options", None)
    (destination / "rl_agent_config.json").write_text(json.dumps(base_cfg, indent=2), encoding="utf-8")
    report = {
        "base_model": cfg.model,
        "device": str(device),
        "duration_seconds": time.time() - started,
        "config": asdict(cfg),
        "temperatures": temperatures,
        "history": history,
        "metrics": metrics,
        "skipped": skipped,
        "split_sizes": {name: len(items) for name, items in split_items.items()},
    }
    (destination / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    source_manifest = Path(prepared_dir) / "manifest.json"
    if source_manifest.exists():
        shutil.copy2(source_manifest, destination / "dataset_manifest.json")
    return report
