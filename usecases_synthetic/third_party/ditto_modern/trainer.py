from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .data import PairDataset, PairExample
from .metrics import compute_binary_metrics, tune_threshold_for_f1
from .model import load_model, load_tokenizer, save_model
from .runtime import DistContext, is_rank0, save_json


@dataclass
class TrainConfig:
    model_name: str = "roberta-base"
    batch_size: int = 16
    max_len: int = 256
    epochs: int = 5
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_accum_steps: int = 1
    early_stopping_patience: int = 2
    seed: int = 42
    num_workers: int = 2
    fp16: bool = True
    da: str | None = None
    alpha_aug: float = 0.8
    summarize: bool = False
    dk: str | None = None
    spacy_model: str = "en_core_web_sm"
    fixed_threshold: float | None = None


def _amp_settings(
    device_type: str, mixed_precision: bool
) -> Tuple[bool, torch.dtype, bool]:
    """Return ``(enabled, dtype, needs_scaler)`` for mixed-precision training.

    CUDA pairs fp16 autocast with a ``GradScaler`` (fp16's narrow dynamic
    range requires loss scaling). MPS (Apple Silicon) uses bf16 — it has
    fp32 range, so no scaler is needed. Other devices (CPU, XPU without a
    tested path) disable AMP.
    """
    if not mixed_precision:
        return False, torch.float32, False
    if device_type == "cuda":
        return True, torch.float16, True
    if device_type == "mps":
        return True, torch.bfloat16, False
    return False, torch.float32, False


def _gather_objects(local_obj, ctx: DistContext):
    if not ctx.enabled:
        return [local_obj]
    gathered = [None for _ in range(ctx.world_size)]
    dist.all_gather_object(gathered, local_obj)
    return gathered


def _make_loader(
    dataset: PairDataset,
    cfg: TrainConfig,
    ctx: DistContext,
    train: bool,
    num_workers: Optional[int] = None,
):
    if ctx.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=train,
            drop_last=False,
        )
    else:
        sampler = (
            torch.utils.data.RandomSampler(dataset)
            if train
            else SequentialSampler(dataset)
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers if num_workers is None else num_workers,
        pin_memory=torch.cuda.is_available(),
        # R7 (2026-05-27): pass the bound-instance ``dataset.pad`` so
        # the collate fn has access to ``self.tokenizer.pad_token_id``.
        # Pre-R7 ``PairDataset.pad`` was a @staticmethod that hard-coded
        # token id 0 as pad — silently masked the CLS token's attention.
        collate_fn=dataset.pad,
    )
    return loader, sampler


def _dedupe_prob_records(
    records: List[Tuple[int, float, int]],
) -> Tuple[List[int], List[float], List[int]]:
    by_idx: Dict[int, Tuple[float, int]] = {}
    for idx, prob, label in records:
        if idx not in by_idx:
            by_idx[idx] = (prob, label)
    idxs = sorted(by_idx.keys())
    probs = [float(by_idx[i][0]) for i in idxs]
    labels = [int(by_idx[i][1]) for i in idxs]
    return idxs, probs, labels


def evaluate_loop(
    model,
    examples: Sequence[PairExample],
    cfg: TrainConfig,
    ctx: DistContext,
    tokenizer=None,
    threshold: float | None = None,
) -> Dict[str, object]:
    if tokenizer is None:
        raise ValueError("tokenizer is required")

    dataset = PairDataset(examples, tokenizer=tokenizer, max_len=cfg.max_len, da=None)
    loader, _ = _make_loader(dataset, cfg, ctx, train=False)

    model.eval()
    local_records: List[Tuple[int, float, int]] = []
    local_loss = 0.0
    local_count = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) != 5:
                raise RuntimeError("Eval batch format unexpected")
            x, att, y, idxs, _weights = batch
            x = x.to(ctx.device)
            att = att.to(ctx.device)
            y = y.to(ctx.device)

            logits = model(input_ids=x, attention_mask=att)
            probs_pos = torch.softmax(logits, dim=-1)[:, 1]
            losses = F.cross_entropy(logits, y, reduction="none")

            local_records.extend(
                list(
                    zip(
                        idxs.cpu().tolist(),
                        probs_pos.detach().cpu().tolist(),
                        y.detach().cpu().tolist(),
                    )
                )
            )
            local_loss += float(losses.sum().detach().cpu().item())
            local_count += int(y.size(0))

    gathered_records = _gather_objects(local_records, ctx)
    gathered_loss = _gather_objects((local_loss, local_count), ctx)

    if not is_rank0(ctx):
        return {}

    flat_records: List[Tuple[int, float, int]] = []
    for chunk in gathered_records:
        flat_records.extend(chunk)

    idxs, probs, labels = _dedupe_prob_records(flat_records)
    if threshold is None:
        preds = [1 if p >= 0.5 else 0 for p in probs]
        used_th = 0.5
    else:
        preds = [1 if p > float(threshold) else 0 for p in probs]
        used_th = float(threshold)

    metrics = compute_binary_metrics(labels, preds)
    total_loss = sum(x[0] for x in gathered_loss)
    total_count = max(1, sum(x[1] for x in gathered_loss))

    metrics["loss"] = total_loss / total_count
    metrics["idxs"] = idxs
    metrics["probs"] = probs
    metrics["labels"] = labels
    metrics["preds"] = preds
    metrics["threshold"] = used_th
    return metrics


def train_loop(
    train_examples: Sequence[PairExample],
    val_examples: Sequence[PairExample],
    cfg: TrainConfig,
    run_dir: Path,
    ctx: DistContext,
    test_examples: Optional[Sequence[PairExample]] = None,
    sample_weights: Optional[Dict[int, float]] = None,
) -> Dict[str, object]:
    run_dir = Path(run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg.model_name)
    base_model = load_model(cfg.model_name, alpha_aug=cfg.alpha_aug)
    base_model.to(ctx.device)

    model = base_model
    if ctx.enabled:
        model = DDP(
            base_model,
            device_ids=[ctx.local_rank],
            output_device=ctx.local_rank,
            find_unused_parameters=False,
        )

    amp_enabled, amp_dtype, amp_needs_scaler = _amp_settings(ctx.device.type, cfg.fp16)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_needs_scaler)

    # MPS + bf16 autocast silently produces NaN when DataLoader uses worker
    # subprocesses (spawn on macOS interacts badly with the autocast amp
    # cache). num_workers=0 keeps the tokenisation in the main process and
    # avoids the issue; the cost is negligible for EM-sized datasets.
    train_num_workers = cfg.num_workers
    if (
        ctx.device.type == "mps"
        and amp_enabled
        and amp_dtype == torch.bfloat16
        and train_num_workers > 0
        and is_rank0(ctx)
    ):
        print(
            "[ditto] MPS bf16 autocast + DataLoader workers corrupts grads; "
            f"forcing num_workers=0 for training (was {train_num_workers})."
        )
        train_num_workers = 0

    train_dataset = PairDataset(
        train_examples,
        tokenizer=tokenizer,
        max_len=cfg.max_len,
        weights=sample_weights,
        da=cfg.da,
    )
    train_loader, train_sampler = _make_loader(
        train_dataset, cfg, ctx, train=True, num_workers=train_num_workers
    )

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(
        1, (len(train_loader) * cfg.epochs) // max(1, cfg.grad_accum_steps)
    )
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.5
    no_improve = 0
    history: List[Dict[str, object]] = []

    if is_rank0(ctx):
        save_json(
            run_dir / "config.json", {**asdict(cfg), "world_size": ctx.world_size}
        )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if ctx.enabled:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_items = 0
        optimizer.zero_grad(set_to_none=True)

        iterator: Iterable = train_loader
        if is_rank0(ctx):
            iterator = tqdm(
                train_loader, desc=f"train epoch {epoch}/{cfg.epochs}", leave=False
            )

        for step, batch in enumerate(iterator, start=1):
            if len(batch) == 7:
                x1, att1, x2, att2, y, _idxs, weights = batch
                x1 = x1.to(ctx.device)
                att1 = att1.to(ctx.device)
                x2 = x2.to(ctx.device)
                att2 = att2.to(ctx.device)
                y = y.to(ctx.device)
                weights = weights.to(ctx.device)

                with torch.autocast(
                    device_type=ctx.device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    logits = model(
                        input_ids=x1,
                        attention_mask=att1,
                        input_ids_aug=x2,
                        attention_mask_aug=att2,
                    )
                    per_item_loss = F.cross_entropy(logits, y, reduction="none")
                    loss = (per_item_loss * weights).mean() / max(
                        1, cfg.grad_accum_steps
                    )
            else:
                x, att, y, _idxs, weights = batch
                x = x.to(ctx.device)
                att = att.to(ctx.device)
                y = y.to(ctx.device)
                weights = weights.to(ctx.device)

                with torch.autocast(
                    device_type=ctx.device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    logits = model(input_ids=x, attention_mask=att)
                    per_item_loss = F.cross_entropy(logits, y, reduction="none")
                    loss = (per_item_loss * weights).mean() / max(
                        1, cfg.grad_accum_steps
                    )

            scaler.scale(loss).backward()

            if step % cfg.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            epoch_loss += (
                float(loss.detach().cpu().item())
                * len(y)
                * max(1, cfg.grad_accum_steps)
            )
            epoch_items += len(y)

        eval_threshold = cfg.fixed_threshold if cfg.fixed_threshold is not None else 0.5
        val_metrics_raw = evaluate_loop(
            model,
            val_examples,
            cfg,
            ctx,
            tokenizer=tokenizer,
            threshold=eval_threshold,
        )

        if is_rank0(ctx):
            if cfg.fixed_threshold is not None:
                tuned_th = float(cfg.fixed_threshold)
                tuned_metrics = {
                    "precision": val_metrics_raw["precision"],
                    "recall": val_metrics_raw["recall"],
                    "f1": val_metrics_raw["f1"],
                    "accuracy": val_metrics_raw["accuracy"],
                }
            else:
                tuned_th, tuned_metrics = tune_threshold_for_f1(
                    val_metrics_raw["labels"], val_metrics_raw["probs"]
                )
            train_loss = epoch_loss / max(1, epoch_items)

            entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics_raw["loss"],
                "val_threshold": tuned_th,
                "val_precision": tuned_metrics["precision"],
                "val_recall": tuned_metrics["recall"],
                "val_f1": tuned_metrics["f1"],
                "val_accuracy": tuned_metrics["accuracy"],
            }
            history.append(entry)

            improved = float(tuned_metrics["f1"]) > best_val_f1
            if improved:
                best_val_f1 = float(tuned_metrics["f1"])
                best_epoch = epoch
                best_threshold = float(tuned_th)
                no_improve = 0
                model_to_save = model.module if isinstance(model, DDP) else model
                save_model(model_to_save, tokenizer, checkpoints_dir / "best")
                save_json(checkpoints_dir / "best_metrics.json", entry)
            else:
                no_improve += 1

            save_json(run_dir / "history.json", {"epochs": history})

        stop_tensor = torch.tensor([0], device=ctx.device)
        if is_rank0(ctx) and no_improve >= cfg.early_stopping_patience:
            stop_tensor[0] = 1

        if ctx.enabled:
            dist.broadcast(stop_tensor, src=0)

        if stop_tensor.item() == 1:
            break

    summary: Dict[str, object] = {}
    if is_rank0(ctx):
        summary["best_epoch"] = best_epoch
        summary["best_val_f1"] = best_val_f1
        summary["best_threshold"] = best_threshold

        if test_examples is not None:
            best_model = load_model(
                str(checkpoints_dir / "best"), alpha_aug=cfg.alpha_aug
            )
            best_model.to(ctx.device)
            test_metrics = evaluate_loop(
                best_model,
                test_examples,
                cfg,
                ctx,
                tokenizer=tokenizer,
                threshold=best_threshold,
            )
            summary["test"] = {
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "accuracy": test_metrics["accuracy"],
                "tp": test_metrics["tp"],
                "fp": test_metrics["fp"],
                "fn": test_metrics["fn"],
                "tn": test_metrics["tn"],
                "loss": test_metrics["loss"],
                "threshold": best_threshold,
            }

            idx_to_pair = {ex.idx: ex.pair_id for ex in test_examples}
            pred_df = pd.DataFrame(
                {
                    "idx": test_metrics["idxs"],
                    "pair_id": [
                        idx_to_pair.get(i, f"idx-{i}") for i in test_metrics["idxs"]
                    ],
                    "gold": test_metrics["labels"],
                    "pred": test_metrics["preds"],
                    "prob": test_metrics["probs"],
                }
            )
            pred_df.to_csv(run_dir / "predictions.csv", index=False)

        save_json(run_dir / "metrics.json", summary)

    return summary
