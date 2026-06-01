#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch.distributed as dist
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from usecases_synthetic.third_party.ditto_modern.data import (
    examples_to_ditto_lines,
    load_wdc_json_gz,
    wdc_to_pair_examples,
)
from usecases_synthetic.third_party.ditto_modern.knowledge import inject_knowledge
from usecases_synthetic.third_party.ditto_modern.runtime import (
    DistContext,
    cleanup_distributed,
    create_run_dir,
    init_distributed,
    is_rank0,
    resolve_run_dir,
    set_seed,
)
from usecases_synthetic.third_party.ditto_modern.summarize import summarize_examples
from usecases_synthetic.third_party.ditto_modern.trainer import TrainConfig, train_loop

DEFAULT_FIELDS = ["title", "brand", "description", "price", "priceCurrency"]


def _parse_fields(raw: str) -> List[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("At least one field must be provided")
    return vals


def _resolve_field_scope(
    fields: List[str],
    domain: str | None,
    *,
    cli_fields_given: bool,
) -> List[str]:
    """Apply the R10-I ``--domain`` train-fields wiring.

    When ``domain`` is set, the field scope is sourced from the canonical
    wide committee list (``committee_ditto_fields`` == ``DOMAIN_TEXT_COLS``
    minus Ditto-reserved names), overriding any stale narrow ``fields``
    default carried in ``--config``. An explicit ``--fields`` that disagrees
    with the canonical scope is a hard error, so the wiring can't be
    silently bypassed with a narrow list. Without ``--domain`` the parsed
    ``fields`` pass through unchanged (legacy behaviour).
    """
    if not domain:
        return fields
    from usecases_synthetic.scripts.ditto.prepare_em_training_data import (
        committee_ditto_fields,
    )

    canonical = committee_ditto_fields(domain)
    if cli_fields_given and fields != canonical:
        raise SystemExit(
            "[train.py] --fields does not match the canonical wide committee "
            f"scope for domain {domain!r}.\n"
            f"  --fields : {fields}\n"
            f"  expected : {canonical}\n"
            "Omit --fields to use the committee scope, or correct the list."
        )
    return canonical


def _load_yaml_config(path: str | None) -> Dict[str, object]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping")
    return data


def _pick(name: str, cli_value, cfg: Dict[str, object], default):
    if cli_value is not None:
        return cli_value
    if name in cfg:
        return cfg[name]
    return default


def _normalize_da_op(da_value: str | None) -> str | None:
    if da_value is None:
        return None
    op = str(da_value).strip().lower()
    if not op:
        return None
    aliases = {
        "delete": "del",
    }
    op = aliases.get(op, op)
    allowed = {
        "all",
        "del",
        "swap",
        "drop_len",
        "drop_sym",
        "drop_same",
        "drop_token",
        "ins",
        "append_col",
        "drop_col",
    }
    if op not in allowed:
        raise ValueError(f"--da must be one of {sorted(allowed)} (or alias: delete)")
    return op


def _broadcast_run_dir(ctx: DistContext, run_dir: Path | None) -> Path:
    if not ctx.enabled:
        if run_dir is None:
            raise ValueError("run_dir is required when not distributed")
        return run_dir
    payload = [str(run_dir) if is_rank0(ctx) else ""]
    dist.broadcast_object_list(payload, src=0)
    return Path(payload[0])


def _load_sample_weights(path: str, train_examples) -> Dict[int, float]:
    df = pd.read_csv(path)
    if "pair_id" not in df.columns or "sample_weight" not in df.columns:
        raise ValueError(
            "sample weights CSV must have pair_id and sample_weight columns"
        )
    w = dict(zip(df["pair_id"].astype(str), df["sample_weight"].astype(float)))
    return {ex.idx: float(w.get(ex.pair_id, 1.0)) for ex in train_examples}


def _write_ditto_text(run_dir: Path, split_name: str, examples) -> None:
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / f"{split_name}.txt"
    lines = examples_to_ditto_lines(examples)
    with open(out_path, "w") as f:
        for line in lines:
            f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train modernized Ditto-style matcher on WDC json.gz"
    )
    parser.add_argument("--train-json-gz", required=True)
    parser.add_argument("--val-json-gz", required=True)
    parser.add_argument("--test-json-gz", default=None)
    parser.add_argument("--config", default="configs/ditto/default_train.yaml")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", default="output/ditto_runs")
    parser.add_argument(
        "--domain",
        default=None,
        help=(
            "When set, the serialization field scope is sourced from the "
            "canonical wide committee list (DOMAIN_TEXT_COLS == "
            "em_matching ditto_plm.fields == sc_block.text_cols), overriding "
            "any stale narrow 'fields' default in --config. This is the "
            "R10-I train-fields wiring: it makes a baseline (R10-H) retrain "
            "train on the same wide surface wide inference serializes, so it "
            "can't silently train on a narrow field set. If --fields is ALSO "
            "passed it must match the canonical scope exactly (else error)."
        ),
    )
    parser.add_argument("--fields", default=None)
    parser.add_argument("--max-field-len", type=int, default=None)
    parser.add_argument("--sample-weights-csv", default=None)
    parser.add_argument(
        "--da",
        default=None,
        help="Ditto data augmentation op (e.g., all, del, swap, drop_col)",
    )
    parser.add_argument(
        "--alpha-aug", type=float, default=None, help="MixDA beta distribution alpha"
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        default=None,
        help="Enable Ditto text summarization",
    )
    parser.add_argument(
        "--dk", default=None, help="Domain knowledge injector: product or general"
    )
    parser.add_argument(
        "--spacy-model",
        default=None,
        help="spaCy model for DK injector (default from config or en_core_web_sm)",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Enable DDP (also auto-enabled under torchrun)",
    )
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help=(
            "Disable per-epoch threshold tuning; use this fixed sigmoid "
            "threshold for both checkpoint selection (val F1 at this "
            "threshold) and the test-set evaluation. Typical value: 0.5."
        ),
    )
    args = parser.parse_args()

    config_map = _load_yaml_config(args.config)

    model_name = _pick("model_name", args.model_name, config_map, "roberta-base")
    batch_size = int(_pick("batch_size", args.batch_size, config_map, 16))
    max_len = int(_pick("max_len", args.max_len, config_map, 256))
    epochs = int(_pick("epochs", args.epochs, config_map, 5))
    lr = float(_pick("lr", args.lr, config_map, 2e-5))
    weight_decay = float(_pick("weight_decay", args.weight_decay, config_map, 0.01))
    warmup_ratio = float(_pick("warmup_ratio", args.warmup_ratio, config_map, 0.1))
    grad_accum_steps = int(
        _pick("grad_accum_steps", args.grad_accum_steps, config_map, 1)
    )
    early_stopping_patience = int(
        _pick("early_stopping_patience", args.early_stopping_patience, config_map, 2)
    )
    seed = int(_pick("seed", args.seed, config_map, 42))
    num_workers = int(_pick("num_workers", args.num_workers, config_map, 2))
    fp16_cfg = bool(_pick("fp16", None, config_map, True))
    fields_raw = _pick("fields", args.fields, config_map, ",".join(DEFAULT_FIELDS))
    max_field_len = int(_pick("max_field_len", args.max_field_len, config_map, 350))
    da = _normalize_da_op(_pick("da", args.da, config_map, None))
    alpha_aug = float(_pick("alpha_aug", args.alpha_aug, config_map, 0.8))
    summarize = bool(_pick("summarize", args.summarize, config_map, False))
    dk = _pick("dk", args.dk, config_map, None)
    spacy_model = _pick("spacy_model", args.spacy_model, config_map, "en_core_web_sm")

    ctx = init_distributed(explicit_ddp=args.ddp)
    set_seed(seed + ctx.rank)

    run_dir = create_run_dir(args.output_dir, prefix="run") if is_rank0(ctx) else None
    run_dir = _broadcast_run_dir(ctx, run_dir)

    fields = _parse_fields(fields_raw)

    # R10-I: --domain sources the field scope from the canonical wide
    # committee list so a baseline/variant Ditto retrain trains on exactly
    # the surface wide inference serializes (see _resolve_field_scope).
    fields = _resolve_field_scope(
        fields, args.domain, cli_fields_given=args.fields is not None
    )
    if args.domain and is_rank0(ctx):
        print(
            f"[train.py] --domain {args.domain}: wide committee field "
            f"scope ({len(fields)} fields): {','.join(fields)}"
        )

    train_df = load_wdc_json_gz(args.train_json_gz)
    val_df = load_wdc_json_gz(args.val_json_gz)
    test_df = load_wdc_json_gz(args.test_json_gz) if args.test_json_gz else None

    train_examples = wdc_to_pair_examples(
        train_df, fields=fields, max_field_len=max_field_len
    )
    val_examples = wdc_to_pair_examples(
        val_df, fields=fields, max_field_len=max_field_len
    )
    test_examples = (
        wdc_to_pair_examples(test_df, fields=fields, max_field_len=max_field_len)
        if test_df is not None
        else None
    )

    if summarize:
        train_examples, val_examples, test_examples = summarize_examples(
            train_examples=train_examples,
            val_examples=val_examples,
            test_examples=test_examples,
            lm=model_name,
            max_len=max_len,
        )

    if dk is not None:
        dk_norm = str(dk).strip().lower()
        if dk_norm not in {"product", "general"}:
            raise ValueError("--dk must be one of: product, general")
        train_examples = inject_knowledge(
            train_examples, dk=dk_norm, spacy_model=spacy_model
        )
        val_examples = inject_knowledge(
            val_examples, dk=dk_norm, spacy_model=spacy_model
        )
        if test_examples is not None:
            test_examples = inject_knowledge(
                test_examples, dk=dk_norm, spacy_model=spacy_model
            )

    if is_rank0(ctx):
        _write_ditto_text(run_dir, "train", train_examples)
        _write_ditto_text(run_dir, "valid", val_examples)
        if test_examples is not None:
            _write_ditto_text(run_dir, "test", test_examples)

    if ctx.enabled:
        dist.barrier()

    sample_weights = None
    if args.sample_weights_csv:
        sample_weights = _load_sample_weights(args.sample_weights_csv, train_examples)

    cfg = TrainConfig(
        model_name=model_name,
        batch_size=batch_size,
        max_len=max_len,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        grad_accum_steps=grad_accum_steps,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
        num_workers=num_workers,
        fp16=(fp16_cfg and not args.no_fp16),
        da=da,
        alpha_aug=alpha_aug,
        summarize=summarize,
        dk=dk,
        spacy_model=spacy_model,
        fixed_threshold=args.fixed_threshold,
    )

    summary = train_loop(
        train_examples=train_examples,
        val_examples=val_examples,
        cfg=cfg,
        run_dir=run_dir,
        ctx=ctx,
        test_examples=test_examples,
        sample_weights=sample_weights,
    )

    if is_rank0(ctx):
        print(f"Run directory: {run_dir}")
        if summary:
            print(summary)

    cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
