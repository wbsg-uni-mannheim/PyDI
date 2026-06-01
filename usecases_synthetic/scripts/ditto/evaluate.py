#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from usecases_synthetic.third_party.ditto_modern.data import (
    load_wdc_json_gz,
    wdc_to_pair_examples,
)
from usecases_synthetic.third_party.ditto_modern.knowledge import inject_knowledge
from usecases_synthetic.third_party.ditto_modern.model import load_model, load_tokenizer
from usecases_synthetic.third_party.ditto_modern.runtime import (
    cleanup_distributed,
    create_run_dir,
    init_distributed,
    is_rank0,
    save_json,
    set_seed,
)
from usecases_synthetic.third_party.ditto_modern.trainer import (
    TrainConfig,
    evaluate_loop,
)
from usecases_synthetic.third_party.ditto_modern.summarize import summarize_examples

DEFAULT_FIELDS = ["title", "brand", "description", "price", "priceCurrency"]
DEFAULT_TEST = "data/wdc/wdcproducts80cc20rnd100un_gs.json.gz"


def _parse_fields(raw: str):
    out = [x.strip() for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("At least one field is required")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate modernized Ditto-style matcher"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to saved checkpoint directory"
    )
    parser.add_argument("--test-json-gz", default=DEFAULT_TEST)
    parser.add_argument("--output-dir", default="output/ditto_runs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
    parser.add_argument("--max-field-len", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for positive class",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Force-enable summarization during eval",
    )
    parser.add_argument(
        "--dk", default=None, help="Force DK injector during eval: product|general"
    )
    parser.add_argument(
        "--spacy-model", default=None, help="spaCy model for DK injector"
    )
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()

    ctx = init_distributed(explicit_ddp=args.ddp)
    set_seed(args.seed + ctx.rank)

    run_dir = create_run_dir(args.output_dir, prefix="eval") if is_rank0(ctx) else None
    if ctx.enabled:
        obj = [str(run_dir) if is_rank0(ctx) else ""]
        dist.broadcast_object_list(obj, src=0)
        run_dir = Path(obj[0])

    fields = _parse_fields(args.fields)

    run_cfg = {}
    parent_cfg = Path(args.checkpoint).resolve().parents[1] / "config.json"
    if parent_cfg.exists():
        with open(parent_cfg, "r") as f:
            run_cfg = json.load(f)

    test_df = load_wdc_json_gz(args.test_json_gz)
    test_examples = wdc_to_pair_examples(
        test_df, fields=fields, max_field_len=args.max_field_len
    )

    use_summarize = bool(args.summarize or run_cfg.get("summarize", False))
    dk = args.dk if args.dk is not None else run_cfg.get("dk", None)
    spacy_model = (
        args.spacy_model
        if args.spacy_model is not None
        else run_cfg.get("spacy_model", "en_core_web_sm")
    )

    if use_summarize:
        # build summarizer index from test only when evaluating independently
        _, _, test_examples = summarize_examples(
            train_examples=[],
            val_examples=[],
            test_examples=test_examples,
            lm=str(run_cfg.get("model_name", args.checkpoint)),
            max_len=args.max_len,
        )

    if dk is not None:
        dk_norm = str(dk).strip().lower()
        if dk_norm not in {"product", "general"}:
            raise ValueError("--dk must be one of: product, general")
        test_examples = inject_knowledge(
            test_examples, dk=dk_norm, spacy_model=spacy_model
        )

    tokenizer = load_tokenizer(args.checkpoint)
    model = load_model(args.checkpoint)
    model.to(ctx.device)

    cfg = TrainConfig(batch_size=args.batch_size, max_len=args.max_len)

    threshold = args.threshold
    if threshold is None:
        bm = Path(args.checkpoint).parent / "best_metrics.json"
        if bm.exists():
            with open(bm, "r") as f:
                payload = json.load(f)
            threshold = float(payload.get("val_threshold", 0.5))
        else:
            threshold = 0.5

    metrics = evaluate_loop(
        model, test_examples, cfg, ctx, tokenizer=tokenizer, threshold=threshold
    )

    if is_rank0(ctx):
        idx_to_pair = {ex.idx: ex.pair_id for ex in test_examples}
        pred_df = pd.DataFrame(
            {
                "idx": metrics["idxs"],
                "pair_id": [idx_to_pair.get(i, f"idx-{i}") for i in metrics["idxs"]],
                "gold": metrics["labels"],
                "pred": metrics["preds"],
                "prob": metrics["probs"],
            }
        )
        pred_path = run_dir / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        out = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
            "loss": metrics["loss"],
            "threshold": threshold,
            "summarize": use_summarize,
            "dk": dk,
            "spacy_model": spacy_model if dk is not None else None,
            "checkpoint": str(args.checkpoint),
            "test_json_gz": str(args.test_json_gz),
        }
        save_json(run_dir / "metrics.json", out)
        print(f"Evaluation run: {run_dir}")
        print(out)

    cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
