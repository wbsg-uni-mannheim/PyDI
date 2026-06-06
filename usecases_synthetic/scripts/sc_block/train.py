#!/usr/bin/env python3
"""Train a per-domain SC-Block encoder (supervised contrastive blocking).

Implements R5 EM blocking sub-D (B8) from
``plans/plan_s1_scale.md``: produce one HuggingFace-format checkpoint
per domain at
``cache/sc_block_checkpoints/<domain>/run_<ts>/checkpoints/best/`` and
symlink ``cache/sc_block_checkpoints/<domain>/best/`` to the winning
run so :class:`usecases_synthetic.lib.sc_block_blocker.SCBlockBlocker`
can load it via
:func:`transformers.AutoModel.from_pretrained`.

Recipe (user-signed-off 2026-05-10)
-----------------------------------

- **Encoder**: ``roberta-base`` (RoBERTa tokenizer handles
  international names well; 125M params; ~20-30 min/domain on Apple
  Silicon MPS).
- **Loss**: paper-faithful supervised contrastive (SupCon, temperature
  0.07). Cluster-balanced batches: 32 distinct clusters x 2 records =
  batch 64. Singletons excluded by default so every anchor has at
  least one in-batch positive.
- **Hard negatives**: in-batch random only (v1).
- **Hyperparameters**: lr 2e-5, weight_decay 0.01, warmup_ratio 0.1,
  epochs 10, max_len 128, fp32 (MPS autocast is unreliable).
- **Per-domain field set** (matches the per-domain Ditto fields so
  train ↔ inference share the serialization shape):

    - companies: ``[name, country, city, industry, founded]``
    - games: ``[name, platform, genres, developer, releaseYear]``
    - music: ``[name, artist, release-date, release-country, duration]``

Selection
---------

Per-epoch eval embeds the source pair declared by ``--eval-pair`` and
materializes :class:`SCBlockBlocker` with
``top_k=50, threshold=0.3`` against the EM val split. The best-val
checkpoint is saved via
``model.save_pretrained(<run>/checkpoints/best)`` plus the matching
tokenizer.

Usage
-----

::

    python usecases_synthetic/scripts/sc_block/train.py \
        --domain companies \
        --eval-pair forbes,dbpedia \
        --epochs 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from usecases_synthetic.lib.column_mapping import apply_column_mapping  # noqa: E402
from usecases_synthetic.lib.domain_config import (  # noqa: E402
    USECASES_DIR,
    data_root_for_domain,
)
from usecases_synthetic.lib.loaders import (  # noqa: E402
    load_domain_sources,
    read_em_gold_csv,
)
from usecases_synthetic.lib.sc_block_blocker import SCBlockBlocker  # noqa: E402
from usecases_synthetic.lib.sc_block_train import (  # noqa: E402
    DOMAIN_TEXT_COLS,
    SC_BLOCK_TEXT_COLS_OVERRIDE,
    ClusterBalancedSampler,
    build_record_clusters,
    build_train_records,
    supcon_loss,
)


def _em_dir_for_domain(domain: str) -> Path:
    """Resolve the per-domain EM gold directory.

    Honors the ``data_root`` override declared in
    ``config/domains/<domain>.yaml`` so domains routed under
    ``usecases_synthetic/usecases/`` (e.g. products) are found
    correctly without the trainer hardcoding the outer
    ``usecases/<domain>`` path.
    """
    root = data_root_for_domain(domain) or USECASES_DIR
    return root / domain / "input" / "entitymatching"


logger = logging.getLogger("sc_block.train")


CACHE_DIR = REPO_ROOT / "usecases_synthetic" / "cache" / "sc_block_checkpoints"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_blocking_column_mapping(domain: str) -> dict[str, dict[str, str]]:
    """Read ``column_mapping`` from the per-domain EM blocking YAML."""
    suffix = "" if domain == "companies" else f"_{domain}"
    path = (
        REPO_ROOT
        / "usecases_synthetic"
        / "config"
        / "committees"
        / f"em_blocking_committee{suffix}.yaml"
    )
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    static = raw.get("column_mapping") or {}
    out: dict[str, dict[str, str]] = {}
    for src, mapping in static.items():
        out[src] = dict(mapping or {})
    return out


def _list_em_train_files(domain: str) -> list[Path]:
    """Return every ``*_train.csv`` in the domain's entitymatching dir.

    Walks the domain's EM gold dir (honoring ``data_root`` overrides
    declared in ``config/domains/<domain>.yaml``) and any ``train_test/``
    subfolder so the games-domain split convention is picked up
    alongside the canonical layout.
    """
    em_dir = _em_dir_for_domain(domain)
    out: list[Path] = []
    if not em_dir.exists():
        return out
    for path in em_dir.rglob("*_train.csv"):
        if "old" in path.parts:
            continue
        if path.name.endswith("_small.csv"):
            continue
        out.append(path)
    return sorted(out)


def _parse_pair_from_filename(name: str) -> tuple[str, str] | None:
    """Extract ``(src1, src2)`` from a ``<src1>_2_<src2>_train.csv`` name.

    Falls back to splitting on ``_train.csv`` when the ``_2_``
    separator is absent (e.g. games ``metacritic_dbpedia_train.csv``
    in the ``train_test/`` subdir).
    """
    base = name
    if base.endswith("_train.csv"):
        base = base[: -len("_train.csv")]
    elif base.endswith("_val.csv"):
        base = base[: -len("_val.csv")]
    else:
        return None
    if "_2_" in base:
        a, b = base.split("_2_", 1)
        return (a, b)
    parts = base.split("_")
    if len(parts) < 2:
        return None
    return (parts[0], parts[1])


def _load_em_pair_splits(
    domain: str,
    pair: tuple[str, str],
) -> dict[str, pd.DataFrame]:
    """Locate ``train`` + ``val`` CSVs for a pair, modulo orientation.

    Searches both the canonical ``input/entitymatching/`` dir and the
    games-domain ``train_test/`` subfolder. Tries both ``<src1>_2_
    <src2>`` and ``<src2>_2_<src1>`` orderings, plus the unsuffixed
    ``<src1>_<src2>`` variant used in ``train_test/``. Honors the
    per-domain ``data_root`` override.
    """
    em_dir = _em_dir_for_domain(domain)
    sub_dirs = [em_dir, em_dir / "train_test"]
    src1, src2 = pair
    out: dict[str, pd.DataFrame] = {}
    for split in ("train", "val"):
        for sub in sub_dirs:
            if not sub.exists():
                continue
            src1_c = src1.replace("_", "")
            src2_c = src2.replace("_", "")
            candidates = [
                sub / f"{src1}_2_{src2}_{split}.csv",
                sub / f"{src2}_2_{src1}_{split}.csv",
                sub / f"{src1}_{src2}_{split}.csv",
                sub / f"{src2}_{src1}_{split}.csv",
                # Underscore-condensed forms for the 2026 papers domain
                # (filename uses ``openalex`` for the source named
                # ``open_alex``).
                sub / f"{src1_c}_{src2_c}_{split}.csv",
                sub / f"{src1}_{src2_c}_{split}.csv",
                sub / f"{src1_c}_{src2}_{split}.csv",
                sub / f"{src2_c}_{src1_c}_{split}.csv",
            ]
            for cand in candidates:
                if cand.exists():
                    out[split] = read_em_gold_csv(cand)
                    break
            if split in out:
                break
    return out


def _swap_id_columns(
    df: pd.DataFrame, current: tuple[str, str], desired: tuple[str, str]
) -> pd.DataFrame:
    """Swap id1/id2 if the file orientation does not match the desired pair."""
    if current == desired:
        return df
    out = df.copy()
    out["id1"], out["id2"] = out["id2"], out["id1"]
    return out


def _load_domain_data(
    domain: str,
    text_cols: list[str],
) -> tuple[
    dict[str, pd.DataFrame],
    dict[tuple[str, str], pd.DataFrame],
    dict[tuple[str, str], dict[str, pd.DataFrame]],
]:
    """Return (sources_mapped, em_train_by_pair, em_splits_by_pair).

    ``sources_mapped`` carries canonical column names. ``em_train_by_
    pair`` is the per-pair train DataFrame keyed by canonical pair
    direction. ``em_splits_by_pair`` carries every loaded split (train
    + val) per pair, with id columns aligned to the canonical
    direction.
    """
    raw_sources = load_domain_sources(domain)
    rename_map = _load_blocking_column_mapping(domain)
    sources_mapped: dict[str, pd.DataFrame] = {}
    for name, df in raw_sources.items():
        mapping = rename_map.get(name, {})
        sources_mapped[name] = (
            apply_column_mapping(df, mapping) if mapping else df.copy()
        )
        missing = [c for c in text_cols if c not in sources_mapped[name].columns]
        if missing:
            logger.warning(
                "source %s missing text_cols %s after column_mapping; "
                "those fields will serialize as empty",
                name,
                missing,
            )
            for col in missing:
                sources_mapped[name][col] = pd.NA

    em_train_by_pair: dict[tuple[str, str], pd.DataFrame] = {}
    em_splits_by_pair: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    seen_train: set[Path] = set()
    for train_path in _list_em_train_files(domain):
        if train_path in seen_train:
            continue
        seen_train.add(train_path)
        parsed = _parse_pair_from_filename(train_path.name)
        if parsed is None:
            logger.warning("skipping unrecognised EM filename: %s", train_path.name)
            continue
        # Resolve underscore-condensed source tokens (e.g. ``openalex``
        # in ``dblp_openalex_train.csv`` for the source named
        # ``open_alex``) so papers' EM gold matches the configured
        # source list.
        if parsed[0] not in sources_mapped or parsed[1] not in sources_mapped:
            sources_collapsed = {s.replace("_", ""): s for s in sources_mapped}
            normalized = (
                sources_collapsed.get(parsed[0].replace("_", ""), parsed[0]),
                sources_collapsed.get(parsed[1].replace("_", ""), parsed[1]),
            )
            if normalized[0] in sources_mapped and normalized[1] in sources_mapped:
                parsed = normalized
        if parsed[0] not in sources_mapped or parsed[1] not in sources_mapped:
            logger.warning(
                "EM file %s references unknown sources %s; skipping",
                train_path.name,
                parsed,
            )
            continue
        splits = _load_em_pair_splits(domain, parsed)
        if "train" not in splits:
            logger.warning("EM pair %s has no train file at expected paths", parsed)
            continue
        em_train_by_pair[parsed] = splits["train"]
        em_splits_by_pair[parsed] = splits
    return sources_mapped, em_train_by_pair, em_splits_by_pair


# ---------------------------------------------------------------------------
# Eval (held-out source pair)
# ---------------------------------------------------------------------------


def _evaluate_recall(
    model_path: Path | None,
    sources_mapped: dict[str, pd.DataFrame],
    eval_pair: tuple[str, str],
    eval_splits: dict[str, pd.DataFrame],
    text_cols: list[str],
    *,
    encoder_fn: Any | None = None,
    top_k: int = 50,
    threshold: float = 0.3,
    device: str | None = None,
) -> dict[str, float]:
    """Embed source pair + score pair_recall vs the val split.

    Either ``model_path`` (HF checkpoint) OR ``encoder_fn`` (test
    injection) must be provided.

    Returns
    -------
    dict
        ``{"pair_recall": float, "n_candidates": int, "n_positives": int}``.
    """
    src1, src2 = eval_pair
    if "val" in eval_splits and not eval_splits["val"].empty:
        gold = eval_splits["val"]
    elif "train" in eval_splits:
        gold = eval_splits["train"]
    else:
        raise ValueError(f"no val/train split available for {eval_pair}")

    blocker = SCBlockBlocker(
        sources_mapped[src1],
        sources_mapped[src2],
        id_column="id",
        text_cols=text_cols,
        checkpoint_path=str(model_path) if model_path is not None else None,
        top_k=top_k,
        threshold=threshold,
        encoder=encoder_fn,
        device=device,
        index_backend="sklearn",
    )
    candidates = blocker.materialize()

    def _is_positive(v: Any) -> bool:
        # Accept bool True, int/float 1, and the string forms
        # ``"true"`` / ``"1"`` (case-insensitive, whitespace-stripped).
        # Papers' EM gold uses integer 0/1 labels (per the canonical
        # header-bearing schema), while the older synthetic header-less
        # CSVs use string ``true``/``false``.
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return int(v) == 1
        s = str(v).strip().lower()
        return s in {"true", "1"}

    pos = gold[gold["label"].apply(_is_positive)]
    pos_set = set(zip(pos["id1"].astype(str).tolist(), pos["id2"].astype(str).tolist()))
    cand_set = set(
        zip(
            candidates["id1"].astype(str).tolist(),
            candidates["id2"].astype(str).tolist(),
        )
    )
    intersect = pos_set & cand_set
    recall = (len(intersect) / len(pos_set)) if pos_set else 0.0
    return {
        "pair_recall": float(recall),
        "n_candidates": int(len(cand_set)),
        "n_positives": int(len(pos_set)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _autoselect_device(torch_mod: Any) -> Any:
    if torch_mod.cuda.is_available():
        return torch_mod.device("cuda")
    mps = getattr(torch_mod.backends, "mps", None)
    if (
        mps is not None
        and getattr(mps, "is_available", lambda: False)()
        and getattr(mps, "is_built", lambda: False)()
    ):
        return torch_mod.device("mps")
    return torch_mod.device("cpu")


def train(
    *,
    domain: str,
    eval_pair: tuple[str, str],
    output_dir: Path,
    model_name: str = "roberta-base",
    epochs: int = 10,
    batch_clusters: int = 32,
    records_per_cluster: int = 2,
    max_len: int = 128,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    temperature: float = 0.07,
    seed: int = 42,
    eval_top_k: int = 50,
    eval_threshold: float = 0.3,
    device_override: str | None = None,
    log_every: int = 25,
    data_override: (
        tuple[
            dict[str, pd.DataFrame],
            dict[tuple[str, str], pd.DataFrame],
            dict[tuple[str, str], dict[str, pd.DataFrame]],
        ]
        | None
    ) = None,
) -> dict[str, Any]:
    """Train one SC-Block encoder for a domain.

    Returns a summary dict (saved to ``metrics.json`` in the run dir).
    """
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    # Blocking text_cols may be narrower than the matching field set (e.g.
    # papers blocks on [title] only); fall back to DOMAIN_TEXT_COLS otherwise.
    text_cols = SC_BLOCK_TEXT_COLS_OVERRIDE.get(domain, DOMAIN_TEXT_COLS[domain])
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = (
        torch.device(device_override) if device_override else _autoselect_device(torch)
    )

    logger.info(
        "Domain=%s model=%s device=%s seed=%d", domain, model_name, device, seed
    )
    logger.info("text_cols=%s", text_cols)

    # ----- Data prep
    # R10-G: ``data_override`` lets a variant-retrain caller inject
    # pre-loaded (K8-resolved) variant sources + corner_filled splits so
    # the variant encoder trains on the perturbed data without this
    # function reaching into the baseline ``usecases/<domain>/`` dir.
    if data_override is not None:
        sources_mapped, em_train_by_pair, em_splits_by_pair = data_override
    else:
        sources_mapped, em_train_by_pair, em_splits_by_pair = _load_domain_data(
            domain, text_cols
        )
    if eval_pair not in em_splits_by_pair:
        # Tolerate reversed orientation
        reversed_pair = (eval_pair[1], eval_pair[0])
        if reversed_pair in em_splits_by_pair:
            eval_pair = reversed_pair
        else:
            raise ValueError(
                f"eval-pair {eval_pair} not found among {list(em_splits_by_pair)}"
            )

    record_to_cluster = build_record_clusters(em_train_by_pair, sources_mapped)
    records = build_train_records(sources_mapped, record_to_cluster, text_cols)
    # O(n) cluster-size tally via Counter. The previous form computed
    # ``n_multi`` with a nested ``sum(1 for x in records ...)`` per record —
    # O(n_records^2), which silently hung for hours on large domains (papers
    # has ~182k records => ~3.3e10 ops, single-core, before the blocker init).
    _cluster_sizes = Counter(r.cluster_id for r in records)
    n_clusters = len(_cluster_sizes)
    n_multi = sum(1 for _sz in _cluster_sizes.values() if _sz >= 2)
    logger.info(
        "n_records=%d n_clusters=%d n_clusters_with_>=2_records=%d",
        len(records),
        n_clusters,
        n_multi,
    )

    sampler = ClusterBalancedSampler(
        records,
        clusters_per_batch=batch_clusters,
        records_per_cluster=records_per_cluster,
        shuffle=True,
        drop_last=True,
        seed=seed,
    )
    batch_size = sampler.batch_size
    logger.info("batch_size=%d n_batches_per_epoch=%d", batch_size, len(sampler))
    if len(sampler) == 0:
        raise RuntimeError(
            "no eligible training batches; check that the EM gold has "
            "enough positive pairs to form clusters of size >= 2"
        )

    # ----- Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(len(sampler), 1)
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Pre-collect cluster ids + texts so the inner loop only tokenises.
    texts = [r.text for r in records]
    cluster_arr = np.asarray([r.cluster_id for r in records], dtype=np.int64)

    def _collate(batch_indices: list[int]) -> dict[str, Any]:
        batch_texts = [texts[i] for i in batch_indices]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "cluster_ids": torch.tensor(cluster_arr[batch_indices], dtype=torch.long),
        }

    def _batch_iter(epoch: int) -> Any:
        sampler.set_epoch(epoch)
        for batch_idx in sampler:
            yield _collate(batch_idx)

    best_val_recall = -1.0
    best_epoch = -1
    history: list[dict[str, Any]] = []
    t_train_start = time.monotonic()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_steps = 0
        t_epoch = time.monotonic()
        for step, batch in enumerate(_batch_iter(epoch)):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            cluster_ids = batch["cluster_ids"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=input_ids, attention_mask=attn)
            hidden = outputs.last_hidden_state
            cls = hidden[:, 0, :]
            z = F.normalize(cls, dim=-1)
            loss = supcon_loss(z, cluster_ids, temperature=temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            n_steps += 1
            if (step + 1) % log_every == 0:
                logger.info(
                    "epoch=%d step=%d/%d loss=%.4f lr=%.2e",
                    epoch,
                    step + 1,
                    len(sampler),
                    running_loss / max(n_steps, 1),
                    scheduler.get_last_lr()[0],
                )
        avg_loss = running_loss / max(n_steps, 1)
        epoch_secs = time.monotonic() - t_epoch

        # Eval: save current to a tmp dir, evaluate via SCBlockBlocker.
        eval_tmp = run_dir / f"_eval_epoch_{epoch:02d}"
        eval_tmp.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(eval_tmp)
        tokenizer.save_pretrained(eval_tmp)

        # Evaluate.
        t_eval = time.monotonic()
        model.eval()
        eval_metrics = _evaluate_recall(
            eval_tmp,
            sources_mapped,
            eval_pair,
            em_splits_by_pair[eval_pair],
            text_cols,
            top_k=eval_top_k,
            threshold=eval_threshold,
            device=str(device),
        )
        eval_secs = time.monotonic() - t_eval
        history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "val_pair_recall": eval_metrics["pair_recall"],
                "val_n_candidates": eval_metrics["n_candidates"],
                "val_n_positives": eval_metrics["n_positives"],
                "epoch_secs": round(epoch_secs, 1),
                "eval_secs": round(eval_secs, 1),
            }
        )
        logger.info(
            "epoch=%d done: loss=%.4f val_pair_recall=%.4f "
            "(n_candidates=%d, n_positives=%d, epoch=%.1fs, eval=%.1fs)",
            epoch,
            avg_loss,
            eval_metrics["pair_recall"],
            eval_metrics["n_candidates"],
            eval_metrics["n_positives"],
            epoch_secs,
            eval_secs,
        )
        if eval_metrics["pair_recall"] > best_val_recall:
            best_val_recall = eval_metrics["pair_recall"]
            best_epoch = epoch
            best_dir = checkpoints_dir / "best"
            if best_dir.exists():
                import shutil

                shutil.rmtree(best_dir)
            # Move the eval_tmp dir to best/ (rename is cheap).
            eval_tmp.rename(best_dir)
        else:
            # Cleanup the per-epoch tmp checkpoint.
            import shutil

            shutil.rmtree(eval_tmp, ignore_errors=True)

    total_secs = time.monotonic() - t_train_start
    summary = {
        "domain": domain,
        "model_name": model_name,
        "device": str(device),
        "eval_pair": list(eval_pair),
        "text_cols": text_cols,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "temperature": temperature,
        "max_len": max_len,
        "seed": seed,
        "n_records": len(records),
        "n_clusters": n_clusters,
        "best_epoch": best_epoch,
        "best_val_pair_recall": best_val_recall,
        "wall_secs": round(total_secs, 1),
        "history": history,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Maintain a stable `best/` symlink at the domain level.
    domain_best = output_dir / "best"
    if domain_best.is_symlink() or domain_best.exists():
        # If it exists as a real directory (legacy), don't clobber.
        if domain_best.is_symlink():
            domain_best.unlink()
    if best_epoch >= 0:
        rel_best = (run_dir / "checkpoints" / "best").relative_to(output_dir)
        try:
            os.symlink(rel_best, domain_best)
        except OSError as exc:  # pragma: no cover - platform guard
            logger.warning("could not create best/ symlink: %s", exc)

    logger.info(
        "Done. Best epoch=%d val_pair_recall=%.4f wall=%.1fs run=%s",
        best_epoch,
        best_val_recall,
        total_secs,
        run_dir,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_DEFAULT_EVAL_PAIRS: dict[str, tuple[str, str]] = {
    "companies": ("forbes", "dbpedia"),
    "games": ("metacritic", "dbpedia"),
    "music": ("musicbrainz", "discogs"),
    # products: anchor pair is products_1 ↔ products_2 (the largest
    # authored pair: 812 ↔ 812 rows, ~1800 train pairs).
    "products": ("products_1", "products_2"),
    # papers: anchor pair is dblp ↔ crossref; both have ~60k records
    # with the canonical 15-attr target schema.
    "papers": ("dblp", "crossref"),
}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=list(DOMAIN_TEXT_COLS))
    parser.add_argument(
        "--eval-pair",
        default=None,
        help="Comma-separated <src1>,<src2> used for per-epoch validation. "
        "Defaults: companies=forbes,dbpedia; games=metacritic,dbpedia; "
        "music=musicbrainz,discogs.",
    )
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-clusters", type=int, default=32)
    parser.add_argument("--records-per-cluster", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-top-k", type=int, default=50)
    parser.add_argument("--eval-threshold", type=float, default=0.3)
    parser.add_argument(
        "--device", default=None, help="cpu / cuda / mps. Auto-detect by default."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to cache/sc_block_checkpoints/<domain>/.",
    )
    args = parser.parse_args()

    if args.eval_pair is not None:
        parts = [p.strip() for p in args.eval_pair.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("--eval-pair must be <src1>,<src2>")
        eval_pair = (parts[0], parts[1])
    else:
        eval_pair = _DEFAULT_EVAL_PAIRS[args.domain]

    out_dir = args.output_dir or (CACHE_DIR / args.domain)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = train(
        domain=args.domain,
        eval_pair=eval_pair,
        output_dir=out_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_clusters=args.batch_clusters,
        records_per_cluster=args.records_per_cluster,
        max_len=args.max_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        temperature=args.temperature,
        seed=args.seed,
        eval_top_k=args.eval_top_k,
        eval_threshold=args.eval_threshold,
        device_override=args.device,
    )

    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
