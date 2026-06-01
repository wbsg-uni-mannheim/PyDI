"""Ditto-based PyDI matcher adapter (synthetic-local infrastructure).

This module lives under ``usecases_synthetic/lib/`` rather than
``PyDI/entitymatching/`` because the Ditto runtime is vendored as synthetic-use
infrastructure (see ``usecases_synthetic/third_party/ditto_modern/``). If the
Ditto adapter later generalises beyond the synthetic pipeline, the promotion
path is to move ``DittoMatcher`` into ``PyDI/entitymatching/ditto.py`` and
re-export it from ``PyDI.entitymatching``.

Used as the PLM member of the EM matching committee (see
``config/committees/em_matching_committee.yaml`` and the ``ditto_plm`` entry).
Inference-only; training happens via
``usecases_synthetic/scripts/ditto/train.py``.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd
import torch

from PyDI.entitymatching.base import BaseMatcher, CorrespondenceSet

from usecases_synthetic.third_party.ditto_modern.data import (
    RESERVED_SERIALIZATION_FIELDS,
    serialize_entity,
)
from usecases_synthetic.third_party.ditto_modern.model import load_model, load_tokenizer

logger = logging.getLogger(__name__)

_CACHE_HEADER = ("id1", "id2", "score")


def _auto_select_device() -> torch.device:
    """Pick the best available torch device for Ditto inference.

    Preference order: CUDA → MPS (Apple Silicon) → CPU. MPS is only
    selected when both ``torch.backends.mps.is_available()`` and
    ``torch.backends.mps.is_built()`` are true so users without a built
    MPS backend transparently fall back to CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if (
        mps_backend is not None
        and getattr(mps_backend, "is_available", lambda: False)()
        and getattr(mps_backend, "is_built", lambda: False)()
    ):
        return torch.device("mps")
    return torch.device("cpu")


class DittoMatcher(BaseMatcher):
    """PyDI adapter around a trained Ditto checkpoint (inference only).

    The matcher loads a checkpoint produced by
    ``usecases_synthetic/scripts/ditto/train.py`` lazily (on the first
    ``match`` call) and caches it across subsequent calls so that the committee
    runner's per-source-pair invocations do not reload the model each time.

    Parameters
    ----------
    checkpoint_path : str or Path
        Directory containing ``model.pt``, ``model_config.json``, and tokenizer
        files (the ``checkpoints/best/`` sub-directory of a training run).
    fields : sequence of str
        Canonical schema fields to serialize, in the same order they were
        passed to the trainer. Both ``df_left`` and ``df_right`` are expected
        to carry these column names after the committee runner's
        ``column_mapping`` rename step.
    max_len : int, default 256
        Maximum pair-sequence length (tokens) fed to the encoder.
    max_field_len : int, default 350
        Per-field character truncation applied before tokenisation.
    batch_size : int, default 16
        Inference batch size.
    device : str, optional
        Torch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"mps"``,
        ...). If ``None``, auto-detects the best available accelerator in the
        order CUDA → MPS (Apple Silicon) → CPU.
    cache_dir : str, Path, or False, optional
        Per-batch inference cache root. Every batch of Ditto inference
        is appended to a CSV keyed by a content hash of the candidate
        set + checkpoint state, so a killed run can resume on the next
        call without re-scoring already-evaluated pairs. The committee's
        full EM stage at domain scale (tens of thousands of pairs × ~10
        ms per pair on CPU/MPS) can take an hour or more, and a SIGINT
        or lid-close otherwise loses every pair scored so far. Cache
        files are partitioned by ``(checkpoint_resolved_path,
        checkpoint_mtime, max_len, max_field_len, fields,
        candidate_pair_set)``, so any change to those inputs uses a
        fresh cache file. Threshold is *not* part of the key — scores
        are cached unfiltered and the threshold is applied on the
        returned DataFrame, so re-running with a different threshold
        reuses cached scores. Defaults to a synthetic-local location
        (``usecases_synthetic/cache/ditto_inference/``) — pass
        ``cache_dir=False`` to disable caching entirely (e.g. for unit
        tests that want a stateless matcher).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        fields: Sequence[str],
        max_len: int = 256,
        max_field_len: int = 350,
        batch_size: int = 16,
        device: Optional[str] = None,
        cache_dir: Optional[str | Path | bool] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        # Drop any field whose name collides with a Ditto WDC reserved
        # metadata key (e.g. music's ``label`` record-label attribute). The
        # WDC pair format uses bare ``label`` / ``pair_id`` / ``id`` for the
        # pair metadata, and the training data builder
        # (``wdc_to_pair_examples`` / ``committee_ditto_fields``) likewise
        # excludes these, so dropping them here keeps inference serialization
        # byte-identical to what the checkpoint was trained on. Other
        # committee members serialize the full field list; this filter is
        # Ditto-specific. See em_matching_committee_music.yaml header note.
        requested = list(fields)
        self.fields: list[str] = [
            f for f in requested if f not in RESERVED_SERIALIZATION_FIELDS
        ]
        dropped = [f for f in requested if f in RESERVED_SERIALIZATION_FIELDS]
        if dropped:
            logger.info(
                "DittoMatcher: dropping reserved field name(s) %s from the "
                "serialization scope (not representable in Ditto's WDC format)",
                dropped,
            )
        self.max_len = int(max_len)
        self.max_field_len = int(max_field_len)
        self.batch_size = int(batch_size)
        self._device_arg = device
        self.cache_dir: Path | None = self._resolve_cache_dir(cache_dir)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: torch.device | None = None

    @staticmethod
    def _resolve_cache_dir(arg: str | Path | bool | None) -> Path | None:
        """Pick the inference cache directory.

        ``None`` (the default) → synthetic-local
        ``usecases_synthetic/cache/ditto_inference/`` resolved from this
        module's location upward. ``False`` (or ``"off"``) → caching
        disabled. Any other str/Path → that exact path.
        """
        if arg is False or (isinstance(arg, str) and arg.lower() == "off"):
            return None
        if arg is None:
            # Default: <repo_root>/usecases_synthetic/cache/ditto_inference.
            # This file lives at usecases_synthetic/lib/ditto_matcher.py;
            # parents[1] is usecases_synthetic/.
            module_root = Path(__file__).resolve().parents[1]
            return module_root / "cache" / "ditto_inference"
        return Path(arg)

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"DittoMatcher checkpoint not found: {self.checkpoint_path}"
            )
        if self._device_arg is not None:
            self._device = torch.device(self._device_arg)
        else:
            self._device = _auto_select_device()
        self._tokenizer = load_tokenizer(str(self.checkpoint_path))
        model = load_model(str(self.checkpoint_path))
        model.eval()
        model.to(self._device)
        self._model = model
        logger.info(
            "DittoMatcher loaded checkpoint %s on %s",
            self.checkpoint_path,
            self._device,
        )

    def _pair_text(
        self,
        left_row: pd.Series,
        right_row: pd.Series,
    ) -> tuple[str, str]:
        """Build the ``COL ... VAL ...`` left/right strings for a pair.

        Missing values (``None`` / float ``NaN``) are skipped so the field
        drops out of the serialization, mirroring the training-data builder
        (``prepare_em_training_data._committee_record`` maps the same nulls
        to ``""``). Without this, ``serialize_entity`` would ``str()`` a NaN
        cell into a literal ``"COL <field> VAL nan"`` token at inference that
        the checkpoint never saw in training — a train/inference mismatch
        that the wide R10-I field scope (many sparse numeric columns:
        price, vram_gb, durations, ...) makes pervasive.
        """
        left_rec: dict[str, object] = {}
        right_rec: dict[str, object] = {}
        for field in self.fields:
            if field in left_row.index:
                val = left_row[field]
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    left_rec[f"{field}_left"] = val
            if field in right_row.index:
                val = right_row[field]
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    right_rec[f"{field}_right"] = val
        left_text = serialize_entity(left_rec, "left", self.fields, self.max_field_len)
        right_text = serialize_entity(
            right_rec, "right", self.fields, self.max_field_len
        )
        return left_text, right_text

    def _score_batch(self, texts: list[tuple[str, str]]) -> list[float]:
        assert self._tokenizer is not None and self._model is not None
        assert self._device is not None
        lefts = [t[0] for t in texts]
        rights = [t[1] for t in texts]
        # R7 (2026-05-27): use HF tokenizer's standard padding convention
        # (pad_token_id, attention_mask via tokenizer). Pre-R7 the
        # trainer (PairDataset.pad) hard-coded pad=0 + masked CLS, and
        # this code path emulated that buggy convention to keep
        # inference consistent (R6-4). Trainer is now fixed (uses
        # tokenizer.pad_token_id), so revert to the standard HF call.
        # Checkpoints trained pre-R7 are still buggy — they need
        # retraining (R7-4) before this matcher gives correct outputs.
        enc = self._tokenizer(
            lefts,
            rights,
            max_length=self.max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)
        with torch.no_grad():
            logits = self._model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)[:, 1]
        return [float(p) for p in probs.cpu().tolist()]

    def _value_lookup(self, df: pd.DataFrame, id_column: str) -> dict[str, str]:
        """Map ``str(id) -> '|'.join(str(field values))`` for one source frame.

        Used by :meth:`_cache_key` so the same candidate pair scored against
        *different* source record values (e.g. the same id at successive
        variant levels, after K1/K5/K6 perturbation) produces a distinct
        cache key. Only fields present in *df* are included; first
        occurrence wins on duplicate ids, matching the ``.loc[...].iloc[0]``
        lookup in :meth:`match`.
        """
        if df.empty:
            return {}
        fields = [f for f in self.fields if f in df.columns]
        ids = df[id_column].astype(str).tolist()
        if fields:
            values = df[fields].astype(str).apply("\x1f".join, axis=1).tolist()
        else:
            values = ["" for _ in ids]
        lookup: dict[str, str] = {}
        for id_str, value in zip(ids, values, strict=True):
            lookup.setdefault(id_str, value)
        return lookup

    def _cache_key(
        self,
        candidates: pd.DataFrame,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_column: str,
    ) -> str:
        """Compute a stable content hash for the inference cache filename.

        The hash combines every input that affects a Ditto inference
        score: the resolved checkpoint path + its mtime (so retrains
        invalidate the cache), the tokeniser/serialiser knobs, the
        canonical-schema field order, the sorted candidate-pair set, AND
        the source record values behind each pair. Threshold is excluded
        — scores are cached pre-filter.

        R10-F (2026-05-29): the per-pair source *values* were previously
        omitted, so the same (id1, id2) pair scored against perturbed
        records across variant levels reused the first level's cached
        score — the flat ditto_plm curve in the products step-5 audit.
        Hashing the field values keys the cache on content, not just ids.
        """
        h = hashlib.sha256()
        h.update(str(self.checkpoint_path.resolve()).encode())
        if self.checkpoint_path.exists():
            h.update(str(int(self.checkpoint_path.stat().st_mtime)).encode())
        h.update(str(self.max_len).encode())
        h.update(str(self.max_field_len).encode())
        h.update(",".join(self.fields).encode())
        left_lookup = self._value_lookup(df_left, id_column)
        right_lookup = self._value_lookup(df_right, id_column)
        pairs = sorted(
            zip(
                candidates["id1"].astype(str).tolist(),
                candidates["id2"].astype(str).tolist(),
                strict=True,
            )
        )
        for id1, id2 in pairs:
            h.update(id1.encode())
            h.update(b"|")
            h.update(id2.encode())
            h.update(b"\n")
            h.update(left_lookup.get(id1, "<absent>").encode())
            h.update(b"\x1eL")
            h.update(right_lookup.get(id2, "<absent>").encode())
            h.update(b"\x1eR")
        return h.hexdigest()[:16]

    def _cache_path_for(
        self,
        candidates: pd.DataFrame,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_column: str,
    ) -> Path:
        """Return the per-candidate-set cache CSV path."""
        assert self.cache_dir is not None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(candidates, df_left, df_right, id_column)
        return self.cache_dir / f"ditto_inference_{key}.csv"

    @staticmethod
    def _load_cache(cache_path: Path) -> dict[tuple[str, str], float]:
        """Read prior batch scores from disk; tolerate a partial trailing line."""
        if not cache_path.exists():
            return {}
        scores: dict[tuple[str, str], float] = {}
        with cache_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return {}
            if tuple(header) != _CACHE_HEADER:
                logger.warning(
                    "DittoMatcher: unexpected cache header %s in %s — ignoring "
                    "cache and starting fresh",
                    header,
                    cache_path,
                )
                return {}
            for row in reader:
                if len(row) != 3:
                    # A truncated last line from a SIGKILL mid-write is
                    # the only realistic source of malformed rows.
                    continue
                try:
                    scores[(row[0], row[1])] = float(row[2])
                except ValueError:
                    continue
        return scores

    @staticmethod
    def _append_cache(
        cache_path: Path,
        new_rows: list[tuple[str, str, float]],
    ) -> None:
        """Append a batch's scores to the cache file with a final fsync.

        The header is written on first creation. fsync after the batch
        write ensures a SIGKILL after this returns cannot drop the
        batch's rows from the OS write buffer.
        """
        if not new_rows:
            return
        write_header = not cache_path.exists()
        with cache_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(_CACHE_HEADER)
            for id1, id2, score in new_rows:
                writer.writerow([str(id1), str(id2), f"{float(score):.10g}"])
            f.flush()
            os.fsync(f.fileno())

    def match(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: Iterable[pd.DataFrame] | pd.DataFrame,
        id_column: str,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> CorrespondenceSet:
        """Score candidate pairs with the Ditto checkpoint.

        Parameters
        ----------
        df_left, df_right : pandas.DataFrame
            Source DataFrames, expected to carry ``id_column`` and every entry
            in ``self.fields`` as column names. The committee runner applies
            its ``column_mapping`` before calling this.
        candidates : pandas.DataFrame or Iterable[pandas.DataFrame]
            Candidate pair batches with ``id1`` / ``id2`` columns.
        id_column : str
            Name of the id column in ``df_left`` / ``df_right``.
        threshold : float, default 0.5
            Minimum softmax class-1 probability retained in the output.
        **kwargs
            Accepted for compatibility with the committee runner
            (``comparators`` / ``weights`` passed by the runner are ignored).

        Returns
        -------
        CorrespondenceSet
            DataFrame with columns ``id1``, ``id2``, ``score``, ``notes``.
        """
        self._validate_inputs(df_left, df_right, id_column)
        self._ensure_loaded()

        if isinstance(candidates, pd.DataFrame):
            batches: list[pd.DataFrame] = [candidates]
        else:
            batches = list(candidates)

        for batch in batches:
            if not batch.empty and (
                "id1" not in batch.columns or "id2" not in batch.columns
            ):
                raise ValueError("candidate batch must have 'id1' and 'id2' columns")

        non_empty = [b for b in batches if not b.empty]
        if non_empty:
            combined = pd.concat(
                [b[["id1", "id2"]] for b in non_empty], ignore_index=True
            )
        else:
            combined = pd.DataFrame(columns=["id1", "id2"])

        # Cache lookup: when configured, prior batches' scores are
        # restored so a killed run can resume without re-scoring.
        cache_path: Path | None = None
        scores_by_pair: dict[tuple[str, str], float] = {}
        if self.cache_dir is not None and not combined.empty:
            cache_path = self._cache_path_for(combined, df_left, df_right, id_column)
            scores_by_pair = self._load_cache(cache_path)
            if scores_by_pair:
                logger.info(
                    "DittoMatcher: resumed %d cached scores from %s",
                    len(scores_by_pair),
                    cache_path,
                )

        left_index = df_left.set_index(id_column, drop=False)
        right_index = df_right.set_index(id_column, drop=False)

        # Build pair_records, skipping pairs already scored in cache.
        pair_records: list[tuple[Any, Any, str, str]] = []
        valid_pair_keys: set[tuple[str, str]] = set()
        for row in combined.itertuples(index=False):
            id1 = row.id1
            id2 = row.id2
            key = (str(id1), str(id2))
            valid_pair_keys.add(key)
            if key in scores_by_pair:
                continue
            if id1 not in left_index.index or id2 not in right_index.index:
                continue
            left_row = left_index.loc[id1]
            right_row = right_index.loc[id2]
            if isinstance(left_row, pd.DataFrame):
                left_row = left_row.iloc[0]
            if isinstance(right_row, pd.DataFrame):
                right_row = right_row.iloc[0]
            left_text, right_text = self._pair_text(left_row, right_row)
            pair_records.append((id1, id2, left_text, right_text))

        # Process remaining in batches; persist after each batch.
        for chunk_start in range(0, len(pair_records), self.batch_size):
            chunk = pair_records[chunk_start : chunk_start + self.batch_size]
            scores = self._score_batch([(l, r) for _, _, l, r in chunk])
            new_rows: list[tuple[str, str, float]] = []
            for (id1, id2, _, _), score in zip(chunk, scores, strict=True):
                key = (str(id1), str(id2))
                scores_by_pair[key] = score
                new_rows.append((str(id1), str(id2), score))
            if cache_path is not None and new_rows:
                self._append_cache(cache_path, new_rows)

        out_rows: list[dict[str, Any]] = []
        for key, score in scores_by_pair.items():
            # Restrict to current call's pair set so a stale cache
            # (e.g. from a wider candidate set) cannot leak extra rows
            # into this call's output. The cache key already includes
            # the candidate set hash so this is defence-in-depth.
            if key not in valid_pair_keys:
                continue
            if score >= threshold:
                out_rows.append(
                    {
                        "id1": key[0],
                        "id2": key[1],
                        "score": score,
                        "notes": "ditto_plm",
                    }
                )

        if not out_rows:
            return pd.DataFrame(columns=["id1", "id2", "score", "notes"])
        return pd.DataFrame(out_rows, columns=["id1", "id2", "score", "notes"])
