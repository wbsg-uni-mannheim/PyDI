"""Modernized Ditto-style entity matching utilities."""

from .data import WDC_COLUMNS, load_wdc_json_gz, wdc_to_pair_examples
from .metrics import compute_binary_metrics
from .pseudolabels import build_pseudolabels
from .trainer import TrainConfig, train_loop, evaluate_loop

__all__ = [
    "WDC_COLUMNS",
    "load_wdc_json_gz",
    "wdc_to_pair_examples",
    "compute_binary_metrics",
    "build_pseudolabels",
    "TrainConfig",
    "train_loop",
    "evaluate_loop",
]
