from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist


@dataclass
class DistContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    mps_backend = getattr(torch.backends, "mps", None)
    if (
        mps_backend is not None
        and getattr(mps_backend, "is_available", lambda: False)()
    ):
        torch.mps.manual_seed(seed)


def _auto_select_device() -> torch.device:
    """Pick the best available torch device for Ditto training.

    Preference order: CUDA → MPS (Apple Silicon) → CPU. MPS is gated on
    both ``torch.backends.mps.is_available()`` and ``is_built()`` so
    environments without a built MPS backend fall back to CPU silently.
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


def init_distributed(explicit_ddp: bool = False) -> DistContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = explicit_ddp or world_size > 1

    if ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return DistContext(True, rank, world_size, local_rank, device)

    device = _auto_select_device()
    return DistContext(False, 0, 1, 0, device)


def cleanup_distributed(ctx: DistContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_rank0(ctx: DistContext) -> bool:
    return ctx.rank == 0


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(base_dir: str | Path, prefix: str = "run") -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / f"{prefix}_{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (base / "LATEST_RUN").write_text(str(run_dir))
    return run_dir


def resolve_run_dir(path: str | Path) -> Path:
    p = Path(path)
    if (
        (p / "config.json").exists()
        or (p / "checkpoints").exists()
        or (p / "predictions.csv").exists()
    ):
        return p
    latest = p / "LATEST_RUN"
    if not latest.exists():
        raise FileNotFoundError(f"No run directory found in {p}; missing LATEST_RUN")
    run_dir = Path(latest.read_text().strip())
    if not run_dir.exists():
        raise FileNotFoundError(f"LATEST_RUN points to missing directory: {run_dir}")
    return run_dir


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
