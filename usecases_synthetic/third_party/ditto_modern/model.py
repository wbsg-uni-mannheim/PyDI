from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

LM_ALIASES = {
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
}


def resolve_model_name(model_name_or_path: str) -> str:
    return LM_ALIASES.get(model_name_or_path, model_name_or_path)


class DittoModel(nn.Module):
    """Ditto-style encoder + linear classifier with optional MixDA."""

    def __init__(self, model_name: str = "roberta-base", alpha_aug: float = 0.8):
        super().__init__()
        self.model_name = resolve_model_name(model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name)
        self.alpha_aug = float(alpha_aug)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 2)

    def _encode_cls(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        input_ids_aug: torch.Tensor | None = None,
        attention_mask_aug: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids_aug is not None:
            combined_ids = torch.cat([input_ids, input_ids_aug], dim=0)
            if attention_mask is not None and attention_mask_aug is not None:
                combined_mask = torch.cat([attention_mask, attention_mask_aug], dim=0)
            else:
                combined_mask = None
            enc = self._encode_cls(combined_ids, combined_mask)
            batch_size = input_ids.size(0)
            enc1 = enc[:batch_size]
            enc2 = enc[batch_size:]
            lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * lam + enc2 * (1.0 - lam)
        else:
            enc = self._encode_cls(input_ids, attention_mask)

        return self.classifier(enc)


def load_tokenizer(model_name_or_path: str):
    resolved = resolve_model_name(model_name_or_path)
    return AutoTokenizer.from_pretrained(resolved, use_fast=True)


def load_model(model_name_or_path: str, alpha_aug: float = 0.8):
    p = Path(model_name_or_path)
    model_path = p / "model.pt"
    config_path = p / "model_config.json"

    if model_path.exists() and config_path.exists():
        with open(config_path, "r") as f:
            cfg = json.load(f)
        model = DittoModel(model_name=cfg["model_name"], alpha_aug=float(cfg.get("alpha_aug", alpha_aug)))
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        return model

    return DittoModel(model_name=model_name_or_path, alpha_aug=alpha_aug)


def save_model(model: DittoModel, tokenizer, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out / "model.pt")
    with open(out / "model_config.json", "w") as f:
        json.dump({"model_name": model.model_name, "alpha_aug": model.alpha_aug}, f, indent=2)

    tokenizer.save_pretrained(out)
