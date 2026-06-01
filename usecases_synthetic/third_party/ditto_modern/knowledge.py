from __future__ import annotations

from dataclasses import replace
from typing import List, Sequence

from .data import PairExample


def _load_spacy(model_name: str):
    import spacy

    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is required but not installed. "
            f"Install with: python -m spacy download {model_name}"
        ) from exc


class DKInjector:
    def __init__(self, name: str, spacy_model: str = "en_core_web_sm"):
        self.name = name
        self.nlp = _load_spacy(spacy_model)

    def transform(self, entry: str) -> str:
        return entry


class ProductDKInjector(DKInjector):
    def transform(self, entry: str) -> str:
        res = ""
        doc = self.nlp(entry)
        start_indices = {}

        for ent in doc.ents:
            start, _, label = ent.start, ent.end, ent.label_
            if label in ["NORP", "GPE", "LOC", "PERSON", "PRODUCT"]:
                start_indices[start] = "PRODUCT"
            if label in ["DATE", "QUANTITY", "TIME", "PERCENT", "MONEY"]:
                start_indices[start] = "NUM"

        for idx, token in enumerate(doc):
            if idx in start_indices:
                res += start_indices[idx] + " "

            if token.like_num:
                try:
                    val = float(token.text.replace(",", ""))
                    if val == round(val):
                        res += f"{int(val)} "
                    else:
                        res += f"{val:.2f} "
                except Exception:
                    res += token.text + " "
            elif len(token.text) >= 7 and any(ch.isdigit() for ch in token.text):
                res += "ID " + token.text + " "
            else:
                res += token.text + " "
        return res.strip()


class GeneralDKInjector(DKInjector):
    def transform(self, entry: str) -> str:
        res = ""
        doc = self.nlp(entry)
        start_indices = {}

        for ent in doc.ents:
            start, _, label = ent.start, ent.end, ent.label_
            if label in ["PERSON", "ORG", "LOC", "PRODUCT", "DATE", "QUANTITY", "TIME"]:
                start_indices[start] = label

        for idx, token in enumerate(doc):
            if idx in start_indices:
                res += start_indices[idx] + " "

            if token.like_num:
                try:
                    val = float(token.text.replace(",", ""))
                    if val == round(val):
                        res += f"{int(val)} "
                    else:
                        res += f"{val:.2f} "
                except Exception:
                    res += token.text + " "
            elif len(token.text) >= 7 and any(ch.isdigit() for ch in token.text):
                res += "ID " + token.text + " "
            else:
                res += token.text + " "
        return res.strip()


def inject_knowledge(
    examples: Sequence[PairExample],
    dk: str,
    spacy_model: str = "en_core_web_sm",
) -> List[PairExample]:
    if dk == "product":
        injector = ProductDKInjector(dk, spacy_model=spacy_model)
    else:
        injector = GeneralDKInjector(dk, spacy_model=spacy_model)

    out: List[PairExample] = []
    for ex in examples:
        out.append(replace(ex, left=injector.transform(ex.left), right=injector.transform(ex.right)))
    return out
