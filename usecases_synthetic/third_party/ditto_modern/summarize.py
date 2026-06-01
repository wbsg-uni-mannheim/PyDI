from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Iterable, List, Sequence, Set

from sklearn.feature_extraction.text import TfidfVectorizer

from .data import PairExample
from .model import load_tokenizer

try:
    from nltk.corpus import stopwords

    _STOPWORDS: Set[str] = set(stopwords.words("english"))
except Exception:
    _STOPWORDS = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
    }


class Summarizer:
    """Ditto summarizer adapted for in-memory pair examples."""

    def __init__(self, lm: str):
        self.tokenizer = load_tokenizer(lm)
        self.len_cache = {}
        self.vocab = {}
        self.idf = []

    def build_index_from_examples(self, examples: Iterable[PairExample]) -> None:
        content = []
        for ex in examples:
            content.append(ex.left)
            content.append(ex.right)
        if not content:
            self.vocab = {}
            self.idf = []
            return
        vectorizer = TfidfVectorizer().fit(content)
        self.vocab = vectorizer.vocabulary_
        self.idf = vectorizer.idf_

    def get_len(self, word: str) -> int:
        if word in self.len_cache:
            return self.len_cache[word]
        length = len(self.tokenizer.tokenize(word))
        self.len_cache[word] = length
        return length

    def _summarize_entry(self, entry: str, token_scores: Counter, max_len: int) -> str:
        token_cnt = Counter(entry.split(" "))
        total_len = token_cnt["COL"] + token_cnt["VAL"]

        subset = Counter()
        for token in set(token_cnt.keys()):
            subset[token] = token_scores[token]
        subset = subset.most_common(max_len)

        topk_tokens_copy = set()
        for word, _ in subset:
            bert_len = self.get_len(word)
            if total_len + bert_len > max_len:
                break
            total_len += bert_len
            topk_tokens_copy.add(word)

        out_tokens = []
        for token in entry.split(" "):
            if token in ["COL", "VAL"]:
                out_tokens.append(token)
            elif token in topk_tokens_copy:
                out_tokens.append(token)
                topk_tokens_copy.remove(token)
        return " ".join(out_tokens)

    def summarize_pair(self, left: str, right: str, max_len: int = 256) -> tuple[str, str]:
        cnt = Counter()
        for sent in [left, right]:
            for token in sent.split(" "):
                if token in ["COL", "VAL"] or token.lower() in _STOPWORDS:
                    continue
                if token in self.vocab:
                    cnt[token] += self.idf[self.vocab[token]]

        left_su = self._summarize_entry(left, cnt, max_len=max_len)
        right_su = self._summarize_entry(right, cnt, max_len=max_len)
        return left_su, right_su


def summarize_examples(
    train_examples: Sequence[PairExample],
    val_examples: Sequence[PairExample],
    test_examples: Sequence[PairExample] | None,
    lm: str,
    max_len: int,
) -> tuple[List[PairExample], List[PairExample], List[PairExample] | None]:
    summarizer = Summarizer(lm=lm)
    corpus = list(train_examples) + list(val_examples) + (list(test_examples) if test_examples is not None else [])
    summarizer.build_index_from_examples(corpus)

    def _apply(examples: Sequence[PairExample]) -> List[PairExample]:
        out: List[PairExample] = []
        for ex in examples:
            l, r = summarizer.summarize_pair(ex.left, ex.right, max_len=max_len)
            out.append(replace(ex, left=l, right=r))
        return out

    out_train = _apply(train_examples)
    out_val = _apply(val_examples)
    out_test = _apply(test_examples) if test_examples is not None else None
    return out_train, out_val, out_test
