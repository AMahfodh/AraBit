"""Builds root / pattern / proclitic / enclitic vocabularies from the morphology
cache (src/data/cache.py), plus a frequency-capped orthographic-word vocabulary
for MATE's word-level generation target (see results/NOTES.md "MATE
input/output granularity" for why MATE operates at the orthographic-token
level, per IMPLEMENTATION.md sec:3.2 / AraBit-1.58.tex's problem formulation).

Index convention: index 0 is always <unk_*> in every vocab, matching
src/model/mate.py's padding_idx=0 / <unk> convention. Consumed by
scripts/00_build_vocabs.py.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

UNK = "<unk>"
# Reserved word-vocab special tokens for MATE fine-tuning (scripts/
# stage1_finetune_and_eval.py): <sep> marks source/target boundary, <eos>
# marks end-of-generation (must be a real id, not aliased to <unk>, or the
# model can never learn to emit a distinct stop signal), <pad> for padding.
SPECIALS = ["<unk>", "<pad>", "<sep>", "<eos>"]


def _build_vocab_from_values(
    values: list[str], max_size: int | None = None, specials: list[str] = (UNK,)
) -> dict[str, int]:
    counts = Counter(values)
    n_reserved = len(specials)
    most_common = counts.most_common(max_size - n_reserved if max_size else None)
    vocab = {tok: i for i, tok in enumerate(specials)}
    for value, _ in most_common:
        if value not in vocab:  # real corpus text should never literally equal a special token, but don't collide if it does
            vocab[value] = len(vocab)
    return vocab


def build_morph_vocabs(cache_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    roots = [r for r in cache_df["root_id"] if r is not None]
    patterns = [p for p in cache_df["pattern_id"] if p is not None]
    proclitics = [c for row in cache_df["prc_ids"] for c in row]
    enclitics = [c for row in cache_df["enc_ids"] for c in row]

    return dict(
        root=_build_vocab_from_values(roots),
        pattern=_build_vocab_from_values(patterns),
        proclitic=_build_vocab_from_values(proclitics),
        enclitic=_build_vocab_from_values(enclitics),
    )


def build_word_vocab(sentences: list[str], max_size: int = 4000) -> dict[str, int]:
    """Frequency-capped orthographic-word vocabulary, for MATE's word-level
    input/output granularity. Corpus-frequency weighted (unlike the
    type-level morph vocabs above), since output-softmax size and coverage
    both depend on real word frequency, not just the set of distinct words.
    """
    words = [w for s in sentences for w in s.split()]
    return _build_vocab_from_values(words, max_size=max_size, specials=SPECIALS)


def save_vocabs(vocabs: dict[str, dict[str, int]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocabs, f, ensure_ascii=False, indent=1)


def load_vocabs(path: str | Path) -> dict[str, dict[str, int]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
