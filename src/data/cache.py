"""Morphological analysis cache.

Critical path — see docs/AraBit-1.58_IMPLEMENTATION.md sec:3.3: CAMeL's disambiguator
runs at ~hundreds of tokens/sec on CPU (measured 2,322.9 tok/s single-core in
this repo, see results/NOTES.md), so pre-training without this cache will not
finish at Stage 2 scale. Stored as Parquet with columns:
    surface_form, left_context_hash, prc_ids, enc_ids, root_id, pattern_id, conf, bpe_ids
Keyed on (surface_form, left_context_hash) — analysis is context-dependent; a
type-level (surface-form-only) cache silently degrades quality. Stage 0/1 use
a type-level shortcut (left_context_hash is always the empty string) for
speed on a small corpus — recorded as a threat to validity per
IMPLEMENTATION.md sec:3.3, see results/NOTES.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_COLUMNS = [
    "surface_form",
    "left_context_hash",
    "prc_ids",
    "enc_ids",
    "root_id",
    "pattern_id",
    "conf",
    "bpe_ids",
]


def write_cache(rows: list[dict], path: str | Path) -> None:
    df = pd.DataFrame(rows, columns=CACHE_COLUMNS)
    df.to_parquet(path, index=False)


def read_cache(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def cache_lookup(df: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """Builds an in-memory (surface_form, left_context_hash) -> row dict index
    for fast per-token lookup during training data preparation."""
    return {
        (r.surface_form, r.left_context_hash): r._asdict()
        for r in df.itertuples(index=False)
    }
