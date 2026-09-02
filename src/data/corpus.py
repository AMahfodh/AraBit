"""Pre-training corpus loading and filtering.

Pre-training corpus is Arabic Wikipedia, pinned dump (results/NOTES.md
"Decisions", 2026-08-31): `wikimedia/wikipedia` config `20231101.ar`, loaded
via HF `datasets` streaming (so this never bulk-downloads the full dump - only
what's iterated). See docs/AraBit-1.58_IMPLEMENTATION.md sec:0 for the
"matched conditions" requirement: this corpus must be identical across all
four front_end x precision cells.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

_SENT_SPLIT = re.compile(r"(?<=[.!؟])\s+")

PRETRAIN_CORPUS = dict(name="wikimedia/wikipedia", config="20231101.ar")


def stream_sentences(min_len: int = 15, article_limit: int | None = None) -> Iterator[str]:
    """Yields Arabic Wikipedia sentences, streaming (no bulk download).

    `article_limit` bounds how many articles are pulled from the stream, as a
    safety cap for callers that only need a fixed sentence/token budget
    (Stage 0 measurements did this explicitly to keep the download bounded -
    see results/NOTES.md).
    """
    from datasets import load_dataset

    ds = load_dataset(
        PRETRAIN_CORPUS["name"], PRETRAIN_CORPUS["config"], split="train", streaming=True
    )
    for i, row in enumerate(ds):
        if article_limit is not None and i >= article_limit:
            return
        for sentence in _SENT_SPLIT.split(row["text"]):
            sentence = sentence.strip()
            if len(sentence) >= min_len:
                yield sentence
