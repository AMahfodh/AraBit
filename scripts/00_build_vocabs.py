"""Builds root/pattern/proclitic/enclitic vocabs (from the morphology cache)
plus the frequency-capped word vocab (from the raw corpus) — src/tokenization/vocab.py.

Despite the 00/01 numbering in IMPLEMENTATION.md sec:2's layout, this script's
actual data dependency is on 01_cache_morphology.py's output, not the other
way around — the numbering appears to reflect pipeline-stage *concept* order
(vocab exists before model construction), not literal run order. Run
01_cache_morphology.py first. Noted in results/NOTES.md.

Usage: python -m scripts.stage1_build_vocabs --cache <path> --out <path> [--article-limit N]
"""

from __future__ import annotations

import argparse

from src.data.cache import read_cache
from src.data.corpus import stream_sentences
from src.tokenization.vocab import build_morph_vocabs, build_word_vocab, save_vocabs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--article-limit", type=int, default=300)
    ap.add_argument("--word-vocab-size", type=int, default=4000)
    args = ap.parse_args()

    cache_df = read_cache(args.cache)
    vocabs = build_morph_vocabs(cache_df)

    sentences = list(stream_sentences(article_limit=args.article_limit))
    vocabs["word"] = build_word_vocab(sentences, max_size=args.word_vocab_size)

    save_vocabs(vocabs, args.out)
    for k, v in vocabs.items():
        print(f"{k:10s}: {len(v)} entries")
    print(f"wrote vocabs to {args.out}")


if __name__ == "__main__":
    main()
