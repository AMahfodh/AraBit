"""Runs CAMeL morphological analysis over the pre-training corpus once,
multiprocessed across all cores, and writes the Parquet cache
(src/data/cache.py).

See docs/AraBit-1.58_IMPLEMENTATION.md sec:3.3: measure throughput on 10k
sentences first, extrapolate, and report the number back before launching the
full run (done — see results/NOTES.md: 2,322.9 tok/s single-core, measured).

Stage 1 (pipeline validation) shortcut, logged per sec:3.3's own instruction
("if you take the type-level shortcut for speed, record that decision as a
threat to validity"): this cache is TYPE-level (surface_form only,
left_context_hash always ""), not context-level — CAMeL's MLE disambiguator
already picks its single best analysis per surface form without needing
neighbouring tokens re-passed per occurrence, and re-analysing every token
occurrence in context for a small pipeline-validation corpus is not worth the
cost yet. Revisit for Stage 2 if context-dependent disambiguation accuracy
matters at that point.

Usage: python -m scripts.stage1_cache_morphology --out <path> [--article-limit N]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import time
from pathlib import Path

from src.data.corpus import stream_sentences

_WORD_SPLIT = re.compile(r"\s+")

_disambiguator = None
_bpe_tokenizer = None


def _init_worker(bpe_path: str):
    global _disambiguator, _bpe_tokenizer
    from camel_tools.disambig.mle import MLEDisambiguator
    from tokenizers import Tokenizer

    _disambiguator = MLEDisambiguator.pretrained()
    _bpe_tokenizer = Tokenizer.from_file(bpe_path)


def _analyze_word(word: str) -> dict:
    bpe_ids = _bpe_tokenizer.encode(word).ids
    no_analysis = dict(
        surface_form=word,
        left_context_hash="",
        prc_ids=[],
        enc_ids=[],
        root_id=None,
        pattern_id=None,
        conf=0.0,
        bpe_ids=bpe_ids,
    )
    try:
        analyses = _disambiguator.disambiguate([word])[0].analyses
    except Exception:
        # camel_tools' analyzer can raise (observed: re.error "bad escape \s"
        # in _combined_backoff_analyses) on malformed input words - e.g.
        # Wikipedia markup remnants that survived sentence splitting. Treated
        # the same as "no analysis" (IMPLEMENTATION.md sec:3.2's own
        # unanalysable-token case), not a pipeline-crashing error. See
        # results/NOTES.md.
        return no_analysis
    if not analyses:
        return no_analysis
    a = analyses[0].analysis
    prc_ids = [a.get(f"prc{i}", "0") for i in range(4) if a.get(f"prc{i}", "0") != "0"]
    enc_ids = [a.get("enc0", "0")] if a.get("enc0", "0") != "0" else []
    return dict(
        surface_form=word,
        left_context_hash="",
        prc_ids=prc_ids,
        enc_ids=enc_ids,
        root_id=a.get("root"),
        pattern_id=a.get("pattern"),
        conf=1.0,
        bpe_ids=bpe_ids,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bpe-tokenizer", required=True)
    ap.add_argument("--article-limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    args = ap.parse_args()

    print(f"streaming up to {args.article_limit} articles...")
    sentences = list(stream_sentences(article_limit=args.article_limit))
    words = sorted({w for s in sentences for w in _WORD_SPLIT.split(s) if w})
    print(f"{len(sentences)} sentences, {len(words)} unique surface forms to analyze")

    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(args.bpe_tokenizer,)) as pool:
        rows = pool.map(_analyze_word, words, chunksize=64)
    elapsed = time.time() - t0
    n_unanalyzed = sum(1 for r in rows if r["conf"] == 0.0)
    print(f"analyzed {len(rows)} unique surface forms in {elapsed:.1f}s "
          f"({len(rows)/elapsed:.1f} types/sec, {args.workers} workers)")
    print(f"unanalyzed (no analysis or analyzer error): {n_unanalyzed} "
          f"({100*n_unanalyzed/len(rows):.1f}%)")

    from src.data.cache import write_cache

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_cache(rows, args.out)
    print(f"wrote cache to {args.out}")


if __name__ == "__main__":
    main()
