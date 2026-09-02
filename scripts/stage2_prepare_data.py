"""Stage 2: prepare the pre-training corpus, BPE tokenizers, morphology
cache, and vocabs at Stage 2 scale (~6000 articles, targeting >=10M words
per docs/AraBit-1.58_IMPLEMENTATION.md sec:4 Stage 2 / user-confirmed 10M
tokens/seed budget, see results/NOTES.md).

Usage: python -m scripts.stage2_prepare_data
"""

from __future__ import annotations

import multiprocessing as mp
import re
import time
from pathlib import Path

from src.data.cache import write_cache
from src.data.corpus import stream_sentences
from src.tokenization.bpe import matched_vocab_size, train_bpe
from src.tokenization.vocab import build_morph_vocabs, build_word_vocab, save_vocabs

ARTICLE_LIMIT = 6000
OUT_DIR = Path("data/stage2")
FALLBACK_VOCAB_SIZE = 8000
WORD_VOCAB_SIZE = 12000  # larger than Stage 1's 4000 given ~10x more corpus
D_MODEL = 512
D_ROOT, D_PATTERN, D_CLITIC = 48, 48, 24  # scaled up modestly from Stage 1's 32/32/16 for d_model=512

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
    no_analysis = dict(surface_form=word, left_context_hash="", prc_ids=[], enc_ids=[],
                        root_id=None, pattern_id=None, conf=0.0, bpe_ids=bpe_ids)
    try:
        analyses = _disambiguator.disambiguate([word])[0].analyses
    except Exception:
        return no_analysis
    if not analyses:
        return no_analysis
    a = analyses[0].analysis
    prc_ids = [a.get(f"prc{i}", "0") for i in range(4) if a.get(f"prc{i}", "0") != "0"]
    enc_ids = [a.get("enc0", "0")] if a.get("enc0", "0") != "0" else []
    return dict(surface_form=word, left_context_hash="", prc_ids=prc_ids, enc_ids=enc_ids,
                root_id=a.get("root"), pattern_id=a.get("pattern"), conf=1.0, bpe_ids=bpe_ids)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"streaming {ARTICLE_LIMIT} articles...")
    t0 = time.time()
    sentences = list(stream_sentences(article_limit=ARTICLE_LIMIT))
    n_words = sum(len(s.split()) for s in sentences)
    print(f"  {len(sentences):,} sentences, {n_words:,} words in {time.time()-t0:.1f}s")

    print("training BPE tokenizers...")
    fallback_tok = train_bpe(sentences, FALLBACK_VOCAB_SIZE)
    fallback_tok.save(str(OUT_DIR / "bpe_tokenizer.json"))
    print(f"  fallback tokenizer: {fallback_tok.get_vocab_size()} vocab")

    print("caching morphology (multiprocessed)...")
    words = sorted({w for s in sentences for w in _WORD_SPLIT.split(s) if w})
    print(f"  {len(words):,} unique surface forms")
    t0 = time.time()
    with mp.Pool(mp.cpu_count(), initializer=_init_worker,
                 initargs=(str(OUT_DIR / "bpe_tokenizer.json"),)) as pool:
        rows = pool.map(_analyze_word, words, chunksize=64)
    elapsed = time.time() - t0
    n_unanalyzed = sum(1 for r in rows if r["conf"] == 0.0)
    print(f"  analyzed {len(rows):,} types in {elapsed:.1f}s ({len(rows)/elapsed:.1f} types/sec), "
          f"{n_unanalyzed} unanalyzed ({100*n_unanalyzed/len(rows):.2f}%)")
    write_cache(rows, OUT_DIR / "morph_cache.parquet")

    print("building vocabs...")
    import pandas as pd
    cache_df = pd.DataFrame(rows)
    vocabs = build_morph_vocabs(cache_df)
    vocabs["word"] = build_word_vocab(sentences, max_size=WORD_VOCAB_SIZE)
    save_vocabs(vocabs, OUT_DIR / "vocabs.json")
    for k, v in vocabs.items():
        print(f"  {k:10s}: {len(v)} entries")

    n_bpe_control = matched_vocab_size(
        n_root=len(vocabs["root"]), n_pattern=len(vocabs["pattern"]),
        n_proclitic=len(vocabs["proclitic"]), n_enclitic=len(vocabs["enclitic"]),
        d_root=D_ROOT, d_pattern=D_PATTERN, d_clitic=D_CLITIC,
        n_bpe_fallback=FALLBACK_VOCAB_SIZE, d_model=D_MODEL,
    )
    print(f"matched BPE control vocab size: {n_bpe_control}")
    control_tok = train_bpe(sentences, n_bpe_control)
    control_tok.save(str(OUT_DIR / "bpe_control_tokenizer.json"))
    print(f"  control tokenizer: {control_tok.get_vocab_size()} vocab")

    print(f"\nStage 2 data prep complete. Files in {OUT_DIR}/")


if __name__ == "__main__":
    main()
