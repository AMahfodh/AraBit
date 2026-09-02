"""Remediation Phase 5, Item 1: standardize decoding across every system.

Before this fix, greedy decoding used a different max_new_tokens cap per
system: 40 for the bpe_* cells, 60 for the mate_* cells (both via
src/eval/batched_generate.py's old per-function defaults), and 64 for every
Stage 3 external baseline (AraBART/AraT5/AraBART+NF4, via
remediation2_full_split_eval.py's MAX_TGT_LEN). Since all systems use greedy
decoding (num_beams=1) and generation already stops per-example at <eos>
(finished_at tracking in batched_generate.py), raising a cap only affects
examples that would otherwise have been truncated mid-sentence - it cannot
make an already-finished generation longer. Standardizing on 64 (the
baselines' value - already the largest, so nothing regresses) and re-running
the 4 own-model cells only; baselines are already at 64 and unaffected by
this fix, so results/remediation2/baseline_results_full.csv is left as-is.

Reuses remediation2_full_split_eval.py's score_own_cells() unchanged (its
generate_bpe_batched/generate_mate_batched calls use the module's defaults,
now 64 for both) and overwrites results/remediation2/'s own-cell generation
TSVs + quality_results_full.csv's own-cell rows in place, preserving the
baseline rows already there.

Usage: python -m scripts.remediation5_standardize_decoding
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from scripts.stage2_finetune_and_eval import load_samer_pairs
from scripts.remediation2_full_split_eval import score_own_cells, DATA_DIR, OUT_DIR, D_MODEL
from src.data.cache import read_cache
from src.data.mate_batch import MorphIndex
from src.model.mate import MATEConfig
from src.tokenization.vocab import load_vocabs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    test_pairs = load_samer_pairs("test")
    eval_sources = [s for s, _ in test_pairs]
    eval_refs = [[t] for _, t in test_pairs]
    print(f"FULL test split: {len(eval_sources)} examples")

    control_tok = Tokenizer.from_file(str(DATA_DIR / "bpe_control_tokenizer.json"))
    fallback_tok = Tokenizer.from_file(str(DATA_DIR / "bpe_tokenizer.json"))
    vocabs = load_vocabs(DATA_DIR / "vocabs.json")
    cache_df = read_cache(DATA_DIR / "morph_cache.parquet")
    morph_index = MorphIndex(cache_df, vocabs)
    mate_cfg = MATEConfig(
        n_proclitic=len(vocabs["proclitic"]), n_enclitic=len(vocabs["enclitic"]),
        n_root=len(vocabs["root"]), n_pattern=len(vocabs["pattern"]),
        n_bpe=fallback_tok.get_vocab_size(),
        d_clitic=24, d_root=48, d_pattern=48, d_model=D_MODEL,
    )

    t0 = time.time()
    own_sari, own_bs = score_own_cells(eval_sources, eval_refs, control_tok, fallback_tok, morph_index, mate_cfg, device)
    print(f"\nown-cell wall time: {(time.time()-t0)/60:.1f} min")

    # Preserve the existing baseline rows (unaffected by this fix - already
    # at 64), overwrite only the own-cell rows.
    existing_baseline_rows = []
    old_csv = OUT_DIR / "quality_results_full.csv"
    if old_csv.exists():
        with open(old_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["cell"] not in own_sari:
                    existing_baseline_rows.append(row)

    with open(OUT_DIR / "quality_results_full.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["cell", "sari_mean", "sari_std", "sari_seeds", "bertscore_mean", "bertscore_std", "n_eval"]
        w.writerow(header)
        for cell in own_sari:
            sm, ss = statistics.mean(own_sari[cell]), (statistics.stdev(own_sari[cell]) if len(own_sari[cell]) > 1 else 0.0)
            bm, bs_ = statistics.mean(own_bs[cell]), (statistics.stdev(own_bs[cell]) if len(own_bs[cell]) > 1 else 0.0)
            w.writerow([cell, f"{sm:.4f}", f"{ss:.4f}", ";".join(f"{s:.4f}" for s in own_sari[cell]),
                        f"{bm:.4f}", f"{bs_:.4f}", len(eval_sources)])
        for row in existing_baseline_rows:
            w.writerow([row[h] for h in header])

    print("\n=== Remediation Phase 5 Item 1: standardized-decoding summary ===")
    for cell in own_sari:
        sm, ss = statistics.mean(own_sari[cell]), (statistics.stdev(own_sari[cell]) if len(own_sari[cell]) > 1 else 0.0)
        bm, bs_ = statistics.mean(own_bs[cell]), (statistics.stdev(own_bs[cell]) if len(own_bs[cell]) > 1 else 0.0)
        print(f"{cell:14s}: SARI={sm:.2f}+/-{ss:.2f}  BERTScore={bm:.4f}+/-{bs_:.4f}")


if __name__ == "__main__":
    main()
