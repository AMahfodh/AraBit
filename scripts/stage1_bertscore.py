"""Adds BERTScore-F1 to Stage 1's SAMER quality eval, using the generations
already saved by scripts/stage1_finetune_and_eval.py
(results/stage1/generations_*.tsv) — no need to re-run fine-tuning/generation.

Usage: python -m scripts.stage1_bertscore
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.eval.quality import bertscore_f1

OUT_DIR = Path("results/stage1")
CELLS = ["bpe_fp16", "bpe_ternary", "mate_fp16", "mate_ternary"]


def main():
    results = {}
    for cell in CELLS:
        gens, refs = [], []
        n_dropped = 0
        with open(OUT_DIR / f"generations_{cell}.tsv", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if not row["generated"].strip():
                    # bert_score's sent_encode crashes on a genuinely empty
                    # string (tries tokenizer.build_inputs_with_special_tokens,
                    # incompatible with this repo's transformers version) -
                    # drop rather than pad with a fake placeholder.
                    n_dropped += 1
                    continue
                gens.append(row["generated"])
                refs.append(row["reference"])
        print(f"[{cell}] scoring {len(gens)} generations ({n_dropped} empty, dropped)...")
        f1 = bertscore_f1(gens, refs)
        print(f"  BERTScore-F1 = {f1:.4f}")
        results[cell] = f1

    # merge into quality_results.csv
    rows = []
    with open(OUT_DIR / "quality_results.csv") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["bertscore_f1"]
        for row in reader:
            row["bertscore_f1"] = f"{results[row['cell']]:.4f}"
            rows.append(row)
    with open(OUT_DIR / "quality_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\n=== BERTScore-F1 summary ===")
    for cell, f1 in results.items():
        print(f"{cell:14s}: {f1:.4f}")
    print(f"\nmerged into {OUT_DIR}/quality_results.csv")


if __name__ == "__main__":
    main()
