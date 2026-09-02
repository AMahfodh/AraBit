"""Remediation Phase 1: BERTScore for the 6 corrected mate checkpoints,
same generations already saved by scripts/remediation1_finetune_eval_mate.py.

Usage: python -m scripts.remediation1_bertscore
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from src.eval.quality import bertscore_f1

OUT_DIR = Path("results/stage2_corrected")
CELLS = ["mate_fp16", "mate_ternary"]
SEEDS = [0, 1, 2]


def main():
    per_cell = {}
    for cell in CELLS:
        per_cell[cell] = []
        for seed in SEEDS:
            gens, refs = [], []
            with open(OUT_DIR / f"generations_{cell}_seed{seed}.tsv", encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    if not row["generated"].strip():
                        continue
                    gens.append(row["generated"])
                    refs.append(row["reference"])
            print(f"[{cell}_seed{seed}] scoring {len(gens)} generations...")
            f1 = bertscore_f1(gens, refs)
            print(f"  BERTScore-F1 = {f1:.4f}")
            per_cell[cell].append(f1)

    with open(OUT_DIR / "quality_results.csv") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["bertscore_mean", "bertscore_std"]
        rows = list(reader)
    for row in rows:
        scores = per_cell[row["cell"]]
        row["bertscore_mean"] = f"{statistics.mean(scores):.4f}"
        row["bertscore_std"] = f"{statistics.stdev(scores):.4f}" if len(scores) > 1 else "0.0000"
    with open(OUT_DIR / "quality_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\n=== Corrected BERTScore-F1 summary (mean +/- std, 3 seeds) ===")
    for cell, scores in per_cell.items():
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(f"{cell:14s}: {mean:.4f} +/- {std:.4f}  (seeds: {[f'{s:.4f}' for s in scores]})")


if __name__ == "__main__":
    main()
