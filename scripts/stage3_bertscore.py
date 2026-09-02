"""Adds BERTScore-F1 to Stage 3's baseline eval (AraBART, AraT5, AraBART+NF4),
using the generations already saved.

Usage: python -m scripts.stage3_bertscore
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.eval.quality import bertscore_f1

OUT_DIR = Path("results/stage3")
FILES = {
    "AraBART": "generations_AraBART.tsv",
    "AraT5": "generations_AraT5.tsv",
    "AraBART_nf4": "generations_AraBART_nf4.tsv",
}


def main():
    results = {}
    for name, fname in FILES.items():
        gens, refs = [], []
        with open(OUT_DIR / fname, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if not row["generated"].strip():
                    continue
                gens.append(row["generated"])
                refs.append(row["reference"])
        print(f"[{name}] scoring {len(gens)} generations...")
        f1 = bertscore_f1(gens, refs)
        print(f"  BERTScore-F1 = {f1:.4f}")
        results[name] = f1

    rows = []
    with open(OUT_DIR / "baseline_results.csv") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["bertscore_f1"]
        for row in reader:
            row["bertscore_f1"] = f"{results.get(row['model'], float('nan')):.4f}" if row["model"] in results else "NA"
            rows.append(row)
    with open(OUT_DIR / "baseline_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\n=== Stage 3 BERTScore-F1 summary ===")
    for name, f1 in results.items():
        print(f"{name:14s}: {f1:.4f}")


if __name__ == "__main__":
    main()
