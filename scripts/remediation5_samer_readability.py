"""Remediation Phase 5, Item 2: SAMER levelled-lexicon readability metric.

Scores source, reference, and every system's generated output on the full
3,277-example SAMER test split (results/remediation2/generations_*_full.tsv)
with src/eval/readability.py's SAMERReadabilityScorer. Covers the 4 own
cells (3 seeds each, mean+/-std to match how Table 4 already reports SARI)
plus the 2 baselines still in Table 4's main body after Item 4 (AraBART,
AraBART_nf4) - AraT5 excluded, consistent with Item 4's decision to drop it
from the main table entirely.

Must run AFTER Item 1's remediation5_standardize_decoding.py, since that
script overwrites the own-cell generations_*_full.tsv files this one reads -
running before would score the pre-standardization (40/60-cap) generations.

Usage: python -m scripts.remediation5_samer_readability
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from src.eval.readability import SAMERReadabilityScorer

IN_DIR = Path("results/remediation2")
OUT_DIR = Path("results/remediation5")
OWN_CELLS = ["bpe_fp16", "bpe_ternary", "mate_fp16", "mate_ternary"]
SEEDS = [0, 1, 2]
BASELINES = ["AraBART", "AraBART_nf4"]


def read_tsv(path: Path) -> tuple[list[str], list[str], list[str]]:
    sources, generated, references = [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sources.append(row["source"])
            generated.append(row["generated"])
            references.append(row["reference"])
    return sources, generated, references


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scorer = SAMERReadabilityScorer()

    # Source/reference are identical across every file (same eval split, same
    # order) - read once from the first own-cell seed-0 file.
    sources, _, references = read_tsv(IN_DIR / f"generations_{OWN_CELLS[0]}_seed0_full.tsv")
    print(f"{len(sources)} examples")

    results: dict[str, dict] = {}

    print("\n[source] scoring...")
    results["source"] = scorer.score_corpus(sources)
    print(f"  {results['source']}")

    print("[reference] scoring...")
    results["reference"] = scorer.score_corpus(references)
    print(f"  {results['reference']}")

    for cell in OWN_CELLS:
        seed_results = []
        for seed in SEEDS:
            _, gen, _ = read_tsv(IN_DIR / f"generations_{cell}_seed{seed}_full.tsv")
            r = scorer.score_corpus(gen)
            seed_results.append(r)
            print(f"[{cell} seed{seed}] mean_level={r['mean_level']:.4f} "
                  f"prop_level1={r['prop_level1']:.4f} coverage={r['coverage']:.4f}")
        mean_levels = [r["mean_level"] for r in seed_results]
        prop_level1s = [r["prop_level1"] for r in seed_results]
        results[cell] = dict(
            mean_level_mean=statistics.mean(mean_levels),
            mean_level_std=statistics.stdev(mean_levels) if len(mean_levels) > 1 else 0.0,
            prop_level1_mean=statistics.mean(prop_level1s),
            prop_level1_std=statistics.stdev(prop_level1s) if len(prop_level1s) > 1 else 0.0,
            per_seed=seed_results,
        )

    for name in BASELINES:
        _, gen, _ = read_tsv(IN_DIR / f"generations_{name}_full.tsv")
        r = scorer.score_corpus(gen)
        results[name] = r
        print(f"[{name}] mean_level={r['mean_level']:.4f} prop_level1={r['prop_level1']:.4f} "
              f"coverage={r['coverage']:.4f}")

    with open(OUT_DIR / "samer_readability_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(OUT_DIR / "samer_readability_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "mean_level", "mean_level_std", "prop_level1", "prop_level1_std", "coverage"])
        for name in ["source", "reference"] + OWN_CELLS + BASELINES:
            r = results[name]
            if name in OWN_CELLS:
                w.writerow([name, f"{r['mean_level_mean']:.4f}", f"{r['mean_level_std']:.4f}",
                            f"{r['prop_level1_mean']:.4f}", f"{r['prop_level1_std']:.4f}", ""])
            else:
                w.writerow([name, f"{r['mean_level']:.4f}", "", f"{r['prop_level1']:.4f}", "", f"{r['coverage']:.4f}"])

    print(f"\nwrote {OUT_DIR}/samer_readability_results.{{json,csv}}")


if __name__ == "__main__":
    main()
