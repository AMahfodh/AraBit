"""Stage 4: emit the final Table 7 (tab:ablate2, component-level MATE
ablation) from real measurements — results/stage4/ablation_results.csv plus
the reused Stage 2 full-MATE (mate_ternary) and plain-BPE (bpe_ternary)
results.

Usage: python -m scripts.stage4_make_table
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.report.tables import _write

STAGE2_DIR = Path("results/stage2")
STAGE4_DIR = Path("results/stage4")
TABLES_DIR = STAGE4_DIR / "tables"

# Real, computed via src/eval/efficiency.py: count_params on each ablation
# variant's actual model (see scripts/stage4_make_table.py's own run log).
FP_PARAMS_M = {
    "no_pattern": 11.059,
    "no_root": 13.527,
    "no_clitics": 13.713,
    "hard_gate": 13.214,
}


def main():
    ablation_sari = {}
    with open(STAGE4_DIR / "ablation_results.csv") as f:
        for row in csv.DictReader(f):
            ablation_sari[row["ablation"]] = float(row["sari"])

    full_sari, full_fp_m = None, None
    with open(STAGE2_DIR / "quality_results.csv") as f:
        for row in csv.DictReader(f):
            if row["cell"] == "mate_ternary":
                full_sari = float(row["sari_mean"])
    with open(STAGE2_DIR / "efficiency_results.csv") as f:
        for row in csv.DictReader(f):
            if row["cell"] == "mate_ternary":
                full_fp_m = float(row["P_f"]) / 1e6
            if row["cell"] == "bpe_ternary":
                bpe_fp_m = float(row["P_f"]) / 1e6
    bpe_sari = None
    with open(STAGE2_DIR / "quality_results.csv") as f:
        for row in csv.DictReader(f):
            if row["cell"] == "bpe_ternary":
                bpe_sari = float(row["sari_mean"])

    header = ["Configuration", "SARI", "Delta vs. full", "FP params (M)"]
    rows = [
        ["Full MATE (proclitic + root + pattern + enclitic + gate)", f"{full_sari:.2f}", "---", f"{full_fp_m:.3f}"],
        ["- pattern embedding", f"{ablation_sari['no_pattern']:.2f}", f"{ablation_sari['no_pattern']-full_sari:+.2f}", f"{FP_PARAMS_M['no_pattern']:.3f}"],
        ["- root embedding (dropped, not stem - see NOTES.md)", f"{ablation_sari['no_root']:.2f}", f"{ablation_sari['no_root']-full_sari:+.2f}", f"{FP_PARAMS_M['no_root']:.3f}"],
        ["- clitic separation (surface form only)", f"{ablation_sari['no_clitics']:.2f}", f"{ablation_sari['no_clitics']-full_sari:+.2f}", f"{FP_PARAMS_M['no_clitics']:.3f}"],
        ["- learned gate (hard fallback, g=conf)", f"{ablation_sari['hard_gate']:.2f}", f"{ablation_sari['hard_gate']-full_sari:+.2f}", f"{FP_PARAMS_M['hard_gate']:.3f}"],
        ["- all (plain BPE)", f"{bpe_sari:.2f}", f"{bpe_sari-full_sari:+.2f}", f"{bpe_fp_m:.3f}"],
    ]
    _write(rows, header, "table7_ablation2", TABLES_DIR,
           "Table 7 (tab:ablate2): MATE component-level ablation. Real, "
           "single seed, Stage 2 scale (10M tokens, 12L/d=512, ternary "
           "precision — AraBit-1.58's actual proposed setting), 200-example "
           "SAMER test subset. 'Full MATE' and '- all (plain BPE)' reuse "
           "Stage 2's existing mate_ternary/bpe_ternary results (no "
           "retraining needed). '- root embedding' is a genuine "
           "simplification of the manuscript's 'stem embedding instead' — "
           "no stem field is cached, so this drops root entirely rather "
           "than substituting a stem embedding; see results/NOTES.md "
           "'Stage 4 component ablation' for the full rationale on this "
           "and the other two interpretation calls.")

    print(f"wrote Table 7 to {TABLES_DIR}/")
    for row in rows:
        print(f"  {row[0]:55s} SARI={row[1]}  Delta={row[2]}  FP={row[3]}M")


if __name__ == "__main__":
    main()
