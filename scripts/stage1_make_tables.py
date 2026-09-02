"""Stage 1: emit Tables 4-7 + HYPOTHESIS.md from the real measurements already
on disk (results/stage1/quality_results.csv, efficiency_results.csv), per
docs/AraBit-1.58_IMPLEMENTATION.md sec:5's reporting contract. Human
evaluation (Table 6) dropped from the manuscript entirely, see
results/NOTES.md — ablate1/ablate2 shifted down to Table 6/7.

Not scripts/05_make_tables.py (the numbered pipeline stage) — this is Stage
1's own driver over Stage-1-scale data; see results/NOTES.md for why the
stage1_*.py scripts are kept separate from the general numbered pipeline.

Usage: python -m scripts.stage1_make_tables
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.report.tables import (
    table4_quality,
    table5_efficiency,
    table6_ablation1,
    table7_ablation2,
    write_hypothesis_md,
)

STAGE1_DIR = Path("results/stage1")
TABLES_DIR = STAGE1_DIR / "tables"


def main():
    cell_sari = {}
    cell_bertscore = {}
    with open(STAGE1_DIR / "quality_results.csv") as f:
        for row in csv.DictReader(f):
            cell_sari[row["cell"]] = float(row["sari"])
            if row.get("bertscore_f1"):
                cell_bertscore[row["cell"]] = float(row["bertscore_f1"])

    cell_eff = {}
    with open(STAGE1_DIR / "efficiency_results.csv") as f:
        for row in csv.DictReader(f):
            cell_eff[row["cell"]] = dict(
                p_t=int(row["P_t"]), p_f=int(row["P_f"]),
                weight_mem_mb=float(row["weight_mem_MB"]),
                model_ms_per_tok=float(row["latency_model_ms_per_tok"]),
                e2e_ms_per_tok=float(row["latency_e2e_ms_per_tok"]),
                joules_per_seq=float(row["energy_J_per_seq"]) if row["energy_J_per_seq"] != "NA" else float("nan"),
            )

    table4_quality(cell_sari, TABLES_DIR, cell_bertscore)
    table5_efficiency(cell_eff, TABLES_DIR)
    delta_ternary, delta_fp16 = table6_ablation1(cell_sari, TABLES_DIR)
    table7_ablation2(TABLES_DIR)
    write_hypothesis_md(delta_ternary, delta_fp16, STAGE1_DIR)

    print(f"wrote Tables 4-7 to {TABLES_DIR}/ and HYPOTHESIS.md to {STAGE1_DIR}/")
    print(f"\nDelta_ternary={delta_ternary:+.2f}  Delta_fp16={delta_fp16:+.2f}  "
          f"(see {STAGE1_DIR}/HYPOTHESIS.md for the required caution note)")


if __name__ == "__main__":
    main()
