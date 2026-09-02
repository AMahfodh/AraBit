"""Stage 2: emit Tables 4-7 + HYPOTHESIS.md from the real Stage 2 measurements
(results/stage2/quality_results.csv, efficiency_results.csv,
bootstrap_significance.csv), with mean +/- std over 3 seeds and the real
paired bootstrap significance test — the actual reporting contract per
docs/AraBit-1.58_IMPLEMENTATION.md sec:5, now achievable with the full
3-seed protocol (unlike Stage 1's single-seed pipeline check). Human
evaluation (Table 6) dropped from the manuscript entirely, see
results/NOTES.md — ablate1/ablate2 shifted down to Table 6/7.

Usage: python -m scripts.stage2_make_tables
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.report.tables import TBD, _write, table5_efficiency, table7_ablation2

STAGE2_DIR = Path("results/stage2")
TABLES_DIR = STAGE2_DIR / "tables"


def main():
    cell_sari, cell_bs = {}, {}
    with open(STAGE2_DIR / "quality_results.csv") as f:
        for row in csv.DictReader(f):
            cell_sari[row["cell"]] = (float(row["sari_mean"]), float(row["sari_std"]))
            if row.get("bertscore_mean"):
                cell_bs[row["cell"]] = (float(row["bertscore_mean"]), float(row["bertscore_std"]))

    cell_eff = {}
    with open(STAGE2_DIR / "efficiency_results.csv") as f:
        for row in csv.DictReader(f):
            cell_eff[row["cell"]] = dict(
                p_t=int(row["P_t"]), p_f=int(row["P_f"]),
                weight_mem_mb=float(row["weight_mem_MB"]),
                model_ms_per_tok=float(row["latency_model_ms_per_tok"]),
                e2e_ms_per_tok=float(row["latency_e2e_ms_per_tok"]),
                joules_per_seq=float(row["energy_J_per_seq"]) if row["energy_J_per_seq"] != "NA" else float("nan"),
            )

    with open(STAGE2_DIR / "bootstrap_significance.csv") as f:
        boot_row = next(csv.DictReader(f))

    def sari_str(cell):
        m, s = cell_sari[cell]
        return f"{m:.2f} +/- {s:.2f}"

    def bs_str(cell):
        if cell not in cell_bs:
            return TBD
        m, s = cell_bs[cell]
        return f"{m:.4f} +/- {s:.4f}"

    # Table 4
    header = ["Model", "Precision", "SARI", "BERTScore-F1", "OSMAN", "BLEU (reph.)"]
    rows = [
        ["AraBART", "FP16", TBD, TBD, TBD, TBD],
        ["AraT5", "FP16", TBD, TBD, TBD, TBD],
        ["FFT-Seq2Seq", "FP16", TBD, TBD, TBD, TBD],
        ["Switch-Arabic", "FP16", TBD, TBD, TBD, TBD],
        ["AraBART + GPTQ", "4-bit", TBD, TBD, TBD, TBD],
        ["B4: ours, BPE", "FP16", sari_str("bpe_fp16"), bs_str("bpe_fp16"), TBD, TBD],
        ["B2: ours, MATE", "FP16", sari_str("mate_fp16"), bs_str("mate_fp16"), TBD, TBD],
        ["B3: ours, BPE", "1.58-bit", sari_str("bpe_ternary"), bs_str("bpe_ternary"), TBD, TBD],
        ["AraBit-1.58 (MATE)", "1.58-bit", sari_str("mate_ternary"), bs_str("mate_ternary"), TBD, TBD],
    ]
    _write(rows, header, "table4_quality", TABLES_DIR,
           "Table 4 (tab:quality): Generation quality, mean +/- std over 3 "
           "seeds (Stage 2, 10M tokens/seed, small model, 200-example SAMER "
           "test subset). External baselines unmeasured, TBD.")

    table5_efficiency(cell_eff, TABLES_DIR, scale_note="Stage 2, small model (12L, d=512), 6000-article corpus")

    # Table 6 (2x2) with real std
    delta_fp16 = cell_sari["mate_fp16"][0] - cell_sari["bpe_fp16"][0]
    delta_ternary = cell_sari["mate_ternary"][0] - cell_sari["bpe_ternary"][0]
    header6 = ["", "BPE front end", "MATE front end", "Delta (MATE gain)"]
    rows6 = [
        ["FP16 weights", f"{sari_str('bpe_fp16')} (B4)", f"{sari_str('mate_fp16')} (B2)", f"{delta_fp16:+.2f}"],
        ["Ternary weights", f"{sari_str('bpe_ternary')} (B3)", f"{sari_str('mate_ternary')} (AraBit-1.58)", f"{delta_ternary:+.2f}"],
    ]
    _write(rows6, header6, "table6_ablation1", TABLES_DIR,
           "Table 6 (tab:ablate1): the central 2x2 ablation, mean +/- std "
           "over 3 seeds, real Stage 2 result (not a pipeline-validation "
           "placeholder). Paired bootstrap significance test (seed 0, "
           f"1000 resamples): SARI(mate_ternary)-SARI(bpe_ternary) = "
           f"{boot_row['diff']}, p(diff<=0) = {boot_row['p_value_diff_le_0']} "
           "— no evidence MATE beats BPE at ternary precision.")

    table7_ablation2(TABLES_DIR)

    diff_h1 = delta_ternary - delta_fp16
    supported = "YES" if diff_h1 > 0 else "NO"
    hyp_text = (
        f"Delta_ternary = {delta_ternary:+.2f}\n"
        f"Delta_fp16    = {delta_fp16:+.2f}\n"
        f"Delta_ternary - Delta_fp16 = {diff_h1:+.2f}\n"
        f"H1 supported: {supported}\n\n"
        "This is the real Stage 2 result: 4 cells x 3 seeds, 10M tokens/seed,\n"
        "12-layer/d=512 model, real SAMER L5->L3 fine-tuning, 200-example test\n"
        "subset (not the full 3277 — a practical eval-time cap, not a data-\n"
        "availability gap), real paired bootstrap significance test.\n\n"
        f"Bootstrap test (seed 0, 1000 resamples): SARI(mate_ternary) - \n"
        f"SARI(bpe_ternary) = {boot_row['diff']}, p(diff<=0) = {boot_row['p_value_diff_le_0']}.\n\n"
        "VERDICT: H1's *shape* holds directionally (MATE's shortfall vs BPE\n"
        "narrows at ternary precision), but MATE does not outperform BPE at\n"
        "either precision in absolute SARI/BERTScore terms, and the bootstrap\n"
        "test finds no evidence favouring MATE at ternary precision specifically\n"
        "(p=0.939 - the resampled distribution overwhelmingly favours BPE).\n"
        "At this token budget and model scale, H1 is NOT supported. Report\n"
        "this as the actual finding (IMPLEMENTATION.md sec:5: 'If H1 is not\n"
        "supported, the paper still gets written and it reports that') rather\n"
        "than tuning the design until it comes out favourably.\n"
    )
    with open(STAGE2_DIR / "HYPOTHESIS.md", "w", encoding="utf-8") as f:
        f.write(hyp_text)

    print(f"wrote Tables 4-7 to {TABLES_DIR}/ and HYPOTHESIS.md to {STAGE2_DIR}/")
    print(f"\nDelta_ternary={delta_ternary:+.2f}  Delta_fp16={delta_fp16:+.2f}  H1 supported: {supported}")
    print(f"(directional shape only — see HYPOTHESIS.md: MATE loses outright at both precisions)")


if __name__ == "__main__":
    main()
