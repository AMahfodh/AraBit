"""Merges Stage 2's own-model results with Stage 3's external baselines into
the final, most complete Tables 4 and 5. Tables 6/7/8 are unchanged from
Stage 2 (they don't involve baselines) — see results/stage2/tables/.

Usage: python -m scripts.stage3_make_tables
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.report.tables import TBD, _write

STAGE2_DIR = Path("results/stage2")
STAGE3_DIR = Path("results/stage3")
TABLES_DIR = STAGE3_DIR / "tables"


def main():
    cell_sari, cell_bs = {}, {}
    with open(STAGE2_DIR / "quality_results.csv") as f:
        for row in csv.DictReader(f):
            cell_sari[row["cell"]] = (float(row["sari_mean"]), float(row["sari_std"]))
            if row.get("bertscore_mean"):
                cell_bs[row["cell"]] = (float(row["bertscore_mean"]), float(row["bertscore_std"]))

    baseline = {}
    with open(STAGE3_DIR / "baseline_results.csv") as f:
        for row in csv.DictReader(f):
            baseline[row["model"]] = dict(sari=float(row["sari"]), bertscore=float(row.get("bertscore_f1", "nan")) if row.get("bertscore_f1") not in (None, "NA") else None)

    baseline_eff = {}
    with open(STAGE3_DIR / "baseline_efficiency.csv") as f:
        for row in csv.DictReader(f):
            baseline_eff[row["model"]] = dict(n_params=int(row["n_params"]), mem_mb=float(row["mem_MB"]), latency=float(row["latency_ms_per_tok"]))

    cell_eff = {}
    with open(STAGE2_DIR / "efficiency_results.csv") as f:
        for row in csv.DictReader(f):
            cell_eff[row["cell"]] = dict(
                p_t=int(row["P_t"]), p_f=int(row["P_f"]), weight_mem_mb=float(row["weight_mem_MB"]),
                model_ms_per_tok=float(row["latency_model_ms_per_tok"]),
                e2e_ms_per_tok=float(row["latency_e2e_ms_per_tok"]),
                joules_per_seq=float(row["energy_J_per_seq"]) if row["energy_J_per_seq"] != "NA" else float("nan"),
            )

    def sari_str(cell):
        m, s = cell_sari[cell]
        return f"{m:.2f} +/- {s:.2f}"

    def bs_str(cell):
        if cell not in cell_bs:
            return TBD
        m, s = cell_bs[cell]
        return f"{m:.4f} +/- {s:.4f}"

    def base_sari(name):
        return f"{baseline[name]['sari']:.2f}" + (" (degenerate generation, see NOTES.md)" if name == "AraT5" else "")

    def base_bs(name):
        v = baseline[name].get("bertscore")
        return f"{v:.4f}" if v is not None else TBD

    # Table 4
    header = ["Model", "Precision", "SARI", "BERTScore-F1", "OSMAN", "BLEU (reph.)"]
    rows = [
        ["AraBART", "FP16", base_sari("AraBART"), base_bs("AraBART"), TBD, TBD],
        ["AraT5", "FP16", base_sari("AraT5"), base_bs("AraT5"), TBD, TBD],
        ["FFT-Seq2Seq", "FP16", TBD, TBD, TBD, TBD],
        ["Switch-Arabic", "FP16", TBD, TBD, TBD, TBD],
        ["AraBART + bitsandbytes-NF4 (not GPTQ, see NOTES.md)", "4-bit", base_sari("AraBART_nf4"), base_bs("AraBART_nf4"), TBD, TBD],
        ["B4: ours, BPE", "FP16", sari_str("bpe_fp16"), bs_str("bpe_fp16"), TBD, TBD],
        ["B2: ours, MATE", "FP16", sari_str("mate_fp16"), bs_str("mate_fp16"), TBD, TBD],
        ["B3: ours, BPE", "1.58-bit", sari_str("bpe_ternary"), bs_str("bpe_ternary"), TBD, TBD],
        ["AraBit-1.58 (MATE)", "1.58-bit", sari_str("mate_ternary"), bs_str("mate_ternary"), TBD, TBD],
    ]
    _write(rows, header, "table4_quality", TABLES_DIR,
           "Table 4 (tab:quality), final: Stage 2 own-model results (mean "
           "+/- std, 3 seeds) plus Stage 3 external baselines (single run "
           "each, real fine-tuning on the same SAMER split/200-example "
           "eval subset). AraT5's number reflects a diagnosed, unresolved "
           "degenerate-generation issue in the checkpoint — see "
           "results/NOTES.md, do not read it as AraT5's true capability. "
           "AraBART+NF4 uses bitsandbytes 4-bit, not GPTQ/AWQ (unavailable "
           "on this machine, see results/NOTES.md). FFT-Seq2Seq/"
           "Switch-Arabic skipped (user's own prior work, not available in "
           "this repo). OSMAN/BLEU(reph.) not implemented/not applicable.")

    # Table 5
    header5 = ["Model", "P_t (M)", "P_f (M)", "Mem (MB)", "Lat. model (ms/tok)",
               "Lat. e2e (ms/tok)", "Energy (J/seq)"]

    def own_row(name, cell):
        r = cell_eff[cell]
        return [name, f"{r['p_t']/1e6:.3f}", f"{r['p_f']/1e6:.3f}", f"{r['weight_mem_mb']:.2f}",
                f"{r['model_ms_per_tok']:.4f}", f"{r['e2e_ms_per_tok']:.4f}",
                f"{r['joules_per_seq']:.4f}" if r["joules_per_seq"] == r["joules_per_seq"] else TBD]

    def base_row(name, key):
        r = baseline_eff[key]
        return [name, "--", f"{r['n_params']/1e6:.3f}", f"{r['mem_mb']:.2f}", f"{r['latency']:.4f}", TBD, TBD]

    rows5 = [
        base_row("AraBART", "AraBART"),
        base_row("AraBART + bitsandbytes-NF4", "AraBART_nf4"),
        own_row("B2: FP16 + MATE", "mate_fp16"),
        own_row("B3: ternary + BPE", "bpe_ternary"),
        own_row("AraBit-1.58 (MATE, ternary)", "mate_ternary"),
        own_row("(also measured) B4: FP16 + BPE", "bpe_fp16"),
        base_row("(also measured) AraT5", "AraT5"),
    ]
    _write(rows5, header5, "table5_efficiency", TABLES_DIR,
           "Table 5 (tab:efficiency), final: Stage 2 own cells (real, "
           "small model, mean over cell architecture - not seed-dependent) "
           "plus Stage 3 baselines. AraBART/AraT5 rows report total params "
           "(no ternary/full-precision split - these are standard "
           "fp32/4-bit models, not this project's own architecture) and "
           "measured latency/memory; Lat. e2e and Energy not measured for "
           "external baselines (TBD).")

    print(f"wrote final Tables 4-5 to {TABLES_DIR}/")


if __name__ == "__main__":
    main()
