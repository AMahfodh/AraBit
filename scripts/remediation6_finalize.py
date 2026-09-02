"""Remediation Phase 6: reporting hygiene. Pulls together every real,
final number produced by Phases 0-5 (output-head fix, full-split
re-evaluation, the actual pre-specified H1 test, standardized decoding,
SAMER readability, dropped OSMAN, excluded AraT5) and regenerates Tables
4-7 + the final results/HYPOTHESIS.md via src/report/tables.py - the single
reproducible harness this project's tables are supposed to come from,
instead of hand-assembling numbers into the manuscript directly.

Does NOT re-run stage1_make_tables.py / stage2_make_tables.py - those are
frozen historical snapshots of what was actually reported at each earlier
stage (results/stage1/, results/stage2/), kept on disk as a record of how
understanding evolved, not overwritten. This script's output
(results/final/) is the new, final, authoritative one.

Usage: python -m scripts.remediation6_finalize
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.report.tables import (
    table4_quality,
    table5_efficiency,
    table6_ablation1,
    table7_ablation2,
    write_final_hypothesis_md,
)

OUT_DIR = Path("results/final")
TABLES_DIR = OUT_DIR / "tables"


def read_quality_full(path: Path) -> dict[str, tuple[float, float]]:
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["cell"]] = (float(row["sari_mean"]), float(row["sari_std"]))
    return out


def read_bertscore_full(path: Path) -> dict[str, tuple[float, float]]:
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["cell"]] = (float(row["bertscore_mean"]), float(row["bertscore_std"]))
    return out


def read_baseline_full(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    sari, bertscore = {}, {}
    name_map = {"AraBART": "AraBART", "AraBART_nf4": "AraBART + NF4"}  # AraT5 deliberately excluded
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] not in name_map:
                continue
            label = name_map[row["model"]]
            sari[label] = float(row["sari"])
            bertscore[label] = float(row["bertscore"])
    return sari, bertscore


def read_readability(path: Path) -> tuple[dict, dict]:
    own, baseline = {}, {}
    name_map = {"AraBART": "AraBART", "AraBART_nf4": "AraBART + NF4"}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["system"] in ("source", "reference"):
                continue
            if row["mean_level_std"]:
                own[row["system"]] = (float(row["mean_level"]), float(row["mean_level_std"]))
            elif row["system"] in name_map:
                baseline[name_map[row["system"]]] = float(row["mean_level"])
    return own, baseline


def read_efficiency(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["cell"]] = dict(
                p_t=int(row["P_t"]), p_f=int(row["P_f"]),
                weight_mem_mb=float(row["weight_mem_MB"]),
                model_ms_per_tok=float(row["latency_model_ms_per_tok"]),
                e2e_ms_per_tok=float(row["latency_e2e_ms_per_tok"]),
                joules_per_seq=float(row["energy_J_per_seq"]) if row["energy_J_per_seq"] not in ("NA", "") else float("nan"),
            )
    return out


def read_baseline_efficiency(path: Path) -> dict[str, dict]:
    out = {}
    name_map = {"AraBART": "AraBART", "AraBART_nf4": "AraBART + NF4"}  # AraT5 was never a Table 5 row
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] not in name_map:
                continue
            out[name_map[row["model"]]] = dict(
                p_t=0, p_f=int(row["n_params"]),
                weight_mem_mb=float(row["mem_MB"]),
                model_ms_per_tok=float(row["latency_ms_per_tok"]),
            )
    return out


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Table 4: quality (SARI, BERTScore, Readability) ----
    cell_sari = read_quality_full(Path("results/remediation2/quality_results_full.csv"))
    cell_bertscore = read_bertscore_full(Path("results/remediation2/quality_results_full.csv"))
    baseline_sari, baseline_bertscore = read_baseline_full(Path("results/remediation2/baseline_results_full.csv"))
    cell_readability, baseline_readability = read_readability(Path("results/remediation5/samer_readability_results.csv"))

    table4_quality(
        cell_sari, TABLES_DIR, cell_bertscore, cell_readability,
        baseline_sari, baseline_bertscore,
        caption=(
            "Table 4 (tab:quality): Generation quality, FINAL numbers "
            "(Remediation Phases 0-5). Own-model SARI/BERTScore/Readability: "
            "mean +/- std over 3 seeds, full 3,277-example SAMER test split, "
            "output-head confound fixed, standardized 64-token greedy "
            "decoding. AraBART/AraBART+NF4: single measurement, same full "
            "split. AraT5 excluded (degenerate output, see Threats to "
            "Validity). AraBART+GPTQ and BLEU (rephrasing task) remain TBD "
            "(never run/measured). Readability = mean SAMER lemma level, "
            "lower = simpler; read jointly with SARI, not in isolation "
            "(results/NOTES.md, Remediation Phase 5 Item 2)."))

    # ---- Table 5: efficiency ----
    cell_eff = read_efficiency(Path("results/stage2/efficiency_results.csv"))  # bpe_*, unaffected by the confound fix
    cell_eff_mate_corrected = read_efficiency(Path("results/stage2_corrected/efficiency_results_mate_only.csv"))
    cell_eff.update(cell_eff_mate_corrected)  # mate_* overwritten with corrected (post-output-head-fix) numbers
    baseline_eff = read_baseline_efficiency(Path("results/stage3/baseline_efficiency.csv"))

    table5_efficiency(
        cell_eff, TABLES_DIR,
        scale_note="Stage 2 scale (12L, d=512, 6000-article corpus); mate_* "
                    "efficiency re-measured post-output-head-fix (Remediation "
                    "Phase 1), bpe_* unaffected and unchanged",
        baseline_eff=baseline_eff,
        caption=(
            "Table 5 (tab:efficiency): FINAL numbers. P_t/P_f/Mem follow "
            "eq:mem's ternary-packing accounting for our own cells; AraBART/"
            "AraBART+NF4's Mem is the real measured resident weight memory "
            "(not eq:mem-derived - NF4 is a different quantization scheme "
            "from this repo's ternary BitLinear, so p_t=0/p_f=n_params for "
            "baselines is a labeling convenience, not a claim they use the "
            "same packing). AraBART+GPTQ was infeasible (no Windows wheels/"
            "compiler) and never measured - stays TBD, not silently dropped."))

    # ---- Table 6: central 2x2 ablation, from the SAME full-split data Table 4 and the H1 test use ----
    table6_ablation1(
        cell_sari, TABLES_DIR,
        caption=(
            "Table 6 (tab:ablate1): the central 2x2 ablation, FINAL numbers. "
            "Mean +/- std over 3 seeds, full 3,277-example test split, "
            "output-head-confound-fixed, standardized decoding - the exact "
            "data results/final/HYPOTHESIS.md's H1 test is computed from. "
            "The Delta column is a simple mean-vs-mean difference for "
            "readability; see HYPOTHESIS.md for the actual paired "
            "interaction test, MDE, and two-level bootstrap CI - do not "
            "read H1 support/non-support from this table's Delta column "
            "alone."))

    # ---- Table 7: component-level MATE ablation ----
    with open("results/stage4_corrected/ablation_results.csv", newline="") as f:
        ablation_sari = {row["ablation"]: float(row["sari"]) for row in csv.DictReader(f)}
    with open("results/stage2_corrected/quality_results.csv", newline="") as f:
        mate_ternary_seed0 = next(
            float(row["sari_seeds"].split(";")[0]) for row in csv.DictReader(f) if row["cell"] == "mate_ternary")
    with open("results/stage2/quality_results.csv", newline="") as f:
        bpe_ternary_seed0 = next(
            float(row["sari_seeds"].split(";")[0]) for row in csv.DictReader(f) if row["cell"] == "bpe_ternary")
    with open("results/remediation6/ablation_params.json") as f:
        params = json.load(f)

    label_map = {
        "Full MATE (proclitic + root + pattern + enclitic + gate)": ("Full MATE", mate_ternary_seed0),
        "- pattern embedding": ("- pattern embedding", ablation_sari["no_pattern"]),
        "- root embedding (stem embedding instead)": ("- root embedding", ablation_sari["no_root"]),
        "- clitic separation (surface form only)": ("- clitic separation", ablation_sari["no_clitics"]),
        "- learned gate (hard fallback)": ("- learned gate", ablation_sari["hard_gate"]),
        "- all (plain BPE)": ("- all (plain BPE)", bpe_ternary_seed0),
    }
    config_sari = {full_label: sari for full_label, (_, sari) in label_map.items()}
    config_p_f = {full_label: params[param_key]["p_f"] for full_label, (param_key, _) in label_map.items()}

    table7_ablation2(
        TABLES_DIR, config_sari, config_p_f,
        caption=(
            "Table 7 (tab:ablate2): MATE component-level ablation, FINAL "
            "numbers (Remediation Phase 1.2.3 re-run post-output-head-fix). "
            "Single seed (seed 0), ternary precision, 200-example eval "
            "subset (not the full split) throughout - Full MATE and plain-"
            "BPE reference points use the same seed-0/200-example "
            "measurement for a fair comparison, not the 3-seed full-split "
            "means reported in Table 4/6. FP params = P_f (full-precision "
            "remainder), computed directly from each saved checkpoint "
            "(results/remediation6/ablation_params.{json,csv}), not "
            "re-measured/re-trained."))

    # ---- Final HYPOTHESIS.md, the real pre-specified H1 test ----
    with open("results/remediation3/hypothesis_test_results.json") as f:
        h1_result = json.load(f)
    write_final_hypothesis_md(
        h1_result, OUT_DIR,
        provenance="results/remediation2/quality_results_full.csv (full 3,277-example "
                    "test split, 3 seeds, output-head-confound-fixed, standardized "
                    "64-token decoding) -> results/remediation3/hypothesis_test_results.json")

    print(f"wrote Tables 4-7 to {TABLES_DIR}/ and the final HYPOTHESIS.md to {OUT_DIR}/")


if __name__ == "__main__":
    main()
