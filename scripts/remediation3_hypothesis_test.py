"""Remediation Phase 3: test the hypothesis that was actually pre-specified.

H1: Delta_p = SARI(mate, p) - SARI(bpe, p); H1 claims Delta_ternary > Delta_fp16,
i.e. the INTERACTION I = Delta_ternary - Delta_fp16 > 0. The existing bootstrap
(results/stage2/bootstrap_significance.csv) tested a different quantity
(SARI(mate_ternary) - SARI(bpe_ternary) at seed 0 only) - not H1 itself.

Uses Phase 2's full-split (3,277 examples), corrected (Phase 1) per-seed
results - the most complete, accurate data available, not the 200-example
subset or the pre-correction numbers.

1. Paired-across-seeds interaction test: I(s) = [SARI(mate_ternary,s) -
   SARI(bpe_ternary,s)] - [SARI(mate_fp16,s) - SARI(bpe_fp16,s)] for each
   seed s in {0,1,2}. Mean, std, SE, paired t-test against zero (df=2).
2. Two-level bootstrap: resamples seeds AND test examples within each
   resampled seed, to propagate both variance sources into the interaction
   estimate. Uses real per-example SARI from the saved full-split
   generations (results/remediation2/generations_*_full.tsv), not just the
   corpus-level means.
3. Minimum detectable effect at alpha=0.05, computed from the REAL SE
   above (t_crit(df=2)*SE), not the plan's own back-of-envelope ~1.5
   orientation figure.

Usage: python -m scripts.remediation3_hypothesis_test
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from src.eval.quality import sari

IN_DIR = Path("results/remediation2")
OUT_DIR = Path("results/remediation3")
CELLS = ["bpe_fp16", "bpe_ternary", "mate_fp16", "mate_ternary"]
SEEDS = [0, 1, 2]
N_BOOTSTRAP = 10000
ALPHA = 0.05


def load_full_split_examples(cell: str, seed: int) -> list[tuple[str, str, str]]:
    """Returns list of (source, generated, reference) from the saved
    full-split generations - the real per-example data the corpus-level
    SARI numbers were computed from."""
    rows = []
    with open(IN_DIR / f"generations_{cell}_seed{seed}_full.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append((row["source"], row["generated"], row["reference"]))
    return rows


def per_example_sari_array(rows: list[tuple[str, str, str]]) -> np.ndarray:
    """Precomputes each example's SARI score ONCE. Critical for bootstrap
    performance: a naive implementation that recomputes sari() inside every
    bootstrap iteration would need ~4 cells * 3 seeds * 3277 examples =
    ~39,300 sari() calls PER iteration, ~393 million total over 10,000
    iterations - likely hours to days. SARI's per-example score only
    depends on that example's own (source, generated, reference), not on
    which other examples are in a given resample, so it can be computed
    once and reused - every bootstrap iteration then just indexes/means a
    plain numpy array."""
    return np.array([sari(s, g, [r]) for s, g, r in rows])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load per-seed corpus SARI directly from Phase 2's own results (already
    # computed correctly there) for the headline numbers.
    sari_by_cell_seed: dict[str, dict[int, float]] = {c: {} for c in CELLS}
    with open(IN_DIR / "quality_results_full.csv") as f:
        for row in csv.DictReader(f):
            seeds_vals = [float(x) for x in row["sari_seeds"].split(";")]
            for seed, val in zip(SEEDS, seeds_vals):
                sari_by_cell_seed[row["cell"]][seed] = val

    print("Per-seed SARI (full split, corrected):")
    for cell in CELLS:
        print(f"  {cell:14s}: {[f'{sari_by_cell_seed[cell][s]:.4f}' for s in SEEDS]}")

    # ---- 1. Paired-across-seeds interaction test ----
    I_values = []
    for s in SEEDS:
        delta_ternary = sari_by_cell_seed["mate_ternary"][s] - sari_by_cell_seed["bpe_ternary"][s]
        delta_fp16 = sari_by_cell_seed["mate_fp16"][s] - sari_by_cell_seed["bpe_fp16"][s]
        I_values.append(delta_ternary - delta_fp16)
    print(f"\nPer-seed interaction I(s) = Delta_ternary(s) - Delta_fp16(s): "
          f"{[f'{v:.4f}' for v in I_values]}")

    n = len(I_values)
    mean_I = statistics.mean(I_values)
    std_I = statistics.stdev(I_values)  # sample std, n-1
    se_I = std_I / (n ** 0.5)
    df = n - 1
    t_stat = mean_I / se_I
    p_two_tailed = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df))
    p_one_tailed = 1 - scipy_stats.t.cdf(t_stat, df)  # H1 predicts I > 0, one-sided
    t_crit_two = scipy_stats.t.ppf(1 - ALPHA / 2, df)
    t_crit_one = scipy_stats.t.ppf(1 - ALPHA, df)

    print(f"\n=== 1. Paired-across-seeds interaction test (df={df}) ===")
    print(f"mean(I) = {mean_I:.4f}")
    print(f"std(I)  = {std_I:.4f}")
    print(f"SE(I)   = {se_I:.4f}")
    print(f"t-statistic = {t_stat:.4f}")
    print(f"two-tailed p = {p_two_tailed:.5f}  (t_crit={t_crit_two:.4f})")
    print(f"one-tailed p (H1: I>0) = {p_one_tailed:.5f}  (t_crit={t_crit_one:.4f})")
    print(f"significant at alpha={ALPHA}: two-tailed={p_two_tailed < ALPHA}, one-tailed={p_one_tailed < ALPHA}")

    # ---- 3. Minimum detectable effect (computed from the real SE above) ----
    mde_two_tailed = t_crit_two * se_I
    mde_one_tailed = t_crit_one * se_I
    print(f"\n=== 3. Minimum detectable effect at alpha={ALPHA} (real, from measured SE) ===")
    print(f"MDE (two-tailed) = t_crit(df={df})*SE = {t_crit_two:.4f}*{se_I:.4f} = {mde_two_tailed:.4f} SARI points")
    print(f"MDE (one-tailed) = {t_crit_one:.4f}*{se_I:.4f} = {mde_one_tailed:.4f} SARI points")
    print(f"Observed |I| = {abs(mean_I):.4f} {'>' if abs(mean_I) > mde_two_tailed else '<='} MDE(two-tailed)")

    # ---- 2. Two-level bootstrap (resample seeds AND examples) ----
    print(f"\n=== 2. Two-level bootstrap ({N_BOOTSTRAP} resamples) ===")
    print("loading real per-example generations and precomputing per-example SARI "
          "(once - not inside the bootstrap loop, see per_example_sari_array's docstring)...")
    # per_example[cell][seed] -> np.ndarray of length n_examples
    per_example: dict[str, dict[int, np.ndarray]] = {c: {} for c in CELLS}
    for cell in CELLS:
        for s in SEEDS:
            rows = load_full_split_examples(cell, s)
            per_example[cell][s] = per_example_sari_array(rows)
    n_examples = len(per_example["bpe_fp16"][0])
    print(f"  {n_examples} examples per cell/seed, precomputed for all {len(CELLS)} cells x {len(SEEDS)} seeds")

    rng = np.random.default_rng(0)
    boot_I = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        # resample seeds with replacement
        resampled_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        # resample the SAME example indices across all 4 cells within this
        # bootstrap replicate (paired resampling - each cell scored on the
        # same resampled example set, consistent with how corpus SARI itself
        # is a per-example mean over one shared test set)
        example_idx = rng.integers(0, n_examples, size=n_examples)

        deltas = {}
        for precision in ("fp16", "ternary"):
            mate_scores = [per_example[f"mate_{precision}"][rs][example_idx].mean() for rs in resampled_seeds]
            bpe_scores = [per_example[f"bpe_{precision}"][rs][example_idx].mean() for rs in resampled_seeds]
            deltas[precision] = float(np.mean(mate_scores)) - float(np.mean(bpe_scores))
        boot_I[b] = deltas["ternary"] - deltas["fp16"]

        if (b + 1) % 2000 == 0:
            print(f"  {b+1}/{N_BOOTSTRAP} resamples done")

    boot_I = boot_I.tolist()

    boot_I.sort()
    ci_lo = boot_I[int(0.025 * N_BOOTSTRAP)]
    ci_hi = boot_I[int(0.975 * N_BOOTSTRAP)]
    boot_mean = statistics.mean(boot_I)
    p_boot_le_zero = sum(1 for v in boot_I if v <= 0) / N_BOOTSTRAP

    print(f"\nbootstrap mean(I) = {boot_mean:.4f}")
    print(f"95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"P(I <= 0) under resampling = {p_boot_le_zero:.5f}")

    # ---- save everything ----
    results = dict(
        per_seed_sari={c: sari_by_cell_seed[c] for c in CELLS},
        per_seed_interaction=I_values,
        paired_t_test=dict(
            mean_I=mean_I, std_I=std_I, se_I=se_I, df=df, t_stat=float(t_stat),
            p_two_tailed=float(p_two_tailed), p_one_tailed=float(p_one_tailed),
            t_crit_two_tailed=float(t_crit_two), t_crit_one_tailed=float(t_crit_one),
            significant_two_tailed=bool(p_two_tailed < ALPHA),
            significant_one_tailed=bool(p_one_tailed < ALPHA),
        ),
        minimum_detectable_effect=dict(
            mde_two_tailed=mde_two_tailed, mde_one_tailed=mde_one_tailed,
            observed_exceeds_mde_two_tailed=bool(abs(mean_I) > mde_two_tailed),
        ),
        two_level_bootstrap=dict(
            n_bootstrap=N_BOOTSTRAP, mean=boot_mean, ci_95_lo=ci_lo, ci_95_hi=ci_hi,
            p_le_zero=p_boot_le_zero,
        ),
    )
    with open(OUT_DIR / "hypothesis_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_DIR}/hypothesis_test_results.json")


if __name__ == "__main__":
    main()
