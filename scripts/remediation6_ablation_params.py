"""Remediation Phase 6: P_t/P_f parameter breakdown for Table 7 (component-
level MATE ablation), computed from already-trained checkpoints - no new
training or GPU time needed.

Table 7's own generator (scripts/remediation1_ablation.py) recorded each
ablation variant's TOTAL parameter count (n_params = sum of every parameter,
ternary and full-precision together), but the manuscript's "FP params (M)"
column means specifically the full-precision remainder (P_f in Table 5's
sense - everything that is NOT a BitLinear.weight, which src/eval/
efficiency.py: count_params() already knows how to isolate). That split was
never computed per ablation variant. Since every checkpoint already exists
on disk (results/stage4_corrected/checkpoint_ablation_*.pt,
results/stage2_corrected/checkpoint_ft_mate_ternary_seed0.pt,
results/stage2/checkpoint_ft_bpe_ternary_seed0.pt), this just re-builds each
model architecture (matching the config each was trained with), loads the
saved weights, and calls count_params() - no forward pass, no GPU required.

Usage: python -m scripts.remediation6_ablation_params
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.eval.efficiency import count_params
from src.model.arabit import AraBit
from src.model.mate import MATEConfig
from src.tokenization.vocab import load_vocabs

DATA_DIR = Path("data/stage2")
D_MODEL, N_LAYERS, N_HEADS, D_FF = 512, 12, 8, 2048
OUT_DIR = Path("results/remediation6")


def build_mate_cfg(vocabs, fallback_tok, ablation: str) -> MATEConfig:
    return MATEConfig(
        n_proclitic=len(vocabs["proclitic"]), n_enclitic=len(vocabs["enclitic"]),
        n_root=len(vocabs["root"]), n_pattern=len(vocabs["pattern"]),
        n_bpe=fallback_tok.get_vocab_size(),
        d_clitic=24, d_root=48, d_pattern=48, d_model=D_MODEL,
        ablation=ablation,
    )


def main():
    device = torch.device("cpu")  # counting params needs no GPU
    control_tok = Tokenizer.from_file(str(DATA_DIR / "bpe_control_tokenizer.json"))
    fallback_tok = Tokenizer.from_file(str(DATA_DIR / "bpe_tokenizer.json"))
    vocabs = load_vocabs(DATA_DIR / "vocabs.json")
    vocab_size = control_tok.get_vocab_size()

    results = {}

    configs = [
        ("Full MATE", "mate", build_mate_cfg(vocabs, fallback_tok, "full"),
         Path("results/stage2_corrected/checkpoint_ft_mate_ternary_seed0.pt")),
        ("- pattern embedding", "mate", build_mate_cfg(vocabs, fallback_tok, "no_pattern"),
         Path("results/stage4_corrected/checkpoint_ablation_no_pattern.pt")),
        ("- root embedding", "mate", build_mate_cfg(vocabs, fallback_tok, "no_root"),
         Path("results/stage4_corrected/checkpoint_ablation_no_root.pt")),
        ("- clitic separation", "mate", build_mate_cfg(vocabs, fallback_tok, "no_clitics"),
         Path("results/stage4_corrected/checkpoint_ablation_no_clitics.pt")),
        ("- learned gate", "mate", build_mate_cfg(vocabs, fallback_tok, "hard_gate"),
         Path("results/stage4_corrected/checkpoint_ablation_hard_gate.pt")),
        ("- all (plain BPE)", "bpe", None,
         Path("results/stage2/checkpoint_ft_bpe_ternary_seed0.pt")),
    ]

    for name, front_end, mate_cfg, ckpt_path in configs:
        model = AraBit(front_end, "ternary", vocab_size, D_MODEL, N_LAYERS, N_HEADS, D_FF, mate_cfg).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        p_t, p_f = count_params(model)
        results[name] = dict(p_t=p_t, p_f=p_f, p_total=p_t + p_f)
        print(f"{name:24s}: P_t={p_t:,}  P_f={p_f:,}  total={p_t+p_f:,}")
        del model

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "ablation_params.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "ablation_params.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["configuration", "p_t", "p_f", "p_total"])
        for name, r in results.items():
            w.writerow([name, r["p_t"], r["p_f"], r["p_total"]])
    print(f"\nwrote {OUT_DIR}/ablation_params.{{json,csv}}")


if __name__ == "__main__":
    main()
