"""Remediation Phase 2: re-score every own-model cell (corrected, 3 seeds
each) and every Stage 3 baseline on the FULL 3,277-example SAMER test split,
using the new batched generation (src/eval/batched_generate.py) instead of
the 200-example subset used throughout Stages 1-4 and Remediation Phase 1.

Generation only, no retraining - reuses the already fine-tuned checkpoints
from results/stage2 (bpe cells, unaffected by Phase 1), results/
stage2_corrected (mate cells, corrected), and results/stage3 (external
baselines).

Usage: python -m scripts.remediation2_full_split_eval
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from scripts.stage2_finetune_and_eval import load_samer_pairs
from src.data.cache import read_cache
from src.data.mate_batch import MorphIndex
from src.eval.batched_generate import generate_bpe_batched, generate_mate_batched
from src.eval.quality import bertscore_f1, corpus_sari
from src.model.arabit import AraBit
from src.model.mate import MATEConfig
from src.tokenization.vocab import load_vocabs

DATA_DIR = Path("data/stage2")
STAGE2_DIR = Path("results/stage2")
STAGE2C_DIR = Path("results/stage2_corrected")
STAGE3_DIR = Path("results/stage3")
OUT_DIR = Path("results/remediation2")
D_MODEL, N_LAYERS, N_HEADS, D_FF = 512, 12, 8, 2048
BATCH_SIZE = 32
SEEDS = [0, 1, 2]


def score_own_cells(eval_sources, eval_refs, control_tok, fallback_tok, morph_index, mate_cfg, device):
    per_cell_sari, per_cell_bs, all_gens = {}, {}, {}
    vocab_size = control_tok.get_vocab_size()

    configs = [
        ("bpe", "fp16", STAGE2_DIR),
        ("bpe", "ternary", STAGE2_DIR),
        ("mate", "fp16", STAGE2C_DIR),
        ("mate", "ternary", STAGE2C_DIR),
    ]
    for front_end, precision, ckpt_dir in configs:
        cell = f"{front_end}_{precision}"
        per_cell_sari[cell], per_cell_bs[cell] = [], []
        for seed in SEEDS:
            cell_seed = f"{cell}_seed{seed}"
            print(f"\n[{cell_seed}] generating on full split ({len(eval_sources)} examples)...")
            model = AraBit(front_end, precision, vocab_size, D_MODEL, N_LAYERS, N_HEADS, D_FF, mate_cfg).to(device)
            state = torch.load(ckpt_dir / f"checkpoint_ft_{cell_seed}.pt", map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()

            t0 = time.time()
            if front_end == "bpe":
                gen = generate_bpe_batched(model, eval_sources, control_tok, device, batch_size=BATCH_SIZE)
            else:
                gen = generate_mate_batched(
                    model, eval_sources, control_tok, fallback_tok, morph_index, device, batch_size=BATCH_SIZE
                )
            gen_time = time.time() - t0

            score = corpus_sari(eval_sources, gen, eval_refs)
            print(f"    SARI={score:.2f}  generation took {gen_time:.1f}s")
            per_cell_sari[cell].append(score)
            all_gens[cell_seed] = gen

            with open(OUT_DIR / f"generations_{cell_seed}_full.tsv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["source", "generated", "reference"])
                for s, g, r in zip(eval_sources, gen, [r[0] for r in eval_refs]):
                    w.writerow([s, g, r])
            del model
            torch.cuda.empty_cache()

        # BERTScore over the concatenation of all 3 seeds' non-empty generations, scored per-seed
        for seed in SEEDS:
            gens = all_gens[f"{cell}_seed{seed}"]
            pairs = [(g, r[0]) for g, r in zip(gens, eval_refs) if g.strip()]
            if pairs:
                f1 = bertscore_f1([p[0] for p in pairs], [p[1] for p in pairs])
            else:
                f1 = float("nan")
            per_cell_bs[cell].append(f1)
            print(f"    seed{seed} BERTScore-F1={f1:.4f}")

    return per_cell_sari, per_cell_bs


def score_baselines(eval_sources, eval_refs, device):
    results = {}
    MAX_SRC_LEN, MAX_TGT_LEN, GEN_BATCH = 64, 64, 16

    def gen_hf(model, tokenizer, sources):
        gens = []
        for i in range(0, len(sources), GEN_BATCH):
            batch_src = sources[i : i + GEN_BATCH]
            enc = tokenizer(batch_src, padding=True, truncation=True, max_length=MAX_SRC_LEN, return_tensors="pt").to(device)
            out_ids = model.generate(**enc, max_new_tokens=MAX_TGT_LEN, num_beams=1)
            gens.extend(tokenizer.batch_decode(out_ids, skip_special_tokens=True))
        return gens

    for name, ckpt_dir, extra in [
        ("AraBART", STAGE3_DIR / "checkpoint_AraBART", {}),
        ("AraT5", STAGE3_DIR / "checkpoint_AraT5", {}),
    ]:
        print(f"\n[{name}] generating on full split...")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).to(device).eval()
        t0 = time.time()
        gen = gen_hf(model, tokenizer, eval_sources)
        gen_time = time.time() - t0
        score = corpus_sari(eval_sources, gen, eval_refs)
        pairs = [(g, r[0]) for g, r in zip(gen, eval_refs) if g.strip()]
        bs = bertscore_f1([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else float("nan")
        print(f"    SARI={score:.2f}  BERTScore-F1={bs:.4f}  ({gen_time:.1f}s)")
        results[name] = dict(sari=score, bertscore=bs)
        with open(OUT_DIR / f"generations_{name}_full.tsv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["source", "generated", "reference"])
            for s, g, r in zip(eval_sources, gen, [r[0] for r in eval_refs]):
                w.writerow([s, g, r])
        del model
        torch.cuda.empty_cache()

    print("\n[AraBART_nf4] generating on full split...")
    tokenizer = AutoTokenizer.from_pretrained(STAGE3_DIR / "checkpoint_AraBART")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(STAGE3_DIR / "checkpoint_AraBART", quantization_config=bnb_config, device_map={"": 0})
    t0 = time.time()
    gen = gen_hf(model, tokenizer, eval_sources)
    gen_time = time.time() - t0
    score = corpus_sari(eval_sources, gen, eval_refs)
    pairs = [(g, r[0]) for g, r in zip(gen, eval_refs) if g.strip()]
    bs = bertscore_f1([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else float("nan")
    print(f"    SARI={score:.2f}  BERTScore-F1={bs:.4f}  ({gen_time:.1f}s)")
    results["AraBART_nf4"] = dict(sari=score, bertscore=bs)
    with open(OUT_DIR / "generations_AraBART_nf4_full.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source", "generated", "reference"])
        for s, g, r in zip(eval_sources, gen, [r[0] for r in eval_refs]):
            w.writerow([s, g, r])
    del model
    torch.cuda.empty_cache()

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    test_pairs = load_samer_pairs("test")
    eval_sources = [s for s, _ in test_pairs]
    eval_refs = [[t] for _, t in test_pairs]
    print(f"FULL test split: {len(eval_sources)} examples")

    control_tok = Tokenizer.from_file(str(DATA_DIR / "bpe_control_tokenizer.json"))
    fallback_tok = Tokenizer.from_file(str(DATA_DIR / "bpe_tokenizer.json"))
    vocabs = load_vocabs(DATA_DIR / "vocabs.json")
    cache_df = read_cache(DATA_DIR / "morph_cache.parquet")
    morph_index = MorphIndex(cache_df, vocabs)
    mate_cfg = MATEConfig(
        n_proclitic=len(vocabs["proclitic"]), n_enclitic=len(vocabs["enclitic"]),
        n_root=len(vocabs["root"]), n_pattern=len(vocabs["pattern"]),
        n_bpe=fallback_tok.get_vocab_size(),
        d_clitic=24, d_root=48, d_pattern=48, d_model=D_MODEL,
    )

    overall_t0 = time.time()
    own_sari, own_bs = score_own_cells(eval_sources, eval_refs, control_tok, fallback_tok, morph_index, mate_cfg, device)
    baseline_results = score_baselines(eval_sources, eval_refs, device)
    print(f"\ntotal wall time: {(time.time()-overall_t0)/60:.1f} min")

    with open(OUT_DIR / "quality_results_full.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "sari_mean", "sari_std", "sari_seeds", "bertscore_mean", "bertscore_std", "n_eval"])
        for cell in own_sari:
            sm, ss = statistics.mean(own_sari[cell]), (statistics.stdev(own_sari[cell]) if len(own_sari[cell]) > 1 else 0.0)
            bm, bs_ = statistics.mean(own_bs[cell]), (statistics.stdev(own_bs[cell]) if len(own_bs[cell]) > 1 else 0.0)
            w.writerow([cell, f"{sm:.4f}", f"{ss:.4f}", ";".join(f"{s:.4f}" for s in own_sari[cell]),
                        f"{bm:.4f}", f"{bs_:.4f}", len(eval_sources)])

    with open(OUT_DIR / "baseline_results_full.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "sari", "bertscore", "n_eval"])
        for name, r in baseline_results.items():
            w.writerow([name, f"{r['sari']:.4f}", f"{r['bertscore']:.4f}", len(eval_sources)])

    print("\n=== Remediation Phase 2 full-split summary ===")
    for cell in own_sari:
        sm, ss = statistics.mean(own_sari[cell]), (statistics.stdev(own_sari[cell]) if len(own_sari[cell]) > 1 else 0.0)
        bm, bs_ = statistics.mean(own_bs[cell]), (statistics.stdev(own_bs[cell]) if len(own_bs[cell]) > 1 else 0.0)
        print(f"{cell:14s}: SARI={sm:.2f}+/-{ss:.2f}  BERTScore={bm:.4f}+/-{bs_:.4f}")
    for name, r in baseline_results.items():
        print(f"{name:14s}: SARI={r['sari']:.2f}  BERTScore={r['bertscore']:.4f}")


if __name__ == "__main__":
    main()
