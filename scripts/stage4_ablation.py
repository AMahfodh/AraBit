"""Stage 4: component-level MATE ablation (Table 7/tab:ablate2), single seed
(seed0), Stage 2 scale (10M tokens, 12L/d=512 model), ternary precision
(matches AraBit-1.58's actual proposed setting) — user-confirmed 2026-09-01
after a corrected ~2.5h time estimate.

Reuses Stage 2's exact hyperparameters/data (scripts/stage2_pretrain_all_cells.py,
scripts/stage2_finetune_and_eval.py) for the 4 new variants (no_pattern,
no_root, no_clitics, hard_gate — see src/model/mate.py's module docstring
for what each means concretely). "Full MATE" reuses the existing
mate_ternary_seed0 result; "- all (plain BPE)" reuses the existing
bpe_ternary_seed0 result — neither needs retraining.

Usage: python -m scripts.stage4_ablation
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from scripts.stage2_finetune_and_eval import (
    BPE_MAX_LEN,
    MATE_MAX_LEN,
    finetune as ft2_finetune,
    generate_mate,
    load_samer_pairs,
    make_mate_examples,
)
from scripts.stage2_pretrain_all_cells import (
    ARTICLE_LIMIT,
    BATCH_SIZE,
    D_FF,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    PEAK_LR_TERNARY,
    SEQ_LEN,
    TARGET_TOKENS_PER_SEED,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    build_word_sequences,
)
from src.data.cache import read_cache
from src.data.corpus import stream_sentences
from src.data.mate_batch import MorphIndex
from src.eval.quality import corpus_sari
from src.model.arabit import AraBit
from src.model.mate import MATEConfig
from src.tokenization.vocab import load_vocabs
from src.train.schedule import lr_at_step

DATA_DIR = Path("data/stage2")
OUT_DIR = Path("results/stage4")
SEED = 0
EVAL_CAP = 200
ABLATIONS = ["no_pattern", "no_root", "no_clitics", "hard_gate"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"streaming {ARTICLE_LIMIT} articles...")
    sentences = list(stream_sentences(article_limit=ARTICLE_LIMIT))
    fallback_tok = Tokenizer.from_file(str(DATA_DIR / "bpe_tokenizer.json"))
    vocabs = load_vocabs(DATA_DIR / "vocabs.json")
    cache_df = read_cache(DATA_DIR / "morph_cache.parquet")
    morph_index = MorphIndex(cache_df, vocabs)
    word_ids, words_grid = build_word_sequences(sentences, vocabs["word"])
    word_vocab_size = len(vocabs["word"])
    print(f"word sequences: {word_ids.shape}")

    train_pairs = load_samer_pairs("train")
    test_pairs = load_samer_pairs("test")
    eval_sources = [s for s, _ in test_pairs][:EVAL_CAP]
    eval_refs = [[t] for _, t in test_pairs][:EVAL_CAP]
    mate_train_ex = make_mate_examples(train_pairs, MATE_MAX_LEN)

    results = {}
    overall_t0 = time.time()
    for ablation in ABLATIONS:
        print(f"\n{'='*20} ablation={ablation} {'='*20}")
        mate_cfg = MATEConfig(
            n_proclitic=len(vocabs["proclitic"]), n_enclitic=len(vocabs["enclitic"]),
            n_root=len(vocabs["root"]), n_pattern=len(vocabs["pattern"]),
            n_bpe=fallback_tok.get_vocab_size(),
            d_clitic=24, d_root=48, d_pattern=48, d_model=D_MODEL,
            ablation=ablation,
        )

        # --- pretrain (mirrors scripts/stage2_pretrain_all_cells.py exactly) ---
        torch.manual_seed(SEED)
        model = AraBit("mate", "ternary", word_vocab_size, D_MODEL, N_LAYERS, N_HEADS, D_FF, mate_cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR_TERNARY, weight_decay=WEIGHT_DECAY)
        n_seq = word_ids.shape[0]
        target_steps = TARGET_TOKENS_PER_SEED // (BATCH_SIZE * SEQ_LEN)
        total_steps = min(target_steps, n_seq // BATCH_SIZE)
        torch.manual_seed(SEED)
        perm = torch.randperm(n_seq)
        print(f"  pretraining: {total_steps} steps")
        t0 = time.time()
        for step in range(total_steps):
            idx = perm[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
            seqs = word_ids[idx].to(device)
            input_words = [w for i in idx.tolist() for w in words_grid[i][:-1]]
            from src.data.mate_batch import build_mate_batch
            batch = build_mate_batch(input_words, morph_index, device)
            prefix_lens = torch.randint(0, (SEQ_LEN - 1) // 2, (BATCH_SIZE,), device=device)
            targets = seqs[:, 1:]

            lr = lr_at_step(step, total_steps, PEAK_LR_TERNARY, WARMUP_RATIO)
            for g in opt.param_groups:
                g["lr"] = lr
            logits = model(batch, prefix_lens, BATCH_SIZE, SEQ_LEN - 1)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, word_vocab_size), targets.reshape(-1))
            if not torch.isfinite(loss):
                print(f"    STEP {step}: loss {loss.item()} -- COLLAPSE")
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % max(1, total_steps // 5) == 0 or step == total_steps - 1:
                print(f"    step {step:5d}/{total_steps}  loss {loss.item():.4f}  ({time.time()-t0:.0f}s)")
        pretrain_loss = loss.item()
        print(f"  pretrain done in {time.time()-t0:.1f}s, final loss {pretrain_loss:.4f}")

        # --- fine-tune + eval (mirrors scripts/stage2_finetune_and_eval.py) ---
        t0 = time.time()
        ft2_finetune(model, mate_train_ex, "mate", "ternary", device, morph_index, None)
        print(f"  fine-tuned in {time.time()-t0:.1f}s")

        t0 = time.time()
        gen = generate_mate(model, eval_sources, morph_index, device)
        score = corpus_sari(eval_sources, gen, eval_refs)
        print(f"  SARI={score:.2f}  generation took {time.time()-t0:.1f}s")

        n_params = sum(p.numel() for p in model.parameters())
        results[ablation] = dict(sari=score, n_params=n_params, pretrain_loss=pretrain_loss)
        torch.save(model.state_dict(), OUT_DIR / f"checkpoint_ablation_{ablation}.pt")
        with open(OUT_DIR / f"generations_ablation_{ablation}.tsv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["source", "generated", "reference"])
            for s, gtext, r in zip(eval_sources, gen, [r[0] for r in eval_refs]):
                w.writerow([s, gtext, r])

        del model, opt
        torch.cuda.empty_cache()
        print(f"  ({time.time()-overall_t0:.0f}s elapsed total)")

    with open(OUT_DIR / "ablation_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ablation", "sari", "n_params", "pretrain_final_loss"])
        for ablation, r in results.items():
            w.writerow([ablation, f"{r['sari']:.4f}", r["n_params"], f"{r['pretrain_loss']:.4f}"])

    print("\n=== Stage 4 ablation summary ===")
    for ablation, r in results.items():
        print(f"{ablation:12s}: SARI={r['sari']:.2f}  params={r['n_params']:,}")
    print(f"\ntotal wall time: {(time.time()-overall_t0)/60:.1f} min")


if __name__ == "__main__":
    main()
