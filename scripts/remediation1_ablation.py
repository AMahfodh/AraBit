"""Remediation Phase 1.2.3: re-run Table 7's component-level MATE ablation
(no_pattern, no_root, no_clitics, hard_gate) with the output-head fix —
shared 13,699-entry BPE vocab, BPE-granularity sequences, single seed
(seed0), ternary precision, Stage 2 scale — mirroring
scripts/remediation1_pretrain_mate.py + remediation1_finetune_eval_mate.py.

"Full MATE" and "- all (plain BPE)" reuse the corrected mate_ternary
(results/stage2_corrected) and the existing bpe_ternary
(results/stage2) respectively — neither needs retraining.

Usage: python -m scripts.remediation1_ablation
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from scripts.remediation1_finetune_eval_mate import finetune, generate, make_mate_examples
from scripts.remediation1_pretrain_mate import build_mate_word_sequences
from scripts.stage2_finetune_and_eval import load_samer_pairs
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
    build_bpe_sequences,
)
from src.data.cache import read_cache
from src.data.corpus import stream_sentences
from src.data.mate_batch import MorphIndex, build_mate_batch
from src.eval.quality import corpus_sari
from src.model.arabit import AraBit
from src.model.mate import MATEConfig
from src.tokenization.vocab import load_vocabs
from src.train.schedule import lr_at_step

DATA_DIR = Path("data/stage2")
OUT_DIR = Path("results/stage4_corrected")
SEED = 0
EVAL_CAP = 200
MAX_LEN = 96
ABLATIONS = ["no_pattern", "no_root", "no_clitics", "hard_gate"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"streaming {ARTICLE_LIMIT} articles...")
    sentences = list(stream_sentences(article_limit=ARTICLE_LIMIT))
    fallback_tok = Tokenizer.from_file(str(DATA_DIR / "bpe_tokenizer.json"))
    control_tok = Tokenizer.from_file(str(DATA_DIR / "bpe_control_tokenizer.json"))
    vocabs = load_vocabs(DATA_DIR / "vocabs.json")
    cache_df = read_cache(DATA_DIR / "morph_cache.parquet")
    morph_index = MorphIndex(cache_df, vocabs)

    bpe_sequences = build_bpe_sequences(sentences, control_tok)
    words_grid = build_mate_word_sequences(sentences, control_tok)
    assert len(words_grid) == bpe_sequences.shape[0]
    vocab_size = control_tok.get_vocab_size()
    print(f"shared sequences: {bpe_sequences.shape}")

    train_pairs = load_samer_pairs("train")
    test_pairs = load_samer_pairs("test")
    eval_sources = [s for s, _ in test_pairs][:EVAL_CAP]
    eval_refs = [[t] for _, t in test_pairs][:EVAL_CAP]
    train_ex = make_mate_examples(train_pairs, control_tok, MAX_LEN)

    def mate_step_fn(idx):
        seqs = bpe_sequences[idx].to(device)
        input_words = [w for i in idx.tolist() for w in words_grid[i][:-1]]
        batch = build_mate_batch(input_words, morph_index, device)
        prefix_lens = torch.randint(0, (SEQ_LEN - 1) // 2, (BATCH_SIZE,), device=device)
        return batch, seqs[:, 1:], prefix_lens

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

        # --- pretrain ---
        torch.manual_seed(SEED)
        model = AraBit("mate", "ternary", vocab_size, D_MODEL, N_LAYERS, N_HEADS, D_FF, mate_cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR_TERNARY, weight_decay=WEIGHT_DECAY)
        n_seq = bpe_sequences.shape[0]
        total_steps = min(TARGET_TOKENS_PER_SEED // (BATCH_SIZE * SEQ_LEN), n_seq // BATCH_SIZE)
        torch.manual_seed(SEED)
        perm = torch.randperm(n_seq)
        print(f"  pretraining: {total_steps} steps")
        t0 = time.time()
        for step in range(total_steps):
            idx = perm[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
            batch, targets, prefix_lens = mate_step_fn(idx)
            lr = lr_at_step(step, total_steps, PEAK_LR_TERNARY, WARMUP_RATIO)
            for g in opt.param_groups:
                g["lr"] = lr
            logits = model(batch, prefix_lens, BATCH_SIZE, SEQ_LEN - 1)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
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

        # --- fine-tune + eval ---
        t0 = time.time()
        finetune(model, train_ex, "ternary", device, morph_index, vocab_size)
        print(f"  fine-tuned in {time.time()-t0:.1f}s")

        t0 = time.time()
        gen = generate(model, eval_sources, control_tok, fallback_tok, morph_index, device)
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

    print("\n=== Remediation Phase 1 ablation summary ===")
    for ablation, r in results.items():
        print(f"{ablation:12s}: SARI={r['sari']:.2f}  params={r['n_params']:,}")
    print(f"\ntotal wall time: {(time.time()-overall_t0)/60:.1f} min")


if __name__ == "__main__":
    main()
