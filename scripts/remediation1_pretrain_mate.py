"""Remediation Phase 1.2.1: re-pretrain mate_fp16/mate_ternary (3 seeds each,
6 runs) with the output-head confound fixed — see results/NOTES.md "output-
head confound, confirmed" and the Phase 1 remediation plan.

Fix: MATE cells now predict over the SAME 13,699-entry BPE vocabulary as the
BPE cells, over the SAME BPE-tokenized target sequences (byte-identical
tokenization, reusing scripts/stage2_pretrain_all_cells.py:
build_bpe_sequences with the same control_tok). MATE's *input* embedding
stays word-level (one morphological embedding per whitespace word, per
IMPLEMENTATION.md sec:3.2); each BPE subword position of a word gets a
COPY of that word's embedding (src/data/mate_batch.py:
words_per_bpe_position). This is an input-representation-only change — MATE
no longer shortens the sequence or narrows the output space relative to BPE;
those were confounds, not part of the actual hypothesis under test.

BPE cells (bpe_fp16/bpe_ternary) are NOT re-run: Phase 1.0 confirmed
`matched_vocab_size()` is input-side only and unaffected by this fix (old
13699, recomputed 13699, unchanged).

Old (confounded) results are NOT overwritten - written to
results/stage2_corrected/ alongside the original results/stage2/.

Usage: python -m scripts.remediation1_pretrain_mate
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from scripts.stage2_pretrain_all_cells import (
    ARTICLE_LIMIT,
    BATCH_SIZE,
    D_FF,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    PEAK_LR_FP16,
    PEAK_LR_TERNARY,
    SEQ_LEN,
    TARGET_TOKENS_PER_SEED,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    build_bpe_sequences,
)
from src.data.cache import read_cache
from src.data.corpus import stream_sentences
from src.data.mate_batch import MorphIndex, build_mate_batch, words_per_bpe_position
from src.model.arabit import AraBit
from src.model.mate import MATEConfig
from src.tokenization.vocab import load_vocabs
from src.train.schedule import lr_at_step

DATA_DIR = Path("data/stage2")
OUT_DIR = Path("results/stage2_corrected")
SEEDS = [0, 1, 2]


def build_mate_word_sequences(sentences: list[str], tokenizer: Tokenizer):
    """Parallel to build_bpe_sequences, but tracks the source word for every
    BPE position instead of (or alongside) the BPE id. Byte-identical
    tokenization to build_bpe_sequences (verified: whole-sentence encoding ==
    word-by-word concat for this tokenizer, see words_per_bpe_position's
    docstring) - the id sequence itself is unused here since we reuse the
    already-built bpe_sequences tensor directly for targets.
    """
    words: list[str] = []
    for s in sentences:
        words.append("<bos>")
        words.extend(words_per_bpe_position(s, tokenizer))
        words.append("<eos>")
    n_seq = len(words) // SEQ_LEN
    words = words[: n_seq * SEQ_LEN]
    return [words[i * SEQ_LEN : (i + 1) * SEQ_LEN] for i in range(n_seq)]


def train_cell_seed(precision, seed, vocab_size, device, mate_cfg, n_sequences, step_fn):
    torch.manual_seed(seed)
    model = AraBit("mate", precision, vocab_size, D_MODEL, N_LAYERS, N_HEADS, D_FF, mate_cfg).to(device)
    peak_lr = PEAK_LR_TERNARY if precision == "ternary" else PEAK_LR_FP16
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=WEIGHT_DECAY)

    target_steps = TARGET_TOKENS_PER_SEED // (BATCH_SIZE * SEQ_LEN)
    steps_available = n_sequences // BATCH_SIZE
    total_steps = min(target_steps, steps_available)
    cell = f"mate_{precision}_seed{seed}"
    print(f"\n[{cell}] {total_steps} steps (target={target_steps}, available={steps_available}), "
          f"peak_lr={peak_lr:.2e}, params={sum(p.numel() for p in model.parameters()):,}")

    torch.manual_seed(seed)
    perm = torch.randperm(n_sequences)
    curve = []
    t0 = time.time()
    for step in range(total_steps):
        idx = perm[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        batch, targets, prefix_lens = step_fn(idx)

        lr = lr_at_step(step, total_steps, peak_lr, WARMUP_RATIO)
        for g in opt.param_groups:
            g["lr"] = lr

        logits = model(batch, prefix_lens, BATCH_SIZE, SEQ_LEN - 1)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        if not torch.isfinite(loss):
            print(f"  STEP {step}: loss is {loss.item()} -- COLLAPSE, aborting")
            curve.append((step, float("nan")))
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append((step, loss.item()))
        if step % max(1, total_steps // 10) == 0 or step == total_steps - 1:
            print(f"  step {step:5d}/{total_steps}  loss {loss.item():.4f}  lr {lr:.2e}  ({time.time()-t0:.0f}s)")

    print(f"[{cell}] finished in {time.time()-t0:.1f}s")
    return curve, model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"streaming {ARTICLE_LIMIT} articles...")
    sentences = list(stream_sentences(article_limit=ARTICLE_LIMIT))
    print(f"{len(sentences):,} sentences")

    fallback_tok = Tokenizer.from_file(str(DATA_DIR / "bpe_tokenizer.json"))
    control_tok = Tokenizer.from_file(str(DATA_DIR / "bpe_control_tokenizer.json"))
    vocabs = load_vocabs(DATA_DIR / "vocabs.json")
    cache_df = read_cache(DATA_DIR / "morph_cache.parquet")
    morph_index = MorphIndex(cache_df, vocabs)

    bpe_sequences = build_bpe_sequences(sentences, control_tok)  # SAME as BPE cells, byte-identical
    words_grid = build_mate_word_sequences(sentences, control_tok)
    assert len(words_grid) == bpe_sequences.shape[0], "MATE word grid and BPE id grid must align 1:1"
    print(f"shared bpe-granularity sequences: {bpe_sequences.shape} "
          f"({bpe_sequences.numel():,} tokens) - now identical shape for mate and bpe cells")

    mate_cfg = MATEConfig(
        n_proclitic=len(vocabs["proclitic"]), n_enclitic=len(vocabs["enclitic"]),
        n_root=len(vocabs["root"]), n_pattern=len(vocabs["pattern"]),
        n_bpe=fallback_tok.get_vocab_size(),
        d_clitic=24, d_root=48, d_pattern=48, d_model=D_MODEL,
    )
    vocab_size = control_tok.get_vocab_size()  # SAME as bpe cells now: 13699

    def mate_step_fn(idx):
        seqs = bpe_sequences[idx].to(device)
        input_words = [w for i in idx.tolist() for w in words_grid[i][:-1]]
        batch = build_mate_batch(input_words, morph_index, device)
        prefix_lens = torch.randint(0, (SEQ_LEN - 1) // 2, (BATCH_SIZE,), device=device)
        return batch, seqs[:, 1:], prefix_lens

    overall_t0 = time.time()
    results = {}
    for precision in ("fp16", "ternary"):
        for seed in SEEDS:
            curve, model = train_cell_seed(
                precision, seed, vocab_size, device, mate_cfg, bpe_sequences.shape[0], mate_step_fn
            )
            cell = f"mate_{precision}_seed{seed}"
            results[cell] = curve
            torch.save(model.state_dict(), OUT_DIR / f"checkpoint_{cell}.pt")
            with open(OUT_DIR / f"pretrain_loss_{cell}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "loss"])
                w.writerows(curve)
            del model
            torch.cuda.empty_cache()
            print(f"  ({time.time()-overall_t0:.0f}s elapsed total)")

    print("\n=== Remediation Phase 1 mate pretraining summary ===")
    import math
    for cell, curve in results.items():
        finite = [l for _, l in curve if math.isfinite(l)]
        print(f"{cell:20s}: start={finite[0]:.4f} end={finite[-1]:.4f} "
              f"descended={finite[-1] < finite[0]} steps={len(curve)}")
    print(f"\ntotal wall time: {(time.time()-overall_t0)/60:.1f} min")


if __name__ == "__main__":
    main()
