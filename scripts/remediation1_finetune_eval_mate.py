"""Remediation Phase 1.2.2: fine-tune + evaluate the 6 corrected
mate_fp16/mate_ternary checkpoints on real SAMER (L5 -> L3), same protocol
as scripts/stage2_finetune_and_eval.py but with the output-head fix (shared
13,699-entry BPE vocab, BPE-granularity sequences — see
scripts/remediation1_pretrain_mate.py's module docstring and
results/NOTES.md "Remediation Phase 1").

Generation now proceeds BPE-token-by-BPE-token. Source tokens have known
words (real morphology lookup via words_per_bpe_position); already-
generated continuation tokens don't have a known word boundary, so they're
fed back through MorphIndex.lookup_bpe_only (see src/data/mate_batch.py) -
the same E_fb-fallback path unanalysable real words already use, applied
honestly to "no word is knowable yet" rather than guessed at.

BPE cells are unaffected and not re-run or re-evaluated (Phase 1.0).

Usage: python -m scripts.remediation1_finetune_eval_mate
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from scripts.stage2_finetune_and_eval import load_samer_pairs
from src.data.cache import read_cache
from src.data.mate_batch import MorphIndex, build_mate_batch, words_per_bpe_position
from src.eval.quality import corpus_sari
from src.model.arabit import AraBit
from src.model.mate import MATEConfig
from src.tokenization.vocab import load_vocabs
from src.train.schedule import lr_at_step

DATA_DIR = Path("data/stage2")
IN_DIR = Path("results/stage2_corrected")
OUT_DIR = Path("results/stage2_corrected")
D_MODEL, N_LAYERS, N_HEADS, D_FF = 512, 12, 8, 2048
MAX_LEN = 96  # BPE-granularity, so longer than the old word-granularity MATE_MAX_LEN=64
BATCH_SIZE = 8
FT_EPOCHS = 3
FT_LR_FP16 = 1e-4
FT_LR_TERNARY = 2e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.1
MAX_NEW_TOKENS = 60  # BPE tokens, so more than the old word-granularity MAX_NEW_TOKENS=40
EVAL_CAP = 200
SEEDS = [0, 1, 2]


def make_mate_examples(pairs, tokenizer: Tokenizer, max_len: int):
    """Mirrors scripts/stage2_finetune_and_eval.py: make_bpe_examples exactly
    (same seq = [src_ids, bos, tgt_ids, eos, pad...] structure, same
    tokenizer), plus a parallel words-per-position list for MATE's
    embedding. Source and target words are BOTH real here (teacher forcing
    - ground truth is known), unlike generation."""
    bos_id, eos_id, pad_id = (tokenizer.token_to_id(t) for t in ("<bos>", "<eos>", "<pad>"))
    examples = []
    for src, tgt in pairs:
        src_ids = tokenizer.encode(src).ids
        tgt_ids = tokenizer.encode(tgt).ids
        seq = src_ids + [bos_id] + tgt_ids + [eos_id]
        if len(seq) + 1 > max_len or not tgt_ids:
            continue
        prefix_len = len(src_ids) + 1
        words = words_per_bpe_position(src, tokenizer) + ["<bos>"] + words_per_bpe_position(tgt, tokenizer) + ["<eos>"]
        assert len(words) == len(seq), "word-per-position must align 1:1 with the BPE id sequence"
        pad_len = max_len - len(seq)
        seq = seq + [pad_id] * pad_len
        words = words + ["<pad>"] * pad_len
        examples.append((seq, words, prefix_len, prefix_len + len(tgt_ids) + 1))
    return examples


def mate_collate(batch_examples, morph_index, device, vocab_size):
    seqs = torch.tensor([e[0] for e in batch_examples], dtype=torch.long)
    words_grid = [e[1] for e in batch_examples]
    prefix_lens = torch.tensor([e[2] for e in batch_examples], dtype=torch.long)
    seq_len = seqs.shape[1] - 1

    input_items = [w for row in words_grid for w in row[:-1]]
    batch = build_mate_batch(input_items, morph_index, device)
    targets = seqs[:, 1:].clone()
    for i, (pl, te) in enumerate(zip(prefix_lens.tolist(), [e[3] for e in batch_examples])):
        mask = torch.ones(targets.shape[1], dtype=torch.bool)
        mask[pl - 1 : te - 1] = False
        targets[i][mask] = -100
    return batch, targets.to(device), prefix_lens.to(device), seq_len


def finetune(model, examples, precision, device, morph_index, vocab_size):
    peak_lr = FT_LR_TERNARY if precision == "ternary" else FT_LR_FP16
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=WEIGHT_DECAY)
    n = len(examples)
    steps_per_epoch = n // BATCH_SIZE
    total_steps = steps_per_epoch * FT_EPOCHS
    step = 0
    t0 = time.time()
    for epoch in range(FT_EPOCHS):
        perm = torch.randperm(n)
        for b in range(steps_per_epoch):
            idx = perm[b * BATCH_SIZE : (b + 1) * BATCH_SIZE].tolist()
            batch_ex = [examples[i] for i in idx]
            batch, targets, prefix_lens, seq_len = mate_collate(batch_ex, morph_index, device, vocab_size)
            lr = lr_at_step(step, total_steps, peak_lr, WARMUP_RATIO)
            for g in opt.param_groups:
                g["lr"] = lr
            logits = model(batch, prefix_lens, len(batch_ex), seq_len)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size), targets.reshape(-1), ignore_index=-100
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        print(f"    epoch {epoch+1}/{FT_EPOCHS}  loss {loss.item():.4f}")
    print(f"    fine-tuned in {time.time()-t0:.1f}s ({total_steps} steps)")


@torch.no_grad()
def generate(model, sources, tokenizer, fallback_tokenizer, morph_index, device, max_new_tokens=MAX_NEW_TOKENS):
    """`tokenizer` is the *control* BPE tokenizer (shared output vocab,
    13,699). `fallback_tokenizer` is MATE's own E_fb fallback tokenizer
    (8,000) - a DIFFERENT vocabulary. Every generated token must be
    decoded via `tokenizer` then re-encoded via `fallback_tokenizer`
    before being used for MATE's embedding of that position
    (MorphIndex.lookup_bpe_only) - see that method's docstring for the
    CUDA crash this fixes."""
    bos_id, eos_id = tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")
    outputs = []
    model.eval()
    for src in sources:
        src_ids = tokenizer.encode(src).ids
        src_words = words_per_bpe_position(src, tokenizer)
        assert len(src_words) == len(src_ids)
        items = list(src_words) + ["<bos>"]  # known words for the prefix
        seq_ids = src_ids + [bos_id]
        prefix_len = torch.tensor([len(seq_ids)], device=device)
        for _ in range(max_new_tokens):
            batch = build_mate_batch(items, morph_index, device)
            logits = model(batch, prefix_len, 1, len(items))
            next_id = logits[0, -1].argmax().item()
            if next_id == eos_id:
                break
            seq_ids.append(next_id)
            piece_text = tokenizer.decode([next_id])
            fallback_ids = fallback_tokenizer.encode(piece_text).ids if piece_text.strip() else []
            items.append(fallback_ids)  # generated token: word unknown, routed via lookup_bpe_only
        outputs.append(tokenizer.decode(seq_ids[len(src_ids) + 1 :]))
    model.train()
    return outputs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_pairs = load_samer_pairs("train")
    test_pairs = load_samer_pairs("test")
    print(f"SAMER train pairs: {len(train_pairs)}, test pairs: {len(test_pairs)}")

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
    vocab_size = control_tok.get_vocab_size()

    train_ex = make_mate_examples(train_pairs, control_tok, MAX_LEN)
    print(f"usable train examples: {len(train_ex)}")

    eval_sources = [s for s, _ in test_pairs][:EVAL_CAP]
    eval_refs = [[t] for _, t in test_pairs][:EVAL_CAP]

    per_seed_sari = {}
    for precision in ("fp16", "ternary"):
        cell = f"mate_{precision}"
        per_seed_sari[cell] = []
        for seed in SEEDS:
            cell_seed = f"{cell}_seed{seed}"
            print(f"\n[{cell_seed}] loading corrected pretrained checkpoint + fine-tuning...")
            torch.manual_seed(seed)
            model = AraBit("mate", precision, vocab_size, D_MODEL, N_LAYERS, N_HEADS, D_FF, mate_cfg).to(device)
            state = torch.load(IN_DIR / f"checkpoint_{cell_seed}.pt", map_location=device, weights_only=True)
            model.load_state_dict(state)

            finetune(model, train_ex, precision, device, morph_index, vocab_size)

            t0 = time.time()
            gen = generate(model, eval_sources, control_tok, fallback_tok, morph_index, device)
            gen_time = time.time() - t0
            score = corpus_sari(eval_sources, gen, eval_refs)
            print(f"    SARI={score:.2f}  generation took {gen_time:.1f}s")
            per_seed_sari[cell].append(score)

            torch.save(model.state_dict(), OUT_DIR / f"checkpoint_ft_{cell_seed}.pt")
            with open(OUT_DIR / f"generations_{cell_seed}.tsv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["source", "generated", "reference"])
                for s, g, r in zip(eval_sources, gen, [r[0] for r in eval_refs]):
                    w.writerow([s, g, r])
            del model
            torch.cuda.empty_cache()

    with open(OUT_DIR / "quality_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "sari_mean", "sari_std", "sari_seeds", "n_eval"])
        for cell, scores in per_seed_sari.items():
            mean = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            w.writerow([cell, f"{mean:.4f}", f"{std:.4f}", ";".join(f"{s:.4f}" for s in scores), len(eval_sources)])

    print("\n=== Remediation Phase 1 mate fine-tune + eval summary (mean +/- std, 3 seeds) ===")
    for cell, scores in per_seed_sari.items():
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(f"{cell:14s}: SARI = {mean:.2f} +/- {std:.2f}  (seeds: {[f'{s:.2f}' for s in scores]})")


if __name__ == "__main__":
    main()
