"""Stage 1: fine-tune all 4 pretrained cells on the real SAMER Corpus
(L5 -> L3 simplification), generate on the test split, and score with real
SARI (+ BERTScore where feasible). Emits Table 4-shaped numbers per cell.

Prefix-LM fine-tuning: sequence = [source tokens, SEP, target tokens, PAD...],
prefix_len = len(source)+1 (source+SEP get bidirectional attention per
build_prefix_lm_mask), loss computed only on target positions (ignore_index
on source/SEP/PAD). Right-padding is safe under the causal+prefix-bidirectional
mask without any extra masking: a real (non-pad) query position can only
attend to keys at or before it (causal) or within the prefix (bidirectional),
and padding always sits after the prefix+target, so it is never a valid key
for any real query. See results/NOTES.md.

Not one of the numbered scripts/00-05 — see results/NOTES.md, same reasoning
as scripts/stage0_tiny_train.py / stage1_pretrain_all_cells.py.

Usage: python -m scripts.stage1_finetune_and_eval
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.data.cache import read_cache
from src.data.mate_batch import MorphIndex
from src.eval.quality import corpus_sari
from src.model.arabit import AraBit
from src.model.mate import MATEConfig
from src.tokenization.vocab import load_vocabs
from src.train.schedule import lr_at_step

SAMER_DIR = Path("docs/samer-simplification-corpus-v1/data")
D_MODEL, N_LAYERS, N_HEADS, D_FF = 256, 4, 4, 1024
BPE_MAX_LEN = 64
MATE_MAX_LEN = 48
BATCH_SIZE = 16
FT_EPOCHS = 3
FT_LR_FP16 = 1e-4
FT_LR_TERNARY = 2e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.1
SEED = 0
MAX_NEW_TOKENS = 40
OUT_DIR = Path("results/stage1")


def load_samer_pairs(split: str) -> list[tuple[str, str]]:
    pairs = []
    with open(SAMER_DIR / f"{split}.tsv", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            l5, l3 = row["L5"].strip(), row["L3"].strip()
            if l5 and l3:
                pairs.append((l5, l3))
    return pairs


def make_bpe_examples(pairs, tokenizer: Tokenizer, max_len: int):
    """seq = [src, <bos>, tgt, <eos>, <pad>...]. <bos> marks "start of
    target" (bidirectional prefix boundary); <eos> is included in the loss
    region so the model learns to emit it, giving generation a real stopping
    signal (see this script's module docstring on why this matters)."""
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")
    examples = []
    for src, tgt in pairs:
        src_ids = tokenizer.encode(src).ids
        tgt_ids = tokenizer.encode(tgt).ids
        seq = src_ids + [bos_id] + tgt_ids + [eos_id]
        if len(seq) + 1 > max_len or not tgt_ids:  # +1 for the shift-by-one target
            continue
        prefix_len = len(src_ids) + 1
        pad_len = max_len - len(seq)
        seq = seq + [pad_id] * pad_len
        examples.append((seq, prefix_len, prefix_len + len(tgt_ids) + 1))  # +1 includes <eos>
    return examples


def make_mate_examples(pairs, word_vocab: dict[str, int], max_len: int):
    """words = [src_words, "<sep>", tgt_words, "<eos>", "<pad>"...] — mirrors
    make_bpe_examples' <bos>/<eos> roles at the word level."""
    examples = []
    for src, tgt in pairs:
        src_w = src.split()
        tgt_w = tgt.split()
        words = src_w + ["<sep>"] + tgt_w + ["<eos>"]
        if len(words) + 1 > max_len or not tgt_w:
            continue
        prefix_len = len(src_w) + 1
        pad_len = max_len - len(words)
        words = words + ["<pad>"] * pad_len
        examples.append((words, prefix_len, prefix_len + len(tgt_w) + 1))  # +1 includes <eos>
    return examples


def bpe_collate(batch_examples, device, pad_id):
    seqs = torch.tensor([e[0] for e in batch_examples], dtype=torch.long)
    prefix_lens = torch.tensor([e[1] for e in batch_examples], dtype=torch.long)
    tgt_end = [e[2] for e in batch_examples]

    input_ids = seqs[:, :-1].to(device)
    targets = seqs[:, 1:].clone()
    for i, (pl, te) in enumerate(zip(prefix_lens.tolist(), tgt_end)):
        mask = torch.ones(targets.shape[1], dtype=torch.bool)
        mask[pl - 1 : te - 1] = False  # keep loss only here (shifted index)
        targets[i][mask] = -100
    return dict(token_ids=input_ids), targets.to(device), prefix_lens.to(device)


def mate_collate(batch_examples, morph_index: MorphIndex, device):
    from src.data.mate_batch import build_mate_batch

    words_grid = [e[0] for e in batch_examples]
    prefix_lens = torch.tensor([e[1] for e in batch_examples], dtype=torch.long)
    tgt_end = [e[2] for e in batch_examples]
    seq_len = len(words_grid[0])

    input_words = [w for row in words_grid for w in row[:-1]]
    batch = build_mate_batch(input_words, morph_index, device)
    targets = torch.tensor(
        [[morph_index.word_id(w) for w in row[1:]] for row in words_grid], dtype=torch.long
    )
    for i, (pl, te) in enumerate(zip(prefix_lens.tolist(), tgt_end)):
        mask = torch.ones(targets.shape[1], dtype=torch.bool)
        mask[pl - 1 : te - 1] = False
        targets[i][mask] = -100
    return batch, targets.to(device), prefix_lens.to(device), seq_len - 1


def finetune_cell(model, examples, front_end, precision, device, morph_index=None, pad_id=None):
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
            batch_examples = [examples[i] for i in idx]

            if front_end == "bpe":
                batch, targets, prefix_lens = bpe_collate(batch_examples, device, pad_id)
                seq_len = batch["token_ids"].shape[1]
            else:
                batch, targets, prefix_lens, seq_len = mate_collate(batch_examples, morph_index, device)

            lr = lr_at_step(step, total_steps, peak_lr, WARMUP_RATIO)
            for g in opt.param_groups:
                g["lr"] = lr

            logits = model(batch, prefix_lens, len(batch_examples), seq_len)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        print(f"    epoch {epoch+1}/{FT_EPOCHS}  loss {loss.item():.4f}")
    print(f"    fine-tuned in {time.time()-t0:.1f}s ({total_steps} steps)")


@torch.no_grad()
def generate_bpe(model, sources, tokenizer, device, max_new_tokens=MAX_NEW_TOKENS):
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    outputs = []
    model.eval()
    for src in sources:
        src_ids = tokenizer.encode(src).ids
        seq = src_ids + [bos_id]
        prefix_len = torch.tensor([len(seq)], device=device)
        for _ in range(max_new_tokens):
            t = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(dict(token_ids=t), prefix_len, 1, t.shape[1])
            next_id = logits[0, -1].argmax().item()
            if next_id == eos_id:
                break
            seq.append(next_id)
        outputs.append(tokenizer.decode(seq[len(src_ids) + 1 :]))
    model.train()
    return outputs


@torch.no_grad()
def generate_mate(model, sources, morph_index, device, max_new_tokens=MAX_NEW_TOKENS):
    from src.data.mate_batch import build_mate_batch

    id_to_word = {i: w for w, i in morph_index.word_vocab.items()}
    outputs = []
    model.eval()
    for src in sources:
        words = src.split() + ["<sep>"]
        prefix_len = torch.tensor([len(words)], device=device)
        gen_words = []
        for _ in range(max_new_tokens):
            batch = build_mate_batch(words, morph_index, device)
            logits = model(batch, prefix_len, 1, len(words))
            next_id = logits[0, -1].argmax().item()
            next_word = id_to_word.get(next_id, "<unk>")
            if next_word == "<eos>":
                break
            gen_words.append(next_word)
            words = words + [next_word]
        outputs.append(" ".join(gen_words))
    model.train()
    return outputs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_pairs = load_samer_pairs("train")
    test_pairs = load_samer_pairs("test")
    print(f"SAMER train pairs: {len(train_pairs)}, test pairs: {len(test_pairs)}")

    control_tok = Tokenizer.from_file("data/stage1/bpe_control_tokenizer.json")
    fallback_tok = Tokenizer.from_file("data/stage1/bpe_tokenizer.json")
    vocabs = load_vocabs("data/stage1/vocabs.json")
    cache_df = read_cache("data/stage1/morph_cache.parquet")
    morph_index = MorphIndex(cache_df, vocabs)
    mate_cfg = MATEConfig(
        n_proclitic=len(vocabs["proclitic"]), n_enclitic=len(vocabs["enclitic"]),
        n_root=len(vocabs["root"]), n_pattern=len(vocabs["pattern"]),
        n_bpe=fallback_tok.get_vocab_size(),
        d_clitic=16, d_root=32, d_pattern=32, d_model=D_MODEL,
    )
    pad_id = control_tok.token_to_id("<pad>")

    bpe_train_ex = make_bpe_examples(train_pairs, control_tok, BPE_MAX_LEN)
    mate_train_ex = make_mate_examples(train_pairs, vocabs["word"], MATE_MAX_LEN)
    print(f"usable examples after length filtering: bpe={len(bpe_train_ex)}, mate={len(mate_train_ex)}")

    eval_sources = [s for s, _ in test_pairs][:200]  # cap for Stage 1 turnaround time
    eval_refs = [[t] for s, t in test_pairs][:200]
    print(f"eval set (capped): {len(eval_sources)} examples")

    results = {}
    for front_end, vocab_size, train_ex in [
        ("bpe", control_tok.get_vocab_size(), bpe_train_ex),
        ("mate", len(vocabs["word"]), mate_train_ex),
    ]:
        for precision in ("fp16", "ternary"):
            cell = f"{front_end}_{precision}"
            print(f"\n[{cell}] loading pretrained checkpoint + fine-tuning...")
            torch.manual_seed(SEED)
            model = AraBit(front_end, precision, vocab_size, D_MODEL, N_LAYERS, N_HEADS, D_FF, mate_cfg).to(device)
            state = torch.load(OUT_DIR / f"checkpoint_{cell}.pt", map_location=device)
            model.load_state_dict(state)

            finetune_cell(model, train_ex, front_end, precision, device, morph_index, pad_id)

            print(f"    generating on {len(eval_sources)} test sources...")
            t0 = time.time()
            if front_end == "bpe":
                gen = generate_bpe(model, eval_sources, control_tok, device)
            else:
                gen = generate_mate(model, eval_sources, morph_index, device)
            gen_time = time.time() - t0

            score = corpus_sari(eval_sources, gen, eval_refs)
            print(f"    SARI={score:.2f}  generation took {gen_time:.1f}s")
            results[cell] = dict(sari=score, gen_time_s=gen_time, n_eval=len(eval_sources))

            torch.save(model.state_dict(), OUT_DIR / f"checkpoint_ft_{cell}.pt")
            with open(OUT_DIR / f"generations_{cell}.tsv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["source", "generated", "reference"])
                for s, g, r in zip(eval_sources, gen, [r[0] for r in eval_refs]):
                    w.writerow([s, g, r])

    with open(OUT_DIR / "quality_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "sari", "n_eval", "gen_time_s"])
        for cell, r in results.items():
            w.writerow([cell, f"{r['sari']:.4f}", r["n_eval"], f"{r['gen_time_s']:.1f}"])

    print("\n=== Stage 1 fine-tune + eval summary (real SAMER L5->L3) ===")
    for cell, r in results.items():
        print(f"{cell:14s}: SARI={r['sari']:.2f}  n={r['n_eval']}  gen_time={r['gen_time_s']:.1f}s")


if __name__ == "__main__":
    main()
