"""Stage 2: fine-tune all 12 pretrained checkpoints (4 cells x 3 seeds) on
the real SAMER Corpus, generate on the test split, score with real SARI,
and report mean +/- std with a paired bootstrap significance test between
AraBit-1.58 (mate, ternary) and the ternary BPE control (B3), per
docs/AraBit-1.58_IMPLEMENTATION.md sec:0 / sec:3.6.

Same prefix-LM fine-tuning scheme as scripts/stage1_finetune_and_eval.py —
see that script's module docstring for why right-padding is safe under the
prefix-LM mask without extra masking.

Usage: python -m scripts.stage2_finetune_and_eval
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.data.cache import read_cache
from src.data.mate_batch import MorphIndex
from src.eval.quality import corpus_sari, sari
from src.model.arabit import AraBit
from src.model.mate import MATEConfig
from src.tokenization.vocab import load_vocabs
from src.train.schedule import lr_at_step

SAMER_DIR = Path("docs/samer-simplification-corpus-v1/data")
DATA_DIR = Path("data/stage2")
IN_DIR = Path("results/stage2")
OUT_DIR = Path("results/stage2")
D_MODEL, N_LAYERS, N_HEADS, D_FF = 512, 12, 8, 2048
BPE_MAX_LEN = 80
MATE_MAX_LEN = 64
BATCH_SIZE = 8
FT_EPOCHS = 3
FT_LR_FP16 = 1e-4
FT_LR_TERNARY = 2e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.1
MAX_NEW_TOKENS = 40
EVAL_CAP = 200
SEEDS = [0, 1, 2]
N_BOOTSTRAP = 1000


def load_samer_pairs(split: str) -> list[tuple[str, str]]:
    pairs = []
    with open(SAMER_DIR / f"{split}.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            l5, l3 = row["L5"].strip(), row["L3"].strip()
            if l5 and l3:
                pairs.append((l5, l3))
    return pairs


def make_bpe_examples(pairs, tokenizer: Tokenizer, max_len: int):
    bos_id, eos_id, pad_id = (tokenizer.token_to_id(t) for t in ("<bos>", "<eos>", "<pad>"))
    examples = []
    for src, tgt in pairs:
        src_ids, tgt_ids = tokenizer.encode(src).ids, tokenizer.encode(tgt).ids
        seq = src_ids + [bos_id] + tgt_ids + [eos_id]
        if len(seq) + 1 > max_len or not tgt_ids:
            continue
        prefix_len = len(src_ids) + 1
        seq = seq + [pad_id] * (max_len - len(seq))
        examples.append((seq, prefix_len, prefix_len + len(tgt_ids) + 1))
    return examples


def make_mate_examples(pairs, max_len: int):
    examples = []
    for src, tgt in pairs:
        src_w, tgt_w = src.split(), tgt.split()
        words = src_w + ["<sep>"] + tgt_w + ["<eos>"]
        if len(words) + 1 > max_len or not tgt_w:
            continue
        prefix_len = len(src_w) + 1
        words = words + ["<pad>"] * (max_len - len(words))
        examples.append((words, prefix_len, prefix_len + len(tgt_w) + 1))
    return examples


def bpe_collate(batch_examples, device, pad_id):
    seqs = torch.tensor([e[0] for e in batch_examples], dtype=torch.long)
    prefix_lens = torch.tensor([e[1] for e in batch_examples], dtype=torch.long)
    input_ids = seqs[:, :-1].to(device)
    targets = seqs[:, 1:].clone()
    for i, (pl, te) in enumerate(zip(prefix_lens.tolist(), [e[2] for e in batch_examples])):
        mask = torch.ones(targets.shape[1], dtype=torch.bool)
        mask[pl - 1 : te - 1] = False
        targets[i][mask] = -100
    return dict(token_ids=input_ids), targets.to(device), prefix_lens.to(device)


def mate_collate(batch_examples, morph_index: MorphIndex, device):
    from src.data.mate_batch import build_mate_batch

    words_grid = [e[0] for e in batch_examples]
    prefix_lens = torch.tensor([e[1] for e in batch_examples], dtype=torch.long)
    seq_len = len(words_grid[0])
    input_words = [w for row in words_grid for w in row[:-1]]
    batch = build_mate_batch(input_words, morph_index, device)
    targets = torch.tensor(
        [[morph_index.word_id(w) for w in row[1:]] for row in words_grid], dtype=torch.long
    )
    for i, (pl, te) in enumerate(zip(prefix_lens.tolist(), [e[2] for e in batch_examples])):
        mask = torch.ones(targets.shape[1], dtype=torch.bool)
        mask[pl - 1 : te - 1] = False
        targets[i][mask] = -100
    return batch, targets.to(device), prefix_lens.to(device), seq_len - 1


def finetune(model, examples, front_end, precision, device, morph_index=None, pad_id=None):
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
            if front_end == "bpe":
                batch, targets, prefix_lens = bpe_collate(batch_ex, device, pad_id)
                seq_len = batch["token_ids"].shape[1]
            else:
                batch, targets, prefix_lens, seq_len = mate_collate(batch_ex, morph_index, device)
            lr = lr_at_step(step, total_steps, peak_lr, WARMUP_RATIO)
            for g in opt.param_groups:
                g["lr"] = lr
            logits = model(batch, prefix_lens, len(batch_ex), seq_len)
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
    bos_id, eos_id = tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")
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


def paired_bootstrap_sari(sources, gen_a, gen_b, refs, n_bootstrap=N_BOOTSTRAP, seed=0):
    """P(mean SARI(a) - mean SARI(b) <= 0 under resampling), i.e. a one-sided
    test of whether a's SARI is significantly greater than b's."""
    rng = random.Random(seed)
    n = len(sources)
    sari_a = [sari(sources[i], gen_a[i], refs[i]) for i in range(n)]
    sari_b = [sari(sources[i], gen_b[i], refs[i]) for i in range(n)]
    observed_diff = sum(sari_a) / n - sum(sari_b) / n
    count_le_zero = 0
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        diff = sum(sari_a[i] for i in idx) / n - sum(sari_b[i] for i in idx) / n
        if diff <= 0:
            count_le_zero += 1
    p_value = count_le_zero / n_bootstrap
    return observed_diff, p_value


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
    pad_id = control_tok.token_to_id("<pad>")

    bpe_train_ex = make_bpe_examples(train_pairs, control_tok, BPE_MAX_LEN)
    mate_train_ex = make_mate_examples(train_pairs, MATE_MAX_LEN)
    print(f"usable examples: bpe={len(bpe_train_ex)}, mate={len(mate_train_ex)}")

    eval_sources = [s for s, _ in test_pairs][:EVAL_CAP]
    eval_refs = [[t] for _, t in test_pairs][:EVAL_CAP]
    print(f"eval set (capped): {len(eval_sources)} examples")

    all_generations = {}
    per_seed_sari = {}
    for front_end, vocab_size, train_ex in [
        ("bpe", control_tok.get_vocab_size(), bpe_train_ex),
        ("mate", len(vocabs["word"]), mate_train_ex),
    ]:
        for precision in ("fp16", "ternary"):
            cell = f"{front_end}_{precision}"
            per_seed_sari[cell] = []
            for seed in SEEDS:
                cell_seed = f"{cell}_seed{seed}"
                print(f"\n[{cell_seed}] loading + fine-tuning...")
                torch.manual_seed(seed)
                model = AraBit(front_end, precision, vocab_size, D_MODEL, N_LAYERS, N_HEADS, D_FF, mate_cfg).to(device)
                state = torch.load(IN_DIR / f"checkpoint_{cell_seed}.pt", map_location=device, weights_only=True)
                model.load_state_dict(state)

                finetune(model, train_ex, front_end, precision, device, morph_index, pad_id)

                t0 = time.time()
                if front_end == "bpe":
                    gen = generate_bpe(model, eval_sources, control_tok, device)
                else:
                    gen = generate_mate(model, eval_sources, morph_index, device)
                gen_time = time.time() - t0

                score = corpus_sari(eval_sources, gen, eval_refs)
                print(f"    SARI={score:.2f}  generation took {gen_time:.1f}s")
                per_seed_sari[cell].append(score)
                all_generations[cell_seed] = gen

                torch.save(model.state_dict(), OUT_DIR / f"checkpoint_ft_{cell_seed}.pt")
                with open(OUT_DIR / f"generations_{cell_seed}.tsv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(["source", "generated", "reference"])
                    for s, g, r in zip(eval_sources, gen, [r[0] for r in eval_refs]):
                        w.writerow([s, g, r])
                del model
                torch.cuda.empty_cache()

    import statistics

    with open(OUT_DIR / "quality_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "sari_mean", "sari_std", "sari_seeds", "n_eval"])
        for cell, scores in per_seed_sari.items():
            mean = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            w.writerow([cell, f"{mean:.4f}", f"{std:.4f}", ";".join(f"{s:.4f}" for s in scores), len(eval_sources)])

    print("\n=== Stage 2 fine-tune + eval summary (mean +/- std over 3 seeds) ===")
    for cell, scores in per_seed_sari.items():
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(f"{cell:14s}: SARI = {mean:.2f} +/- {std:.2f}  (seeds: {[f'{s:.2f}' for s in scores]})")

    # Paired bootstrap significance: AraBit-1.58 (mate_ternary) vs B3 (bpe_ternary), seed0 pairing
    print("\n=== Paired bootstrap significance (seed 0, AraBit-1.58 vs B3 ternary BPE) ===")
    gen_mate_t = all_generations["mate_ternary_seed0"]
    gen_bpe_t = all_generations["bpe_ternary_seed0"]
    diff, p = paired_bootstrap_sari(eval_sources, gen_mate_t, gen_bpe_t, eval_refs)
    print(f"SARI(mate_ternary) - SARI(bpe_ternary) = {diff:+.2f}, bootstrap p(diff<=0) = {p:.4f}")
    with open(OUT_DIR / "bootstrap_significance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "diff", "p_value_diff_le_0", "n_bootstrap"])
        w.writerow(["mate_ternary_seed0 - bpe_ternary_seed0", f"{diff:.4f}", f"{p:.4f}", N_BOOTSTRAP])


if __name__ == "__main__":
    main()
