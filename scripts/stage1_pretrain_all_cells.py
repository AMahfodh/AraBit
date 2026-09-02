"""Stage 1: full 2x2 (front_end x precision) on the tiny model with the
small (300-article) corpus, per docs/AraBit-1.58_IMPLEMENTATION.md sec:4
Stage 1 ("confirm the whole harness runs end to end... absolute numbers
will be poor").

Not one of the numbered scripts/00-05 (see results/NOTES.md, same reasoning
as scripts/stage0_tiny_train.py) — Stage 1's pretraining is still a
feasibility/harness check, not the general Hydra multi-seed driver
scripts/02_pretrain.py is meant to become for Stage 2.

Usage: python -m scripts.stage1_pretrain_all_cells
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.data.cache import read_cache
from src.data.corpus import stream_sentences
from src.data.mate_batch import MorphIndex, build_mate_batch
from src.model.arabit import AraBit
from src.model.mate import MATEConfig
from src.tokenization.vocab import load_vocabs
from src.train.schedule import lr_at_step

ARTICLE_LIMIT = 300
SEQ_LEN = 128
BATCH_SIZE = 16
D_MODEL, N_LAYERS, N_HEADS, D_FF = 256, 4, 4, 1024
PEAK_LR_FP16 = 3e-4
PEAK_LR_TERNARY = 2 * PEAK_LR_FP16
WARMUP_RATIO = 0.02
WEIGHT_DECAY = 0.1
SEED = 0
OUT_DIR = Path("results/stage1")


def build_bpe_sequences(sentences: list[str], tokenizer: Tokenizer) -> torch.Tensor:
    bos, eos = tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")
    ids: list[int] = []
    for s in sentences:
        ids.append(bos)
        ids.extend(tokenizer.encode(s).ids)
        ids.append(eos)
    n_seq = len(ids) // SEQ_LEN
    ids = ids[: n_seq * SEQ_LEN]
    return torch.tensor(ids, dtype=torch.long).view(n_seq, SEQ_LEN)


def build_word_sequences(sentences: list[str], word_vocab: dict[str, int]):
    words_flat: list[str] = []
    for s in sentences:
        words_flat.extend(s.split())
    n_seq = len(words_flat) // SEQ_LEN
    words_flat = words_flat[: n_seq * SEQ_LEN]
    word_ids = torch.tensor([word_vocab.get(w, 0) for w in words_flat], dtype=torch.long).view(n_seq, SEQ_LEN)
    words_grid = [words_flat[i * SEQ_LEN : (i + 1) * SEQ_LEN] for i in range(n_seq)]
    return word_ids, words_grid


def train_cell(
    front_end: str,
    precision: str,
    vocab_size: int,
    device: torch.device,
    mate_cfg: MATEConfig | None,
    n_sequences: int,
    step_fn,  # (perm_idx_for_step) -> (batch_dict, targets)
) -> tuple[list[tuple[int, float]], AraBit]:
    torch.manual_seed(SEED)
    model = AraBit(front_end, precision, vocab_size, D_MODEL, N_LAYERS, N_HEADS, D_FF, mate_cfg).to(device)
    peak_lr = PEAK_LR_TERNARY if precision == "ternary" else PEAK_LR_FP16
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=WEIGHT_DECAY)

    total_steps = n_sequences // BATCH_SIZE
    perm = torch.randperm(n_sequences)
    cell = f"{front_end}_{precision}"
    print(f"\n[{cell}] {total_steps} steps, peak_lr={peak_lr:.2e}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")

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
            print(f"  step {step:4d}/{total_steps}  loss {loss.item():.4f}  lr {lr:.2e}")

    print(f"[{cell}] finished in {time.time()-t0:.1f}s")
    return curve, model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sentences = list(stream_sentences(article_limit=ARTICLE_LIMIT))
    print(f"{len(sentences)} sentences")

    fallback_tok = Tokenizer.from_file("data/stage1/bpe_tokenizer.json")
    control_tok = Tokenizer.from_file("data/stage1/bpe_control_tokenizer.json")
    vocabs = load_vocabs("data/stage1/vocabs.json")
    cache_df = read_cache("data/stage1/morph_cache.parquet")
    morph_index = MorphIndex(cache_df, vocabs)

    bpe_sequences = build_bpe_sequences(sentences, control_tok)
    word_ids, words_grid = build_word_sequences(sentences, vocabs["word"])
    print(f"bpe sequences: {bpe_sequences.shape}, word sequences: {word_ids.shape}")

    mate_cfg = MATEConfig(
        n_proclitic=len(vocabs["proclitic"]), n_enclitic=len(vocabs["enclitic"]),
        n_root=len(vocabs["root"]), n_pattern=len(vocabs["pattern"]),
        n_bpe=fallback_tok.get_vocab_size(),
        d_clitic=16, d_root=32, d_pattern=32, d_model=D_MODEL,
    )

    def bpe_step_fn(idx):
        seqs = bpe_sequences[idx].to(device)
        prefix_lens = torch.randint(0, (SEQ_LEN - 1) // 2, (BATCH_SIZE,), device=device)
        return dict(token_ids=seqs[:, :-1]), seqs[:, 1:], prefix_lens

    def mate_step_fn(idx):
        seqs = word_ids[idx].to(device)
        input_words = [w for i in idx.tolist() for w in words_grid[i][:-1]]
        batch = build_mate_batch(input_words, morph_index, device)
        prefix_lens = torch.randint(0, (SEQ_LEN - 1) // 2, (BATCH_SIZE,), device=device)
        return batch, seqs[:, 1:], prefix_lens

    results = {}
    for front_end, vocab_size, n_seq, step_fn in [
        ("bpe", control_tok.get_vocab_size(), bpe_sequences.shape[0], bpe_step_fn),
        ("mate", len(vocabs["word"]), word_ids.shape[0], mate_step_fn),
    ]:
        for precision in ("fp16", "ternary"):
            curve, model = train_cell(
                front_end, precision, vocab_size, device, mate_cfg, n_seq, step_fn
            )
            cell = f"{front_end}_{precision}"
            results[cell] = curve
            torch.save(model.state_dict(), OUT_DIR / f"checkpoint_{cell}.pt")
            with open(OUT_DIR / f"pretrain_loss_{cell}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "loss"])
                w.writerows(curve)

    print("\n=== Stage 1 pretraining summary ===")
    import math
    for cell, curve in results.items():
        finite = [l for _, l in curve if math.isfinite(l)]
        print(f"{cell:14s}: start={finite[0]:.4f} end={finite[-1]:.4f} "
              f"descended={finite[-1] < finite[0]} steps={len(curve)}")


if __name__ == "__main__":
    main()
