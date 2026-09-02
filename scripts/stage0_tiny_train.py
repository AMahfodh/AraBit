"""Stage 0(e): train a tiny model (4 layers, d=256) on real Arabic Wikipedia
tokens in both ternary and fp16, to confirm the ternary loss actually descends
and does not collapse, per docs/AraBit-1.58_IMPLEMENTATION.md sec:4 Stage 0(e).

Not one of the numbered scripts/00-05 pipeline stages (see results/NOTES.md) -
this is a one-off feasibility check, separate from the general Hydra-driven
scripts/02_pretrain.py that Stage 2's real multi-cell/multi-seed runs will use.

Usage: python -m scripts.stage0_tiny_train
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.data.corpus import stream_sentences
from src.model.arabit import AraBit
from src.tokenization.bpe import train_bpe
from src.train.schedule import lr_at_step

TARGET_TOKENS = 10_000_000
ARTICLE_CAP = 3000
VOCAB_SIZE = 8000
SEQ_LEN = 256
BATCH_SIZE = 32
N_LAYERS, D_MODEL, N_HEADS, D_FF = 4, 256, 4, 1024
PEAK_LR_FP16 = 3e-4
PEAK_LR_TERNARY = 2 * PEAK_LR_FP16  # IMPLEMENTATION.md sec:3.1: ~2x fp16
WARMUP_RATIO = 0.02
WEIGHT_DECAY = 0.1
SEED = 0

OUT_DIR = Path("results/stage0")


def build_token_stream() -> tuple[list[int], Tokenizer]:
    print(f"streaming up to {ARTICLE_CAP} articles from Arabic Wikipedia...")
    t0 = time.time()
    sentences = list(stream_sentences(article_limit=ARTICLE_CAP))
    print(f"  {len(sentences)} sentences in {time.time()-t0:.1f}s")

    print(f"training BPE tokenizer (vocab_size={VOCAB_SIZE})...")
    t0 = time.time()
    tokenizer = train_bpe(sentences, VOCAB_SIZE)
    print(f"  done in {time.time()-t0:.1f}s, actual vocab={tokenizer.get_vocab_size()}")

    print("encoding corpus...")
    t0 = time.time()
    bos, eos = tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")
    ids: list[int] = []
    for s in sentences:
        ids.append(bos)
        ids.extend(tokenizer.encode(s).ids)
        ids.append(eos)
        if len(ids) >= TARGET_TOKENS:
            break
    print(f"  {len(ids)} tokens encoded in {time.time()-t0:.1f}s")
    return ids, tokenizer


def make_batches(ids: list[int]) -> torch.Tensor:
    n_seq = len(ids) // SEQ_LEN
    ids = ids[: n_seq * SEQ_LEN]
    return torch.tensor(ids, dtype=torch.long).view(n_seq, SEQ_LEN)


def train_one_precision(
    precision: str, sequences: torch.Tensor, vocab_size: int, device: torch.device
) -> list[tuple[int, float]]:
    torch.manual_seed(SEED)
    model = AraBit(vocab_size, D_MODEL, N_LAYERS, N_HEADS, D_FF, precision).to(device)
    peak_lr = PEAK_LR_TERNARY if precision == "ternary" else PEAK_LR_FP16
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=WEIGHT_DECAY)

    n_seq = sequences.shape[0]
    total_steps = n_seq // BATCH_SIZE
    perm = torch.randperm(n_seq)

    print(f"\n[{precision}] {total_steps} steps, peak_lr={peak_lr:.2e}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    curve = []
    t0 = time.time()
    for step in range(total_steps):
        batch_idx = perm[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        batch = sequences[batch_idx].to(device)
        prefix_lens = torch.randint(0, SEQ_LEN // 2, (BATCH_SIZE,), device=device)

        lr = lr_at_step(step, total_steps, peak_lr, WARMUP_RATIO)
        for g in opt.param_groups:
            g["lr"] = lr

        logits = model(batch[:, :-1], prefix_lens)
        targets = batch[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size), targets.reshape(-1)
        )

        if not torch.isfinite(loss):
            print(f"  STEP {step}: loss is {loss.item()} -- COLLAPSE, aborting")
            curve.append((step, float("nan")))
            break

        opt.zero_grad()
        loss.backward()
        opt.step()

        curve.append((step, loss.item()))
        if step % max(1, total_steps // 20) == 0 or step == total_steps - 1:
            print(f"  step {step:5d}/{total_steps}  loss {loss.item():.4f}  lr {lr:.2e}")

    print(f"[{precision}] finished in {time.time()-t0:.1f}s")
    return curve


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids, tokenizer = build_token_stream()
    sequences = make_batches(ids)
    print(f"\n{sequences.shape[0]} sequences of length {SEQ_LEN} "
          f"({sequences.numel()} tokens actually used)")

    results = {}
    for precision in ("fp16", "ternary"):
        curve = train_one_precision(precision, sequences, tokenizer.get_vocab_size(), device)
        results[precision] = curve
        with open(OUT_DIR / f"loss_{precision}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "loss"])
            w.writerows(curve)

    print("\n=== Stage 0(e) summary ===")
    for precision, curve in results.items():
        finite = [l for _, l in curve if math.isfinite(l)]
        collapsed = len(finite) < len(curve)
        descended = len(finite) >= 2 and finite[-1] < finite[0]
        print(
            f"{precision:8s}: start={finite[0]:.4f} end={finite[-1]:.4f} "
            f"descended={descended} collapsed={collapsed} steps_completed={len(curve)}"
        )


if __name__ == "__main__":
    main()
