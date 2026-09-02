"""Stage 3: fine-tune external baselines (AraBART, AraT5) on the real SAMER
Corpus (L5 -> L3), same task/splits/eval subset as Stage 2's own cells, per
docs/AraBit-1.58_IMPLEMENTATION.md sec:4 Stage 3.

FFT-Seq2Seq and Switch-Arabic (the user's own prior work) are skipped —
no code/checkpoints available in this repo, user-confirmed 2026-09-01, see
results/NOTES.md.

Usage: python -m scripts.stage3_finetune_baselines
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.eval.quality import corpus_sari
from src.train.schedule import lr_at_step

SAMER_DIR = Path("docs/samer-simplification-corpus-v1/data")
OUT_DIR = Path("results/stage3")
MAX_SRC_LEN = 64
MAX_TGT_LEN = 64
BATCH_SIZE = 2
GRAD_ACCUM = 8  # effective batch 16 — smaller per-step batch for headroom on
# this 8GB card (AraT5-base peaked at 5.86GB at batch=4 in a smoke test,
# too close to the limit alongside ~2.3GB other processes already use)
FT_EPOCHS = 3
LR = 3e-5
EVAL_CAP = 200
MODELS = {
    "AraBART": "moussaKam/AraBART",
    "AraT5": "UBC-NLP/AraT5-base",
}


def load_samer_pairs(split: str) -> list[tuple[str, str]]:
    pairs = []
    with open(SAMER_DIR / f"{split}.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            l5, l3 = row["L5"].strip(), row["L3"].strip()
            if l5 and l3:
                pairs.append((l5, l3))
    return pairs


def finetune_and_eval(name: str, model_id: str, train_pairs, eval_sources, eval_refs, device):
    print(f"\n=== {name} ({model_id}) ===")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded in {time.time()-t0:.1f}s, {n_params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n = len(train_pairs)
    steps_per_epoch = n // BATCH_SIZE
    total_micro_steps = steps_per_epoch * FT_EPOCHS
    total_opt_steps = total_micro_steps // GRAD_ACCUM
    print(f"  fine-tuning: {total_micro_steps} micro-steps / {total_opt_steps} optimizer steps "
          f"({FT_EPOCHS} epochs x {steps_per_epoch} steps/epoch)")

    model.train()
    t0 = time.time()
    micro_step = 0
    opt_step = 0
    import random
    rng = random.Random(0)
    for epoch in range(FT_EPOCHS):
        order = list(range(n))
        rng.shuffle(order)
        for b in range(steps_per_epoch):
            idx = order[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            srcs = [train_pairs[i][0] for i in idx]
            tgts = [train_pairs[i][1] for i in idx]
            enc = tokenizer(srcs, padding=True, truncation=True, max_length=MAX_SRC_LEN, return_tensors="pt").to(device)
            lbl = tokenizer(text_target=tgts, padding=True, truncation=True, max_length=MAX_TGT_LEN, return_tensors="pt").to(device)
            labels = lbl.input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            micro_step += 1
            if micro_step % GRAD_ACCUM == 0:
                # gradient clipping + warmup+cosine LR - both missing from the
                # first attempt, which let AraT5's full fine-tune (282M
                # params) diverge into repeating two garbage tokens for every
                # input (verified by inspecting results/stage3/generations_
                # AraT5.tsv). Full fine-tuning a large pretrained model
                # without either is a known recipe for this failure mode.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                lr = lr_at_step(opt_step, total_opt_steps, LR, warmup_ratio=0.06)
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.step()
                opt.zero_grad()
                opt_step += 1
        print(f"    epoch {epoch+1}/{FT_EPOCHS}  loss {out.loss.item():.4f}  lr {lr:.2e}  ({time.time()-t0:.0f}s elapsed)")
    print(f"  fine-tuned in {time.time()-t0:.1f}s")

    model.eval()
    t0 = time.time()
    gens = []
    with torch.no_grad():
        for i in range(0, len(eval_sources), BATCH_SIZE):
            batch_src = eval_sources[i : i + BATCH_SIZE]
            enc = tokenizer(batch_src, padding=True, truncation=True, max_length=MAX_SRC_LEN, return_tensors="pt").to(device)
            out_ids = model.generate(**enc, max_new_tokens=MAX_TGT_LEN, num_beams=1)
            gens.extend(tokenizer.batch_decode(out_ids, skip_special_tokens=True))
    gen_time = time.time() - t0

    score = corpus_sari(eval_sources, gens, eval_refs)
    print(f"  SARI={score:.2f}  generation took {gen_time:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR / f"checkpoint_{name}")
    tokenizer.save_pretrained(OUT_DIR / f"checkpoint_{name}")
    with open(OUT_DIR / f"generations_{name}.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source", "generated", "reference"])
        for s, g, r in zip(eval_sources, gens, [r[0] for r in eval_refs]):
            w.writerow([s, g, r])

    return dict(sari=score, n_params=n_params, gen_time_s=gen_time)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_pairs = load_samer_pairs("train")
    test_pairs = load_samer_pairs("test")
    eval_sources = [s for s, _ in test_pairs][:EVAL_CAP]
    eval_refs = [[t] for _, t in test_pairs][:EVAL_CAP]
    print(f"train pairs: {len(train_pairs)}, eval (capped): {len(eval_sources)}")

    results = {}
    for name, model_id in MODELS.items():
        results[name] = finetune_and_eval(name, model_id, train_pairs, eval_sources, eval_refs, device)

    with open(OUT_DIR / "baseline_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "sari", "n_params", "gen_time_s"])
        for name, r in results.items():
            w.writerow([name, f"{r['sari']:.4f}", r["n_params"], f"{r['gen_time_s']:.1f}"])

    print("\n=== Stage 3 baseline summary ===")
    for name, r in results.items():
        print(f"{name:10s}: SARI={r['sari']:.2f}  params={r['n_params']:,}  gen_time={r['gen_time_s']:.1f}s")


if __name__ == "__main__":
    main()
