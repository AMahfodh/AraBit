"""Stage 3: apply bitsandbytes 4-bit (NF4) quantization to the fine-tuned
AraBART checkpoint and re-evaluate on the same SAMER test subset.

bitsandbytes-NF4, not GPTQ/AWQ (user-confirmed 2026-09-01) - see
results/NOTES.md "GPTQ/AWQ unavailable" for why.

Usage: python -m scripts.stage3_quantize_arabart
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from src.eval.quality import corpus_sari
from scripts.stage3_finetune_baselines import load_samer_pairs

CHECKPOINT = Path("results/stage3/checkpoint_AraBART")
OUT_DIR = Path("results/stage3")
MAX_SRC_LEN = 64
MAX_TGT_LEN = 64
BATCH_SIZE = 4
EVAL_CAP = 200


def main():
    device = torch.device("cuda")
    test_pairs = load_samer_pairs("test")
    eval_sources = [s for s, _ in test_pairs][:EVAL_CAP]
    eval_refs = [[t] for _, t in test_pairs][:EVAL_CAP]

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT, quantization_config=bnb_config, device_map={"": 0})
    print(f"loaded 4-bit model in {time.time()-t0:.1f}s")

    mem_bytes = torch.cuda.memory_allocated()
    print(f"GPU memory after load: {mem_bytes/1e6:.2f} MB")

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
    print(f"SARI={score:.2f}  generation took {gen_time:.1f}s  ({len(eval_sources)} examples)")

    with open(OUT_DIR / "generations_AraBART_nf4.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source", "generated", "reference"])
        for s, g, r in zip(eval_sources, gens, [r[0] for r in eval_refs]):
            w.writerow([s, g, r])

    with open(OUT_DIR / "baseline_results.csv", "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(["AraBART_nf4", f"{score:.4f}", "139221504", f"{gen_time:.1f}"])

    print(f"\nGPU memory (4-bit): {mem_bytes/1e6:.2f} MB")


if __name__ == "__main__":
    main()
