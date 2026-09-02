"""Stage 3: real memory/latency measurement for AraBART, AraT5, and
AraBART+bitsandbytes-NF4.

Usage: python -m scripts.stage3_efficiency
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

OUT_DIR = Path("results/stage3")
SAMPLE_SRC = "كان الطقس جميلا في ذلك اليوم من أيام الربيع، وذهب الرجل إلى السوق."


@torch.no_grad()
def measure_latency(model, tokenizer, device, n_warmup=10, n_iters=20):
    enc = tokenizer([SAMPLE_SRC], return_tensors="pt").to(device)
    for _ in range(n_warmup):
        model.generate(**enc, max_new_tokens=20, num_beams=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = model.generate(**enc, max_new_tokens=20, num_beams=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_tokens = out.shape[1] * n_iters
    return 1000.0 * elapsed / n_tokens


def main():
    device = torch.device("cuda")
    results = {}

    for name, checkpoint in [("AraBART", OUT_DIR / "checkpoint_AraBART"), ("AraT5", OUT_DIR / "checkpoint_AraT5")]:
        print(f"\n[{name}]")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device).eval()
        n_params = sum(p.numel() for p in model.parameters())
        mem_mb = n_params * 4 / 1e6  # fp32
        lat = measure_latency(model, tokenizer, device)
        print(f"  params={n_params:,}  fp32_mem={mem_mb:.2f}MB  latency={lat:.4f}ms/tok")
        results[name] = dict(n_params=n_params, mem_mb=mem_mb, latency_ms_per_tok=lat)
        del model
        torch.cuda.empty_cache()

    print("\n[AraBART_nf4]")
    tokenizer = AutoTokenizer.from_pretrained(OUT_DIR / "checkpoint_AraBART")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(OUT_DIR / "checkpoint_AraBART", quantization_config=bnb_config, device_map={"": 0})
    mem_mb_actual = torch.cuda.memory_allocated() / 1e6
    lat = measure_latency(model, tokenizer, device)
    print(f"  actual_gpu_mem={mem_mb_actual:.2f}MB  latency={lat:.4f}ms/tok")
    results["AraBART_nf4"] = dict(n_params=results["AraBART"]["n_params"], mem_mb=mem_mb_actual, latency_ms_per_tok=lat)

    with open(OUT_DIR / "baseline_efficiency.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_params", "mem_MB", "latency_ms_per_tok"])
        for name, r in results.items():
            w.writerow([name, r["n_params"], f"{r['mem_mb']:.2f}", f"{r['latency_ms_per_tok']:.4f}"])

    print("\n=== Stage 3 efficiency summary ===")
    for name, r in results.items():
        print(f"{name:14s}: params={r['n_params']:,}  mem={r['mem_mb']:.2f}MB  latency={r['latency_ms_per_tok']:.4f}ms/tok")


if __name__ == "__main__":
    main()
