"""Stage 2: real memory/latency/energy measurement (Table 5) for all 4 cells
at the small-model scale, per docs/AraBit-1.58_IMPLEMENTATION.md sec:3.5.
Measured on seed0's fine-tuned checkpoint per cell — memory/latency/energy
are architecture properties, not seed-dependent.

Usage: python -m scripts.stage2_efficiency
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.data.cache import read_cache
from src.data.mate_batch import MorphIndex, build_mate_batch
from src.eval.efficiency import (
    count_params,
    measure_gpu_energy_joules,
    measure_latency_model_only_ms_per_token,
    weight_memory_bytes,
)
from src.model.arabit import AraBit
from src.model.mate import MATEConfig
from src.tokenization.vocab import load_vocabs

D_MODEL, N_LAYERS, N_HEADS, D_FF = 512, 12, 8, 2048
DATA_DIR = Path("data/stage2")
OUT_DIR = Path("results/stage2")
SAMPLE_SENTENCES = [
    "كان الطقس جميلا في ذلك اليوم من أيام الربيع",
    "ذهب الرجل إلى السوق ليشتري بعض الحاجيات الضرورية لأسرته",
    "تحدثت المعلمة مع الطلاب عن أهمية القراءة في حياتهم اليومية",
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

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

    from camel_tools.disambig.mle import MLEDisambiguator
    disambiguator = MLEDisambiguator.pretrained()

    def camel_preprocess_ms(sentences):
        t0 = time.perf_counter()
        n_tokens = 0
        for s in sentences:
            for w in s.split():
                n_tokens += 1
                try:
                    disambiguator.disambiguate([w])
                except Exception:
                    pass
        return 1000.0 * (time.perf_counter() - t0), n_tokens

    def bpe_preprocess_ms(sentences, tok):
        t0 = time.perf_counter()
        n_tokens = 0
        for s in sentences:
            n_tokens += len(tok.encode(s).ids)
        return 1000.0 * (time.perf_counter() - t0), n_tokens

    results = {}
    for front_end, vocab_size in [("bpe", control_tok.get_vocab_size()), ("mate", len(vocabs["word"]))]:
        for precision in ("fp16", "ternary"):
            cell = f"{front_end}_{precision}"
            print(f"\n[{cell}]")
            model = AraBit(front_end, precision, vocab_size, D_MODEL, N_LAYERS, N_HEADS, D_FF, mate_cfg).to(device)
            state = torch.load(OUT_DIR / f"checkpoint_ft_{cell}_seed0.pt", map_location=device, weights_only=True)
            model.load_state_dict(state)

            p_t, p_f = count_params(model)
            mem_bytes = weight_memory_bytes(p_t, p_f)
            print(f"  P_t={p_t:,}  P_f={p_f:,}  weight_mem={mem_bytes/1e6:.2f} MB")

            if front_end == "bpe":
                seq_len = 32
                dummy = dict(token_ids=torch.randint(0, vocab_size, (1, seq_len), device=device))
            else:
                all_words = " ".join(SAMPLE_SENTENCES).split()
                words = (all_words * 2)[:32]
                seq_len = len(words)
                dummy = build_mate_batch(words, morph_index, device)
            prefix_lens = torch.tensor([seq_len // 2], device=device)

            model_ms_per_tok = measure_latency_model_only_ms_per_token(
                model, dummy, prefix_lens, 1, seq_len, device
            )
            print(f"  model-only latency: {model_ms_per_tok:.4f} ms/token")

            if front_end == "mate":
                pre_ms, pre_tokens = camel_preprocess_ms(SAMPLE_SENTENCES)
            else:
                pre_ms, pre_tokens = bpe_preprocess_ms(SAMPLE_SENTENCES, control_tok)
            pre_ms_per_tok = pre_ms / max(1, pre_tokens)
            e2e_ms_per_tok = model_ms_per_tok + pre_ms_per_tok
            print(f"  front-end preprocess: {pre_ms_per_tok:.4f} ms/token")
            print(f"  end-to-end latency: {e2e_ms_per_tok:.4f} ms/token")

            def run_n(n=20):
                for _ in range(n):
                    with torch.no_grad():
                        model(dummy, prefix_lens, 1, seq_len)

            joules, wall_s = measure_gpu_energy_joules(lambda: run_n(20))
            joules_per_seq = joules / 20 if joules == joules else float("nan")
            print(f"  energy: {joules_per_seq:.4f} J/sequence" if joules == joules else "  energy: NA")

            results[cell] = dict(
                p_t=p_t, p_f=p_f, weight_mem_mb=mem_bytes / 1e6,
                model_ms_per_tok=model_ms_per_tok, e2e_ms_per_tok=e2e_ms_per_tok,
                joules_per_seq=joules_per_seq,
            )
            del model
            torch.cuda.empty_cache()

    with open(OUT_DIR / "efficiency_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "P_t", "P_f", "weight_mem_MB", "latency_model_ms_per_tok",
                    "latency_e2e_ms_per_tok", "energy_J_per_seq"])
        for cell, r in results.items():
            w.writerow([cell, r["p_t"], r["p_f"], f"{r['weight_mem_mb']:.3f}",
                        f"{r['model_ms_per_tok']:.4f}", f"{r['e2e_ms_per_tok']:.4f}",
                        f"{r['joules_per_seq']:.4f}" if r["joules_per_seq"] == r["joules_per_seq"] else "NA"])

    print("\n=== Stage 2 efficiency summary ===")
    for cell, r in results.items():
        print(f"{cell:14s}: P_t={r['p_t']:,} P_f={r['p_f']:,} mem={r['weight_mem_mb']:.2f}MB "
              f"model={r['model_ms_per_tok']:.3f}ms/tok e2e={r['e2e_ms_per_tok']:.3f}ms/tok")


if __name__ == "__main__":
    main()
