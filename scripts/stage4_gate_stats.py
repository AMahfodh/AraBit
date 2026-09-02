"""Stage 4: gate-behaviour analysis — mean MATE gate value by token category
(native Arabic, foreign, digit, named entity), on the fine-tuned Stage 2
mate_fp16/mate_ternary checkpoints (seed0), over the same 200-example SAMER
test subset used throughout.

Usage: python -m scripts.stage4_gate_stats
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from scripts.stage2_finetune_and_eval import load_samer_pairs
from src.data.cache import read_cache
from src.data.mate_batch import MorphIndex, build_mate_batch
from src.eval.errors import categorize_tokens, gate_stats_by_category
from src.model.mate import MATE, MATEConfig
from src.tokenization.vocab import load_vocabs

DATA_DIR = Path("data/stage2")
IN_DIR = Path("results/stage2")
OUT_DIR = Path("results/stage4")
D_MODEL = 512
EVAL_CAP = 200


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from tokenizers import Tokenizer

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

    test_pairs = load_samer_pairs("test")
    sources = [s for s, _ in test_pairs][:EVAL_CAP]
    all_words = [w for s in sources for w in s.split()]
    print(f"{len(sources)} sentences, {len(all_words)} words")

    print("running NER (CAMeL Tools ner-arabert)...")
    from camel_tools.ner import NERecognizer
    ner = NERecognizer.pretrained()
    ner_labels = []
    for s in sources:
        words = s.split()
        ner_labels.extend(ner.predict_sentence(words))
    categories = categorize_tokens(all_words, ner_labels)
    cat_counts = {c: categories.count(c) for c in set(categories)}
    print(f"category counts: {cat_counts}")

    results = {}
    for precision in ("fp16", "ternary"):
        print(f"\n[mate_{precision}]")
        mate = MATE(mate_cfg).to(device)
        full_state = torch.load(
            IN_DIR / f"checkpoint_ft_mate_{precision}_seed0.pt", map_location=device, weights_only=True
        )
        mate_state = {k[len("embed."):]: v for k, v in full_state.items() if k.startswith("embed.")}
        mate.load_state_dict(mate_state)
        mate.eval()

        gate_values = []
        BATCH = 512
        with torch.no_grad():
            for i in range(0, len(all_words), BATCH):
                batch_words = all_words[i : i + BATCH]
                batch = build_mate_batch(batch_words, morph_index, device)
                _, g = mate(**batch, return_gate=True)
                gate_values.extend(g.cpu().tolist())

        stats = gate_stats_by_category(gate_values, categories)
        results[precision] = stats
        for cat, s in sorted(stats.items()):
            print(f"  {cat:14s}: mean_gate={s['mean']:.4f}  std={s['std']:.4f}  n={s['n']}")

    with open(OUT_DIR / "gate_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["precision", "category", "mean_gate", "std_gate", "n"])
        for precision, stats in results.items():
            for cat, s in stats.items():
                w.writerow([precision, cat, f"{s['mean']:.4f}", f"{s['std']:.4f}", s["n"]])

    print(f"\nwrote {OUT_DIR}/gate_stats.csv")


if __name__ == "__main__":
    main()
