import json
from easse.sari import corpus_sari, compute_ngram_stats, compute_macro_sari

IN = r"D:\LocalProgramming\BitNet_ClaudeV\data\easse_check\sample_with_repo_sari.json"
OUT = r"D:\LocalProgramming\BitNet_ClaudeV\data\easse_check\easse_compare_results.json"

with open(IN, encoding="utf-8") as f:
    sample = json.load(f)


def easse_sari_matched(source: str, generated: str, reference: str) -> float:
    """Bypasses easse's corpus_sari/get_corpus_sari_operation_scores wrapper,
    which always routes sentences through utils_prep.normalize() (a sacrebleu
    tokenizer - "none" turned out to be an unimplemented stub in this easse
    version, and "13a" splits off punctuation) before counting n-grams. Calls
    compute_ngram_stats/compute_macro_sari directly on the raw, untokenized
    strings instead, since extract_ngrams() itself does only line.split() -
    matching src/eval/quality.py's plain .split() tokenization exactly. Also
    uses use_f1_for_deletion=False to match Xu et al. (2016)'s original
    precision-only deletion term, which is what this repo's sari() implements.
    """
    stats = compute_ngram_stats([source], [generated], [[reference]])
    add_f1, keep_f1, del_score = compute_macro_sari(*stats, use_f1_for_deletion=False)
    return 100.0 * (add_f1 + keep_f1 + del_score) / 3.0


results = []
for ex in sample:
    easse_score = easse_sari_matched(ex["source"], ex["generated"], ex["reference"])
    results.append({**ex, "easse_sari": easse_score, "diff": ex["repo_sari"] - easse_score})

repo_mean = sum(r["repo_sari"] for r in results) / len(results)
easse_mean = sum(r["easse_sari"] for r in results) / len(results)
diffs = [r["diff"] for r in results]
mean_abs_diff = sum(abs(d) for d in diffs) / len(diffs)
max_abs_diff = max(abs(d) for d in diffs)

# Also report easse's own DEFAULT settings (macro, F1-deletion, 13a
# tokenizer+lowercase) for reference, since that's what a reader who just
# `pip install easse; corpus_sari(...)` would get - shows how much of any
# gap is "different formula variant" vs "different tokenization".
default_scores = [
    corpus_sari([ex["source"]], [ex["generated"]], [[ex["reference"]]])
    for ex in sample
]
default_mean = sum(default_scores) / len(default_scores)

summary = dict(
    n_examples=len(results),
    repo_mean_sari=repo_mean,
    easse_matched_settings_mean_sari=easse_mean,
    easse_default_settings_mean_sari=default_mean,
    mean_abs_diff_matched=mean_abs_diff,
    max_abs_diff_matched=max_abs_diff,
    matched_settings="raw .split() tokenization (bypassing easse's normalize wrapper), use_f1_for_deletion=False, macro aggregation",
)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "per_example": results}, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
