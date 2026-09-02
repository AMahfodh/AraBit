"""Quality metrics: SARI, BERTScore, OSMAN readability, BLEU.

See docs/AraBit-1.58_IMPLEMENTATION.md sec:3.6:
  - SARI is primary; record the number of references used; on SAMER evaluate
    per target readability level, then aggregate.
  - BERTScore-F1 with an Arabic encoder (aubmindlab/bert-base-arabertv02),
    not a multilingual default.
  - OSMAN + SAMER's levelled lexicon for level-shift measurement.
  - BLEU only for the rephrasing task, never headlined for simplification.
  - Report mean +/- std over seeds and a paired bootstrap significance test
    between AraBit-1.58 and the ternary BPE control.

`easse` is not installed (see results/NOTES.md: no Python 3.12+ release
exists) — SARI is implemented directly here per Xu et al. (2016), the
IMPLEMENTATION.md-sanctioned fallback.

OSMAN is NOT implemented: it is a specific published readability formula
(Al-Tamimi et al.) and no verified reference implementation was available to
check this implementation against — per the project's "never fabricate a
number for a results table" rule, a plausible-looking but unverified formula
would be worse than an honest gap. `osman_readability` raises
NotImplementedError. Revisit if a citable reference implementation is found.
"""

from __future__ import annotations

from collections import Counter


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _sari_ngram_scores(orig: list[str], sys_out: list[str], refs: list[list[str]], n: int) -> tuple[float, float, float]:
    """Per Xu et al. (2016), 'Optimizing Statistical Machine Translation for
    Text Simplification', TACL. Returns (add_f1, keep_f1, del_precision) for
    this n-gram order.
    """
    orig_ng = _ngrams(orig, n)
    sys_ng = _ngrams(sys_out, n)
    ref_ngs = [_ngrams(r, n) for r in refs]
    n_refs = len(refs)

    # ADD: n-grams in system output, not in original, credited if in any reference.
    sys_add = sys_ng - orig_ng
    ref_add_union = Counter()
    for rng in ref_ngs:
        ref_add_union |= (rng - orig_ng)
    add_tp = sum(c for g, c in sys_add.items() if ref_add_union.get(g, 0) > 0)
    add_precision = add_tp / max(1, sum(sys_add.values()))
    ref_add_total = Counter()
    for rng in ref_ngs:
        ref_add_total |= (rng - orig_ng)
    add_recall_denom = sum(ref_add_total.values())
    add_recall = add_tp / max(1, add_recall_denom) if add_recall_denom > 0 else (1.0 if add_tp == 0 else 0.0)
    add_f1 = 0.0 if (add_precision + add_recall) == 0 else 2 * add_precision * add_recall / (add_precision + add_recall)

    # KEEP: n-grams in both original and system output, scored against how
    # often references also keep them (average count across references).
    orig_keep = orig_ng
    sys_keep = sys_ng & orig_ng
    ref_keep_avg = Counter()
    for g, c in orig_ng.items():
        avg = sum(min(c, rng.get(g, 0)) for rng in ref_ngs) / max(1, n_refs)
        if avg > 0:
            ref_keep_avg[g] = avg
    keep_tp = sum(min(sys_keep.get(g, 0), ref_keep_avg.get(g, 0)) for g in orig_keep)
    keep_precision = keep_tp / max(1, sum(sys_keep.values()))
    keep_recall_denom = sum(ref_keep_avg.values())
    keep_recall = keep_tp / max(1, keep_recall_denom) if keep_recall_denom > 0 else (1.0 if keep_tp == 0 else 0.0)
    keep_f1 = 0.0 if (keep_precision + keep_recall) == 0 else 2 * keep_precision * keep_recall / (keep_precision + keep_recall)

    # DELETE: n-grams in original but not in system output, precision only
    # (per Xu et al.'s definition), scored against how often references also delete them.
    orig_del = orig_ng - sys_ng
    ref_keep_from_orig = Counter()
    for g, c in orig_ng.items():
        avg = sum(min(c, rng.get(g, 0)) for rng in ref_ngs) / max(1, n_refs)
        ref_keep_from_orig[g] = avg
    del_tp = sum(max(0.0, c - ref_keep_from_orig.get(g, 0.0)) for g, c in orig_del.items())
    del_denom = sum(orig_del.values())
    del_precision = del_tp / max(1, del_denom) if del_denom > 0 else 1.0

    return add_f1, keep_f1, del_precision


def sari(source: str, system_output: str, references: list[str]) -> float:
    """SARI score, Xu et al. (2016). Averages Add-F1, Keep-F1, Del-P over
    n-gram orders 1-4, then averages the three components.
    """
    orig = source.split()
    sys_out = system_output.split()
    refs = [r.split() for r in references]

    add_scores, keep_scores, del_scores = [], [], []
    for n in range(1, 5):
        a, k, d = _sari_ngram_scores(orig, sys_out, refs, n)
        add_scores.append(a)
        keep_scores.append(k)
        del_scores.append(d)

    avg_add = sum(add_scores) / 4
    avg_keep = sum(keep_scores) / 4
    avg_del = sum(del_scores) / 4
    return 100.0 * (avg_add + avg_keep + avg_del) / 3.0


def corpus_sari(sources: list[str], system_outputs: list[str], references: list[list[str]]) -> float:
    """Mean SARI over a corpus. `references[i]` is the list of reference(s)
    for example i — record how many references were used (IMPLEMENTATION.md
    sec:3.6) alongside this score, don't just report the number.
    """
    scores = [sari(s, o, r) for s, o, r in zip(sources, system_outputs, references)]
    return sum(scores) / len(scores)


def bertscore_f1(system_outputs: list[str], references: list[str], lang_model: str = "aubmindlab/bert-base-arabertv02") -> float:
    """BERTScore-F1 with an Arabic encoder, per IMPLEMENTATION.md sec:3.6
    ("not a multilingual default"). Downloads `lang_model` from the HF Hub
    on first use if not cached.
    """
    from bert_score import score as bert_score_fn

    # bert_score's internal model2layers registry doesn't know arabertv02
    # (a standard 12-layer BERT-base architecture) - pass num_layers
    # explicitly rather than let it KeyError.
    _, _, f1 = bert_score_fn(
        system_outputs, references, model_type=lang_model, num_layers=12, lang="ar", verbose=False
    )
    return f1.mean().item()


def corpus_bleu(system_outputs: list[str], references: list[list[str]]) -> float:
    """BLEU, rephrasing task only — IMPLEMENTATION.md sec:3.6: do not
    headline for simplification (it correlates poorly with simplicity)."""
    import sacrebleu

    return sacrebleu.corpus_bleu(system_outputs, list(zip(*references))).score


def osman_readability(text: str) -> float:
    raise NotImplementedError(
        "OSMAN readability is not implemented: no verified reference "
        "implementation was available to check a from-scratch implementation "
        "against, and this repo's policy is not to fabricate a formula for a "
        "results table. See this module's docstring."
    )
