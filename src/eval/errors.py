"""Gate statistics: mean MATE gate value g_i by token category.

See docs/AraBit-1.58_IMPLEMENTATION.md sec:3.2 (per-category gate logging)
and AraBit-1.58.tex sec:errors' "Gate behaviour" paragraph. The manuscript's
other two error-analysis directions (analyser-error correlation, failure
taxonomy) were dropped from this manuscript (user-confirmed 2026-09-01,
same reason as human evaluation — both need hand annotation) — see
results/NOTES.md.

Categories match the manuscript exactly: native Arabic words, foreign
transliterations, digits, and named entities (detected with CAMeL Tools'
own `ner-arabert` model, real NER, not a heuristic).
"""

from __future__ import annotations

import re
from collections import defaultdict

_LATIN_RE = re.compile(r"^[A-Za-z][A-Za-z\-']*$")
_DIGIT_RE = re.compile(r"^\d+$")


def categorize_tokens(words: list[str], ner_labels: list[str]) -> list[str]:
    """One category per word: 'named_entity' > 'digit' > 'foreign' > 'native_arabic'
    (named entity checked first since e.g. transliterated foreign names are
    still named entities and that's the more informative category)."""
    categories = []
    for w, label in zip(words, ner_labels):
        if label != "O":
            categories.append("named_entity")
        elif _DIGIT_RE.match(w):
            categories.append("digit")
        elif _LATIN_RE.match(w):
            categories.append("foreign")
        else:
            categories.append("native_arabic")
    return categories


def gate_stats_by_category(gate_values: list[float], categories: list[str]) -> dict[str, dict]:
    buckets = defaultdict(list)
    for g, c in zip(gate_values, categories):
        buckets[c].append(g)
    stats = {}
    for cat, vals in buckets.items():
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
        stats[cat] = dict(mean=mean, std=var ** 0.5, n=n)
    return stats
