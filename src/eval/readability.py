"""SAMER levelled-lexicon readability metric (Al Khalil et al.).

Remediation Phase 5 Item 3 (results/NOTES.md) dropped the unimplemented
OSMAN readability formula (no verified reference implementation existed to
check a from-scratch attempt against - see quality.py's `osman_readability`
docstring) and replaced Table 4's "Readability" column with this instead:
mean CAMeL-lemma readability level against
docs/samer-readability-lexicon-v2/SAMER-Readability-Lexicon-v2.tsv - the
same resource the SAMER simplification corpus itself (Alhafni et al. 2024,
alhafni2024samer) is levelled against, so it is a real, citable, verified
measurement rather than a plausible-looking guess. Lower is better - level 1
is the lexicon's simplest/most frequent band, level 5 the hardest.

Lexicon key format: verified empirically (see results/NOTES.md, Remediation
Phase 5 Item 2) that CAMeL's own `lex` (diacritized lemma) and `pos` fields,
joined as f"{lex}#{pos}", match the lexicon's "lemma#pos" column exactly -
e.g. disambiguating the surface form "في" gives lex="فِي" pos="prep", and
the lexicon's own row key is literally "فِي#prep". No extra normalization
needed.

Uses CAMeL's MLEDisambiguator per-word (not per-sentence-with-context) -
matching scripts/01_cache_morphology.py's own established precedent
("CAMeL's MLE disambiguator already picks its single best analysis per
surface form without needing neighbouring tokens re-passed per occurrence"),
with per-word memoization (a word seen 1,000 times across a corpus is only
disambiguated once) since this metric doesn't need the position-in-sentence
context cache.py's context-level caching is built for.
"""

from __future__ import annotations

import csv
from pathlib import Path

LEXICON_PATH = Path("docs/samer-readability-lexicon-v2/SAMER-Readability-Lexicon-v2.tsv")


def load_samer_lexicon(path: Path | str = LEXICON_PATH) -> dict[str, int]:
    """Parses the SAMER readability lexicon TSV into {"lex#pos": level}."""
    lexicon: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = row.get("lemma#pos")
            level_str = row.get("readability (rounded average)")
            if key and level_str:
                lexicon[key] = int(level_str)
    return lexicon


class SAMERReadabilityScorer:
    """Construct once (loads the CAMeL disambiguator model + lexicon), reuse
    across many score_corpus() calls."""

    def __init__(self, lexicon: dict[str, int] | None = None):
        from camel_tools.disambig.mle import MLEDisambiguator

        self._disambiguator = MLEDisambiguator.pretrained()
        self._lexicon = lexicon if lexicon is not None else load_samer_lexicon()
        self._level_cache: dict[str, int | None] = {}

    def _level_of_word(self, word: str) -> int | None:
        if word in self._level_cache:
            return self._level_cache[word]
        level = None
        try:
            analyses = self._disambiguator.disambiguate([word])[0].analyses
        except Exception:
            # Same defensive handling as scripts/01_cache_morphology.py:
            # CAMeL's analyzer can raise on malformed input (e.g. stray
            # markup/punctuation surviving tokenization) - treated as "no
            # analysis", not a crash.
            analyses = []
        if analyses:
            a = analyses[0].analysis
            key = f"{a.get('lex')}#{a.get('pos')}"
            level = self._lexicon.get(key)
        self._level_cache[word] = level
        return level

    def score_corpus(self, sentences: list[str]) -> dict:
        """Mean lemma-readability level and proportion of level-1 lemmas,
        over every word in `sentences` that has a lexicon hit. Words with no
        analysis or no lexicon entry (numerals, punctuation, OOV names, ...)
        are excluded from the mean, not scored as 0 or skipped silently -
        `coverage` reports what fraction of tokens actually contributed."""
        levels: list[int] = []
        n_tokens = 0
        for sentence in sentences:
            for word in sentence.split():
                n_tokens += 1
                level = self._level_of_word(word)
                if level is not None:
                    levels.append(level)

        n_covered = len(levels)
        return dict(
            mean_level=(sum(levels) / n_covered) if n_covered else float("nan"),
            prop_level1=(sum(1 for lv in levels if lv == 1) / n_covered) if n_covered else float("nan"),
            coverage=(n_covered / n_tokens) if n_tokens else 0.0,
            n_tokens=n_tokens,
            n_covered=n_covered,
        )
