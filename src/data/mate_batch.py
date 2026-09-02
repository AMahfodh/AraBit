"""Turns raw sentences into the flat, EmbeddingBag-style tensors src/model/mate.py:
MATE.forward needs, using the morphology cache (src/data/cache.py) and vocabs
(src/tokenization/vocab.py) built by scripts/01_cache_morphology.py and
scripts/00_build_vocabs.py.

Remediation Phase 1 (2026-09-01, results/NOTES.md "output-head confound"):
MATE's *input* embedding is still computed per orthographic word (one
morphological embedding per whitespace-split word), but the *sequence*
granularity and *output* vocabulary must now match the BPE cells' exactly
(shared 13,699-entry BPE vocab, targets tokenized identically) to remove the
output-vocabulary-reachability confound the pre-correction runs had. See
`words_per_bpe_position` below: each BPE subword position gets the SAME
word-level MATE embedding as every other subword position belonging to the
same word (repeated, not split) — MATE's representation stays word-level,
only the sequence/output alignment changed.
"""

from __future__ import annotations

import pandas as pd
import torch

UNK = "<unk>"


def words_per_bpe_position(sentence: str, tokenizer) -> list[str]:
    """One entry per BPE token `tokenizer.encode(sentence).ids` would produce
    for this sentence, giving the source word for each position (repeated
    across a word's multiple subword pieces). Built by encoding word-by-word
    and repeating - verified empirically to align 1:1 with whole-sentence
    encoding for this project's BPE tokenizers (Whitespace pre-tokenizer
    never merges across word boundaries, so
    `tokenizer.encode(sentence).ids == concat(tokenizer.encode(w).ids for w in sentence.split())`
    exactly), which is what `src/tokenization/bpe.py: train_bpe` /
    `build_bpe_sequences` (scripts/stage2_pretrain_all_cells.py) rely on too.
    """
    words = []
    for w in sentence.split():
        n_pieces = len(tokenizer.encode(w).ids)
        words.extend([w] * n_pieces)
    return words


class MorphIndex:
    """surface_form -> (root_id, pattern_id, conf, prc_ids, enc_ids, bpe_ids),
    all already mapped through the built vocabs (src/tokenization/vocab.py)
    except bpe_ids, which are already tokenizer ids from the cache.
    """

    def __init__(self, cache_df: pd.DataFrame, vocabs: dict[str, dict[str, int]]):
        root_v, pat_v, pro_v, enc_v = (
            vocabs["root"], vocabs["pattern"], vocabs["proclitic"], vocabs["enclitic"]
        )
        self.word_vocab = vocabs["word"]
        self.index: dict[str, dict] = {}
        for r in cache_df.itertuples(index=False):
            self.index[r.surface_form] = dict(
                root_id=root_v.get(r.root_id, 0) if r.root_id is not None else 0,
                pattern_id=pat_v.get(r.pattern_id, 0) if r.pattern_id is not None else 0,
                conf=float(r.conf),
                prc_ids=[pro_v.get(c, 0) for c in r.prc_ids],
                enc_ids=[enc_v.get(c, 0) for c in r.enc_ids],
                bpe_ids=list(r.bpe_ids) if len(r.bpe_ids) > 0 else [0],
            )

    def lookup(self, word: str) -> dict:
        return self.index.get(word, dict(root_id=0, pattern_id=0, conf=0.0, prc_ids=[], enc_ids=[], bpe_ids=[0]))

    def word_id(self, word: str) -> int:
        return self.word_vocab.get(word, 0)

    @staticmethod
    def lookup_bpe_only(fallback_bpe_ids: list[int]) -> dict:
        """Synthetic fallback record for a position whose *word* is not
        known - used during BPE-token-by-token generation (Remediation
        Phase 1, results/NOTES.md), where the continuation being generated
        has no ground-truth word boundaries to look morphology up against.
        Rather than guess at word boundaries from decoded text (fragile:
        this project's BPE has no explicit continuation marker to detect
        them from), feed the model its own just-generated text through the
        same E_fb fallback path unanalysable tokens already use (conf=0,
        root/pattern=<unk>) - the same degrade-gracefully behaviour MATE
        already has for real unanalysable words, applied honestly to the
        case where no word is knowable yet.

        `fallback_bpe_ids` MUST already be ids in E_fb's own vocabulary
        (the *fallback* BPE tokenizer MATE was built with, `cfg.n_bpe` /
        `data/*/bpe_tokenizer.json`) - NOT ids from whatever tokenizer
        produced the generation output (e.g. the *control* BPE tokenizer
        used for the shared output head, a different, differently-sized
        vocabulary). Passing the wrong tokenizer's ids here indexes E_fb
        out of range and crashes CUDA (`EmbeddingBag` assertion) - this bug
        was hit for real during Remediation Phase 1's first generation run
        and is exactly why this docstring says so explicitly. The caller
        (e.g. a generation loop) must decode the generated token via
        whatever tokenizer produced it, then *re-encode* that text through
        the fallback tokenizer to get valid ids here.
        """
        return dict(root_id=0, pattern_id=0, conf=0.0, prc_ids=[], enc_ids=[],
                     bpe_ids=fallback_bpe_ids if fallback_bpe_ids else [0])


def build_mate_batch(items: list, morph_index: MorphIndex, device: torch.device) -> dict[str, torch.Tensor]:
    """`items` is a flat list of length B*T (row-major). Each item is either
    a `str` (a known word - real morphology lookup) or a `list[int]` (ids
    already in the *fallback* tokenizer's vocabulary, for a position whose
    word isn't known yet, e.g. a just-generated continuation token - routed
    through MorphIndex.lookup_bpe_only instead). See that method's
    docstring for why it must be fallback-tokenizer ids, not any other
    tokenizer's.
    """
    prc_ids, prc_offsets = [], []
    enc_ids, enc_offsets = [], []
    bpe_ids, bpe_offsets = [], []
    root_ids, pattern_ids, conf = [], [], []

    for item in items:
        m = morph_index.lookup(item) if isinstance(item, str) else MorphIndex.lookup_bpe_only(item)
        prc_offsets.append(len(prc_ids))
        prc_ids.extend(m["prc_ids"])
        enc_offsets.append(len(enc_ids))
        enc_ids.extend(m["enc_ids"])
        bpe_offsets.append(len(bpe_ids))
        bpe_ids.extend(m["bpe_ids"])
        root_ids.append(m["root_id"])
        pattern_ids.append(m["pattern_id"])
        conf.append(m["conf"])

    def t(x, dtype=torch.long):
        return torch.tensor(x, dtype=dtype, device=device)

    return dict(
        proclitic_ids=t(prc_ids), proclitic_offsets=t(prc_offsets),
        enclitic_ids=t(enc_ids), enclitic_offsets=t(enc_offsets),
        root_ids=t(root_ids), pattern_ids=t(pattern_ids),
        bpe_ids=t(bpe_ids), bpe_offsets=t(bpe_offsets),
        conf=t(conf, dtype=torch.float32).unsqueeze(-1),
    )
