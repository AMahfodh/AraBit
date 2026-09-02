"""Matched-budget BPE control.

See docs/AraBit-1.58_IMPLEMENTATION.md sec:3.4: for the real Stage 2 ablation,
this must be trained on the same pre-training corpus as MATE, matched on
*parameter budget* (not vocab size) per:

    n_bpe_control * d_model  ~=  (n_root*d_root + n_pattern*d_pattern
                                  + (n_pro+n_enc)*d_clitic + W_p params
                                  + n_bpe_fallback*d_model)

Matching vocab size instead of parameters invalidates the comparison - document
the resulting numbers in the manuscript's Table 8. `train_bpe`/`Tokenizer`
below is used as-is (vocab-size-only) for Stage 0(e)'s precision-only smoke
test, which does not compare front ends, so budget-matching does not apply
yet - flagged so this isn't mistaken for the real Stage-2-ready control.
"""

from __future__ import annotations

from collections.abc import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]


def train_bpe(sentences: Iterable[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    return tokenizer


def matched_vocab_size(
    n_root: int,
    n_pattern: int,
    n_proclitic: int,
    n_enclitic: int,
    d_root: int,
    d_pattern: int,
    d_clitic: int,
    n_bpe_fallback: int,
    d_model: int,
) -> int:
    """IMPLEMENTATION.md sec:3.4's matched-parameter-budget formula:

        n_bpe_control * d_model ~= n_root*d_root + n_pattern*d_pattern
                                    + (n_pro+n_enc)*d_clitic + W_p params
                                    + n_bpe_fallback*d_model

    (gate module excluded, per the formula as literally given). Returns the
    BPE control's vocab size. Note per results/NOTES.md "MATE pattern field
    used as-is": n_pattern is BPE-vocab-scale here (CAMeL's raw `pattern`
    feature, not an abstract template class), so this is not the small
    number the manuscript originally envisioned - it is what it is,
    computed from the real vocab, not assumed.
    """
    w_p_in = 2 * d_clitic + d_root + d_pattern
    w_p_params = w_p_in * d_model + d_model  # + bias
    target_params = (
        n_root * d_root
        + n_pattern * d_pattern
        + (n_proclitic + n_enclitic) * d_clitic
        + w_p_params
        + n_bpe_fallback * d_model
    )
    return round(target_params / d_model)
