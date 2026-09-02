"""MATE - the morphology-aware embedding front end.

Implements docs/AraBit-1.58_IMPLEMENTATION.md sec:3.2's forward-pass spec (which
takes precedence over the manuscript's eq. 3-4 where they differ: the gate here is
per-dimension, nn.Linear(2*d_model+1, d_model) -> sigmoid, not the manuscript's
scalar w_g^T gate).

Index convention for root/pattern vocabs: index 0 is reserved for <unk_root> /
<unk_pattern>. tokenization/vocab.py must respect this when building the vocab.

Edge cases this module must handle without crashing (see IMPLEMENTATION.md sec:3.2
and tests/test_mate.py):
  - token has no analysis -> conf=0, root/pattern=<unk> (index 0)
  - token is a digit or Latin string -> tokenization/morph.py assigns conf=0 and
    <unk> root/pattern for these; MATE itself has no special-cased digit/Latin
    logic, it just needs to be robust to conf=0 + <unk> ids, which is the same
    path as "no analysis"
  - token has 3 stacked proclitics (e.g. wa+bi+al) -> all summed via EmbeddingBag
  - root is OOV (unseen radical combination) -> <unk_root>, not a crash

Log the mean gate value per token category every N steps (IMPLEMENTATION.md
sec:3.2 point 5) - that's a training-loop diagnostic, hooked in from src/train/,
not implemented in this module.

Component-level ablation (`MATEConfig.ablation`, AraBit-1.58.tex tab:ablate2,
Stage 4): two of the manuscript's five ablation rows needed a concrete
engineering decision not fully specified by the manuscript text, made and
logged here (see results/NOTES.md "Stage 4 component ablation" for the full
rationale):
  - "root embedding (stem embedding instead)": implemented as simply
    dropping E_root from the concatenation (parallel to how "- pattern
    embedding" drops E_pat), NOT as a genuine separate stem embedding table
    - building one needs a "stem" field cached per token, which
    scripts/01_cache_morphology.py does not currently extract. This measures
    "cost of losing root information," not literally "root vs. stem," and
    is documented as a substitution, not silently treated as identical.
  - "clitic separation (surface form only)": implemented as dropping the
    proclitic/enclitic embeddings (c_plus/c_minus) from the concatenation.
  - "learned gate (hard fallback)": g = conf (broadcast over d_model),
    i.e. a deterministic gate driven directly by analyser confidence
    instead of a learned sigmoid - matches "hard fallback" literally (fully
    trust morphology when the analyser succeeded, fully fall back when it
    didn't), no interpretation gap here.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


ABLATIONS = ("full", "no_pattern", "no_root", "no_clitics", "hard_gate")


class MATEConfig:
    def __init__(
        self,
        n_proclitic: int,
        n_enclitic: int,
        n_root: int,
        n_pattern: int,
        n_bpe: int,
        d_clitic: int,
        d_root: int,
        d_pattern: int,
        d_model: int,
        ablation: str = "full",
    ):
        assert ablation in ABLATIONS
        self.n_proclitic = n_proclitic
        self.n_enclitic = n_enclitic
        self.n_root = n_root
        self.n_pattern = n_pattern
        self.n_bpe = n_bpe
        self.d_clitic = d_clitic
        self.d_root = d_root
        self.d_pattern = d_pattern
        self.d_model = d_model
        self.ablation = ablation


class MATE(nn.Module):
    def __init__(self, cfg: MATEConfig):
        super().__init__()
        self.cfg = cfg
        a = cfg.ablation
        self.use_clitics = a != "no_clitics"
        self.use_root = a != "no_root"
        self.use_pattern = a != "no_pattern"
        self.hard_gate = a == "hard_gate"

        if self.use_clitics:
            self.E_pro = nn.EmbeddingBag(cfg.n_proclitic, cfg.d_clitic, mode="sum")
            self.E_enc = nn.EmbeddingBag(cfg.n_enclitic, cfg.d_clitic, mode="sum")
        if self.use_root:
            self.E_root = nn.Embedding(cfg.n_root, cfg.d_root, padding_idx=0)
        if self.use_pattern:
            self.E_pat = nn.Embedding(cfg.n_pattern, cfg.d_pattern, padding_idx=0)
        self.E_fb = nn.EmbeddingBag(cfg.n_bpe, cfg.d_model, mode="mean")

        w_p_in = (
            (2 * cfg.d_clitic if self.use_clitics else 0)
            + (cfg.d_root if self.use_root else 0)
            + (cfg.d_pattern if self.use_pattern else 0)
        )
        assert w_p_in > 0, "at least one of clitics/root/pattern must remain"
        self.W_p = nn.Linear(w_p_in, cfg.d_model)
        if not self.hard_gate:
            self.gate = nn.Linear(2 * cfg.d_model + 1, cfg.d_model)

    def forward(
        self,
        proclitic_ids: Tensor,
        proclitic_offsets: Tensor,
        enclitic_ids: Tensor,
        enclitic_offsets: Tensor,
        root_ids: Tensor,
        pattern_ids: Tensor,
        bpe_ids: Tensor,
        bpe_offsets: Tensor,
        conf: Tensor,
        return_gate: bool = False,
    ) -> Tensor:
        """All *_ids/*_offsets pairs follow nn.EmbeddingBag's flat+offsets
        convention: one row of ids per token is not assumed, instead ids for
        all tokens in the batch are concatenated and `offsets` marks each
        token's start (this is what lets a variable clitic/BPE-piece count per
        token be handled without padding). root_ids/pattern_ids/conf are one
        value per token (shape [n_tokens], [n_tokens], [n_tokens, 1]).
        """
        parts = []
        if self.use_clitics:
            parts.append(self.E_pro(proclitic_ids, proclitic_offsets))  # eq. 1, c_plus
        if self.use_root:
            parts.append(self.E_root(root_ids))
        if self.use_pattern:
            parts.append(self.E_pat(pattern_ids))
        if self.use_clitics:
            parts.append(self.E_enc(enclitic_ids, enclitic_offsets))  # eq. 1, c_minus

        morph_in = torch.cat(parts, dim=-1)
        e_morph = self.W_p(morph_in)  # eq. 2

        e_fb = self.E_fb(bpe_ids, bpe_offsets)  # mean-pooled BPE fallback per token

        if self.hard_gate:
            g = conf.expand(-1, e_morph.shape[-1])  # "learned gate (hard fallback)": g = conf
        else:
            g = torch.sigmoid(self.gate(torch.cat([e_morph, e_fb, conf], dim=-1)))  # eq. 3
        e = g * e_morph + (1 - g) * e_fb  # eq. 4
        if return_gate:
            # mean over d_model, since g is per-dimension (sec:3.2) but the
            # gate-behaviour analysis (IMPLEMENTATION.md sec:3.2 point 5,
            # AraBit-1.58.tex sec:errors) reports one scalar per token.
            return e, g.mean(dim=-1)
        return e
