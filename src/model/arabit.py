"""Full AraBit model: front_end in {mate, bpe} x precision in {ternary, fp16}, switchable.

These two axes are the *only* things that may vary between the four experiment
cells — see docs/AraBit-1.58_IMPLEMENTATION.md sec:0 ("matched conditions").
Composes mate.py (front_end=mate) or a plain nn.Embedding (front_end=bpe, the
matched-budget control per src/tokenization/bpe.py: matched_vocab_size) for
the front end, and blocks.py (BitLinear-backed when precision=ternary) for
the transformer body.

Decoder-only, prefix-LM objective (results/NOTES.md "Decisions", 2026-08-31):
`forward` takes `prefix_lens` and builds the prefix-LM attention mask via
blocks.py: build_prefix_lm_mask.

front_end=mate operates at orthographic-token granularity (one sequence
position = one whitespace word), front_end=bpe at subword granularity — these
differ in sequence length by design (see results/NOTES.md "MATE input/output
granularity" and AraBit-1.58.tex sec:mate's own "shortens the effective
input" point), so the two arms are matched in layer count/hidden size/corpus/
schedule, not in tokenization granularity itself — that is exactly the
independent variable under study.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.model.bitlinear import RMSNorm
from src.model.blocks import TransformerBlock, build_prefix_lm_mask
from src.model.mate import MATE, MATEConfig


class AraBit(nn.Module):
    def __init__(
        self,
        front_end: str,
        precision: str,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        mate_cfg: MATEConfig | None = None,
    ):
        super().__init__()
        assert front_end in ("mate", "bpe")
        assert precision in ("ternary", "fp16")
        ternary = precision == "ternary"
        self.front_end = front_end

        if front_end == "bpe":
            self.embed = nn.Embedding(vocab_size, d_model)  # full precision, sec:3.1
        else:
            assert mate_cfg is not None and mate_cfg.d_model == d_model
            self.embed = MATE(mate_cfg)  # full precision, sec:3.1

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, ternary) for _ in range(n_layers)]
        )
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # full precision, sec:3.1

    def forward(
        self, batch: dict, prefix_lens: Tensor, batch_size: int, seq_len: int,
        key_valid: Tensor | None = None,
    ) -> Tensor:
        """`key_valid` ([B, T] bool, True=real token): see
        blocks.py: build_prefix_lm_mask's docstring — only needed for
        batched generation with left-padded, variable-length sources."""
        if self.front_end == "bpe":
            x = self.embed(batch["token_ids"])  # [B, T, D]
        else:
            flat = self.embed(
                batch["proclitic_ids"], batch["proclitic_offsets"],
                batch["enclitic_ids"], batch["enclitic_offsets"],
                batch["root_ids"], batch["pattern_ids"],
                batch["bpe_ids"], batch["bpe_offsets"], batch["conf"],
            )  # [B*T, D]
            x = flat.view(batch_size, seq_len, -1)

        attn_mask = build_prefix_lm_mask(prefix_lens, seq_len, key_valid)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.final_norm(x)
        return self.head(x)
