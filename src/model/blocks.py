"""Transformer block: RMSNorm, RoPE, squared-ReLU FFN, prefix-LM attention masking.

See docs/AraBit-1.58_IMPLEMENTATION.md sec:2 (repo layout) and results/NOTES.md
"Decisions" (2026-08-31, user-confirmed): decoder-only architecture, prefix-LM
objective — attention is bidirectional over each sequence's prefix segment and
causal over the continuation that must be generated.

Ternary quantization is applied via BitLinear (bitlinear.py) inside attention
q/k/v/o and the FFN projections only — everything else (RMSNorm, RoPE has no
learned params) stays full precision, per IMPLEMENTATION.md sec:3.1.

Normalization follows BitLinear's own per-projection SubLN design (each
BitLinear/NormLinear owns its own RMSNorm immediately before it, per
IMPLEMENTATION.md sec:3.1's forward-pass pseudocode) rather than a single
shared pre-block norm — this is also what keeps the ternary and fp16 arms
genuinely matched at the sub-layer level (NormLinear is BitLinear's
full-precision twin, see bitlinear.py). AraBit-1.58.tex's Figure 1 draws one
RMSNorm box per sub-layer as a simplified visual; the real topology is one
RMSNorm per projection. Noted in results/NOTES.md.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.model.bitlinear import BitLinear, NormLinear


def build_prefix_lm_mask(prefix_lens: Tensor, seq_len: int, key_valid: Tensor | None = None) -> Tensor:
    """Boolean attention mask, True = attend allowed, shape [batch, seq_len, seq_len].

    For each sequence, positions within the prefix (index < prefix_len) attend
    bidirectionally to the whole prefix; positions in the continuation
    (index >= prefix_len) attend causally (to themselves and everything before).

    `key_valid` (optional, [batch, seq_len] bool, True = real token): batched
    generation (Remediation Phase 2, results/NOTES.md) left-pads shorter
    sources so every sequence in a batch starts generating at the same
    position — without this, a real (non-pad) query could legally attend to
    the artificial left-padding under the plain causal/prefix rule above,
    a train/inference mismatch (this model was never trained with padding
    inside its prefix). ANDing key_valid in removes exactly that: a key
    position is only attendable if BOTH the causal/prefix rule allows it AND
    it isn't padding.
    """
    device = prefix_lens.device
    idx = torch.arange(seq_len, device=device)
    query_idx = idx.view(1, seq_len, 1)
    key_idx = idx.view(1, 1, seq_len)
    prefix_len = prefix_lens.view(-1, 1, 1)

    causal = key_idx <= query_idx
    prefix_bidirectional = (query_idx < prefix_len) & (key_idx < prefix_len)
    mask = causal | prefix_bidirectional
    if key_valid is not None:
        mask = mask & key_valid.view(key_valid.shape[0], 1, seq_len)
    return mask


def apply_rope(x: Tensor, base: float = 10000.0) -> Tensor:
    """Rotary position embedding, applied per attention head.

    x: [batch, heads, seq_len, head_dim] (head_dim must be even).
    """
    *_, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"
    device, dtype = x.device, x.dtype

    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(pos, theta)  # [seq_len, head_dim/2]
    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]

    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ternary: bool):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        Proj = BitLinear if ternary else NormLinear
        self.q_proj = Proj(d_model, d_model)
        self.k_proj = Proj(d_model, d_model)
        self.v_proj = Proj(d_model, d_model)
        self.o_proj = Proj(d_model, d_model)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q), apply_rope(k)

        # attn_mask: [batch, t, t] bool, True = allowed -> additive bias for SDPA
        bias = torch.zeros(attn_mask.shape, dtype=q.dtype, device=q.device)
        bias.masked_fill_(~attn_mask, float("-inf"))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias.unsqueeze(1))

        out = out.transpose(1, 2).reshape(b, t, d)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """Squared-ReLU FFN (IMPLEMENTATION.md sec:2's blocks.py note)."""

    def __init__(self, d_model: int, d_ff: int, ternary: bool):
        super().__init__()
        Proj = BitLinear if ternary else NormLinear
        self.up = Proj(d_model, d_ff)
        self.down = Proj(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.relu(self.up(x))
        return self.down(h * h)


class TransformerBlock(nn.Module):
    """Residual + attention, residual + FFN. No block-level pre-norm: each
    projection inside Attention/FeedForward owns its own RMSNorm (SubLN) —
    see this module's docstring.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, ternary: bool):
        super().__init__()
        self.attn = Attention(d_model, n_heads, ternary)
        self.ffn = FeedForward(d_model, d_ff, ternary)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x = x + self.attn(x, attn_mask)
        x = x + self.ffn(x)
        return x
