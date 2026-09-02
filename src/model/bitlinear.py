"""Ternary quantization + straight-through estimator.

Implements equations (5)-(9) of the manuscript (docs/AraBit-1.58.tex, section
"The Ternary Backbone") per docs/AraBit-1.58_IMPLEMENTATION.md sec:3.1.

Non-negotiable details (do not relax without re-reading IMPLEMENTATION.md sec:3.1):
  - Keep an FP32 master copy of self.weight; gradients accumulate there.
  - RMSNorm before activation quantization, not after.
  - Ternary applies only to attention q/k/v/o and FFN projections. Embeddings,
    output head, RMSNorm gains, and biases stay full precision (BF16 in the spec;
    torch.float32 here since this repo does not yet do mixed-precision training).
  - Peak LR for ternary training ~2x the FP16 value; ~2% warmup (handled in
    src/train/schedule.py, not here).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def weight_quant(w: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    """Eq. (6)-(7): ternary-quantize a weight matrix.

    Returns (w_ternary, gamma) where gamma = mean(|w|) over the whole matrix
    and w_ternary = RoundClip(w / (gamma + eps), -1, +1) in {-1, 0, +1}.
    """
    gamma = w.abs().mean().clamp(min=eps)
    w_q = (w / (gamma + eps)).round().clamp(-1, 1)
    return w_q, gamma


def act_quant(x: Tensor, bits: int = 8, eps: float = 1e-5) -> tuple[Tensor, Tensor, int]:
    """Eq. (8): per-token absmax activation quantization to `bits` bits."""
    Qb = 2 ** (bits - 1)
    gamma_x = x.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    x_q = (x * Qb / gamma_x).clamp(-Qb + eps, Qb - eps)
    return x_q, gamma_x, Qb


class RMSNorm(nn.Module):
    """Full-precision sub-layer norm applied before activation quantization
    (IMPLEMENTATION.md sec:3.1: "Normalisation must come before activation
    quantization, not after.").
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


class BitLinear(nn.Module):
    """Ternary-weight, 8-bit-activation linear layer with STE (eq. 5-9).

    Only ever instantiate this for attention q/k/v/o and FFN up/down
    projections. Embeddings, the output head, and norm layers must use plain
    nn.Linear / nn.Embedding instead — see IMPLEMENTATION.md sec:3.1.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, act_bits: int = 8):
        super().__init__()
        self.act_bits = act_bits
        self.norm = RMSNorm(in_features)
        # FP32 master copy; gradients accumulate here (sec:3.1).
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.norm(x)
        x_q_val, gamma_x, Qb = act_quant(x_norm, bits=self.act_bits)
        x_q = x_norm + (x_q_val - x_norm).detach()

        w_q_val, gamma = weight_quant(self.weight)
        w_q = self.weight + (w_q_val - self.weight).detach()

        y = torch.nn.functional.linear(x_q, w_q) * (gamma * gamma_x / Qb)
        if self.bias is not None:
            y = y + self.bias
        return y


class NormLinear(nn.Module):
    """Full-precision counterpart to BitLinear: identical norm-before-projection
    structure (RMSNorm then linear), no quantization.

    Exists so the fp16 arm of the front_end x precision comparison is truly
    matched to the ternary arm at the *sub-layer* topology level (each
    projection sees its own RMSNorm'd input either way, per BitLinear's own
    per-projection SubLN design in IMPLEMENTATION.md sec:3.1's pseudocode) -
    not just matched in layer count/hidden size. Swapping BitLinear<->NormLinear
    per `precision` is the only thing that should differ between the ternary
    and fp16 configs (IMPLEMENTATION.md sec:0, "matched conditions").
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.norm = RMSNorm(in_features)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.norm(x))
