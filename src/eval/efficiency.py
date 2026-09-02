"""Efficiency measurement: weight memory, latency, energy.

See docs/AraBit-1.58_IMPLEMENTATION.md sec:3.5 - three numbers, never
conflated:
  - Weight memory: P_t*(log2(3)/8) + 2*P_f bytes; report P_t and P_f
    separately (AraBit-1.58.tex eq:mem).
  - Latency, model only: ms/token, forward pass, warm cache, batch=1,
    discard first 20 iterations.
  - Latency, end-to-end: includes front-end preprocessing (CAMeL analysis for
    front_end=mate, BPE tokenization for front_end=bpe) - report both.
  - Energy: J/sequence via pynvml sampling at >=10 Hz, measured not
    estimated from FLOP counts.

Ternary packing here is simulated: BitLinear.weight is stored as an FP32
tensor whose *values* happen to be in {-1,0,1} after quantization, not as a
packed 1.58-bit buffer. P_t is counted correctly (so the memory-accounting
formula below reports the real theoretical bound), but no run in this repo
has measured an actual packed in-memory footprint — say so explicitly in the
manuscript rather than reporting a measured 1.58-bit footprint, per
IMPLEMENTATION.md sec:3.5's own instruction.
"""

from __future__ import annotations

import math
import time

import torch
from torch import nn

from src.model.bitlinear import BitLinear

LOG2_3 = math.log2(3)


def count_params(model: nn.Module) -> tuple[int, int]:
    """Returns (P_t, P_f): ternary params (BitLinear.weight only - NOT its
    RMSNorm/bias, which stay full precision per sec:3.1) vs everything else.
    """
    p_t = 0
    p_f = 0
    ternary_weight_ids = set()
    for module in model.modules():
        if isinstance(module, BitLinear):
            ternary_weight_ids.add(id(module.weight))
    for p in model.parameters():
        if id(p) in ternary_weight_ids:
            p_t += p.numel()
        else:
            p_f += p.numel()
    return p_t, p_f


def weight_memory_bytes(p_t: int, p_f: int) -> float:
    """AraBit-1.58.tex eq:mem: M = P_t*(log2(3)/8) + 2*P_f (P_f in BF16, 2 bytes)."""
    return p_t * (LOG2_3 / 8) + 2 * p_f


@torch.no_grad()
def measure_latency_model_only_ms_per_token(
    model: nn.Module, batch: dict, prefix_lens, batch_size: int, seq_len: int,
    device: torch.device, n_warmup: int = 20, n_iters: int = 50,
) -> float:
    """ms/token, forward pass only, batch=1, discards first n_warmup iterations."""
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(n_warmup):
        model(batch, prefix_lens, batch_size, seq_len)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        model(batch, prefix_lens, batch_size, seq_len)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    model.train()
    return 1000.0 * elapsed / (n_iters * batch_size * seq_len)


def measure_gpu_energy_joules(fn, sample_hz: float = 10.0) -> tuple[float, float]:
    """Runs fn() while sampling GPU power via pynvml at >=sample_hz,
    integrating power(W) * dt(s) = energy(J). Returns (joules, wall_seconds).
    CPU-only fallback: returns (nan, wall_seconds) since pynvml requires an
    NVIDIA GPU - never fabricate an energy number without one.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        t0 = time.perf_counter()
        fn()
        return float("nan"), time.perf_counter() - t0

    import threading

    samples = []
    stop = threading.Event()

    def sampler():
        period = 1.0 / sample_hz
        while not stop.is_set():
            t = time.perf_counter()
            mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
            samples.append((t, mw / 1000.0))  # watts
            time.sleep(period)

    thread = threading.Thread(target=sampler, daemon=True)
    t_start = time.perf_counter()
    thread.start()
    fn()
    t_end = time.perf_counter()
    stop.set()
    thread.join()
    pynvml.nvmlShutdown()

    if len(samples) < 2:
        return float("nan"), t_end - t_start

    joules = 0.0
    for i in range(1, len(samples)):
        dt = samples[i][0] - samples[i - 1][0]
        avg_w = (samples[i][1] + samples[i - 1][1]) / 2.0
        joules += avg_w * dt
    return joules, t_end - t_start
