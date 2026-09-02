"""Pre-training entrypoint. Selects configs/exp/*.yaml + configs/model/*.yaml via
Hydra and calls src/train/pretrain.py.

See docs/AraBit-1.58_IMPLEMENTATION.md §4 for the Stage 0-2 order — do not skip to
Stage 2 (small model, full 2x2, 3 seeds) before Stage 0/1 pass.
"""
