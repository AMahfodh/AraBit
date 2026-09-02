"""Batched greedy generation for both front_ends (Remediation Phase 2,
results/NOTES.md: "cheap, no retraining" wasn't true at batch=1 with no
KV-cache — extrapolated to ~14h for the full SAMER test split. This module
batches B examples per forward pass instead of one at a time, which is
where nearly all the wall-clock time was going (a tiny model on an
under-utilized 8GB GPU at batch=1).

Sources are left-padded to a common length within each batch so every
example starts generating at the same sequence position — with a real
attention-level fix (src/model/blocks.py: build_prefix_lm_mask's
`key_valid` argument) so the model never attends to the artificial padding,
avoiding a train/inference mismatch (this model was trained with only
right-padding, in a region no real position ever attends to).

Still no KV-cache: each step re-runs the full forward pass over the
growing sequence. Only the across-example batching is new. Real speedup
comes from GPU utilization at batch>1, not from reducing the O(T^2) total
compute of the naive loop.
"""

from __future__ import annotations

import torch

from src.data.mate_batch import MorphIndex, build_mate_batch, words_per_bpe_position


@torch.no_grad()
def generate_bpe_batched(model, sources, tokenizer, device, batch_size=16, max_new_tokens=64):
    bos_id, eos_id, pad_id = (tokenizer.token_to_id(t) for t in ("<bos>", "<eos>", "<pad>"))
    outputs = [None] * len(sources)

    for start in range(0, len(sources), batch_size):
        batch_sources = sources[start : start + batch_size]
        B = len(batch_sources)
        src_ids_list = [tokenizer.encode(s).ids for s in batch_sources]
        real_lens = [len(ids) for ids in src_ids_list]
        max_src_len = max(real_lens)

        seqs, valid = [], []
        for ids in src_ids_list:
            pad_len = max_src_len - len(ids)
            seqs.append([pad_id] * pad_len + ids + [bos_id])
            valid.append([False] * pad_len + [True] * (len(ids) + 1))
        cur = torch.tensor(seqs, dtype=torch.long, device=device)
        key_valid = torch.tensor(valid, dtype=torch.bool, device=device)
        prefix_len = max_src_len + 1
        prefix_lens_t = torch.full((B,), prefix_len, dtype=torch.long, device=device)

        done = torch.zeros(B, dtype=torch.bool, device=device)
        finished_at = [None] * B

        for step in range(max_new_tokens):
            T = cur.shape[1]
            logits = model(dict(token_ids=cur), prefix_lens_t, B, T, key_valid=key_valid)
            next_ids = logits[:, -1].argmax(-1)
            next_ids = torch.where(done, torch.full_like(next_ids, pad_id), next_ids)

            newly_done = (next_ids == eos_id) & ~done
            for i in torch.nonzero(newly_done).flatten().tolist():
                finished_at[i] = step
            done = done | (next_ids == eos_id)

            cur = torch.cat([cur, next_ids.unsqueeze(1)], dim=1)
            key_valid = torch.cat([key_valid, torch.ones(B, 1, dtype=torch.bool, device=device)], dim=1)
            if bool(done.all()):
                break

        for i in range(B):
            end = prefix_len + (finished_at[i] if finished_at[i] is not None else max_new_tokens)
            gen_ids = cur[i, prefix_len:end].tolist()
            outputs[start + i] = tokenizer.decode(gen_ids)

    return outputs


@torch.no_grad()
def generate_mate_batched(
    model, sources, tokenizer, fallback_tokenizer, morph_index: MorphIndex, device,
    batch_size=16, max_new_tokens=64,
):
    """`tokenizer` is the control BPE tokenizer (shared output vocab).
    `fallback_tokenizer` is MATE's own E_fb vocabulary - a DIFFERENT,
    smaller tokenizer; see src/data/mate_batch.py: MorphIndex.lookup_bpe_only
    for why re-encoding through it is required, not optional."""
    bos_id, eos_id = tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")
    outputs = [None] * len(sources)

    for start in range(0, len(sources), batch_size):
        batch_sources = sources[start : start + batch_size]
        B = len(batch_sources)
        src_words_list = [words_per_bpe_position(s, tokenizer) for s in batch_sources]
        real_lens = [len(w) for w in src_words_list]
        max_src_len = max(real_lens)

        items_grid, valid = [], []
        for words in src_words_list:
            pad_len = max_src_len - len(words)
            items_grid.append(["<pad>"] * pad_len + list(words) + ["<bos>"])
            valid.append([False] * pad_len + [True] * (len(words) + 1))
        prefix_len = max_src_len + 1
        prefix_lens_t = torch.full((B,), prefix_len, dtype=torch.long, device=device)

        done = torch.zeros(B, dtype=torch.bool, device=device)
        finished_at = [None] * B
        gen_ids_grid = [[] for _ in range(B)]

        for step in range(max_new_tokens):
            T = prefix_len + step
            flat_items = [item for row in items_grid for item in row]
            batch = build_mate_batch(flat_items, morph_index, device)
            key_valid = torch.tensor(valid, dtype=torch.bool, device=device)
            logits = model(batch, prefix_lens_t, B, T, key_valid=key_valid)
            next_ids = logits[:, -1].argmax(-1)

            newly_done = (next_ids == eos_id) & ~done
            for i in torch.nonzero(newly_done).flatten().tolist():
                finished_at[i] = step
            done_list = done.tolist()
            done = done | (next_ids == eos_id)

            for i in range(B):
                if done_list[i]:
                    items_grid[i].append("<pad>")
                    valid[i].append(True)  # harmless: this row's output is already finalized
                    continue
                nid = next_ids[i].item()
                gen_ids_grid[i].append(nid)
                piece_text = tokenizer.decode([nid])
                fallback_ids = fallback_tokenizer.encode(piece_text).ids if piece_text.strip() else []
                items_grid[i].append(fallback_ids)
                valid[i].append(True)

            if bool(done.all()):
                break

        for i in range(B):
            end = finished_at[i] if finished_at[i] is not None else len(gen_ids_grid[i])
            outputs[start + i] = tokenizer.decode(gen_ids_grid[i][:end])

    return outputs
