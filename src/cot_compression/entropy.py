from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from cot_compression.data.answers import (
    cot_token_ids,
    extract_answer_trace,
    prefix_token_ids,
)


def entropy_cache_path(cache_dir: Path, model_name: str) -> Path:
    """Cache filename keyed by model so a wrong-model cache can't load."""
    slug = model_name.replace("/", "__").replace(" ", "_")
    return Path(cache_dir) / f"cot_entropies__{slug}.npz"


def _ragged(
    arrays_by_index: dict[int, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten variable-length per-sample arrays into (indices, offsets, flat,
    lengths). Sample i's slice = ``flat[offsets[i]:offsets[i+1]]``."""
    indices = sorted(arrays_by_index)
    arrays = [arrays_by_index[i].to(torch.float32).cpu().numpy() for i in indices]
    lengths = np.asarray([a.shape[0] for a in arrays], dtype=np.int64)
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    flat = (
        np.concatenate(arrays).astype(np.float32)
        if arrays
        else np.zeros(0, dtype=np.float32)
    )
    return np.asarray(indices, dtype=np.int64), offsets, flat, lengths


def save_entropies_npz(path: Path, arrays_by_index: dict[int, torch.Tensor]) -> None:
    """Store ragged per-sample entropies (sample_index / offsets / entropies)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    indices, offsets, flat, _ = _ragged(arrays_by_index)
    np.savez(path, sample_index=indices, offsets=offsets, entropies=flat)


def save_entropy_cache(path: Path, entropies: dict[int, torch.Tensor]) -> None:
    """Store per-sample CoT-token entropies as a flat npz with offsets.

    sample i's entropies = ``entropies[offsets[i]:offsets[i+1]]``. ``cot_lengths``
    (= per-sample original CoT token count) is stored explicitly; it doubles as
    the "original length saved once" artifact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    indices, offsets, flat, lengths = _ragged(entropies)
    np.savez(
        path,
        sample_index=indices,
        offsets=offsets,
        entropies=flat,
        cot_lengths=lengths.astype(np.int32),
    )


def load_entropy_cache(path: Path) -> dict[int, torch.Tensor]:
    """Inverse of save_entropy_cache: sample_index -> float32 entropy tensor."""
    with np.load(path) as data:
        indices = data["sample_index"]
        offsets = data["offsets"]
        flat = data["entropies"]
        return {
            int(idx): torch.from_numpy(flat[offsets[i] : offsets[i + 1]].copy())
            for i, idx in enumerate(indices)
        }


# Rows (positions) processed per softmax chunk. Bounds the float32 [chunk, vocab]
# working set so a full-vocab (~152k) entropy over long sequences can't OOM,
# regardless of batch*length. ~chunk*vocab*4 bytes per intermediate.
_ENTROPY_CHUNK_ROWS = 4096


def _sequence_entropies(logits: torch.Tensor) -> torch.Tensor:
    """Per-position predictive entropy over the vocab, computed in float32.

    Chunked over the flattened position dimension so the transient float32
    ``[rows, vocab]`` softmax never spans the whole ``[batch, length, vocab]``
    logits at once (that materialization is what OOMs for long CoTs). Values are
    identical to the unchunked computation.
    """
    batch, length, vocab = logits.shape
    flat = logits.reshape(-1, vocab)
    out = torch.empty(flat.shape[0], dtype=torch.float32, device=logits.device)
    for start in range(0, flat.shape[0], _ENTROPY_CHUNK_ROWS):
        chunk = flat[start : start + _ENTROPY_CHUNK_ROWS].float()
        log_probs = F.log_softmax(chunk, dim=-1)
        out[start : start + chunk.shape[0]] = -(log_probs.exp() * log_probs).sum(dim=-1)
    return out.reshape(batch, length)


def _greedy_batches(
    items: Sequence[tuple[int, list[int], int, int]],
    batch_size: int,
    max_batch_tokens: int | None,
) -> Iterable[list[tuple[int, list[int], int, int]]]:
    batch: list[tuple[int, list[int], int, int]] = []
    max_len = 0
    for item in items:
        length = len(item[1])
        would_exceed = (
            max_batch_tokens is not None
            and batch
            and max(max_len, length) * (len(batch) + 1) > max_batch_tokens
        )
        if would_exceed or len(batch) >= batch_size:
            yield batch
            batch, max_len = [], 0
        batch.append(item)
        max_len = max(max_len, length)
    if batch:
        yield batch


def compute_cot_entropies(
    model: Any,
    tokenizer: Any,
    examples: Any,
    sample_indices: Iterable[int],
    batch_size: int,
    max_batch_tokens: int | None,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Batched per-token CoT entropies keyed by dataset ``sample_index``.

    Runs the model over (prompt prefix + CoT) token ids (right-padded with an
    attention mask; causal attention leaves the real-token logits unaffected by
    trailing pad). The CoT is never scored in isolation — the prompt is always
    prepended so each CoT token's entropy is conditioned on the real context.

    The entropy assigned to CoT token ``j`` is the entropy of the predictive
    distribution that **produced** it — i.e. ``logits`` at the *preceding*
    position (which predicts token ``j``), not the distribution at ``j`` (which
    predicts token ``j+1``). For a sequence ``prefix + cot`` of length
    ``prefix_len + cot_len``, that is positions
    ``[prefix_len - 1, prefix_len + cot_len - 1)`` — length ``cot_len``.

    Samples whose trace/CoT can't be extracted, or that have no prompt prefix
    (no distribution can produce the first CoT token), are skipped.
    """
    pad_token_id = int(tokenizer.pad_token_id)
    items: list[tuple[int, list[int], int, int]] = []
    for sample_index in sample_indices:
        trace = extract_answer_trace(examples[sample_index]["messages"])
        if trace is None:
            continue
        try:
            cot_ids = cot_token_ids(trace, tokenizer)
        except ValueError:
            continue
        prefix_ids = prefix_token_ids(trace, tokenizer)
        if not prefix_ids:
            continue
        items.append((sample_index, prefix_ids + cot_ids, len(prefix_ids), len(cot_ids)))

    result: dict[int, torch.Tensor] = {}
    for batch in _greedy_batches(items, batch_size, max_batch_tokens):
        max_len = max(len(full_ids) for _, full_ids, _, _ in batch)
        input_rows, attn_rows = [], []
        for _, full_ids, _, _ in batch:
            pad = max_len - len(full_ids)
            input_rows.append(full_ids + [pad_token_id] * pad)
            attn_rows.append([1] * len(full_ids) + [0] * pad)
        input_tensor = torch.tensor(input_rows, dtype=torch.long, device=device)
        attention_tensor = torch.tensor(attn_rows, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(
                input_ids=input_tensor, attention_mask=attention_tensor
            ).logits
            entropies = _sequence_entropies(logits)

        for row, (sample_index, _, prefix_len, cot_len) in enumerate(batch):
            # Entropy that produced CoT token j = logits[prefix_len + j - 1].
            result[sample_index] = entropies[
                row, prefix_len - 1 : prefix_len + cot_len - 1
            ].cpu()
    return result
