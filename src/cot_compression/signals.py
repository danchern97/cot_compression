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

# The per-token signals that patching/pooling can key off. "entropy" is the
# predictive entropy of the distribution that produced a CoT token; "surprisal"
# is -log p of the realized token under that same distribution. Both come out of
# one forward pass and one log_softmax (see _sequence_signals).
SIGNALS: tuple[str, ...] = ("entropy", "surprisal")

# npz filename stem per signal. entropy keeps its historical name so a cache
# built before surprisal existed still loads unchanged.
_SIGNAL_STEM = {"entropy": "cot_entropies", "surprisal": "cot_surprisals"}


def signal_cache_path(cache_dir: Path, model_name: str, signal: str) -> Path:
    """Cache filename keyed by model *and* signal so a wrong cache can't load."""
    if signal not in _SIGNAL_STEM:
        raise ValueError(f"Unknown signal: {signal!r}. Expected one of {SIGNALS}.")
    slug = model_name.replace("/", "__").replace(" ", "_")
    return Path(cache_dir) / f"{_SIGNAL_STEM[signal]}__{slug}.npz"


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


def save_signal_cache(path: Path, values: dict[int, torch.Tensor]) -> None:
    """Store per-sample CoT-token signal values as a flat npz with offsets.

    sample i's values = ``values[offsets[i]:offsets[i+1]]``. ``cot_lengths`` (=
    per-sample original CoT token count) is stored explicitly; it doubles as the
    "original length saved once" artifact. The values live under ``values``;
    ``load_signal_cache`` also reads the legacy ``entropies`` key so an entropy
    cache written before this refactor still loads.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    indices, offsets, flat, lengths = _ragged(values)
    np.savez(
        path,
        sample_index=indices,
        offsets=offsets,
        values=flat,
        cot_lengths=lengths.astype(np.int32),
    )


def load_signal_cache(path: Path) -> dict[int, torch.Tensor]:
    """Inverse of save_signal_cache: sample_index -> float32 signal tensor.

    Reads the current ``values`` key, falling back to the legacy ``entropies``
    key so a pre-refactor entropy cache is loaded transparently.
    """
    with np.load(path) as data:
        indices = data["sample_index"]
        offsets = data["offsets"]
        flat = data["values"] if "values" in data else data["entropies"]
        return {
            int(idx): torch.from_numpy(flat[offsets[i] : offsets[i + 1]].copy())
            for i, idx in enumerate(indices)
        }


# Rows (positions) processed per softmax chunk. Bounds the float32 [chunk, vocab]
# working set so a full-vocab (~152k) signal softmax over long sequences can't
# OOM, regardless of batch*length. ~chunk*vocab*4 bytes per intermediate.
_SIGNAL_CHUNK_ROWS = 4096


def _sequence_signals(
    logits: torch.Tensor,
    targets: torch.Tensor,
    signals: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Per-position ``[batch, length]`` signal tensors, computed in float32.

    ``entropy`` is the predictive entropy of each position's distribution;
    ``surprisal`` is ``-log p`` of the realized token, where ``targets[b, p]``
    is the id that position ``p`` produced (i.e. ``input_ids[b, p+1]``). Both are
    read off the *same* float32 ``log_softmax``, so requesting both costs one
    forward and one softmax, not two.

    Chunked over the flattened position dimension so the transient float32
    ``[rows, vocab]`` softmax never spans the whole ``[batch, length, vocab]``
    logits at once (that materialization is what OOMs for long CoTs). Values are
    identical to the unchunked computation.
    """
    batch, length, vocab = logits.shape
    flat = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)
    outs = {
        name: torch.empty(flat.shape[0], dtype=torch.float32, device=logits.device)
        for name in signals
    }
    for start in range(0, flat.shape[0], _SIGNAL_CHUNK_ROWS):
        chunk = flat[start : start + _SIGNAL_CHUNK_ROWS].float()
        log_probs = F.log_softmax(chunk, dim=-1)
        stop = start + chunk.shape[0]
        if "entropy" in outs:
            outs["entropy"][start:stop] = -(log_probs.exp() * log_probs).sum(dim=-1)
        if "surprisal" in outs:
            gathered = log_probs.gather(
                1, flat_targets[start:stop].unsqueeze(1)
            ).squeeze(1)
            outs["surprisal"][start:stop] = -gathered
    return {name: out.reshape(batch, length) for name, out in outs.items()}


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


def compute_cot_signals(
    model: Any,
    tokenizer: Any,
    examples: Any,
    sample_indices: Iterable[int],
    batch_size: int,
    max_batch_tokens: int | None,
    device: torch.device,
    signals: Sequence[str] = SIGNALS,
) -> dict[str, dict[int, torch.Tensor]]:
    """Batched per-token CoT ``signals`` keyed by ``signal`` then ``sample_index``.

    Runs the model over (prompt prefix + CoT) token ids (right-padded with an
    attention mask; causal attention leaves the real-token logits unaffected by
    trailing pad). The CoT is never scored in isolation — the prompt is always
    prepended so each CoT token's signal is conditioned on the real context.

    The value assigned to CoT token ``j`` comes from the predictive distribution
    that **produced** it — i.e. ``logits`` at the *preceding* position (which
    predicts token ``j``), not the distribution at ``j`` (which predicts token
    ``j+1``). For a sequence ``prefix + cot`` of length ``prefix_len + cot_len``,
    that is positions ``[prefix_len - 1, prefix_len + cot_len - 1)`` — length
    ``cot_len``. Surprisal uses the same slice: the target of position ``p`` is
    ``full_ids[p + 1]``, so the surprisal read at ``prefix_len + j - 1`` is
    ``-log p`` of exactly CoT token ``j``.

    Samples whose trace/CoT can't be extracted, or that have no prompt prefix
    (no distribution can produce the first CoT token), are skipped.
    """
    unknown = [name for name in signals if name not in SIGNALS]
    if unknown:
        raise ValueError(f"Unknown signals {unknown}. Expected a subset of {SIGNALS}.")
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
        items.append(
            (sample_index, prefix_ids + cot_ids, len(prefix_ids), len(cot_ids))
        )

    result: dict[str, dict[int, torch.Tensor]] = {name: {} for name in signals}
    for batch in _greedy_batches(items, batch_size, max_batch_tokens):
        max_len = max(len(full_ids) for _, full_ids, _, _ in batch)
        input_rows, attn_rows = [], []
        for _, full_ids, _, _ in batch:
            pad = max_len - len(full_ids)
            input_rows.append(full_ids + [pad_token_id] * pad)
            attn_rows.append([1] * len(full_ids) + [0] * pad)
        input_tensor = torch.tensor(input_rows, dtype=torch.long, device=device)
        attention_tensor = torch.tensor(attn_rows, dtype=torch.long, device=device)
        # target[p] = token id that position p predicts = input_ids[p + 1]. The
        # last column has no successor; pad it (that column is never sliced into
        # the CoT range, which ends at prefix_len + cot_len - 1 <= max_len - 1).
        target_tensor = torch.roll(input_tensor, shifts=-1, dims=1)
        target_tensor[:, -1] = pad_token_id

        with torch.no_grad():
            logits = model(
                input_ids=input_tensor, attention_mask=attention_tensor
            ).logits
            values = _sequence_signals(logits, target_tensor, signals)

        for row, (sample_index, _, prefix_len, cot_len) in enumerate(batch):
            # Value that produced CoT token j is read at logits[prefix_len + j - 1].
            span = slice(prefix_len - 1, prefix_len + cot_len - 1)
            for name in signals:
                result[name][sample_index] = values[name][row, span].cpu()
    return result
