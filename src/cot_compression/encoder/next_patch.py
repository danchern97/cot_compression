"""Auxiliary next-patch supervision: predict patch j+1 from codes 0..j.

The objective is: from ``[prompt; z_0..z_j]``, predict the tokens of patch *j+1*,
teacher-forced, through the **frozen** decoder, with ordinary cross-entropy. It
exists because answer likelihood alone is a very thin signal -- a measured 0.247
supervised tokens per latent slot -- which is why the held-out gap moves so slowly
and is nearly insensitive to codebook quality.

Two facts force the design, and both are worth stating before the code:

**One sequence per (row, patch) is ~227x a training step.** Each patch would re-read
the prompt and the growing code prefix: ``sum_j (P + j + rho)`` is ~1.38 M tokens
per row against Pass A's 6.1 K. The ``K^2/2`` term is fatal, so every patch has to
be supervised inside ONE forward pass.

**One forward with a plain causal mask leaks.** Patch *j+1* would attend to patches
1..*j* in raw form, and raw tokens are strictly more informative than their own
lossy summaries -- the codes become redundant and the encoder's gradient collapses
toward zero. So the pass needs a block mask that lets a patch see the prompt, the
codes *before* it, and its own prefix, and nothing else.

Everything here is pure index arithmetic over the batch tensors: no model, no CUDA
state, and no randomness. That is what makes it testable against the naive
per-patch loop, which is the specification executed literally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


@dataclass(frozen=True)
class NextPatchBatch:
    """The geometry of one aux pass, indexed by COLUMN of the `[B, W]` sequence.

    Row `b` is `[its Pass B prefix (prompt, <think>, codes); its aux tokens; padding]`,
    with the aux block starting at that row's OWN `prefix_len[b]`. Every per-position
    tensor is shaped `[B, W]` and describes a column directly, never "the i-th aux
    token", for two reasons:

    * A row-offset layout cannot be misaligned. The earlier token-indexed form built
      the mask and the RoPE positions as if every aux block began at
      `max(prefix_len)`, which was silently wrong for every shorter row of a
      length-grouped batch while single-row tests passed.
    * The FlexAttention `mask_mod` captures these tensors, so their shapes enter the
      compiled graph. `[B, W]` makes the compiled shape a function of `(B, W)` alone;
      a token-indexed `[B, span]` tensor would recompile on every change of span.
    """

    width: int
    """`W`, the sequence length. At least the natural `max(prefix_len + aux_count)`."""
    is_prefix: Tensor
    """`[B, W]` bool -- a real Pass B prefix token."""
    patch: Tensor
    """`[B, W]` long -- the patch index of an aux token; -1 on prefix and padding."""
    code_limit: Tensor
    """`[B, W]` long -- the last column an aux token may read in the prefix (its
    patch's preceding code); -1 on prefix and padding."""
    position_ids: Tensor
    """`[B, W]` long -- RoPE positions. A patch is numbered as if it directly followed
    its conditioning code, which is where the specification puts it; prefix and
    padding columns keep their own index."""
    aux_rows: Tensor
    """`[N]` -- the row of each aux token, for scattering aux embeddings."""
    aux_columns: Tensor
    """`[N]` -- the column of each aux token."""
    predictor: Tensor
    """`[N]` -- flat row-major index into `[B, W]` of the state predicting each target."""
    targets: Tensor
    """`[N]` -- the token id each predictor must produce."""

    @property
    def num_targets(self) -> int:
        return int(self.targets.numel())


def natural_width(aux_patch: Tensor, prefix_len: Tensor) -> int:
    """The narrowest `W` that holds every row: `max(prefix_len + aux_count)`."""
    return int((prefix_len + (aux_patch >= 0).sum(dim=1)).max())


def bucket_width(width: int, multiple: int | None) -> int:
    """Round `width` up to a multiple, so compiled flex shapes repeat across batches.

    An ADDITIVE ladder, never a geometric one: a power-of-two ladder nearly doubles W
    in the worst case and attention memory grows with W^2 -- that is what took a
    relaunch from W=8977 to 16384 and out of memory. `None` keeps the natural width.
    """
    return width if multiple is None else -(-width // multiple) * multiple


def build_next_patch_batch(
    *,
    slot_positions: Tensor,
    aux_ids: Tensor,
    aux_patch: Tensor,
    prefix_len: Tensor,
    width: int | None = None,
) -> NextPatchBatch:
    """Lay out every row's aux tokens after its own prefix and pair them with targets.

    `aux_ids`/`aux_patch` are `[B, A]`, right-padded with `aux_patch = -1` as
    `collate_encoder_batch` produces them, so row `b`'s valid aux tokens are its
    first `count[b]` entries and land at columns `prefix_len[b] + 0..count[b]-1`.

    The predicting state for a patch's FIRST token is the code before that patch
    (`slot_positions[m-1]`); for every later token it is the previous column. That
    is what makes this a gather rather than the one-position shift
    `chunked_ce_from_hidden` applies. Patch 0 is never a target -- it has no
    preceding code -- and callers drop it when building `aux_ids`.

    `width` pads to a chosen bucket so compiled shapes repeat; `None` is natural.
    """
    device = aux_patch.device
    batch, span = aux_patch.shape
    needed = natural_width(aux_patch, prefix_len)
    width = needed if width is None else width
    if width < needed:
        raise ValueError(f"width={width} is narrower than the natural width {needed}")

    valid = aux_patch >= 0
    index = torch.arange(span, device=device)
    columns = prefix_len.unsqueeze(1) + index  # [B, A]; only meaningful where valid

    # The code each aux token is conditioned on: patch m reads z_{m-1}.
    limit = torch.gather(slot_positions, 1, (aux_patch - 1).clamp_min(0))

    # A change of patch index along the row marks a patch's first token.
    previous = torch.cat(
        [torch.full_like(aux_patch[:, :1], -1), aux_patch[:, :-1]], dim=1
    )
    starts = aux_patch != previous
    predictor = torch.where(starts, limit, columns - 1)
    # Rank within the patch: a running max over run-start indices gives each run's
    # first index, because `aux_patch` is non-decreasing over the valid entries.
    run_start = torch.cummax(torch.where(starts, index, -1), dim=1).values
    position = limit + 1 + (index - run_start)

    rows = torch.arange(batch, device=device).unsqueeze(1).expand(batch, span)
    aux_rows, aux_columns = rows[valid], columns[valid]
    grid = (batch, width)
    column_index = torch.arange(width, device=device).expand(grid)

    def scatter(fill: int, values: Tensor) -> Tensor:
        out = torch.full(grid, fill, dtype=torch.long, device=device)
        return out.index_put((aux_rows, aux_columns), values[valid])

    return NextPatchBatch(
        width=width,
        is_prefix=column_index < prefix_len.unsqueeze(1),
        patch=scatter(-1, aux_patch),
        code_limit=scatter(-1, limit),
        position_ids=column_index.clone().index_put(
            (aux_rows, aux_columns), position[valid]
        ),
        aux_rows=aux_rows,
        aux_columns=aux_columns,
        predictor=aux_rows * width + predictor[valid],
        targets=aux_ids[valid],
    )


def patch_weights(aux_patch: Tensor) -> tuple[Tensor, Tensor]:
    """Per-target weights `1/(P_i * T_ij)`, and each row's supervised-patch count.

    The objective averages CE **within a patch**, then **over a row's patches**, then
    over rows, so a target's weight is the product of those two reciprocals and every
    row's weights sum to 1. `T_ij` is patch *j*'s token count, `P_i` the number of
    supervised patches in row *i* (patch 0 never appears, and `next_patch_subsample`
    may drop others).

    Returned in `aux_patch[valid]` order, which is exactly `targets` order, so the
    weights line up with the CE terms positionally and need no second gather.

    Keyed by *run*, not by patch index: `aux_patch` is non-decreasing along a row, so
    a cumulative sum over "this is a new patch" gives every (row, patch) pair a dense
    id. Keying on the patch index instead would need `aux_patch.max()` -- a host sync
    per micro-batch -- because a subsampled row's indices are sparse (K can be 8,000
    with 40 surviving tokens).
    """
    valid = aux_patch >= 0
    previous = torch.cat(
        [torch.full_like(aux_patch[:, :1], -1), aux_patch[:, :-1]], dim=1
    )
    starts = (aux_patch != previous) & valid
    rows = aux_patch.shape[0]
    row_index = (
        torch.arange(rows, device=aux_patch.device).unsqueeze(1).expand_as(aux_patch)
    )
    flat_starts = starts[valid]
    # A row's first valid entry always starts a run (its `previous` is the -1 fill),
    # so runs never straddle a row boundary.
    run_id = flat_starts.cumsum(0) - 1
    run_tokens = torch.bincount(run_id.clamp_min(0), minlength=1)
    patches_per_row = torch.bincount(row_index[valid][flat_starts], minlength=rows)
    weights = 1.0 / (run_tokens[run_id] * patches_per_row[row_index[valid]]).to(
        torch.float32
    ).clamp_min(1.0)
    return weights, patches_per_row


def next_patch_mask_mod(batch: NextPatchBatch) -> Any:
    """`mask_mod` for FlexAttention encoding the leak-free conditioning.

    | query | may attend to |
    |---|---|
    | prefix column `q` | prefix columns `<= q` (ordinary causal) |
    | aux token of patch `m` | prefix columns `<= code_limit` (prompt and codes
      `z_0..z_{m-1}`), and aux tokens of patch `m` only, `<= q` |
    | padding | itself only |

    `q == kv` is always allowed. A fully masked row yields NaN from the softmax, and
    one NaN poisons the whole batch's gradient even where the loss ignores it -- the
    same guard `CoTEncoder._cross_mask` applies with `allowed[..., 0] = True`.
    """
    is_prefix, patch, code_limit = batch.is_prefix, batch.patch, batch.code_limit

    def mask_mod(b: Tensor, h: Tensor, q: Tensor, kv: Tensor) -> Tensor:
        del h
        causal = kv <= q
        kv_prefix = is_prefix[b, kv]
        q_patch = patch[b, q]
        q_aux = q_patch >= 0
        prefix_to_prefix = is_prefix[b, q] & kv_prefix & causal
        aux_to_prefix = q_aux & kv_prefix & (kv <= code_limit[b, q])
        aux_to_aux = q_aux & (patch[b, kv] == q_patch) & causal
        return prefix_to_prefix | aux_to_prefix | aux_to_aux | (q == kv)

    return mask_mod


def dense_next_patch_mask(batch: NextPatchBatch) -> Tensor:
    """The same rules materialized as `[B, 1, W, W]` booleans, from the same `mask_mod`.

    The reference the rules are asserted against, and the CPU path: FlexAttention has
    no CPU backward, so without this the aux objective could not be tested at all.
    `B * W^2` booleans -- for tests and small shapes only, never a production batch.
    """
    rows, width = batch.is_prefix.shape
    device = batch.is_prefix.device
    b = torch.arange(rows, device=device).view(rows, 1, 1)
    q = torch.arange(width, device=device).view(1, width, 1)
    kv = torch.arange(width, device=device).view(1, 1, width)
    return next_patch_mask_mod(batch)(b, b, q, kv).unsqueeze(1)


def aux_attention_mask(batch: NextPatchBatch, dtype: torch.dtype) -> BlockMask | Tensor:
    """The mask the decoder consumes: a `BlockMask` on CUDA, a dense one on CPU.

    `_compile=True` is not an optimization. Uncompiled, `create_block_mask` evaluates
    `mask_mod` under vmap over the full `[B, W, W]` grid with int64 intermediates --
    16 GiB at B=4, W=16384, which OOM'd the Sep 11 relaunch at step 0 -- whereas the
    compiled form evaluates block by block.
    """
    rows, width = batch.is_prefix.shape
    if batch.is_prefix.is_cuda:
        return create_block_mask(
            next_patch_mask_mod(batch),
            B=rows,
            H=None,
            Q_LEN=width,
            KV_LEN=width,
            device=str(batch.is_prefix.device),
            _compile=True,
        )
    return dense_additive_mask(dense_next_patch_mask(batch), dtype)


def dense_additive_mask(allowed: Tensor, dtype: torch.dtype) -> Tensor:
    """Boolean `allowed` -> the additive `0 / -inf` mask every backend accepts.

    Bool masks work with SDPA but NOT with the `eager` path, which does
    `attn_weights + mask` and would read `True` as `+1`. Additive is correct for
    both. No row can be all `-inf`: `next_patch_mask_mod` always permits `q == kv`.
    """
    return torch.zeros_like(allowed, dtype=dtype).masked_fill_(
        ~allowed, torch.finfo(dtype).min
    )
