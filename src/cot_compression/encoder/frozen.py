"""The frozen decoder: context pass, slot splice, answer cross-entropy.

Two passes per sample, and the asymmetry between them is the whole design:

* **Pass A** runs over ``[prompt; CoT]`` under ``no_grad`` and produces the
  cross-attention memory. The answer is deliberately excluded, which removes any
  path for it to leak into the memory, and mirrors what
  ``signals.compute_cot_signals`` already does.
* **Pass B** runs over the compressed render *with* gradient, with the encoder's
  codes spliced into ``inputs_embeds`` at the placeholder positions, and is
  scored on answer tokens only.

No LLM parameter is ever trained. Gradient still traverses every layer of Pass B,
because ``inputs_embeds`` requires grad -- which is why gradient checkpointing on
the frozen backbone is mandatory rather than an optimization.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Iterator
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers import AttentionInterface
from transformers.integrations.flex_attention import repeat_kv
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from cot_compression.compression import (
    CompressionMethod,
    RandomCompressionMethod,
    SignalWeightedMeanCompressionMethod,
    SimpleMeanCompressionMethod,
    StepMeanCompressionMethod,
)
from cot_compression.data.chat import IGNORE_INDEX
from cot_compression.patching import PatchingMethod
from cot_compression.training.sft import chunked_ce_weighted


def build_latent_init(name: str, patching: PatchingMethod | None) -> CompressionMethod:
    """The latent initializer, as an existing training-free compression method.

    Not a reimplementation: ``materialize`` on these already returns exactly the
    ``[K, d]`` seed the encoder wants, already keyed on ``seed + sample_index`` so
    sample *i* gets the same seed under every method, and already excluding
    special/added tokens from the random draw. Reusing them also means the
    encoder's step-0 behaviour is a *measured* baseline rather than a new one.
    """
    if name == "random":
        return RandomCompressionMethod(patching)
    if name == "simple_mean":
        return SimpleMeanCompressionMethod(patching)
    if name == "step_mean":
        # The same arithmetic mean the training seed computes, so a step checkpoint
        # is scored from the initialization it was trained from.
        return StepMeanCompressionMethod(patching)
    if name in ("surprisal_t0", "entropy_t0"):
        # T=0 collapses the softmax to a one-hot on the highest-signal token, i.e.
        # the single most surprising (or most uncertain) token's embedding stands
        # for the patch. Parameterized by signal so an `entropy_*` patching can be
        # paired with an initializer that reads the SAME cached signal -- mixing
        # the two would need both caches loaded at once, which `_load_signals`
        # refuses by design.
        return SignalWeightedMeanCompressionMethod(
            patching, temperature=0.0, signal=name.removesuffix("_t0")
        )
    raise ValueError(
        f"Unknown latent_init {name!r}; expected random, simple_mean, "
        "step_mean, surprisal_t0 or entropy_t0."
    )


ROUTED_ATTENTION = "cot_compression_routed"


@functools.cache
def _compiled_flex_attention() -> Any:
    """FlexAttention compiled for STATIC shapes. Built once, on first use.

    Owned here rather than taken from transformers, whose `WrappedFlexAttention`
    picks compile flags by torch version: only its <= 2.5.1 and == 2.6.0 branches
    pass `dynamic=False`, so on 2.12 it compiles with automatic dynamic shapes. After
    a few distinct widths Dynamo then lowers the flex BACKWARD with symbolic sizes,
    Inductor prunes every Triton config, and it raises `NoValidChoicesError` with no
    ATEN fallback -- which killed rank 3 of both aux arms at step 291 on 2026-09-08
    (recovered from its result pickle), and hung the other ranks on ALLREDUCE.

    Static shapes recompile per distinct `(B, W)`, which `bucket_width` bounds and
    `configure_aux_compilation` turns into a hard error if the bound is exceeded,
    instead of the silent fallback to eager flex that OOM'd the Sep 11 resume.
    """
    return torch.compile(cast(Any, flex_attention), dynamic=False)


def configure_aux_compilation(recompile_limit: int) -> dict[str, object]:
    """Size Dynamo's recompile cache for the aux shapes, and make overflow fatal.

    Past `recompile_limit`, Dynamo stops compiling a function and runs it EAGERLY.
    For flex attention that means the dense `sdpa_dense` kernel and a full
    `[B, H, W, W]` score matrix -- at aux widths of ~28k that is ~150 GB, so the
    fallback is never a slower success, only a delayed OOM (the Sep 11 resume died
    exactly this way at 64 GiB). `fail_on_recompile_limit_hit` makes it raise at the
    moment the limit is crossed, naming the cause.

    Returns the EFFECTIVE values, read back, for the startup log: a setting that
    silently did not apply is how a limit of 256 was once observed as 8.
    """
    # `Any`: the config module's attributes are typed as their literal defaults.
    dynamo: Any = torch._dynamo.config
    dynamo.recompile_limit = recompile_limit
    dynamo.accumulated_recompile_limit = max(
        int(dynamo.accumulated_recompile_limit), recompile_limit
    )
    dynamo.fail_on_recompile_limit_hit = True
    return {
        "recompile_limit": dynamo.recompile_limit,
        "accumulated_recompile_limit": dynamo.accumulated_recompile_limit,
        "fail_on_recompile_limit_hit": dynamo.fail_on_recompile_limit_hit,
    }


def _flex_attention_forward(
    module: Any,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_mask: BlockMask,
    scaling: float | None = None,
    dropout: float = 0.0,
    **_: Any,
) -> tuple[Tensor, None]:
    """transformers' `flex_attention_forward`, minus the parts this decoder never uses.

    Same contract: `[B, H, T, D]` in, `[B, T, H, D]` out. No attention sinks, no
    softcap and no score mask exist in Qwen3, and the log-sum-exp is only consumed by
    sinks, so it is not requested -- saving its `[B, H, T]` fp32 buffer.
    """
    del module
    if dropout:
        raise ValueError("flex attention here is inference-only; dropout must be 0")
    enable_gqa = True
    heads = query.shape[1]
    if heads & (heads - 1):
        # Mirrors transformers: the GQA kernel path wants a power-of-two head count.
        key = repeat_kv(key, heads // key.shape[1])
        value = repeat_kv(value, heads // value.shape[1])
        enable_gqa = False
    out = _compiled_flex_attention()(
        query,
        key,
        value,
        block_mask=block_mask,
        scale=scaling,
        enable_gqa=enable_gqa,
    )
    return out.transpose(1, 2).contiguous(), None


def _routed_attention_forward(module: Any, query, key, value, attention_mask, **kwargs):
    """Dispatch on the MASK TYPE, statelessly: `BlockMask` -> flex, else -> SDPA.

    A context manager cannot do this job. Gradient checkpointing with
    `use_reentrant=False` recomputes the forward during the BACKWARD pass, long
    after any `with` block around the forward has exited -- so a scoped swap
    restores SDPA and the recomputation then hands it a `BlockMask`, raising
    "attn_mask must be Tensor, not BlockMask". Routing per call has no such window.

    It also keeps Passes A and B on SDPA, which is faster for their plain 2-D
    padding masks, without anyone having to remember to switch back.
    """
    if isinstance(attention_mask, BlockMask):
        return _flex_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


def _install_mask_routing(model: Any) -> None:
    """Pin the decoder to the routing implementation, once, at construction."""
    AttentionInterface.register(ROUTED_ATTENTION, _routed_attention_forward)
    model.set_attn_implementation(ROUTED_ATTENTION)


def assert_no_train_mode_behaviour(model: Any) -> None:
    """Refuse a decoder whose forward differs between train() and eval().

    The backbone is kept in train() mode so gradient checkpointing engages (see
    `FrozenBackbone.__init__`). That is only sound for a model with no dropout and
    no batch normalization; anything else would inject noise into Pass B while
    reporting a frozen, deterministic decoder.
    """
    offenders = []
    for name, value in vars(model.config).items():
        if "dropout" in name and isinstance(value, (int, float)) and value > 0:
            offenders.append(f"config.{name}={value}")
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout) and module.p > 0:
            offenders.append(f"{name}: Dropout(p={module.p})")
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            offenders.append(f"{name}: {type(module).__name__}")
    if offenders:
        raise ValueError(
            "The frozen decoder behaves differently in train() mode, so it cannot "
            "be kept there for gradient checkpointing: " + ", ".join(offenders) + ". "
            "Either use a dropout-free decoder or accept the memory cost of "
            "eval-mode (unckeckpointed) Pass B."
        )


class FrozenBackbone:
    """Wraps the frozen decoder. Deliberately **not** an `nn.Module`.

    The training module holds one of these, and if it were an `nn.Module` the
    decoder's parameters would land in that module's `state_dict()` -- adding
    ~1.2 GB of frozen weights to every checkpoint -- and in DDP's bucket scan.
    A plain class is invisible to both by construction, rather than by remembering
    to filter it out.
    """

    def __init__(
        self,
        model: Any,
        *,
        memory_layer: int = -1,
        ce_chunk_tokens: int = 1024,
        gradient_checkpointing: bool = True,
    ) -> None:
        self.model = model
        self.memory_layer = memory_layer
        self.ce_chunk_tokens = ce_chunk_tokens

        model.requires_grad_(False)
        model.config.use_cache = False
        # Non-reentrant is required: the reentrant variant misbehaves when no
        # parameter requires grad, which is exactly this case. Verified that
        # gradient still reaches `inputs_embeds` and that no parameter collects one.
        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            model.gradient_checkpointing_enable({"use_reentrant": False})

        # The decoder lives in eval() -- it is frozen, and a frozen model parked in
        # train() reads like a bug. Train mode is entered only for the duration of
        # Pass B, by `_checkpointing()`, because HF gates checkpointing on it.
        assert_no_train_mode_behaviour(model)
        model.eval()
        _install_mask_routing(model)

    @contextlib.contextmanager
    def _checkpointing(self) -> Iterator[None]:
        """Enter train() mode just long enough for checkpointing to engage.

        `GradientCheckpointingLayer.__call__` gates on
        `self.gradient_checkpointing and self.training`, and transformers offers no
        override -- so an eval-mode decoder silently skips checkpointing, and Pass B
        then retains activations for every layer because `inputs_embeds` requires
        grad. That failure is invisible: the run merely OOMs or crawls.

        Scoped rather than global for two reasons. A frozen decoder left in train()
        is misleading to read, and Pass A (under `no_grad`) has nothing to
        checkpoint, so it should not pay the mode flip at all. Sound only because
        the decoder is dropout-free, which `assert_no_train_mode_behaviour` checks
        at construction rather than trusting.
        """
        if not self.gradient_checkpointing:
            # Nothing to enable, so do not flip modes at all. Pass B then retains
            # every layer's activations -- roughly 2.2 MB/token at 0.6B, which is
            # only safe if max_batch_tokens is lowered to match.
            yield
            return
        was_training = self.model.training
        self.model.train()
        try:
            yield
        finally:
            self.model.train(was_training)

    @property
    def embedding_weight(self) -> Tensor:
        return self.model.get_input_embeddings().weight

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode_context(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Pass A: hidden states over ``[prompt; CoT]``, detached.

        Attention is causal, so position *j* encodes only the prefix up to *j*;
        with the answer excluded from the input there is no path by which the
        answer can enter the memory.
        """
        if self.memory_layer == -1:
            out = self.model.model(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state.detach()
        out = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Verified on the installed transformers: hidden_states[-1] is
        # last_hidden_state, i.e. post-final-norm, so the indexing is consistent.
        return out.hidden_states[self.memory_layer].detach()

    def splice(
        self,
        input_ids: Tensor,
        codes: Tensor,
        slot_positions: Tensor,
        slot_mask: Tensor,
    ) -> Tensor:
        """Overwrite placeholder positions in ``inputs_embeds`` with the codes.

        Out-of-place ``index_copy`` rather than the eval path's Python double
        loop, which would build O(B*K) autograd nodes -- at K ~ 1500 that is a
        graph large enough to dominate the step. ``index_copy`` with duplicate
        indices is undefined; positions are distinct within a row and rows are
        offset by the sequence length, so they cannot collide (asserted in tests).
        """
        embeds = self.model.get_input_embeddings()(input_ids).detach()
        batch, length, dim = embeds.shape
        rows = torch.arange(batch, device=input_ids.device).unsqueeze(1)
        offsets = (rows * length + slot_positions)[slot_mask]
        flat = embeds.reshape(batch * length, dim)
        spliced = flat.index_copy(0, offsets, codes[slot_mask].to(flat.dtype))
        return spliced.view(batch, length, dim)

    def _decode(
        self,
        inputs_embeds: Tensor,
        attention_mask: Any,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        """One gradient-carrying decoder forward. The single place Pass B runs.

        `attention_mask` is whatever the active backend accepts: a 2-D padding mask
        for SDPA/flash, or a `BlockMask` under FlexAttention.
        """
        with self._checkpointing():
            return self.model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).last_hidden_state

    def next_patch_ce(
        self,
        inputs_embeds: Tensor,
        block_mask: Any,
        predictor: Tensor,
        targets: Tensor,
        position_ids: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """CE for predicting each patch's tokens from the codes before it.

        Returns `[sum(w * CE), sum(CE)]`: the first is the objective's next-patch
        term (`weights` from `patch_weights`, so it is a sum over rows of each row's
        mean per-patch CE), the second its per-token twin for metrics.

        `inputs_embeds` is `[prefix through the last code; selected patch tokens]`
        with the codes already spliced; `block_mask` restricts a patch's tokens to
        the prompt, the codes preceding that patch, and their own prefix.

        `predictor[i]` is the FLAT index (row-major over `[B, T]`) of the hidden
        state that must predict `targets[i]`. A gather, not a shift: a patch's first
        token is predicted by its preceding code, which is not the position before
        it in the sequence.
        """
        hidden = self._decode(inputs_embeds, block_mask, position_ids)
        flat = hidden.flatten(0, 1)
        return chunked_ce_weighted(
            self.model.lm_head,
            flat.index_select(0, predictor),
            targets,
            weights,
            self.ce_chunk_tokens,
        )

    def answer_ce_weighted(
        self, inputs_embeds: Tensor, attention_mask: Tensor, labels: Tensor
    ) -> Tensor:
        """Pass B answer cross-entropy, reduced two ways: `[sum_i mean_t CE, sum_t CE]`.

        The first is the objective's answer term: each row's mean over *its own*
        answer tokens, summed over rows, so that dividing by the step's global row
        count gives every row the same weight whatever its answer length. That
        matters here because answer length is where the CoT stops mattering -- the
        longest-answer quartile of this corpus holds 48% of all answer tokens but
        only 0.30 nats/token of base-vs-no_cot headroom, against 1.07 for the
        shortest. The second is the token-weighted twin, for metrics comparable with
        earlier runs.

        Per-row weighting is a per-token weight of `1/A_i`, so `chunked_ce_weighted`
        does all of it; this method only has to build the weights and pair them with
        the same positions the labels select.
        """
        hidden = self._decode(inputs_embeds, attention_mask)
        targets = labels[:, 1:]
        answer_tokens = (targets != IGNORE_INDEX).sum(-1, keepdim=True)
        weights = (1.0 / answer_tokens.clamp_min(1).float()).expand_as(targets)
        # select_labels: only the answer is supervised here (a measured 21.8% of
        # positions), so running lm_head over the rest is 4.6x wasted work and 4.6x
        # the fp32 logit allocation. Identical value either way.
        keep = (targets.flatten() != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        return chunked_ce_weighted(
            self.model.lm_head,
            hidden[:, :-1].flatten(0, 1).index_select(0, keep),
            targets.flatten().index_select(0, keep),
            weights.flatten().index_select(0, keep),
            self.ce_chunk_tokens,
        )

    def answer_ce(
        self, inputs_embeds: Tensor, attention_mask: Tensor, labels: Tensor
    ) -> Tensor:
        """Pass B: shifted cross-entropy over answer positions only, as a SUM.

        The per-token sum of `answer_ce_weighted`, kept as its own name because
        "answer CE over these rows" reads better than indexing a pair, and because
        the baselines and the eval harness want exactly this number.
        """
        return self.answer_ce_weighted(inputs_embeds, attention_mask, labels)[1]
