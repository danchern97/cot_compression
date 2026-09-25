"""The trainable compressor: an AIAYN decoder stack over latent slots, plus VQ.

The encoder reads the frozen LLM's hidden states for `[prompt; CoT]` through
cross-attention and emits one vector per patch, which is quantized and spliced
into `inputs_embeds` at placeholder positions. Nothing here touches the LLM --
the caller supplies the memory and the latent initialization, both of which are
constants as far as these parameters are concerned.

Baseline configuration, deliberately minimal: no dimensionality projections
anywhere (`d_c == d_m == d_llm`), a gradient-trained codebook rather than EMA, and
only the normalization a pre-norm transformer requires. Every deviation is an
ablation, not a default.

The one thing that is *not* optional any more is position. With
`position_encoding="rope"` both attentions get ordinary rotary encodings, each in
its own coordinate system -- slot index for the latents, Pass A token index for the
memory and for the span each slot summarizes. Without it a slot can only find its
own span by content match, which is the single thing it most needs to do;
`CoTEncoder.rope_positions` documents the choice of query position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MaskMode = Literal["causal", "bidirectional"]
LatentInit = Literal["random", "simple_mean", "surprisal_t0", "entropy_t0"]
LATENT_INITS: tuple[str, ...] = get_args(LatentInit)
PositionEncoding = Literal["none", "rope"]
POSITION_ENCODINGS: tuple[str, ...] = get_args(PositionEncoding)


@dataclass(frozen=True)
class EncoderConfig:
    """Everything that makes a checkpoint incompatible with another.

    Serialized beside the weights; `check_encoder_architecture` compares it on
    resume. `d_llm` is the decoder's hidden size and is the field most likely to
    silently differ -- Qwen3-0.6B is 1024, Qwen3-4B is 2560 -- because the run
    directory carries no model identifier.
    """

    d_llm: int
    n_blocks: int = 2
    n_heads: int = 8
    ffn_mult: int = 4
    codebook_size: int = 64
    # Per-vector RMS of the frozen decoder's INPUT EMBEDDINGS, derived from the
    # loaded model and never configured. `out_norm` is initialized to this instead
    # of to 1.0, because RMSNorm pins its output RMS to `rms(weight)` exactly and
    # `z = q` is spliced straight into the embedding sequence with no rescaling.
    # Leaving it at 1.0 -- Qwen3's *internal* hidden-state scale, not its input
    # embedding scale -- put the codebook 34x outside the only space the frozen
    # decoder understands, which is what made 46 of 64 codes dead from step 1.
    embed_rms: float = 1.0
    self_attn_mask: MaskMode = "causal"
    cross_attn_mask: MaskMode = "causal"
    latent_init: LatentInit = "random"
    # Rotary position encoding on Q and K, in the coordinate system each attention
    # works in: slot index for the latent self-attention, Pass A token index for the
    # cross-attention (see `CoTEncoder.rope_positions`). Defaults to "none" so a
    # checkpoint written before this existed keeps its own semantics --
    # `check_encoder_architecture` then refuses to load it into a RoPE run.
    position_encoding: PositionEncoding = "none"
    # 1e6, matching the decoder's own `rope_theta`, not RoFormer's 10,000: Men et al.
    # (arXiv:2405.14591, Table 2) put the lower bound for 32k context at 6.4e5 at
    # head_dim 128, which is exactly this encoder's head dim at d_llm=1024, n_heads=8
    # -- and `max_length` is 32,768, so the memory really does reach that far.
    rope_theta: float = 1_000_000.0
    # Softens `argmin` into a weighted average of codes early in training; 0.0
    # disables it. Off by default -- an early-collapse mitigation to reach for
    # only once utilization is measured to be failing.
    soft_assign_temperature: float = 0.0
    commit_weight: float = 0.25
    # Weight on the auxiliary next-patch loss, now a ratio between two comparable
    # per-row means: a row's mean per-patch CE against its mean answer CE. It no
    # longer carries the aux/answer TOKEN ratio (~16x on average, 2-29x per step)
    # that the answer-token denominator used to fold in, so values from before that
    # change do not transfer -- the old 1.0 is roughly 16 here.
    next_patch_weight: float = 0.0
    # Probability a patch is supervised. 1.0 supervises every patch; lower rates drop
    # the unselected patches' tokens from the sequence entirely, so this scales the
    # decoder cost rather than only the lm_head.
    next_patch_subsample: float = 1.0
    # EnCodec's `threshold_ema_dead_code`: a code is dead when its usage EMA falls
    # below this ABSOLUTE count. Kept absolute, and kept at 2, because that is what
    # the reference implementations use (Jukebox 1.0, SoundStream/EnCodec/
    # lucidrains 2) -- a fraction-of-fair-share rule is not in the literature and
    # was not shown to be necessary here.
    #
    # Known consequence, deliberately accepted: this is not scale-free, so 2 is
    # 4.2% of a code's fair share at |C|=1024 but 0.26% at |C|=64. Codebook-health
    # metrics are therefore NOT comparable across codebook sizes.
    dead_code_threshold: float = 2.0
    usage_decay: float = 0.99

    def __post_init__(self) -> None:
        if self.d_llm % self.n_heads != 0:
            raise ValueError(
                f"d_llm={self.d_llm} must be divisible by n_heads={self.n_heads}."
            )
        for name in ("self_attn_mask", "cross_attn_mask"):
            value = getattr(self, name)
            if value not in ("causal", "bidirectional"):
                raise ValueError(
                    f"{name} must be causal or bidirectional, got {value!r}."
                )
        if self.latent_init not in LATENT_INITS:
            raise ValueError(
                f"latent_init must be one of {LATENT_INITS}, got {self.latent_init!r}."
            )
        if self.position_encoding not in POSITION_ENCODINGS:
            raise ValueError(
                f"position_encoding must be one of {POSITION_ENCODINGS}, got "
                f"{self.position_encoding!r}."
            )
        if self.position_encoding == "rope":
            if self.rope_theta <= 1.0:
                raise ValueError(f"rope_theta must be > 1, got {self.rope_theta!r}.")
            if (self.d_llm // self.n_heads) % 2:
                raise ValueError(
                    f"rope needs an even head dimension, got "
                    f"{self.d_llm // self.n_heads} (d_llm={self.d_llm}, "
                    f"n_heads={self.n_heads})."
                )
        if self.next_patch_weight < 0.0:
            raise ValueError(
                f"next_patch_weight must be >= 0, got {self.next_patch_weight!r}."
            )
        if not 0.0 <= self.next_patch_subsample <= 1.0:
            raise ValueError(
                "next_patch_subsample is a probability and must be in [0, 1], got "
                f"{self.next_patch_subsample!r}."
            )
        if self.dead_code_threshold <= 0.0:
            raise ValueError(
                f"dead_code_threshold must be positive, got "
                f"{self.dead_code_threshold!r}."
            )
        if self.embed_rms <= 0.0:
            raise ValueError(f"embed_rms must be positive, got {self.embed_rms!r}.")
        if self.codebook_size < 2:
            raise ValueError(f"codebook_size must be >= 2, got {self.codebook_size}.")


@dataclass
class QuantizerOutput:
    z: Tensor
    """Straight-through codes, `[B, K, d]`; equal to the hard codes in value."""
    codebook_loss: Tensor
    """Sum over ROWS of each row's mean over its valid slots (and over dimensions)."""
    commit_loss: Tensor
    """Sum over ROWS of each row's mean over its valid slots (and over dimensions)."""
    indices: Tensor
    """Assigned code per slot, `[B, K]`; garbage where `slot_mask` is False."""
    counts: Tensor
    """Assignments per code this batch, `[codebook_size]`. All-reduce before use."""
    encoder_output: Tensor
    """Pre-quantization `g`, `[B, K, d]`. Dead codes are reseeded from these."""


def _tile(pool: Tensor, wanted: int, generator: torch.Generator) -> Tensor:
    """Jukebox's `_tile`: repeat a too-small pool and jitter the copies.

    Needed because the pool is capped and `codebook_size` reaches 1024. Without
    the jitter the repeats are exact duplicates, so k-means would collapse them
    onto one centroid and hand back fewer distinct codes than asked for.

    The jitter is scaled by the pool's own RMS rather than Jukebox's absolute
    `0.01/sqrt(d)`, because here the code scale is derived from the decoder
    (`embed_rms` ~ 0.029) instead of being a property of a jointly-trained model.
    """
    if pool.shape[0] >= wanted:
        return pool
    repeats = -(-wanted // pool.shape[0])
    tiled = pool.repeat(repeats, 1)
    std = 0.01 * pool.pow(2).mean().sqrt()
    noise = torch.randn(tiled.shape, generator=generator, device=tiled.device)
    return tiled + noise * std


@torch.no_grad()
def kmeans(
    samples: Tensor, num_clusters: int, num_iters: int, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    """Lloyd's algorithm, following EnCodec's `core_vq.kmeans`.

    Distances use the expanded `|x|^2 - 2 x.c + |c|^2` form rather than an
    explicit difference tensor: at 16k samples, 1024 clusters and d=1024 the
    naive `samples[:, None] - means[None]` is 68 GB, while this is 16M floats.

    Empty clusters keep their previous centroid instead of collapsing to zero --
    a zero row would be an out-of-distribution code that nothing ever reclaims.
    """
    samples = samples.float()
    pool = _tile(samples, num_clusters, generator)
    pick = torch.randperm(pool.shape[0], generator=generator, device=pool.device)
    means = pool[pick[:num_clusters]].clone()

    for _ in range(num_iters):
        distances = (
            samples.pow(2).sum(-1, keepdim=True)
            - 2.0 * samples @ means.t()
            + means.pow(2).sum(-1)
        )
        buckets = distances.argmin(dim=-1)
        bins = torch.bincount(buckets, minlength=num_clusters)
        totals = torch.zeros_like(means)
        totals.index_add_(0, buckets, samples)
        empty = bins == 0
        totals /= bins.clamp_min(1).unsqueeze(-1)
        means = torch.where(empty.unsqueeze(-1), means, totals)
    # Bins as well as means: EnCodec seeds `cluster_size` from them, so the usage
    # statistic starts at the truth rather than at a placeholder.
    return means, bins


class VectorQuantizer(nn.Module):
    """Nearest-neighbour VQ with a gradient-trained codebook.

    The codebook is a plain `nn.Parameter`, so DDP all-reduces it like any other
    weight -- which is the reason to prefer gradients over EMA here, since an EMA
    codebook is a *buffer* and `broadcast_buffers=False` would let each rank drift
    its own.

    Full dimension and no l2 normalization, per the baseline. Both raise the
    collapse risk (nearest-neighbour assignment is poorly conditioned in 1024
    dimensions), which is why `counts` is returned every step: utilization is the
    number that tells you whether this configuration is working.
    """

    usage: Tensor
    reseeds: Tensor

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.codebook = nn.Parameter(torch.empty(config.codebook_size, config.d_llm))
        # Placeholder; `init_codebook_from_encoder_outputs` replaces this before
        # the first optimizer step. Scaled to `embed_rms` rather than a bare 0.02
        # so that a caller who skips the k-means init -- a test, or an inference
        # path -- still gets codes in the decoder's embedding range instead of
        # far-out-of-distribution vectors, since with `z = q` the code IS the
        # embedding.
        nn.init.normal_(self.codebook, std=config.embed_rms)
        # Zeros, as EnCodec's `cluster_size`. Not a live statistic yet:
        # `init_codebook_from_encoder_outputs` overwrites it with the k-means bin
        # counts before the first step, exactly as EnCodec's `init_embed_` does.
        self.register_buffer("usage", torch.zeros(config.codebook_size))
        # Cumulative reseeds, for reporting. A rising slope late in training means
        # chronic collapse rather than a transient the restarts absorbed. A BUFFER,
        # not a Python int: an int is absent from `state_dict`, so it silently
        # restarted from zero on every resume and the metric lied across the seam.
        self.register_buffer("reseeds", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def init_codebook_from_encoder_outputs(
        self, pool: Tensor, generator: torch.Generator, iters: int = 50
    ) -> None:
        """Seed codes with k-means centroids of real encoder outputs.

        The SoundStream/EnCodec initialization ("run the k-means algorithm on the
        first training batch and use the learned centroids"), which is what makes
        every reference restart rule safe: the codebook and the vectors it
        quantizes are the same distribution from step 0, so the dead set is a
        handful rather than a majority.

        Seeding from vocabulary embeddings instead -- the previous behaviour --
        is only equivalent once `out_norm` puts encoder outputs at the embedding
        scale. It is, with `embed_rms`; this then satisfies both constraints at
        once, being matched to the encoder cloud AND inside the decoder's
        embedding distribution.
        """
        means, bins = kmeans(pool, self.config.codebook_size, iters, generator)
        self.codebook.copy_(means.to(self.codebook.dtype))
        self.usage.copy_(bins.to(self.usage.dtype))

    @torch.no_grad()
    def update_usage(self, counts: Tensor) -> None:
        """Fold globally-reduced batch counts into the usage EMA.

        Takes already-reduced counts rather than reducing here: the module must
        not care whether it is running under DDP, and a rank-local EMA would make
        dead-code detection differ between ranks.
        """
        decay = self.config.usage_decay
        self.usage.mul_(decay).add_(counts.to(self.usage.dtype), alpha=1.0 - decay)

    @torch.no_grad()
    def reseed_codes(self, rows: Tensor, replacements: Tensor) -> None:
        """Overwrite dead codes and clear their usage history.

        Deliberately takes the replacement vectors rather than sampling them: the
        caller must draw them once and broadcast, because DDP synchronizes
        gradients and not parameters, so a per-rank draw would leave every rank
        with a different codebook and nothing would report it.

        Usage is deliberately NOT reset, matching EnCodec's `replace_`, which
        writes only `embed` and leaves `cluster_size` alone. A code that earns no
        assignments is therefore replaced again on the next step -- EnCodec accepts
        that churn because its codebook is an EMA buffer, where the following
        update pulls a replaced row straight into place. Ours is an nn.Parameter
        under AdamW and settles over hundreds of steps instead, so the churn costs
        more here. Watch `vq/reseeds_total` against `vq/perplexity`; a grace period
        is the fix if it turns out to matter, but nothing measured so far says it
        does.
        """
        if rows.numel() == 0:
            return
        self.codebook.data[rows] = replacements.to(self.codebook.dtype)
        self.reseeds += rows.numel()

    def dead_codes(self) -> Tensor:
        """EnCodec's `expired_codes = cluster_size < threshold_ema_dead_code`."""
        return (self.usage < self.config.dead_code_threshold).nonzero(as_tuple=True)[0]

    def perplexity(self) -> Tensor:
        """`exp(H(usage))`, in `[1, codebook_size]`.

        Moves before utilization does -- a codebook concentrating on a few entries
        loses perplexity while every code is still nominally in use.
        """
        probs = self.usage / self.usage.sum().clamp_min(1e-9)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum()
        return entropy.exp()

    def forward(self, hidden: Tensor, slot_mask: Tensor) -> QuantizerOutput:
        codebook = self.codebook.to(hidden.dtype)
        # ||g - c||^2 expanded. The ||g||^2 term does not affect the argmin, but
        # is kept so `distances` is a true squared distance -- soft assignment
        # below softmaxes over it, where a dropped constant would not cancel.
        distances = (
            hidden.pow(2).sum(-1, keepdim=True)
            - 2.0 * hidden @ codebook.t()
            + codebook.pow(2).sum(-1)
        )
        indices = distances.argmin(dim=-1)
        soft = self.config.soft_assign_temperature > 0.0
        if soft:
            # Differentiable in its own right, so it must NOT be wrapped in the
            # straight-through estimator below -- detaching would cut the very
            # gradient that makes soft assignment worth annealing from.
            weights = torch.softmax(
                -distances / self.config.soft_assign_temperature, dim=-1
            )
            quantized = weights @ codebook
        else:
            quantized = F.embedding(indices, codebook)

        valid = slot_mask.unsqueeze(-1)
        # Mean over dimensions and over each ROW's valid slots, then summed over
        # rows: every term in the objective is a sum over rows of a per-row mean,
        # and the training loop divides by the step's *global* row count. That is
        # what keeps the quotient invariant to how rows were grouped into
        # micro-batches, and it makes `commit_weight` scale-free -- weighted per
        # answer token, as it was before, its strength moved with (L/rho)/A.
        denominator = hidden.shape[-1]
        slots = slot_mask.sum(-1).clamp_min(1)
        codebook_slot = ((hidden.detach() - quantized) * valid).pow(2).sum(
            -1
        ) / denominator
        commit_slot = ((hidden - quantized.detach()) * valid).pow(2).sum(
            -1
        ) / denominator
        codebook_loss = (codebook_slot.sum(-1) / slots).sum()
        commit_loss = (commit_slot.sum(-1) / slots).sum()

        # Straight-through, written as `q + (g - sg[g])` rather than the more
        # common `g + sg[q - g]`. Both have identity gradient to `g` and none to
        # the codebook, but this form is **bitwise** equal to `q` in the forward
        # pass where the common one carries ~1e-7 of float error. Since `z = q` is
        # spliced directly into the decoder with no rescaling, that exactness is
        # what makes the spliced vector *be* the code rather than approximate it.
        z = quantized if soft else quantized.detach() + (hidden - hidden.detach())

        counts = torch.bincount(
            indices[slot_mask], minlength=self.config.codebook_size
        ).to(hidden.device)
        return QuantizerOutput(
            z=z,
            codebook_loss=codebook_loss,
            commit_loss=commit_loss,
            indices=indices,
            counts=counts,
            encoder_output=hidden.detach(),
        )


RopeTables = tuple[Tensor, Tensor]


def rotate_half(x: Tensor) -> Tensor:
    """Split the head dimension in half and rotate: the GPT-NeoX/HF convention.

    The same convention the frozen decoder uses, so "rotated by position p" means
    one thing across both. (RoFormer's original interleaves adjacent pairs instead;
    the two differ only by a fixed permutation of a learned projection's outputs.)
    """
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_tables(positions: Tensor, head_dim: int, theta: float) -> RopeTables:
    """`cos`/`sin` for `positions`, shaped `[B, 1, T, head_dim]` for `[B, H, T, D]` q/k.

    Built in float32 whatever the activations are, as transformers does: the angle
    is `position * theta^(-2i/head_dim)` and positions reach 32,768 here, where
    bf16 cannot even represent consecutive integers.
    """
    exponent = torch.arange(
        0, head_dim, 2, device=positions.device, dtype=torch.float32
    )
    inv_freq = 1.0 / (theta ** (exponent / head_dim))
    angles = positions.float().unsqueeze(-1) * inv_freq
    emb = torch.cat((angles, angles), dim=-1).unsqueeze(-3)
    return emb.cos(), emb.sin()


def apply_rope(x: Tensor, tables: RopeTables | None) -> Tensor:
    """Rotate `[B, H, T, D]` queries or keys. `None` leaves them untouched."""
    if tables is None:
        return x
    cos, sin = tables
    return x * cos.to(x.dtype) + rotate_half(x) * sin.to(x.dtype)


class Attention(nn.Module):
    """Multi-head attention over an explicit boolean mask.

    Hand-rolled rather than `nn.MultiheadAttention` so the mask convention is
    visible: `True` means *attend*, matching `scaled_dot_product_attention` and
    the opposite of `nn.MultiheadAttention`'s `attn_mask`.

    Queries and keys carry their own rotary tables, because in cross-attention they
    live at different positions -- the query at its span's last token, the key at
    its own token index.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        keyvalue: Tensor,
        mask: Tensor | None,
        query_rope: RopeTables | None = None,
        key_rope: RopeTables | None = None,
    ) -> Tensor:
        q = apply_rope(self._heads(self.q_proj(query)), query_rope)
        k = apply_rope(self._heads(self.k_proj(keyvalue)), key_rope)
        v = self._heads(self.v_proj(keyvalue))
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        merged = attended.transpose(1, 2).flatten(2)
        return self.out_proj(merged)


class EncoderBlock(nn.Module):
    """One *Attention Is All You Need* decoder block, minus the output head.

    Pre-norm rather than the 2017 original's post-norm (which needs LR warmup and
    destabilizes with depth) and RMSNorm rather than LayerNorm (matching Qwen3's
    own convention, and cheaper). Cross-attention K/V are projected per block
    straight from full-width memory, so no single lossy projection sits in front
    of every block's view of the trace.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        d = config.d_llm
        self.self_norm = nn.RMSNorm(d)
        self.self_attn = Attention(d, config.n_heads)
        self.cross_norm = nn.RMSNorm(d)
        self.cross_attn = Attention(d, config.n_heads)
        self.ffn_norm = nn.RMSNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, config.ffn_mult * d, bias=False),
            nn.SiLU(),
            nn.Linear(config.ffn_mult * d, d, bias=False),
        )

    def forward(
        self,
        hidden: Tensor,
        memory: Tensor,
        self_mask: Tensor | None,
        cross_mask: Tensor | None,
        slot_rope: RopeTables | None = None,
        span_rope: RopeTables | None = None,
        memory_rope: RopeTables | None = None,
    ) -> Tensor:
        normed = self.self_norm(hidden)
        hidden = hidden + self.self_attn(
            normed, normed, self_mask, slot_rope, slot_rope
        )
        hidden = hidden + self.cross_attn(
            self.cross_norm(hidden), memory, cross_mask, span_rope, memory_rope
        )
        return hidden + self.ffn(self.ffn_norm(hidden))


class CoTEncoder(nn.Module):
    """Latent slots -> quantized codes, in the decoder's input-embedding space.

    Pure with respect to the frozen LLM: `latent_init` and `memory` arrive as
    plain tensors, so nothing here can accidentally backpropagate into the
    decoder. The final norm is not optional -- a pre-norm residual stream is
    unnormalized and grows with depth, and since `z = q` the codebook would
    otherwise have to chase whatever scale the stream happens to reach.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            EncoderBlock(config) for _ in range(config.n_blocks)
        )
        self.out_norm = nn.RMSNorm(config.d_llm)
        # RMSNorm pins its output to `rms(weight)` for ANY input, so this constant
        # -- not the residual stream, and not any branch initialization -- is what
        # sets the scale of every code spliced into the frozen decoder. See
        # `EncoderConfig.embed_rms`. Learnable, so the model can still move it.
        nn.init.constant_(self.out_norm.weight, config.embed_rms)
        self.quantizer = VectorQuantizer(config)

    def forward(
        self,
        latent_init: Tensor,
        memory: Tensor,
        slot_mask: Tensor,
        memory_mask: Tensor,
        cross_limit: Tensor | None = None,
    ) -> QuantizerOutput:
        """
        ``latent_init``  [B, K, d]  seed per slot; a constant, never a parameter.
        ``memory``       [B, M, d]  frozen LLM hidden states over [prompt; CoT].
        ``slot_mask``    [B, K]     True where the slot is real, not padding.
        ``memory_mask``  [B, M]     True where the memory position is real.
        ``cross_limit``  [B, K]     exclusive upper bound on the memory index each
                                    slot may attend to; required when
                                    ``cross_attn_mask == "causal"``, ignored
                                    otherwise.
        """
        # The decoder runs in bf16 while these parameters are fp32 masters, so the
        # inputs arrive in the decoder's dtype. `encode` casts at the boundary
        # rather than relying on an enclosing autocast: a module that is only
        # correct inside a context manager is one that will eventually be called
        # outside one. Under autocast this is harmless -- autocast still picks the
        # matmul dtype.
        return self.quantizer(
            self.encode(latent_init, memory, slot_mask, memory_mask, cross_limit),
            slot_mask,
        )

    def encode(
        self,
        latent_init: Tensor,
        memory: Tensor,
        slot_mask: Tensor,
        memory_mask: Tensor,
        cross_limit: Tensor | None = None,
    ) -> Tensor:
        """Everything up to and including `out_norm`, without quantizing.

        Split out so the k-means codebook initializer can see real pre-quantization
        outputs before a codebook exists to quantize against.
        """
        self_mask = self._self_mask(slot_mask)
        cross_mask = self._cross_mask(slot_mask, memory_mask, cross_limit)
        dtype = self.out_norm.weight.dtype
        hidden = latent_init.to(dtype)
        memory = memory.to(dtype)
        slot_rope, span_rope, memory_rope = self._rope(
            slot_mask.shape[1], memory.shape[1], cross_limit
        )
        for block in self.blocks:
            hidden = block(
                hidden, memory, self_mask, cross_mask, slot_rope, span_rope, memory_rope
            )
        return self.out_norm(hidden)

    def rope_positions(
        self, slots: int, length: int, cross_limit: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """The three position arrays: `(slot index, span end, memory index)`.

        Each attention gets ordinary RoPE in the coordinate system it works in,
        which is not the same system for both:

        * **self-attention over latents -> slot index** `j`. Uniform spacing, no
          dependence on span lengths, and the layout the codes actually occupy in
          Pass B, where they sit at consecutive positions.
        * **cross-attention -> Pass A token index** on both sides: memory column `m`
          is token `m` (Pass A is right-padded, so the column index *is* the
          position, exactly as the decoder reads it), and slot `i` queries from
          `cross_limit_i - 1`, the last token of its own span. Its span then lies at
          relative offsets `0 .. len_i - 1` in every row, so "read my span" is a
          fixed offset a head can learn. Numbering the query by slot index instead
          would mix units: the offset to a slot's own span would drift with `i` (at
          rho=4 on a long trace: -135 at the first slot, -14,139 at the last) and
          start at the row's prompt length, which ranges 28..3,058 here.

        That a slot has two position numbers is not a contradiction: RoPE only needs
        query and key to share a coordinate system within one attention, and each of
        these does. Perceiver AR (arXiv:2202.07765) is the precedent for the
        cross-attention convention -- one latent per trailing input position, causal
        masking, rotary on both heads.
        """
        if cross_limit is None:
            raise ValueError(
                "position_encoding='rope' needs cross_limit (prompt_len + span end "
                "per slot): it is where the cross-attention query positions come "
                "from, in both mask modes."
            )
        device = cross_limit.device
        slot_positions = torch.arange(slots, device=device).unsqueeze(0)
        memory_positions = torch.arange(length, device=device).unsqueeze(0)
        # Padded slots carry cross_limit=1 (collate keeps their attention row
        # non-empty), so this is 0 for them. Their output is discarded anyway.
        return slot_positions, (cross_limit - 1).clamp_min(0), memory_positions

    def _rope(
        self, slots: int, length: int, cross_limit: Tensor | None
    ) -> tuple[RopeTables | None, RopeTables | None, RopeTables | None]:
        """Tables for one forward, shared by every block. `None` when RoPE is off."""
        if self.config.position_encoding != "rope":
            return None, None, None
        head_dim = self.config.d_llm // self.config.n_heads
        theta = self.config.rope_theta
        slot_positions, span_positions, memory_positions = self.rope_positions(
            slots, length, cross_limit
        )
        return (
            rope_tables(slot_positions, head_dim, theta),
            rope_tables(span_positions, head_dim, theta),
            rope_tables(memory_positions, head_dim, theta),
        )

    def _self_mask(self, slot_mask: Tensor) -> Tensor:
        batch, slots = slot_mask.shape
        allowed = slot_mask[:, None, None, :].expand(batch, 1, slots, slots).clone()
        if self.config.self_attn_mask == "causal":
            lower = torch.ones(
                slots, slots, dtype=torch.bool, device=slot_mask.device
            ).tril()
            allowed &= lower
        # A padded slot with no visible key would attend over nothing; let it see
        # itself so the row stays finite. Its output is discarded by slot_mask.
        eye = torch.eye(slots, dtype=torch.bool, device=slot_mask.device)
        return allowed | eye

    def _cross_mask(
        self,
        slot_mask: Tensor,
        memory_mask: Tensor,
        cross_limit: Tensor | None,
    ) -> Tensor:
        batch, slots = slot_mask.shape
        length = memory_mask.shape[1]
        allowed = memory_mask[:, None, None, :].expand(batch, 1, slots, length).clone()
        if self.config.cross_attn_mask == "causal":
            if cross_limit is None:
                raise ValueError(
                    "cross_attn_mask='causal' requires cross_limit "
                    "(prompt_len + span end per slot)."
                )
            positions = torch.arange(length, device=slot_mask.device)
            # [M] < [B, 1, K, 1] broadcasts to [B, 1, K, M]: slot i may attend to
            # memory position j only while j < prompt_len + span_end(i).
            allowed &= positions < cross_limit[:, None, :, None]
        # Position 0 is always inside every prefix, so no row is fully masked.
        allowed[..., 0] = True
        return allowed

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
