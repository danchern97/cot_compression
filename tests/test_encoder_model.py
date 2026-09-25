"""Encoder and quantizer invariants.

Two classes of silent bug live here. A mask that leaks lets a latent read trace
it should not see, which would inflate every result without failing anything; and
a straight-through estimator wired backwards makes the encoder untrainable while
the loss still descends on the codebook alone. Both are asserted directly.
"""

from __future__ import annotations

import pytest
import torch

from cot_compression.encoder.model import CoTEncoder, EncoderConfig, VectorQuantizer

D_LLM = 32


def _config(**overrides) -> EncoderConfig:
    base = {
        "d_llm": D_LLM,
        "n_blocks": 2,
        "n_heads": 4,
        "ffn_mult": 2,
        "codebook_size": 8,
    }
    return EncoderConfig(**{**base, **overrides})


def _batch(batch: int = 2, slots: int = 5, memory: int = 11):
    torch.manual_seed(0)
    return (
        torch.randn(batch, slots, D_LLM),
        torch.randn(batch, memory, D_LLM),
        torch.ones(batch, slots, dtype=torch.bool),
        torch.ones(batch, memory, dtype=torch.bool),
    )


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides",
    [
        {"d_llm": 30, "n_heads": 4},
        {"self_attn_mask": "diagonal"},
        {"cross_attn_mask": "none"},
        {"latent_init": "learned"},
        {"codebook_size": 1},
    ],
)
def test_invalid_config_raises(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


# --------------------------------------------------------------------------- #
# Quantizer
# --------------------------------------------------------------------------- #


def test_straight_through_matches_hard_codes_in_value():
    """`q_tilde == q` exactly forward; they differ only in gradient."""
    quantizer = VectorQuantizer(_config())
    hidden = torch.randn(2, 5, D_LLM, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)
    out = quantizer(hidden, mask)
    hard = quantizer.codebook[out.indices]
    assert torch.equal(out.z, hard)


def test_straight_through_passes_gradient_to_the_encoder_output():
    """Without this the argmin's zero gradient starves everything upstream."""
    quantizer = VectorQuantizer(_config())
    hidden = torch.randn(2, 5, D_LLM, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)
    quantizer(hidden, mask).z.sum().backward()
    assert hidden.grad is not None
    # d(z)/d(hidden) is the identity, so every element gets exactly 1.
    assert torch.allclose(hidden.grad, torch.ones_like(hidden.grad))


def test_codebook_loss_does_not_reach_the_encoder_output():
    """`sg[g]` in the codebook term: it trains codes, never the encoder."""
    quantizer = VectorQuantizer(_config())
    hidden = torch.randn(2, 5, D_LLM, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)
    quantizer(hidden, mask).codebook_loss.backward()
    assert hidden.grad is None or torch.allclose(hidden.grad, torch.zeros_like(hidden))


def test_commit_loss_does_not_reach_the_codebook():
    quantizer = VectorQuantizer(_config())
    hidden = torch.randn(2, 5, D_LLM, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)
    quantizer(hidden, mask).commit_loss.backward()
    grad = quantizer.codebook.grad
    assert grad is None or torch.allclose(grad, torch.zeros_like(grad))


def test_padded_slots_are_excluded_from_losses_and_counts():
    quantizer = VectorQuantizer(_config())
    hidden = torch.randn(2, 6, D_LLM)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[:, 3:] = False
    full = quantizer(hidden, mask)
    # Corrupting only the padded slots must change nothing observable.
    hidden[:, 3:] = 1e4
    again = quantizer(hidden, mask)
    assert torch.allclose(full.codebook_loss, again.codebook_loss)
    assert torch.allclose(full.commit_loss, again.commit_loss)
    assert torch.equal(full.counts, again.counts)
    assert int(full.counts.sum()) == int(mask.sum())


def test_usage_and_perplexity_track_a_collapsed_codebook():
    config = _config(codebook_size=8, usage_decay=0.0)
    quantizer = VectorQuantizer(config)
    collapsed = torch.zeros(config.codebook_size)
    collapsed[0] = 100.0
    quantizer.update_usage(collapsed)
    assert pytest.approx(float(quantizer.perplexity()), abs=1e-4) == 1.0
    assert quantizer.dead_codes().numel() == config.codebook_size - 1

    uniform = torch.full((config.codebook_size,), 100.0)
    quantizer.update_usage(uniform)
    assert (
        pytest.approx(float(quantizer.perplexity()), abs=1e-3) == config.codebook_size
    )
    assert quantizer.dead_codes().numel() == 0


def test_kmeans_init_puts_codes_inside_the_encoder_output_cloud():
    """The property that makes every reference restart rule safe.

    k-means centroids of the encoder's own outputs are, by construction, in the
    same distribution as the vectors they quantize -- so the dead set is a handful
    rather than a majority. Seeding from a *different* distribution is what left
    46 of 64 codes dead from step 1 in the previous campaign.
    """
    quantizer = VectorQuantizer(_config(codebook_size=8))
    generator = torch.Generator().manual_seed(0)
    # Four tight, well-separated clusters: k-means must land inside them.
    centres = torch.randn(4, D_LLM, generator=generator) * 3.0
    pool = (
        centres.repeat_interleave(64, 0)
        + torch.randn(256, D_LLM, generator=generator) * 0.01
    )
    quantizer.init_codebook_from_encoder_outputs(pool, generator, iters=25)

    assert quantizer.codebook.shape == (8, D_LLM)
    assert torch.isfinite(quantizer.codebook).all()
    # Every code must be close to some cluster centre, not stranded between them.
    to_centre = torch.cdist(quantizer.codebook.detach(), centres).min(dim=-1).values
    assert float(to_centre.max()) < 0.5, "a code landed outside every cluster"
    # And the codebook's scale must match the pool's, which is the whole point.
    pool_rms = float(pool.pow(2).mean(-1).sqrt().mean())
    code_rms = float(quantizer.codebook.detach().pow(2).mean(-1).sqrt().mean())
    assert code_rms == pytest.approx(pool_rms, rel=0.1)
    # Usage starts at the threshold, so nothing reads as dead at step 0.
    assert quantizer.dead_codes().numel() == 0


def test_out_norm_puts_encoder_output_at_the_decoder_embedding_scale():
    """RMSNorm pins output RMS to `rms(weight)`, so this constant IS the scale.

    Left at the default 1.0 it is Qwen3's *internal* hidden-state scale, 34x above
    its input-embedding scale -- and `z = q` is spliced into the embedding
    sequence with no rescaling. This is the assertion that would have caught it.
    """
    embed_rms = 0.0289
    encoder = CoTEncoder(_config(codebook_size=8, embed_rms=embed_rms))
    assert float(encoder.out_norm.weight.detach().min()) == pytest.approx(embed_rms)

    torch.manual_seed(0)
    latent = torch.randn(2, 6, D_LLM) * embed_rms
    memory = torch.randn(2, 12, D_LLM) * 3.0
    hidden = encoder.encode(
        latent,
        memory,
        torch.ones(2, 6, dtype=torch.bool),
        torch.ones(2, 12, dtype=torch.bool),
        cross_limit=torch.full((2, 6), 12),
    )
    per_vector = hidden.detach().pow(2).mean(-1).sqrt()
    assert float(per_vector.mean()) == pytest.approx(embed_rms, rel=1e-4)


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #


def _attention_reaches(encoder: CoTEncoder, memory_len: int, slot: int) -> set[int]:
    """Memory positions slot `slot` can influence, by gradient reachability."""
    latent, memory, slot_mask, memory_mask = _batch(1, 5, memory_len)
    memory = memory.clone().requires_grad_(True)
    limit = torch.tensor([[2, 4, 6, 8, 10]])
    out = encoder(latent, memory, slot_mask, memory_mask, cross_limit=limit)
    out.z[0, slot].sum().backward()
    assert memory.grad is not None
    return {
        int(i) for i in (memory.grad[0].abs().sum(-1) > 0).nonzero(as_tuple=True)[0]
    }


def test_causal_cross_attention_cannot_see_past_its_span_end():
    encoder = CoTEncoder(
        _config(self_attn_mask="causal", cross_attn_mask="causal", n_blocks=1)
    )
    # Slot 0's limit is 2, so only memory positions {0, 1} may influence it.
    assert _attention_reaches(encoder, memory_len=11, slot=0) <= {0, 1}


def test_bidirectional_cross_attention_sees_the_whole_trace():
    encoder = CoTEncoder(
        _config(self_attn_mask="causal", cross_attn_mask="bidirectional", n_blocks=1)
    )
    assert _attention_reaches(encoder, memory_len=11, slot=0) == set(range(11))


def test_causal_self_attention_hides_later_slots():
    encoder = CoTEncoder(
        _config(self_attn_mask="causal", cross_attn_mask="bidirectional", n_blocks=1)
    )
    latent, memory, slot_mask, memory_mask = _batch(1, 5, 7)
    latent = latent.clone().requires_grad_(True)
    encoder(latent, memory, slot_mask, memory_mask).z[0, 1].sum().backward()
    assert latent.grad is not None
    reached = {
        int(i) for i in (latent.grad[0].abs().sum(-1) > 0).nonzero(as_tuple=True)[0]
    }
    assert reached <= {0, 1}, f"slot 1 saw future slots: {reached}"


def test_bidirectional_self_attention_sees_every_slot():
    encoder = CoTEncoder(
        _config(
            self_attn_mask="bidirectional", cross_attn_mask="bidirectional", n_blocks=1
        )
    )
    latent, memory, slot_mask, memory_mask = _batch(1, 5, 7)
    latent = latent.clone().requires_grad_(True)
    encoder(latent, memory, slot_mask, memory_mask).z[0, 1].sum().backward()
    assert latent.grad is not None
    reached = {
        int(i) for i in (latent.grad[0].abs().sum(-1) > 0).nonzero(as_tuple=True)[0]
    }
    assert reached == set(range(5))


def test_causal_cross_attention_requires_a_limit():
    encoder = CoTEncoder(_config(cross_attn_mask="causal"))
    latent, memory, slot_mask, memory_mask = _batch()
    with pytest.raises(ValueError, match="requires cross_limit"):
        encoder(latent, memory, slot_mask, memory_mask, cross_limit=None)


def test_output_is_finite_with_padding_everywhere():
    """Padded slots and padded memory must not produce NaN.

    A fully-masked attention row is the classic way to get one, and a single NaN
    would poison the whole batch's gradient even though the loss masks the slot.
    """
    encoder = CoTEncoder(_config(cross_attn_mask="causal"))
    latent, memory, slot_mask, memory_mask = _batch(2, 6, 9)
    slot_mask[:, 4:] = False
    memory_mask[:, 5:] = False
    limit = torch.tensor([[1, 2, 3, 4, 0, 0], [2, 3, 4, 5, 0, 0]])
    out = encoder(latent, memory, slot_mask, memory_mask, cross_limit=limit)
    assert torch.isfinite(out.z).all()
    assert torch.isfinite(out.codebook_loss) and torch.isfinite(out.commit_loss)


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


def test_padded_batch_matches_unbatched_for_valid_slots():
    """Padding must be inert: batching two samples of different K must not move
    either one's codes."""
    encoder = CoTEncoder(
        _config(self_attn_mask="causal", cross_attn_mask="causal", n_blocks=1)
    ).eval()
    torch.manual_seed(1)
    memory = torch.randn(1, 9, D_LLM)
    latent = torch.randn(1, 3, D_LLM)
    limit = torch.tensor([[3, 6, 9]])
    with torch.no_grad():
        alone = encoder(
            latent,
            memory,
            torch.ones(1, 3, dtype=torch.bool),
            torch.ones(1, 9, dtype=torch.bool),
            cross_limit=limit,
        ).z

        padded_latent = torch.cat([latent, torch.randn(1, 2, D_LLM)], dim=1)
        slot_mask = torch.tensor([[True, True, True, False, False]])
        padded_limit = torch.tensor([[3, 6, 9, 1, 1]])
        batched = encoder(
            padded_latent,
            memory,
            slot_mask,
            torch.ones(1, 9, dtype=torch.bool),
            cross_limit=padded_limit,
        ).z
    # Not bitwise: a 5-slot self-attention matmul reduces in a different order
    # than a 3-slot one, so float rounding differs. A tolerance this tight still
    # catches real leakage, which would move values by orders of magnitude more.
    assert torch.allclose(alone, batched[:, :3], rtol=0, atol=1e-5)


def test_reseed_replaces_dead_codes_and_leaves_live_ones_alone():
    quantizer = VectorQuantizer(_config(codebook_size=8, usage_decay=0.0))
    quantizer.update_usage(torch.tensor([50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    dead = quantizer.dead_codes()
    assert dead.tolist() == [2, 3, 4, 5, 6, 7]

    before = quantizer.codebook.detach().clone()
    replacements = torch.randn(dead.numel(), D_LLM)
    quantizer.reseed_codes(dead, replacements)

    assert torch.allclose(quantizer.codebook[dead], replacements)
    # Live codes must be untouched.
    assert torch.equal(quantizer.codebook[:2], before[:2])
    # Usage is NOT reset, so the same codes still read as dead -- EnCodec's
    # behaviour, pinned in test_replacement_leaves_usage_alone_like_encodec.
    assert quantizer.dead_codes().tolist() == dead.tolist()
    assert quantizer.reseeds == dead.numel()


def test_reseed_is_a_noop_when_nothing_is_dead():
    quantizer = VectorQuantizer(_config(codebook_size=8, usage_decay=0.0))
    quantizer.update_usage(torch.full((8,), 10.0))
    before = quantizer.codebook.detach().clone()
    quantizer.reseed_codes(quantizer.dead_codes(), torch.empty(0, D_LLM))
    assert torch.equal(quantizer.codebook, before)
    assert quantizer.reseeds == 0


def test_dead_code_rule_matches_encodec():
    """`expired_codes = cluster_size < threshold_ema_dead_code`, absolute.

    Deliberately NOT a fraction of a code's fair share: no reference implementation
    does that, and it was never shown to be necessary here. The consequence -- that
    the same threshold means 4.2% of fair share at |C|=1024 and 0.26% at |C|=64 --
    is accepted, and is why codebook-health metrics do not compare across sizes.
    """
    quantizer = VectorQuantizer(_config(codebook_size=8, dead_code_threshold=2.0))
    quantizer.usage.copy_(torch.tensor([0.0, 1.0, 1.9, 2.0, 2.1, 50.0, 500.0, 5e4]))
    assert quantizer.dead_codes().tolist() == [0, 1, 2]


def test_replacement_leaves_usage_alone_like_encodec():
    """EnCodec's `replace_` writes only `embed`; `cluster_size` is untouched.

    So a code that earns nothing is replaced again next step. That churn is
    accepted here: it is what the reference does, and nothing measured showed it
    caused harm. `vq/reseeds_total` against `vq/perplexity` is what would.
    """
    quantizer = VectorQuantizer(_config(codebook_size=8, dead_code_threshold=2.0))
    quantizer.usage.copy_(torch.full((8,), 100.0))
    quantizer.usage[:3] = 0.0
    dead = quantizer.dead_codes()
    assert dead.tolist() == [0, 1, 2]

    before = quantizer.usage.clone()
    quantizer.reseed_codes(dead, torch.randn(3, D_LLM))
    assert quantizer.reseeds == 3
    assert torch.equal(quantizer.usage, before), "usage must not be reset"
    assert quantizer.dead_codes().tolist() == [0, 1, 2], "still dead until it earns"


def test_kmeans_init_seeds_usage_from_bin_counts():
    """EnCodec's `init_embed_` copies the k-means bin counts into `cluster_size`.

    The statistic then starts at the truth instead of a placeholder, so the very
    first dead-code check is meaningful. The buffer is zeros beforehand, matching
    EnCodec, because nothing should read it before initialization.
    """
    quantizer = VectorQuantizer(_config(codebook_size=8))
    assert float(quantizer.usage.sum()) == 0.0

    generator = torch.Generator().manual_seed(0)
    pool = torch.randn(256, D_LLM, generator=generator)
    quantizer.init_codebook_from_encoder_outputs(pool, generator, iters=10)
    assert float(quantizer.usage.sum()) == pytest.approx(256.0)
    assert float(quantizer.usage.max()) > 0.0


# --------------------------------------------------------------------------- #
# Per-row normalization
# --------------------------------------------------------------------------- #


def test_quantizer_losses_are_per_row_means_summed_over_rows():
    """Each row contributes its own mean over its own slots, and no more.

    The training loop divides by the step's global ROW count, so a row holding ten
    times the slots must not carry ten times the quantizer weight -- that is what
    made `commit_weight` move with the compression ratio when these were slot sums.
    """
    quantizer = VectorQuantizer(_config())
    torch.manual_seed(0)
    hidden = torch.randn(3, 6, D_LLM)
    mask = torch.ones(3, 6, dtype=torch.bool)
    mask[1, 2:] = False  # a row with 2 valid slots beside two with 6
    out = quantizer(hidden, mask)

    # The literal definition: mean over dimensions and over the row's valid slots.
    quantized = torch.nn.functional.embedding(out.indices, quantizer.codebook)
    expected_row = sum(
        (hidden[r, : int(mask[r].sum())] - quantized[r, : int(mask[r].sum())])
        .pow(2)
        .mean()
        for r in range(3)
    )
    assert float(out.commit_loss.detach()) == pytest.approx(
        float(expected_row), rel=1e-6
    )
    # A short row is not diluted by the long ones: doubling its error moves the
    # per-row total by its own share, 1/3 of the change, not 2/14 of it.
    louder = hidden.clone()
    louder[1, :2] *= 3.0
    grew = quantizer(louder, mask)
    assert float(grew.commit_loss) > float(out.commit_loss)


def test_codebook_geometry_sees_a_collapse_that_usage_would_miss():
    """Duplicate codes must show up even while every code is still being used."""
    from cot_compression.encoder.training import codebook_geometry

    torch.manual_seed(0)
    spread = torch.randn(16, D_LLM)
    healthy = codebook_geometry(spread)
    # Random directions in 32 dimensions are near-orthogonal, not identical.
    assert abs(healthy["vq/cos_mean"]) < 0.4
    assert healthy["vq/cos_nn_mean"] < 0.9
    assert healthy["vq/dist_nn_mean"] > 0.0

    collapsed = spread.clone()
    collapsed[1::2] = collapsed[0::2] + 1e-4  # every code paired with a twin
    merged = codebook_geometry(collapsed)
    assert merged["vq/cos_nn_mean"] == pytest.approx(1.0, abs=1e-3)
    assert merged["vq/dist_nn_mean"] < healthy["vq/dist_nn_mean"]


# --------------------------------------------------------------------------- #
# Positional encoding
# --------------------------------------------------------------------------- #


def _rope_config(**overrides) -> EncoderConfig:
    return _config(position_encoding="rope", **overrides)


def test_rope_positions_are_slot_index_and_span_end():
    """Each attention gets the coordinate system it works in, and nothing else."""
    encoder = CoTEncoder(_rope_config())
    cross_limit = torch.tensor([[4, 9, 13], [2, 6, 6]])
    slots, spans, memory = encoder.rope_positions(3, 20, cross_limit)

    # Self-attention: the slot's index, shared by every row.
    assert torch.equal(slots, torch.arange(3).unsqueeze(0))
    # Cross-attention: the last token of the slot's own span, and the memory's own
    # column. `cross_limit` is exclusive, hence the -1.
    assert torch.equal(spans, cross_limit - 1)
    assert torch.equal(memory, torch.arange(20).unsqueeze(0))


def test_rope_needs_cross_limit_to_know_where_a_slot_is():
    encoder = CoTEncoder(_rope_config(cross_attn_mask="bidirectional"))
    latent, memory, slot_mask, memory_mask = _batch()
    # Even bidirectionally: the mask no longer needs the limit, but the query
    # positions still do, so silently falling back to slot 0 is not an option.
    with pytest.raises(ValueError, match="cross_limit"):
        encoder.encode(latent, memory, slot_mask, memory_mask, cross_limit=None)


def test_rope_attention_depends_only_on_relative_position():
    """Shifting both sides' positions must leave every attention logit unchanged.

    This is the property the span-end anchor buys: a slot finds its own span at
    offsets 0..len-1 whatever the prompt in front of it, so one head can learn "read
    my span" and be right in every row. Numbering the query by slot index instead
    would break it -- the offset from a slot to its span would then move with both
    the slot index and the prompt length.
    """
    from cot_compression.encoder.model import apply_rope, rope_tables

    torch.manual_seed(0)
    head_dim, theta = 16, 1_000_000.0
    query = torch.randn(1, 2, 3, head_dim)
    key = torch.randn(1, 2, 7, head_dim)
    query_positions = torch.tensor([[3, 9, 14]])
    key_positions = torch.arange(7).unsqueeze(0)

    def logits(shift: int) -> torch.Tensor:
        rotated_q = apply_rope(
            query, rope_tables(query_positions + shift, head_dim, theta)
        )
        rotated_k = apply_rope(key, rope_tables(key_positions + shift, head_dim, theta))
        return rotated_q @ rotated_k.transpose(-1, -2)

    torch.testing.assert_close(logits(0), logits(1000), atol=1e-4, rtol=1e-4)
    # And position must actually matter: unrotated logits differ from rotated ones.
    plain = query @ key.transpose(-1, -2)
    assert not torch.allclose(plain, logits(0), atol=1e-3)


def test_rope_moves_a_slot_when_its_span_moves():
    """The cross-attention query position must come from the slot's own span end."""
    torch.manual_seed(0)
    encoder = CoTEncoder(_rope_config(cross_attn_mask="bidirectional")).eval()
    latent, memory, slot_mask, memory_mask = _batch(batch=1, slots=3, memory=12)
    with torch.no_grad():
        near = encoder.encode(
            latent,
            memory,
            slot_mask,
            memory_mask,
            cross_limit=torch.tensor([[4, 8, 12]]),
        )
        far = encoder.encode(
            latent,
            memory,
            slot_mask,
            memory_mask,
            cross_limit=torch.tensor([[2, 6, 10]]),
        )
    # Same content, same mask (bidirectional), different anchors -> different codes.
    assert not torch.allclose(near, far, atol=1e-5)


def test_rope_changes_the_encoding():
    """A run with positions must not silently equal one without them."""
    torch.manual_seed(0)
    latent, memory, slot_mask, memory_mask = _batch()
    cross_limit = torch.tensor([[2, 4, 6, 8, 10], [3, 5, 7, 9, 11]])

    plain = CoTEncoder(_config()).eval()
    roped = CoTEncoder(_rope_config()).eval()
    roped.load_state_dict(plain.state_dict())
    with torch.no_grad():
        without = plain.encode(
            latent, memory, slot_mask, memory_mask, cross_limit=cross_limit
        )
        with_rope = roped.encode(
            latent, memory, slot_mask, memory_mask, cross_limit=cross_limit
        )
    assert not torch.allclose(without, with_rope, atol=1e-4)
