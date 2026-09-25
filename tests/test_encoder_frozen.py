"""End-to-end wiring of the frozen backbone against a real (tiny) Qwen3.

A real model rather than a fake, because the assertions here are precisely about
things a fake would let pass: that non-reentrant gradient checkpointing works
when *no parameter* requires grad, that gradient reaches the encoder through
`inputs_embeds`, and that it reaches no decoder parameter at all.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from cot_compression.encoder.frozen import FrozenBackbone, build_latent_init
from cot_compression.encoder.model import CoTEncoder, EncoderConfig
from cot_compression.patching import UniformPatchingMethod

D_LLM = 64
VOCAB = 512


@pytest.fixture(scope="module")
def backbone() -> FrozenBackbone:
    config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    config.num_hidden_layers = 2
    config.hidden_size = D_LLM
    config.intermediate_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.vocab_size = VOCAB
    config.tie_word_embeddings = True
    return FrozenBackbone(AutoModelForCausalLM.from_config(config), ce_chunk_tokens=16)


def _encoder(**overrides) -> CoTEncoder:
    base = {
        "d_llm": D_LLM,
        "n_blocks": 1,
        "n_heads": 4,
        "ffn_mult": 2,
        "codebook_size": 8,
    }
    return CoTEncoder(EncoderConfig(**{**base, **overrides}))


def test_frozen_decoder_has_no_trainable_parameters(backbone):
    assert not any(p.requires_grad for p in backbone.model.parameters())
    assert backbone.model.model.gradient_checkpointing
    assert backbone.hidden_size == D_LLM


def test_context_pass_is_detached(backbone):
    ids = torch.randint(0, VOCAB, (2, 12))
    memory = backbone.encode_context(ids, torch.ones_like(ids))
    assert memory.shape == (2, 12, D_LLM)
    assert not memory.requires_grad


def test_splice_places_codes_exactly_and_leaves_other_positions_alone(backbone):
    ids = torch.randint(0, VOCAB, (2, 10))
    codes = torch.randn(2, 3, D_LLM)
    positions = torch.tensor([[1, 4, 7], [2, 5, 8]])
    mask = torch.ones(2, 3, dtype=torch.bool)

    spliced = backbone.splice(ids, codes, positions, mask)
    original = backbone.model.get_input_embeddings()(ids).detach()
    for row in range(2):
        for slot in range(3):
            at = int(positions[row, slot])
            assert torch.equal(spliced[row, at], codes[row, slot].to(spliced.dtype))
        untouched = [i for i in range(10) if i not in positions[row].tolist()]
        assert torch.equal(spliced[row, untouched], original[row, untouched])


def test_splice_offsets_cannot_collide():
    """`index_copy` with duplicate indices is undefined behaviour."""
    positions = torch.tensor([[1, 4, 7], [1, 4, 7], [0, 4, 9]])
    mask = torch.ones(3, 3, dtype=torch.bool)
    rows = torch.arange(3).unsqueeze(1)
    offsets = (rows * 10 + positions)[mask]
    assert offsets.unique().numel() == offsets.numel()


def test_padded_slots_are_never_spliced(backbone):
    ids = torch.randint(0, VOCAB, (1, 8))
    codes = torch.full((1, 3, D_LLM), 1e4)
    positions = torch.tensor([[2, 5, 0]])
    mask = torch.tensor([[True, True, False]])
    spliced = backbone.splice(ids, codes, positions, mask)
    original = backbone.model.get_input_embeddings()(ids).detach()
    # Position 0 is the padded slot's placeholder index; it must be untouched.
    assert torch.equal(spliced[0, 0], original[0, 0])


@pytest.mark.parametrize("cross_mask", ["causal", "bidirectional"])
def test_gradient_reaches_the_encoder_but_no_decoder_parameter(backbone, cross_mask):
    """The load-bearing claim of the whole design.

    Gradient must traverse every decoder layer to reach the spliced codes, while
    leaving the decoder's own weights untouched.
    """
    encoder = _encoder(cross_attn_mask=cross_mask, self_attn_mask="causal")
    backbone.model.zero_grad(set_to_none=True)
    encoder.zero_grad(set_to_none=True)

    slots, length = 4, 14
    context = torch.randint(0, VOCAB, (2, 20))
    memory = backbone.encode_context(context, torch.ones_like(context))
    latent = memory[:, :slots].clone()

    out = encoder(
        latent,
        memory,
        torch.ones(2, slots, dtype=torch.bool),
        torch.ones(2, 20, dtype=torch.bool),
        cross_limit=torch.tensor([[5, 10, 15, 20], [5, 10, 15, 20]]),
    )

    ids = torch.randint(0, VOCAB, (2, length))
    positions = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    mask = torch.ones(2, slots, dtype=torch.bool)
    embeds = backbone.splice(ids, out.z, positions, mask)

    labels = torch.full((2, length), -100)
    labels[:, -4:] = ids[:, -4:]
    loss = backbone.answer_ce(embeds, torch.ones_like(ids), labels)
    (loss + out.codebook_loss + 0.25 * out.commit_loss).backward()

    assert torch.isfinite(loss)
    trained = [n for n, p in encoder.named_parameters() if p.grad is not None]
    assert "quantizer.codebook" in trained
    missing = [n for n, p in encoder.named_parameters() if p.grad is None]
    assert not missing, f"encoder parameters received no gradient: {missing}"
    assert all(p.grad is None for p in backbone.model.parameters())


def test_answer_ce_ignores_non_answer_positions(backbone):
    ids = torch.randint(0, VOCAB, (1, 12))
    embeds = backbone.model.get_input_embeddings()(ids).detach()
    mask = torch.ones_like(ids)

    all_ignored = torch.full((1, 12), -100)
    assert float(backbone.answer_ce(embeds, mask, all_ignored)) == 0.0

    scored = all_ignored.clone()
    scored[:, -3:] = ids[:, -3:]
    assert float(backbone.answer_ce(embeds, mask, scored)) > 0.0


def test_latent_init_reuses_the_training_free_methods():
    patching = UniformPatchingMethod(compression_ratio=4.0)
    assert build_latent_init("random", patching).method_family == "random"
    assert build_latent_init("simple_mean", patching).method_family == "simple_mean"
    surprisal = build_latent_init("surprisal_t0", patching)
    assert surprisal.method_family == "surprisal_weighted_mean"
    assert surprisal.weight_signal() == "surprisal"
    with pytest.raises(ValueError, match="Unknown latent_init"):
        build_latent_init("pooled", patching)


def test_encoder_accepts_the_decoders_dtype(backbone):
    """The decoder runs bf16; the encoder holds fp32 masters.

    Regression guard: relying on an enclosing autocast to bridge that made the
    encoder correct only inside a context manager.
    """
    encoder = _encoder(cross_attn_mask="bidirectional")
    ids = torch.randint(0, VOCAB, (1, 10))
    memory = backbone.encode_context(ids, torch.ones_like(ids))
    assert memory.dtype == torch.bfloat16
    out = encoder(
        memory[:, :3].clone(),
        memory,
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 10, dtype=torch.bool),
    )
    assert torch.isfinite(out.z).all()


def test_gradient_checkpointing_actually_engages(backbone):
    """Checkpointing must fire on every layer, and only inside Pass B.

    `GradientCheckpointingLayer.__call__` gates on
    `self.gradient_checkpointing and self.training`, and transformers offers no
    override. An eval-mode decoder skips checkpointing entirely and Pass B retains
    activations for every layer, because `inputs_embeds` requires grad. Nothing
    errors -- the run just OOMs or crawls -- so it is asserted, not trusted.
    """
    # The frozen decoder rests in eval(); train() is entered only for Pass B.
    assert not backbone.model.training, "a frozen decoder must rest in eval()"

    calls = {"n": 0}
    layers = list(backbone.model.model.layers)
    for layer in layers:
        original = layer._gradient_checkpointing_func

        def counting(*args, _original=original, **kwargs):
            calls["n"] += 1
            return _original(*args, **kwargs)

        layer._gradient_checkpointing_func = counting

    ids = torch.randint(0, VOCAB, (1, 16))
    embeds = backbone.model.get_input_embeddings()(ids).detach().requires_grad_(True)
    labels = torch.full((1, 16), -100)
    labels[:, -3:] = ids[:, -3:]
    backbone.answer_ce(embeds, torch.ones_like(ids), labels).backward()

    assert calls["n"] == len(layers), (
        f"checkpointing ran on {calls['n']} of {len(layers)} decoder layers"
    )
    # And the mode flip must not leak past the call that needed it.
    assert not backbone.model.training


def test_context_pass_does_not_enter_train_mode(backbone):
    """Pass A runs under no_grad and has nothing to checkpoint, so it should not
    pay the mode flip -- and must leave the decoder in eval() either way."""
    seen = {}
    original = backbone.model.model.forward

    def record(*args, **kwargs):
        seen["training"] = backbone.model.training
        return original(*args, **kwargs)

    backbone.model.model.forward = record
    try:
        ids = torch.randint(0, VOCAB, (1, 8))
        backbone.encode_context(ids, torch.ones_like(ids))
    finally:
        backbone.model.model.forward = original
    assert seen["training"] is False
    assert not backbone.model.training


def test_a_decoder_with_dropout_is_refused(backbone):
    """train() mode is only sound for a dropout-free decoder; say so loudly."""
    from cot_compression.encoder.frozen import assert_no_train_mode_behaviour

    assert_no_train_mode_behaviour(backbone.model)  # Qwen3 is dropout-free

    backbone.model.config.attention_dropout = 0.1
    try:
        with pytest.raises(ValueError, match="behaves differently in train"):
            assert_no_train_mode_behaviour(backbone.model)
    finally:
        backbone.model.config.attention_dropout = 0.0


def test_label_selection_is_exactly_equivalent_but_cheaper(backbone):
    """Dropping ignored positions before `lm_head` must not change the loss.

    `ignore_index` already contributes zero, so selecting is a pure work saving --
    but only if the shift and the gather compose correctly. On the encoder's data
    only 21.8% of Pass B positions carry a label, so this is a 4.6x reduction in
    `lm_head` work and in the fp32 logit upcast; on SFT it would save nothing,
    which is why it is opt-in rather than the default.
    """
    from cot_compression.training.sft import chunked_ce_from_hidden

    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB, (2, 20))
    hidden = backbone.model.model(
        input_ids=ids, attention_mask=torch.ones_like(ids)
    ).last_hidden_state

    labels = torch.full((2, 20), -100)
    labels[0, -4:] = ids[0, -4:]
    labels[1, -6:] = ids[1, -6:]

    for chunk in (4, 64):
        everything = chunked_ce_from_hidden(
            backbone.model.lm_head, hidden, labels, chunk, select_labels=False
        )
        selected = chunked_ce_from_hidden(
            backbone.model.lm_head, hidden, labels, chunk, select_labels=True
        )
        assert torch.allclose(everything, selected, rtol=0, atol=1e-4), (
            f"chunk={chunk}: {float(everything)} vs {float(selected)}"
        )
        assert float(selected) > 0.0

    # A row with no labels at all must still give exactly zero, not NaN from an
    # empty gather.
    empty = torch.full((2, 20), -100)
    assert (
        float(
            chunked_ce_from_hidden(
                backbone.model.lm_head, hidden, empty, 8, select_labels=True
            )
        )
        == 0.0
    )
