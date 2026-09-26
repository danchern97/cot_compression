"""`train_encoder` end to end, against a tiny real Qwen3.

Fakes the data but not the model: the assertions here are about gradient actually
reaching the encoder through a frozen decoder and about a checkpoint being
resumable, and a stub decoder would let both pass while being wrong.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from cot_compression.encoder.model import CoTEncoder, EncoderConfig
from cot_compression.encoder.training import (
    ENCODER_CONFIG,
    ENCODER_WEIGHTS,
    EncoderPrepDataset,
    SaveableEncoder,
    check_encoder_architecture,
    collate_encoder_batch,
    load_encoder,
    plan_columns,
    train_encoder,
)
from cot_compression.patching import UniformPatchingMethod

TINY = "Qwen/Qwen3-0.6B"


def _tiny_decoder(*_args, **_kwargs):
    config = AutoConfig.from_pretrained(TINY)
    config.num_hidden_layers = 2
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    # >= 16: FlexAttention's Triton kernel refuses a smaller head dimension
    # ("NYI: embedding dimension ... must be at least 16"), so head_dim=8 made the
    # aux pass untestable for a reason that cannot occur in production -- real
    # Qwen3-0.6B has head_dim=128.
    config.head_dim = 16
    config.tie_word_embeddings = True
    return AutoModelForCausalLM.from_config(config)


def _rows(count: int, seed: int) -> Dataset:
    """Rollouts shaped like the real corpus: a think block, then an answer."""
    rng = np.random.default_rng(seed)
    records = []
    for index in range(count):
        cot = " ".join(f"step{int(v)}" for v in rng.integers(0, 50, size=40))
        records.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Question {index}? Reason step by step.",
                    },
                    {
                        "role": "assistant",
                        "content": f"<think>\n{cot}\n</think>\n\nThe answer is {index}.",
                    },
                ],
                "domain": "math",
                "rollout_id": f"p{index:04d}:r0",
            }
        )
    return Dataset.from_list(records)


def _measured(dataset: Dataset, tokenizer) -> Dataset:
    from cot_compression.data.dolci_traces import _measure_row

    return dataset.map(lambda row: _measure_row(row, tokenizer))


def _cfg(tmp_path, **training):
    base = {
        "seed": 1234,
        "deterministic": False,
        "torch_dtype": "float32",
        "autocast_dtype": "bfloat16",
        "decoder_dtype": "float32",
        "attn_implementation": "eager",
        "max_length": 4096,
        "max_batch_tokens": 4096,
        "length_group_size": 64,
        "micro_batch_max_sequences": 2,
        "target_global_batch": 2,
        "max_train_examples": None,
        "max_steps": 2,
        "ce_chunk_tokens": 64,
        # Non-trivial on purpose, so the end-to-end runs exercise padded aux widths.
        "aux_width_multiple": 7,
        "aux_recompile_limit": 128,
        "num_workers": 0,
        "prefetch_factor": 4,
        "warmup_ratio": 0.0,
        "min_warmup_steps": 0,
        "schedule_horizon_steps": None,
        "gradient_clip": 1.0,
        "gradient_checkpointing": True,
        "eval_interval": 100,
        "eval_examples": 4,
        "log_interval": 100,
        "checkpoint_minutes": 0,
        "keep_checkpoints": 2,
        "resume_from_checkpoint": "auto",
    }
    base.update(training)
    return OmegaConf.create(
        {
            "mode": "encoder_train",
            "paths": {"run_dir": str(tmp_path / "enc")},
            "method": {
                "model_name": TINY,
                "use_fast_tokenizer": True,
                "trust_remote_code": False,
            },
            "encoder": {
                "n_blocks": 1,
                "n_heads": 4,
                "ffn_mult": 2,
                "codebook_size": 8,
                "self_attn_mask": "causal",
                "cross_attn_mask": "causal",
                "latent_init": "random",
                "memory_layer": -1,
                "patching": "uniform",
                "commit_weight": 0.25,
                "dead_code_threshold": 2.0,
                "next_patch_weight": 0.0,
                "next_patch_subsample": 1.0,
                "kmeans_iters": 5,
                "reseed_dead_codes": False,
                "usage_decay": 0.99,
                "soft_assign_temperature": 0.0,
            },
            "evaluation": {"methods": {"patching": {"compression_ratio": 4.0}}},
            "training": base,
            "optim": {
                "lr": 1e-3,
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
            "logging": {
                "enabled": False,
                "log_file_name": "encoder.log",
                "log_artifacts": False,
                "max_artifact_mb": 32,
            },
        }
    )


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TINY)


@pytest.fixture
def patched(monkeypatch, tokenizer):
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.setattr(
        "cot_compression.encoder.training.AutoModelForCausalLM",
        type("M", (), {"from_pretrained": staticmethod(_tiny_decoder)}),
    )
    data = DatasetDict(
        {
            "train": _measured(_rows(12, seed=0), tokenizer),
            "val": _measured(_rows(4, seed=1), tokenizer),
        }
    )
    monkeypatch.setattr(
        "cot_compression.encoder.training.load_trace_lengths", lambda cfg: data
    )
    return data


# --------------------------------------------------------------------------- #
# Pieces
# --------------------------------------------------------------------------- #


def test_prep_dataset_produces_aligned_slots(tokenizer):
    rows = _rows(4, seed=3)
    prep = EncoderPrepDataset(
        rows,
        tokenizer,
        UniformPatchingMethod(compression_ratio=4.0),
        seed=7,
        max_length=None,
        vocab_bound=1000,
        latent_init="random",
    )
    sample = prep[0]
    assert sample is not None
    assert sample.num_slots == len(sample.slot_positions) == len(sample.init_ids)
    assert len(sample.cross_limit) == sample.num_slots
    # Positions must be strictly increasing and distinct -- index_copy depends on it.
    assert sample.slot_positions == sorted(set(sample.slot_positions))
    # cross_limit is monotone: each slot sees at least as much as the one before.
    assert sample.cross_limit == sorted(sample.cross_limit)
    assert all(v > 0 for v in sample.cross_limit)


def _signal_patching():
    from cot_compression.compression import build_patching_method

    return build_patching_method(
        "surprisal_threshold",
        OmegaConf.create({"compression_ratio": 4.0, "random": {"max_exponent": 6}}),
    )


def test_prep_dataset_refuses_unsupported_configurations(tokenizer):
    """An unknown initializer must fail at construction, not per sample.

    `simple_mean` used to be the example here, because a worker holds no embedding
    table. It is now supported: the worker ships a bag partition and the main process
    does one `embedding_bag`. So the refusal that remains is for names that mean
    nothing at all.
    """
    with pytest.raises(ValueError, match="latent_init"):
        EncoderPrepDataset(
            _rows(2, seed=0),
            tokenizer,
            UniformPatchingMethod(compression_ratio=4.0),
            seed=1,
            max_length=None,
            vocab_bound=100,
            latent_init="nonesuch",
        )


def test_pooled_inits_ship_a_bag_partition_instead_of_token_ids(tokenizer):
    """What replaced the refusal: a partition the main process can average."""
    for init, per_slot in (("step_mean", False), ("simple_mean", True)):
        prep = EncoderPrepDataset(
            _rows(2, seed=0),
            tokenizer,
            UniformPatchingMethod(compression_ratio=4.0),
            seed=1,
            max_length=None,
            vocab_bound=100,
            latent_init=init,
        )
        sample = prep[0]
        assert sample is not None
        assert sample.init_ids == [], "pooled inits pick no single token"
        # Bags tile the trace, so the flat list is the CoT itself -- which is what
        # keeps this linear rather than quadratic in a step's latent count.
        assert len(sample.init_bag_ids) == len(sample.context_ids) - (
            sample.context_ids.index(sample.init_bag_ids[0])
        )
        assert sample.init_bag_offsets[0] == 0
        assert sample.init_bag_offsets == sorted(set(sample.init_bag_offsets))
        # One bag per latent for `simple_mean`; one per step for `step_mean`. Under
        # uniform patching those coincide, since one span is one step.
        assert len(sample.init_bag_offsets) == sample.num_slots or not per_slot


def test_prep_dataset_demands_the_signal_cache_when_needed(tokenizer):
    """A missing cache must fail at construction, not silently patch uniformly."""
    for patching, init in (
        (_signal_patching(), "random"),
        (UniformPatchingMethod(compression_ratio=4.0), "surprisal_t0"),
    ):
        with pytest.raises(ValueError, match="signal"):
            EncoderPrepDataset(
                _rows(2, seed=0),
                tokenizer,
                patching,
                seed=1,
                max_length=None,
                vocab_bound=100,
                latent_init=init,
                signals=None,
            )


def test_collate_pads_without_letting_padding_speak(tokenizer):
    rows = _rows(3, seed=5)
    prep = EncoderPrepDataset(
        rows,
        tokenizer,
        UniformPatchingMethod(compression_ratio=4.0),
        seed=7,
        max_length=None,
        vocab_bound=1000,
        latent_init="random",
    )
    samples = [prep[i] for i in range(3)]
    assert all(s is not None for s in samples)
    batch = collate_encoder_batch(samples, pad_token_id=0)  # type: ignore[arg-type]

    for row, sample in enumerate(samples):
        assert sample is not None
        assert int(batch["slot_mask"][row].sum()) == sample.num_slots
        # Padded slots must still see memory position 0, or their attention row
        # is fully masked and can produce NaN.
        assert (batch["cross_limit"][row] >= 1).all()
    assert batch["labels"].shape == batch["input_ids"].shape


def test_plan_columns_makes_denominator_the_answer_token_count(tokenizer):
    measured = _measured(_rows(6, seed=9), tokenizer)
    lengths, starts = plan_columns(measured)
    answers = np.asarray(measured["answer_length"], dtype=np.int64)
    assert np.array_equal(lengths - starts, answers)
    prompt = np.asarray(measured["prompt_length"], dtype=np.int64)
    cot = np.asarray(measured["cot_length"], dtype=np.int64)
    assert np.array_equal(lengths, prompt + cot)


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_training_runs_and_publishes_a_resumable_checkpoint(patched, tmp_path):
    train_encoder(_cfg(tmp_path))

    root = tmp_path / "enc" / "checkpoints"
    pointer = (root / "LATEST").read_text().strip()
    assert (root / pointer / "training_state.pt").exists()
    assert (root / pointer / ENCODER_WEIGHTS).exists()
    assert (root / pointer / ENCODER_CONFIG).exists()
    assert (root / "best" / ENCODER_WEIGHTS).exists()
    # best/ is eval-only and must not carry optimizer state.
    assert not (root / "best" / "training_state.pt").exists()

    # The checkpoint must hold the ENCODER only -- not 0.6B of frozen decoder.
    weights = torch.load(root / pointer / ENCODER_WEIGHTS, weights_only=True)
    assert any(key.startswith("quantizer.codebook") for key in weights)
    assert not any("lm_head" in key or "embed_tokens" in key for key in weights)


def test_training_actually_moves_the_encoder(patched, tmp_path):
    """A run that leaves the weights untouched would pass every other check."""
    train_encoder(_cfg(tmp_path))
    root = tmp_path / "enc" / "checkpoints"
    pointer = (root / "LATEST").read_text().strip()
    trained = torch.load(root / pointer / ENCODER_WEIGHTS, weights_only=True)

    config = EncoderConfig(
        **json.loads((root / pointer / ENCODER_CONFIG).read_text(encoding="utf-8"))
    )
    fresh = CoTEncoder(config).state_dict()
    # `trained` comes off whatever device the run used; compare on the CPU copy.
    moved = [k for k in trained if not torch.allclose(trained[k].cpu(), fresh[k].cpu())]
    assert moved, "no encoder parameter changed during training"


def test_architecture_guard_refuses_a_mismatched_checkpoint(tmp_path):
    config = EncoderConfig(d_llm=32, n_blocks=1, n_heads=4, codebook_size=8)
    SaveableEncoder(CoTEncoder(config)).save_pretrained(tmp_path)

    check_encoder_architecture(config, tmp_path)  # same config: silent

    for field, value in (
        ("d_llm", 64),
        ("codebook_size", 16),
        ("latent_init", "simple_mean"),
    ):
        other = EncoderConfig(**{**config.__dict__, field: value})
        with pytest.raises(ValueError, match="different encoder"):
            check_encoder_architecture(other, tmp_path)


def test_saved_encoder_round_trips(tmp_path):
    config = EncoderConfig(d_llm=32, n_blocks=1, n_heads=4, codebook_size=8)
    encoder = CoTEncoder(config)
    SaveableEncoder(encoder).save_pretrained(tmp_path)
    restored = load_encoder(tmp_path, torch.device("cpu"))
    assert restored.config == config
    for key, value in encoder.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


# --------------------------------------------------------------------------- #
# Eval integration
# --------------------------------------------------------------------------- #


def test_learned_method_scores_through_the_eval_harness(tokenizer):
    """A trained encoder must be scorable as an ordinary CompressionMethod.

    This is what makes Phase 1 possible: the learned method lands in the same
    summary.json, under the same join key, on the same rows as base / no_cot /
    simple_mean. A parallel eval path would produce numbers that could not be
    compared against the training-free results already measured.
    """
    from cot_compression.data.answers import (
        cot_token_ids,
        extract_answer_trace,
        prefix_token_ids,
    )
    from cot_compression.encoder.frozen import build_latent_init
    from cot_compression.encoder.method import LearnedCompressionMethod

    decoder = _tiny_decoder().eval()
    d_llm = decoder.config.hidden_size
    encoder = CoTEncoder(
        EncoderConfig(d_llm=d_llm, n_blocks=1, n_heads=4, ffn_mult=2, codebook_size=8)
    ).eval()
    patching = UniformPatchingMethod(compression_ratio=4.0)
    method = LearnedCompressionMethod(
        encoder=encoder,
        patching=patching,
        latent_init=build_latent_init("random", patching),
    )

    assert method.requires_prefix()
    assert method.method_family == "learned"
    # The join key is a data contract shared with every plot and report.
    assert method.name == "learned_enc_L1k8_uniform_cr4"

    trace = extract_answer_trace(_rows(1, seed=11)[0]["messages"])
    assert trace is not None
    cot_ids = cot_token_ids(trace, tokenizer)
    prefix = prefix_token_ids(trace, tokenizer)
    device = torch.device("cpu")

    plan = method.plan(len(cot_ids), 0, 1337, None, device)
    slots = method.materialize(
        cot_ids, 0, 1337, tokenizer, decoder, device, None, prefix
    )
    assert slots is not None
    # plan() runs in a worker and materialize() on the device; a disagreement is
    # exactly what the eval loop drops samples for, so it must not happen here.
    assert slots.shape == (plan.num_slots, d_llm)
    assert torch.isfinite(slots).all()

    # Every spliced slot must BE a codebook row -- the representation is discrete,
    # not merely quantizer-adjacent. This is what the bitwise-exact
    # straight-through formulation buys.
    codebook = encoder.quantizer.codebook.to(slots.dtype)
    for row in slots:
        assert (codebook == row).all(dim=1).any(), "a slot is not a codebook entry"


def test_learned_method_demands_the_prompt(tokenizer):
    """requires_prefix() is a contract; failing loudly beats cross-attending to
    a silently empty prompt."""
    from cot_compression.encoder.frozen import build_latent_init
    from cot_compression.encoder.method import LearnedCompressionMethod

    patching = UniformPatchingMethod(compression_ratio=4.0)
    method = LearnedCompressionMethod(
        encoder=CoTEncoder(EncoderConfig(d_llm=32, n_blocks=1, n_heads=4)),
        patching=patching,
        latent_init=build_latent_init("random", patching),
    )
    with pytest.raises(ValueError, match="needs prefix_ids"):
        method.materialize(
            [1, 2, 3, 4], 0, 1337, tokenizer, None, torch.device("cpu"), None
        )


def test_codebook_usage_updates_during_training_only(patched, tmp_path):
    """`usage` drives perplexity and dead-code detection; unwired it reads zero.

    Also asserts the training/eval split: folding eval batches into the statistic
    would make utilization describe a mixture of two data distributions.
    """
    from cot_compression.encoder.frozen import FrozenBackbone
    from cot_compression.encoder.training import EncoderTrainingModule

    decoder = _tiny_decoder()
    backbone = FrozenBackbone(decoder, ce_chunk_tokens=32)
    encoder = CoTEncoder(
        EncoderConfig(
            d_llm=decoder.config.hidden_size,
            n_blocks=1,
            n_heads=4,
            ffn_mult=2,
            codebook_size=8,
            usage_decay=0.0,
        )
    )
    module = EncoderTrainingModule(encoder, backbone, commit_weight=0.25)

    length, slots = 12, 3
    batch = {
        "context_ids": torch.randint(0, 100, (1, 10)),
        "context_mask": torch.ones(1, 10, dtype=torch.bool),
        "input_ids": torch.randint(0, 100, (1, length)),
        "attention_mask": torch.ones(1, length, dtype=torch.long),
        "labels": torch.full((1, length), -100),
        "slot_positions": torch.tensor([[1, 2, 3]]),
        "slot_mask": torch.ones(1, slots, dtype=torch.bool),
        "cross_limit": torch.tensor([[4, 7, 10]]),
        "init_ids": torch.randint(0, 100, (1, slots)),
    }
    batch["labels"][:, -2:] = batch["input_ids"][:, -2:]

    # Zeros before initialization, as EnCodec's `cluster_size`. Nothing should
    # read it until `init_codebook_from_encoder_outputs` fills it from the k-means
    # bin counts, which is why this is not a live statistic here.
    assert float(encoder.quantizer.usage.sum()) == 0.0

    module.train()
    module(**batch)
    assert module.last_code_counts is not None
    assert int(module.last_code_counts.sum()) == slots
    # decay=0.0 makes the EMA the raw counts, so this is exact, not just non-zero.
    assert float(encoder.quantizer.usage.sum()) == float(slots)
    # With every slot assigned, perplexity must be a real value in [1, |C|].
    assert 1.0 <= float(encoder.quantizer.perplexity()) <= 8.0

    before = encoder.quantizer.usage.clone()
    module.eval()
    with torch.no_grad():
        module(**batch)
    assert torch.equal(encoder.quantizer.usage, before), "eval must not move usage"


def test_plan_epoch_denominator_survives_negative_proxies(tokenizer):
    """`plan_columns` returns a proxy that goes negative on real data.

    2.3% of the corpus answers at greater length than it reasons, so
    `length - answer_length` is negative there. That is fine only because
    `plan_epoch` consumes the array as a difference and never as an index -- this
    pins that contract, so a future change to `plan_epoch` fails here instead of
    silently mis-weighting the loss.
    """
    from cot_compression.training.sft import plan_epoch

    lengths = np.array([100, 40, 200, 60], dtype=np.int64)
    answers = np.array([10, 90, 20, 80], dtype=np.int64)  # rows 1 and 3 go negative
    starts = lengths - answers
    assert (starts < 0).any(), "fixture must exercise the negative case"

    cfg = OmegaConf.create(
        {
            "training": {
                "seed": 0,
                "max_length": 1000,
                "max_batch_tokens": 1000,
                "micro_batch_max_sequences": 2,
                "length_group_size": 4,
                "target_global_batch": 2,
                "max_train_examples": None,
            }
        }
    )
    plan = plan_epoch(lengths, starts, cfg, epoch=0, world_size=1)
    assert sum(plan.denom) == int(answers.sum())
    for (start, stop), denom in zip(plan.steps, plan.denom, strict=True):
        rows = [i for micro in plan.micro[start:stop] for i in micro]
        assert denom == int(answers[rows].sum())


def test_eval_report_overlays_the_model_on_every_reference():
    """Each eval emits the model's losses and every reference, once, under one name.

    The references are constants logged at every eval so that one chart can overlay
    them on the answer CE; nothing else is emitted about them -- no gap curves (with
    constant references those are the same curve shifted) and no aliases.
    """
    from cot_compression.encoder.training import eval_report

    popped = {"loss": 9.0, "answer_ce_row": 3.0, "answer_ce_tok": 3.5}

    class _State:
        module = type("M", (), {"pop_totals": lambda _self, train: popped})()

    baselines = {
        "base": {"row": 2.0, "tok": 2.5},
        "no_cot": {"row": 5.0, "tok": 5.5},
        "surprisal_t0": {"row": 3.5, "tok": 4.0},
    }
    report = eval_report(baselines, _State(), 99.0)
    assert report == {
        "eval/loss": 9.0,
        "eval/answer_ce_row": 3.0,
        "eval/answer_ce_tok": 3.5,
        "eval/base_row": 2.0,
        "eval/base_tok": 2.5,
        "eval/no_cot_row": 5.0,
        "eval/no_cot_tok": 5.5,
        "eval/surprisal_t0_row": 3.5,
        "eval/surprisal_t0_tok": 4.0,
    }
    # The objective comes from the module's own accumulators, not from the value
    # `estimate_loss` hands over -- they agree by construction, so one must win.
    assert report["eval/loss"] == 9.0


def test_baselines_are_computed_on_the_eval_rows(patched, tmp_path, tokenizer):
    """Baselines must come from the same rows `estimate_loss` scores.

    Scored on a different slice they would not be comparable, and the gap would be
    a difference between two unrelated populations.
    """
    from cot_compression.encoder.frozen import FrozenBackbone
    from cot_compression.encoder.training import baseline_answer_losses

    backbone = FrozenBackbone(_tiny_decoder(), ce_chunk_tokens=64)
    rows = patched["val"]
    out = baseline_answer_losses(
        backbone, tokenizer, rows, [[0, 1], [2, 3]], torch.device("cpu")
    )
    # No controls were passed, so none is reported -- absent rather than zero.
    assert set(out) == {"base", "no_cot"}
    # Both normalizations for every name: the objective is per row, while the
    # per-token figure is what earlier runs reported.
    assert all(set(values) == {"row", "tok"} for values in out.values())
    assert out["base"]["row"] > 0 and out["no_cot"]["row"] > 0
    # An untrained tiny decoder has no reason to prefer either, so only finiteness
    # and positivity are asserted -- a real decoder is what makes base < no_cot.
    assert all(np.isfinite(v) for values in out.values() for v in values.values())


def _surprisal_cache(rows, tokenizer):
    """A deterministic stand-in for the precomputed per-CoT-token surprisal cache."""
    from cot_compression.data.answers import cot_token_ids, extract_answer_trace

    cache = {}
    for index in range(len(rows)):
        trace = extract_answer_trace(rows[index]["messages"])
        count = len(cot_token_ids(trace, tokenizer))
        generator = torch.Generator().manual_seed(1000 + index)
        cache[index] = torch.rand(count, generator=generator) * 5.0
    return cache


def test_surprisal_t0_control_is_t0_compression_of_each_span(tokenizer):
    """The control's slot vectors ARE `surprisal_weighted_mean` compression at T=0.

    So the encoder-path control and the training-free evaluation method are one
    method, not two that happen to share a name: same spans, same token, and the
    variance rescale is exactly a no-op for a one-hot weighting.
    """
    from cot_compression.compression import SignalWeightedMeanCompressionMethod
    from cot_compression.data.answers import cot_token_ids, extract_answer_trace

    rows = _rows(3, seed=4)
    cache = _surprisal_cache(rows, tokenizer)
    patching = _signal_patching()
    prep = EncoderPrepDataset(
        rows,
        tokenizer,
        patching,
        seed=7,
        max_length=None,
        vocab_bound=1000,
        latent_init="surprisal_t0",
        signals=cache,
    )
    # fp32: `_tiny_decoder` loads in bf16, and `reduce_patches` returns the weights' dtype.
    table = _tiny_decoder().get_input_embeddings().weight.detach().float()
    method = SignalWeightedMeanCompressionMethod(
        patching, temperature=0.0, signal="surprisal"
    )
    for index in range(len(rows)):
        sample = prep[index]
        assert sample is not None
        cot = torch.tensor(
            cot_token_ids(extract_answer_trace(rows[index]["messages"]), tokenizer)
        )
        spans = patching.split(len(cot), index, 7, cache[index])
        assert len(spans) == sample.num_slots > 1
        for slot, (start, end) in enumerate(spans):
            compressed = method.reduce_patches(
                table[cot[start:end]].unsqueeze(0), cache[index][start:end].unsqueeze(0)
            )[0]
            assert torch.allclose(table[sample.init_ids[slot]], compressed)


def test_baselines_score_every_control_on_the_same_answer_tokens(patched, tokenizer):
    """Each control is reported under its name, and all must share one token set.

    A control that silently dropped a row would average over fewer answer tokens
    than `base`, making its gap a comparison between two populations -- so the
    counts are checked and a mismatch is an error, not a quiet bias.
    """
    from cot_compression.encoder.frozen import FrozenBackbone
    from cot_compression.encoder.training import baseline_answer_losses

    rows = patched["val"]
    cache = _surprisal_cache(rows, tokenizer)
    backbone = FrozenBackbone(_tiny_decoder(), ce_chunk_tokens=64)

    def control(latent_init):
        return EncoderPrepDataset(
            rows,
            tokenizer,
            _signal_patching(),
            seed=7,
            max_length=None,
            vocab_bound=1000,
            latent_init=latent_init,
            signals=cache,
        )

    controls = {"random": control("random"), "surprisal_t0": control("surprisal_t0")}
    batches = [[0, 1], [2, 3]]
    out = baseline_answer_losses(
        backbone, tokenizer, rows, batches, torch.device("cpu"), controls=controls
    )
    assert set(out) == {"base", "no_cot", "random", "surprisal_t0"}
    assert all(
        np.isfinite(v) and v > 0 for values in out.values() for v in values.values()
    )

    class _DropsARow:
        def __init__(self, prep):
            self.prep, self.tokenizer = prep, prep.tokenizer
            # The seed is built for the prep's initializer, so the stub must say which.
            self.latent_init = prep.latent_init

        def __getitem__(self, index):
            return None if index == 1 else self.prep[index]

    with pytest.raises(RuntimeError, match="different answer tokens"):
        baseline_answer_losses(
            backbone,
            tokenizer,
            rows,
            batches,
            torch.device("cpu"),
            controls={"random": _DropsARow(controls["random"])},
        )


def test_signal_cache_from_the_wrong_split_is_refused(tokenizer, tmp_path):
    """A val cache handed to the train split must fail, not silently mis-attach.

    The cache is keyed by row index *within a split*, and every split starts at 0,
    so nothing about a cross-split mix-up is structurally invalid -- it just
    attaches the wrong surprisal to the first rows and drops the rest. This
    actually happened (precompute defaults to the validation split), and it would
    have corrupted two grid arms with no error anywhere.
    """
    from cot_compression.encoder.training import verify_signal_cache

    train = _measured(_rows(6, seed=0), tokenizer)
    other = _measured(_rows(3, seed=99), tokenizer)

    good = {i: torch.zeros(int(n)) for i, n in enumerate(train["cot_length"])}
    verify_signal_cache(good, train, tmp_path / "c.npz", "train")  # silent

    too_few = {i: torch.zeros(int(n)) for i, n in enumerate(other["cot_length"])}
    with pytest.raises(ValueError, match="different split"):
        verify_signal_cache(too_few, train, tmp_path / "c.npz", "train")

    # Right row count, wrong data: the length fingerprint is what catches it.
    wrong = {i: torch.zeros(int(n) + 1) for i, n in enumerate(train["cot_length"])}
    with pytest.raises(ValueError, match="disagrees with split"):
        verify_signal_cache(wrong, train, tmp_path / "c.npz", "train")


def test_resume_reproduces_an_uninterrupted_run(patched, tmp_path):
    """Resume must be a no-op on the *trajectory*, not merely restart cleanly.

    A checkpoint that reloads weights but not the optimizer moments or the
    scheduler position still runs, converges, and looks fine on the loss curve --
    it just walks a different path than the run it claims to continue. The only
    check that catches that is training the same steps twice and comparing.
    """
    straight, split = tmp_path / "a", tmp_path / "b"
    train_encoder(_cfg(straight, max_steps=4))

    train_encoder(_cfg(split, max_steps=2))
    root = split / "enc" / "checkpoints"
    pointer = root / (root / "LATEST").read_text().strip()
    state = torch.load(pointer / "training_state.pt", weights_only=False)
    assert state["global_step"] == 2
    assert state["optimizer"]["state"], "no optimizer moments in the checkpoint"
    assert state["scheduler"]["last_epoch"] == 2

    train_encoder(_cfg(split, max_steps=4))

    def _weights(run_dir):
        checkpoints = run_dir / "enc" / "checkpoints"
        name = (checkpoints / "LATEST").read_text().strip()
        return torch.load(checkpoints / name / ENCODER_WEIGHTS, weights_only=True)

    left, right = _weights(straight), _weights(split)
    assert left.keys() == right.keys()
    for key in left:
        a, b = left[key].cpu(), right[key].cpu()
        if torch.cuda.is_available():
            # CUDA reductions use atomics, so re-running the same steps is not
            # bitwise reproducible. A tight tolerance still catches the failure this
            # test exists for -- a resume that walks a DIFFERENT trajectory because
            # the optimizer moments or the scheduler position were not restored --
            # which moves weights by orders of magnitude more than atomics do.
            assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), f"{key} diverged"
        else:
            assert torch.equal(a, b), f"{key} diverged across resume"
    # The usage EMA is a buffer, so codebook health survives the seam too.
    assert "quantizer.usage" in left


def test_wandb_id_is_stable_across_a_resume(tmp_path):
    """A requeue must append to the same W&B run, not open a second one."""
    from cot_compression.training.logging import _wandb_id

    cfg = _cfg(tmp_path)
    cfg.run_name = "enc_c64_lr0.0003"
    assert _wandb_id(cfg) == _wandb_id(cfg.copy()) == "enc_c64_lr0.0003"


def test_long_run_names_keep_distinct_wandb_ids(tmp_path):
    """Two runs differing only past character 63 must not share a W&B run.

    Plain truncation mapped every encoder arm that differed only in its masks,
    position encoding or lr onto one id, and `resume="allow"` would then append one
    arm's history to another's. Short names keep the id they always had.
    """
    from cot_compression.training.logging import _wandb_id

    cfg = _cfg(tmp_path)
    stem = "rowloss-rope_cr4.0_c1024_surprisal_threshold_surprisal_t0_causal-causal"
    ids = set()
    for tail in (
        "_perope_np1.0_lr0.0003",
        "_pernone_np1.0_lr0.0003",
        "_perope_np0.0_lr0.0003",
    ):
        cfg.run_name = stem + tail
        run_id = _wandb_id(cfg)
        assert len(run_id) <= 64, run_id
        ids.add(run_id)
    assert len(ids) == 3, ids
    cfg.run_name = "enc_c64_lr0.0003"
    assert _wandb_id(cfg) == "enc_c64_lr0.0003"


def test_every_loss_is_reported_once_and_separably():
    """Each term separable, the CE terms in both normalizations, nothing twice.

    Reporting only a total makes a run that is merely shrinking its commitment loss
    look like one that is learning to answer, which is the failure this split
    exists to expose. The CE terms need both normalizations: `*_row` is what is
    optimized, `*_tok` is what compares with earlier runs.
    """
    import cot_compression.encoder.training as mod

    encoder = CoTEncoder(
        EncoderConfig(d_llm=32, n_blocks=1, n_heads=4, codebook_size=8)
    )
    module = mod.EncoderTrainingModule.__new__(mod.EncoderTrainingModule)
    torch.nn.Module.__init__(module)
    module.encoder = encoder
    module.train_totals = None
    module.eval_totals = None
    module.commit_weight = 0.25
    module.next_patch_weight = 2.0
    module.train()

    tensor = torch.tensor
    module._accumulate(
        {
            "answer_row": tensor(3.0),  # 2 rows, so 1.5 per row
            "answer_tok": tensor(7.0),  # 4 answer tokens, so 1.75 per token
            "answer_tokens": tensor(4.0),
            "next_patch_row": tensor(5.0),  # 2 rows with a patch
            "next_patch_tok": tensor(9.0),  # 6 aux tokens
            "aux_tokens": tensor(6.0),
            "codebook_row": tensor(1.0),
            "commit_row": tensor(2.0),
            "rows": tensor(2.0),
            "aux_rows": tensor(2.0),
        }
    )
    popped = module.pop_totals(train=True)
    assert set(popped) == {
        "loss",
        "answer_ce_row",
        "answer_ce_tok",
        "next_patch_ce_row",
        "next_patch_ce_tok",
        "codebook_row",
        "commit_row",
    }
    assert popped["answer_ce_row"] == pytest.approx(1.5)
    assert popped["answer_ce_tok"] == pytest.approx(7.0 / 4.0)
    # Each term keeps its own denominator: aux per row-with-a-patch and per aux
    # token, the quantizer per row.
    assert popped["next_patch_ce_row"] == pytest.approx(2.5)
    assert popped["next_patch_ce_tok"] == pytest.approx(1.5)
    assert popped["codebook_row"] == pytest.approx(0.5)
    assert popped["commit_row"] == pytest.approx(1.0)
    # `loss` is the objective, and the quantizer's share is recoverable from it.
    assert popped["loss"] == pytest.approx(1.5 + 0.5 + 0.25 * 1.0 + 2.0 * 2.5)
    # Popping clears, so the next window cannot double-count.
    assert module.pop_totals(train=True) == {}
    # Train and eval accumulate separately, so an eval never dilutes a train window.
    assert module.pop_totals(train=False) == {}


def test_entropy_and_surprisal_t0_share_one_code_path():
    """`entropy_t0` must be the same initializer, only reading a different signal.

    A separate implementation is how the two silently drift apart; this pins that
    the only difference is which cached signal is consumed.
    """
    from cot_compression.encoder.frozen import build_latent_init
    from cot_compression.encoder.model import LATENT_INITS
    from cot_compression.patching import UniformPatchingMethod

    patching = UniformPatchingMethod(compression_ratio=2.0)
    for name in ("entropy_t0", "surprisal_t0"):
        assert name in LATENT_INITS
        method = build_latent_init(name, patching)
        assert method.temperature == 0.0
        assert method.signal == name.removesuffix("_t0")

    # And the signal the loader must fetch follows the init, not a hardcoded name.
    for name, expected in (("entropy_t0", "entropy"), ("surprisal_t0", "surprisal")):
        assert name.removesuffix("_t0") == expected


def test_dead_codes_metric_reports_the_count_before_reseeding(patched, tmp_path):
    """`vq/dead_codes` must report the count `reseed_dead_codes` acted on.

    `post_step` runs before the logging block, so a metric that re-queries
    `dead_codes()` describes the state *after* replacement rather than the dead set
    that triggered it. Under the old reset-to-threshold rule that read exactly 0
    forever, which hid two full codebook collapses across 1000 steps of both
    production arms. EnCodec's no-reset rule happens to make the two agree today;
    reporting the acted-on count keeps the metric correct either way.
    """
    from cot_compression.encoder.training import codebook_metrics, reseed_dead_codes

    cfg = _cfg(tmp_path)
    cfg.encoder.reseed_dead_codes = True
    encoder = CoTEncoder(
        EncoderConfig(d_llm=32, n_blocks=1, n_heads=4, codebook_size=8)
    )
    module = torch.nn.Module()
    module.encoder = encoder
    module.code_pool = torch.randn(16, 32)
    module.last_dead_count = 0
    module.last_quant_rel_error = None
    module.last_code_counts = None
    # This stub enumerates exactly what `codebook_metrics` reads, so a new metric
    # must be added here too rather than made optional with getattr in production --
    # a defensive read there would hide a genuinely missing attribute.
    module.last_within_step_cos = None
    module.last_across_step_cos = None
    module.last_within_step_pairs = None
    module.pop_totals = lambda train: {}

    encoder.quantizer.usage[:] = 100.0
    encoder.quantizer.usage[:3] = 0.0  # three genuinely dead

    state = SimpleNamespace(
        cfg=cfg,
        module=module,
        global_step=5,
        world_size=1,
        optimizer=torch.optim.AdamW(encoder.parameters()),
        logger=SimpleNamespace(logger=SimpleNamespace(debug=lambda *_: None)),
    )
    reseed_dead_codes(state)
    assert encoder.quantizer.reseeds == 3

    # The metric must report what was reseeded, independently of what `usage`
    # happens to say afterwards. Simulate the replaced codes immediately earning
    # traffic, so a re-query would read 0, and check the metric still says 3.
    encoder.quantizer.usage[:] = 100.0
    assert encoder.quantizer.dead_codes().numel() == 0
    assert codebook_metrics(state)["vq/dead_codes_frac"] == 3 / 8


def test_grad_norm_is_captured_and_logged(patched, tmp_path):
    """`clip_grad_norm_` already computes it; discarding it left both NaNs blind."""
    records: list[dict] = []
    import cot_compression.training.sft_loop as loop

    original = loop.RunLogger.log_metrics

    def spy(self, metrics, step):
        records.append(dict(metrics))
        return original(self, metrics, step)

    loop.RunLogger.log_metrics = spy
    try:
        train_encoder(_cfg(tmp_path, max_steps=2, log_interval=1))
    finally:
        loop.RunLogger.log_metrics = original

    train_records = [r for r in records if "train/grad_norm" in r]
    assert train_records, "train/grad_norm was never logged"
    assert all(r["train/grad_norm"] > 0.0 for r in train_records)
    assert all(math.isfinite(r["train/grad_norm"]) for r in train_records)


def test_resume_honours_config_tunables_not_the_checkpoint_manifest(tmp_path):
    """Tunables must follow the command line, architecture must follow the guard.

    The manifest stores both. Rebuilding from it wholesale pins a resumed run to
    whatever `dead_code_threshold`, `usage_decay` or `commit_weight` were when the
    checkpoint was written, silently ignoring the config -- and every one of those
    is a knob you would want to change mid-campaign.
    """
    written = EncoderConfig(
        d_llm=32, n_blocks=1, n_heads=4, codebook_size=8, dead_code_threshold=1.0
    )
    SaveableEncoder(CoTEncoder(written)).save_pretrained(tmp_path)

    wanted = EncoderConfig(**{**written.__dict__, "dead_code_threshold": 2.0})
    check_encoder_architecture(wanted, tmp_path)  # architecture agrees: silent

    from_manifest = load_encoder(tmp_path, torch.device("cpu"))
    assert from_manifest.config.dead_code_threshold == 1.0, "inference path"

    from_config = load_encoder(tmp_path, torch.device("cpu"), config=wanted)
    assert from_config.config.dead_code_threshold == 2.0, "training path"


# --------------------------------------------------------------------------- #
# Auxiliary next-patch supervision
# --------------------------------------------------------------------------- #


def _aux_fixture():
    """A tiny, fully hand-checkable next-patch batch.

    prefix = [p0, p1, z0, z1, z2] (prefix_len 5, codes at 2/3/4);
    patches: 0 = never a target, 1 = two tokens, 2 = one token.
    """
    from cot_compression.encoder.next_patch import build_next_patch_batch

    return build_next_patch_batch(
        slot_positions=torch.tensor([[2, 3, 4]]),
        aux_ids=torch.tensor([[12, 13, 14]]),
        aux_patch=torch.tensor([[1, 1, 2]]),
        prefix_len=torch.tensor([5]),
    )


# Four rows chosen so no two share a layout: different prefix lengths (the old
# token-indexed geometry was wrong for every row shorter than the longest), a row
# with only two codes, a row whose patch 2 was subsampled away (patch 3 must still
# read z2), and a row with nothing supervised.
_ROWS_SLOT_POSITIONS = [[2, 3, 4], [1, 2, 0], [3, 4, 5], [1, 2, 3]]
_ROWS_PREFIX_LEN = [5, 3, 6, 4]
_ROWS_AUX_PATCH = [[1, 1, 2, -1], [1, 1, -1, -1], [1, 3, 3, 3], [-1, -1, -1, -1]]
_ROWS_AUX_IDS = [[12, 13, 14, -100], [21, 22, -100, -100], [31, 32, 33, 34], [-100] * 4]


def _rows_batch(width=None):
    from cot_compression.encoder.next_patch import build_next_patch_batch

    return build_next_patch_batch(
        slot_positions=torch.tensor(_ROWS_SLOT_POSITIONS),
        aux_ids=torch.tensor(_ROWS_AUX_IDS),
        aux_patch=torch.tensor(_ROWS_AUX_PATCH),
        prefix_len=torch.tensor(_ROWS_PREFIX_LEN),
        width=width,
    )


def test_next_patch_pairs_each_target_with_its_conditioning_code():
    """A patch's FIRST token is predicted by the code before it, not by position-1.

    This is the whole reason the aux path gathers rather than reusing the
    one-position shift in `chunked_ce_from_hidden`.
    """
    plan = _aux_fixture()
    assert plan.width == 8
    # Column-indexed: patch 1 (cols 5, 6) reads z0 at 2; patch 2 (col 7) reads z1.
    assert plan.code_limit.tolist() == [[-1, -1, -1, -1, -1, 2, 2, 3]]
    # token 12 from z0; token 13 from token 12; token 14 from z1.
    assert plan.predictor.tolist() == [2, 5, 3]
    assert plan.targets.tolist() == [12, 13, 14]


def test_aux_mask_forbids_cross_patch_attention():
    """The property the whole design exists for.

    If a patch could see other patches' raw tokens, those are strictly more
    informative than their own lossy codes, the codes become redundant, and the
    encoder's gradient collapses. Also asserts no row is fully masked -- a NaN there
    would poison the batch even where the loss ignores it.
    """
    from cot_compression.encoder.next_patch import dense_next_patch_mask

    mask = dense_next_patch_mask(_aux_fixture())[0, 0]
    # positions: 0,1 prompt | 2,3,4 codes z0..z2 | 5,6 patch 1 | 7 patch 2
    assert mask[5, 2] and not mask[5, 3] and not mask[5, 4], "patch 1 sees z0 only"
    assert mask[7, 2] and mask[7, 3] and not mask[7, 4], "patch 2 sees z0,z1 not z2"
    assert not mask[7, 5] and not mask[7, 6], "patch 2 must not see patch 1"
    assert mask[6, 5], "within a patch, attention is causal"
    assert not mask[5, 6], "and does not run backwards"
    assert bool(mask.any(dim=-1).all()), "no fully masked row"


def test_short_prefix_rows_are_laid_out_after_their_own_prefix():
    """Each row's aux block starts at ITS prefix, in the mask AND the RoPE positions.

    The regression this pins: the token-indexed geometry built both as if every aux
    block began at `max(prefix_len)`, which was silently wrong for every shorter row
    of a length-grouped batch. Padding columns must attend only to themselves and be
    read by nobody, at any padded width.
    """
    from cot_compression.encoder.next_patch import dense_next_patch_mask

    plan = _rows_batch(width=14)
    mask = dense_next_patch_mask(plan)[:, 0]

    # Row 1: prefix [p0, z0, z1]; patch 1 at columns 3, 4 reads z0 only.
    assert mask[1, 3, 0] and mask[1, 3, 1] and not mask[1, 3, 2]
    assert mask[1, 4, 3], "patch 1 token 2 reads patch 1 token 1"
    assert plan.position_ids[1, 3:5].tolist() == [2, 3], "numbered right after z0"

    # Row 2: prefix [p0,p1,p2,z0,z1,z2]; patch 1 at col 6, patch 3 at cols 7-9.
    assert mask[2, 6, 3] and not mask[2, 6, 4], "patch 1 reads z0, not z1"
    assert mask[2, 7, 5] and not mask[2, 7, 6], "patch 3 reads z2, not patch 1"
    assert mask[2, 9, 7] and mask[2, 9, 8], "causal within patch 3"
    assert plan.position_ids[2, 6:10].tolist() == [4, 6, 7, 8]

    padding = ~plan.is_prefix & (plan.patch < 0)
    eye = torch.eye(plan.width, dtype=torch.bool).expand_as(mask)
    assert not bool((mask & padding[:, None, :] & ~eye).any()), "nobody reads padding"
    assert not bool((mask & padding[:, :, None] & ~eye).any()), "padding reads nobody"
    assert bool(mask.any(dim=-1).all()), "no fully masked row"


def test_width_helpers():
    from cot_compression.encoder.next_patch import bucket_width, natural_width

    assert (
        natural_width(torch.tensor(_ROWS_AUX_PATCH), torch.tensor(_ROWS_PREFIX_LEN))
        == 10
    ), "max over rows of prefix + aux count, not max(prefix) + max(aux)"
    assert bucket_width(10, None) == 10
    assert bucket_width(10, 4) == 12 and bucket_width(12, 4) == 12
    with pytest.raises(ValueError, match="narrower"):
        _rows_batch(width=9)


def _aux_training_module(width_multiple, latent_init="random"):
    from cot_compression.encoder.frozen import FrozenBackbone
    from cot_compression.encoder.training import EncoderTrainingModule

    # fp32: this compares against an oracle to 1e-5, beyond bf16's ~3 digits.
    decoder = _tiny_decoder().float()
    backbone = FrozenBackbone(decoder, ce_chunk_tokens=64, gradient_checkpointing=False)
    encoder = CoTEncoder(
        EncoderConfig(
            d_llm=decoder.config.hidden_size,
            n_blocks=1,
            n_heads=4,
            ffn_mult=2,
            codebook_size=8,
            latent_init=latent_init,
        )
    )
    return EncoderTrainingModule(
        encoder,
        backbone,
        commit_weight=0.25,
        next_patch_weight=1.0,
        aux_width_multiple=width_multiple,
    )


@pytest.mark.parametrize("width_multiple", [None, 7])
def test_next_patch_loss_matches_the_naive_per_patch_loop(width_multiple):
    """THE correctness test: the production aux pass == the specification, run literally.

    The oracle builds `[row b's prefix through z_{m-1}; patch m]` as its own sequence
    with an ordinary causal mask and default positions -- exactly "predict patch m
    from the codes before it" -- and reduces it the way the objective does, by
    literally averaging within each patch, then over a row's patches, then summing
    over rows. The single masked pass through `_next_patch_loss`, which gets there by
    weighting every target by `1/(P_i * T_ij)` instead, must agree in the loss AND in
    the gradient reaching `spliced`, which is the only path to the encoder.

    Multi-row with unequal prefixes, a subsampled patch and an empty row, because
    single-row and uniform batches are exactly where the old geometry bug hid. A width
    multiple of 7 forces padding (natural width 10 -> 14).
    """
    from cot_compression.encoder.probe import naive_next_patch_loss

    torch.manual_seed(0)
    module = _aux_training_module(width_multiple)
    spliced = (torch.randn(4, 8, module.backbone.hidden_size) * 0.03).requires_grad_()
    aux_ids = torch.tensor(_ROWS_AUX_IDS)
    aux_patch = torch.tensor(_ROWS_AUX_PATCH)
    slot_positions = torch.tensor(_ROWS_SLOT_POSITIONS)

    expected_pair = naive_next_patch_loss(
        module.backbone, spliced, aux_ids, aux_patch, slot_positions
    )
    actual_pair, count, rows_with_patches = module._next_patch_loss(
        spliced, aux_ids, aux_patch, torch.tensor(_ROWS_PREFIX_LEN), slot_positions
    )
    expected, actual = expected_pair[0], actual_pair[0]
    (got,) = torch.autograd.grad(actual, spliced, retain_graph=True)
    (want,) = torch.autograd.grad(expected, spliced, retain_graph=True)
    loss_error, grad_error = _relative_errors((actual, got), (expected, want))

    # The per-token twin rides along in the same pass and must be the plain sum.
    token_error = float(
        (actual_pair[1] - expected_pair[1]).abs() / expected_pair[1].abs()
    )
    assert token_error < 1e-6, token_error
    # One row of the fixture has no supervised patch, and it must not be counted:
    # its per-row loss does not exist.
    rows_with_any = int((aux_patch >= 0).any(dim=1).sum())
    assert int(rows_with_patches) == rows_with_any < aux_patch.shape[0]

    # Tolerances CALIBRATED, not guessed. In fp32 the correct path differs from the
    # oracle by 7e-8 (loss, relative) and 1.1e-5 (gradient, relative to its max) --
    # the oracle runs causal SDPA on shorter sequences, and Qwen3's RMSNorm and RoPE
    # compute in fp32 internally, so even fp64 floors at ~1e-7. Re-introducing the
    # old bug class (one short row's aux RoPE positions off by one) gives 2.9e-5 and
    # 8.2e-2. Both bounds sit >=14x above that noise and >=29x below that bug.
    assert count == int((aux_patch >= 0).sum())
    assert loss_error < 1e-6, loss_error
    assert grad_error < 1e-3, grad_error


def test_picklable_error_survives_cloudpickle_and_keeps_the_message():
    """A rank's real error must reach submitit's result dump instead of replacing it.

    On 2026-09-08 an InductorError holding weakref'd graph objects made that dump die
    with "cannot pickle 'weakref.ReferenceType' object", and the actual cause had to
    be recovered from a partial pickle.
    """
    import logging
    import weakref

    import cloudpickle

    from cot_compression.encoder.training import picklable_error

    class Holder:
        pass

    holder = Holder()
    try:
        error = RuntimeError("NoValidChoicesError: No choices to select")
        error.graph = weakref.ref(holder)  # type: ignore[attr-defined]
        raise error
    except RuntimeError as caught:
        with pytest.raises(TypeError, match="weakref"):
            cloudpickle.dumps(caught)
        wrapped = picklable_error(caught, 3, logging.getLogger("test"))
    restored = cloudpickle.loads(cloudpickle.dumps(wrapped))
    assert "rank 3" in str(restored) and "No choices to select" in str(restored)


def test_rank0_decision_is_the_local_value_without_a_process_group():
    from types import SimpleNamespace

    from cot_compression.training.sft_loop import _decided_on_rank0

    state = SimpleNamespace(world_size=1, device=torch.device("cpu"))
    assert _decided_on_rank0(state, True) is True  # type: ignore[arg-type]
    assert _decided_on_rank0(state, False) is False  # type: ignore[arg-type]


def test_next_patch_subsample_is_deterministic_and_a_noop_at_one(tokenizer):
    """Selection is keyed on seed + sample_index, like every other draw here."""
    rows = _rows(3, seed=11)
    patching = UniformPatchingMethod(compression_ratio=4.0)

    def build(rate):
        return EncoderPrepDataset(
            rows,
            tokenizer,
            patching,
            seed=7,
            max_length=None,
            vocab_bound=1000,
            latent_init="random",
            next_patch_subsample=rate,
        )

    full, half, off = build(1.0)[0], build(0.5)[0], build(0.0)[0]
    assert full is not None and half is not None and off is not None
    # 1.0 supervises every patch except patch 0, which has no preceding code.
    assert set(full.aux_patch) == set(range(1, full.num_slots))
    assert len(full.aux_ids) == len(full.aux_patch)
    # 0.5 is a strict subset, and 0.0 supervises nothing.
    assert set(half.aux_patch) <= set(full.aux_patch)
    assert len(half.aux_ids) < len(full.aux_ids)
    assert off.aux_ids == [] and off.aux_patch == []
    # Deterministic: same seed and index, same selection.
    assert build(0.5)[0].aux_patch == half.aux_patch
    # prefix_len is Pass B's own prefix, through the last code.
    assert full.prefix_len == full.slot_positions[-1] + 1


def test_training_runs_end_to_end_with_next_patch_supervision(patched, tmp_path):
    """The aux path, exercised for real: block mask, flex attention, backward.

    Everything else about the aux loss is unit-tested against the naive per-patch
    loop; this is the test that the whole thing survives contact with `train_encoder`
    -- collate widths, DDP-less rank handling, the attention-backend swap, and the
    gradient actually flowing.
    """
    cfg = _cfg(tmp_path, max_steps=2, log_interval=1)
    cfg.encoder.next_patch_weight = 1.0
    train_encoder(cfg)

    root = tmp_path / "enc" / "checkpoints"
    pointer = (root / "LATEST").read_text().strip()
    assert (root / pointer / ENCODER_WEIGHTS).exists()


def test_a_failing_rank_surfaces_its_own_error_in_a_picklable_form(patched, tmp_path):
    """The failure path, through `train_encoder` itself.

    An injected fault after step 1 must come out as a RuntimeError that names it and
    survives cloudpickle -- what submitit needs to record it -- instead of the
    original exception object, whose unpicklable payload hid the Sep 8 crash.
    """
    import cloudpickle

    cfg = _cfg(tmp_path, max_steps=3, log_interval=1)
    cfg.training.fault_injection = {"rank": 0, "step": 1}
    with pytest.raises(
        RuntimeError, match="injected fault on rank 0 after step 1"
    ) as info:
        train_encoder(cfg)
    assert "injected fault" in str(cloudpickle.loads(cloudpickle.dumps(info.value)))
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_next_patch_weight_zero_costs_nothing_and_changes_nothing(patched, tmp_path):
    """Disabled means disabled: no aux forward, and the objective is untouched."""
    from cot_compression.encoder.frozen import FrozenBackbone
    from cot_compression.encoder.training import EncoderTrainingModule

    decoder = _tiny_decoder()
    backbone = FrozenBackbone(decoder, ce_chunk_tokens=32)
    encoder = CoTEncoder(
        EncoderConfig(
            d_llm=decoder.config.hidden_size,
            n_blocks=1,
            n_heads=4,
            ffn_mult=2,
            codebook_size=8,
        )
    )
    module = EncoderTrainingModule(
        encoder, backbone, commit_weight=0.25, next_patch_weight=0.0
    )
    loss, tokens, rows = module._next_patch_loss(
        torch.zeros(1, 4, decoder.config.hidden_size),
        torch.tensor([[12]]),
        torch.tensor([[1]]),
        torch.tensor([3]),
        torch.tensor([[1, 2]]),
    )
    assert float(loss.sum()) == 0.0 and int(tokens) == 0 and int(rows) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs CUDA")
def test_flex_and_dense_agree_in_loss_and_gradient_across_shapes():
    """The bridge from the CPU oracle to the production CUDA path.

    FlexAttention has no CPU backward, so the oracle test runs the dense mask. Here the
    real CUDA path -- compiled flex through the routed attention, `create_block_mask`
    compiled, gradient checkpointing ON as in training -- must match the dense mask in
    loss and in the gradient to `spliced`, on the multi-row batch and at several
    widths and row counts.
    """
    from cot_compression.encoder.next_patch import build_next_patch_batch

    device = torch.device("cuda")
    torch.manual_seed(0)
    module = _aux_training_module(width_multiple=16)
    module.backbone.model.to(device)
    module.backbone.gradient_checkpointing = True
    module.backbone.model.gradient_checkpointing_enable({"use_reentrant": False})
    dim = module.backbone.hidden_size

    def batches():
        yield (
            torch.tensor(_ROWS_AUX_IDS),
            torch.tensor(_ROWS_AUX_PATCH),
            torch.tensor(_ROWS_PREFIX_LEN),
            torch.tensor(_ROWS_SLOT_POSITIONS),
            8,
        )
        for rows, codes, per_patch in [(1, 60, 5), (3, 40, 9), (6, 25, 3)]:
            generator = torch.Generator().manual_seed(rows)
            prefix = torch.randint(codes + 2, codes + 30, (rows,), generator=generator)
            slots = torch.stack([torch.arange(p - codes, p) for p in prefix.tolist()])
            patch = torch.arange(1, codes).repeat_interleave(per_patch)
            aux_patch = patch.expand(rows, -1).clone()
            aux_patch[0, -per_patch * 3 :] = -1  # unequal aux counts across rows
            aux_ids = torch.where(aux_patch >= 0, 100 + aux_patch, -100)
            yield aux_ids, aux_patch, prefix, slots, int(prefix.max())

    for aux_ids, aux_patch, prefix_len, slots, length in batches():
        aux_ids, aux_patch = aux_ids.to(device), aux_patch.to(device)
        prefix_len, slots = prefix_len.to(device), slots.to(device)
        spliced = (
            torch.randn(len(prefix_len), length, dim, device=device) * 0.03
        ).requires_grad_()

        flex, count = module._next_patch_loss(
            spliced, aux_ids, aux_patch, prefix_len, slots
        )
        (flex_grad,) = torch.autograd.grad(flex, spliced)

        plan = build_next_patch_batch(
            slot_positions=slots,
            aux_ids=aux_ids,
            aux_patch=aux_patch,
            prefix_len=prefix_len,
            width=flex_width(aux_patch, prefix_len),
        )

        # Self-calibrated. Flex tiles its softmax and SDPA's math backend does not, so
        # fp32 differences are expected; what matters is that they are far below the
        # error a real layout bug produces AT THIS SHAPE. The bug here is the one the
        # column geometry prevents: every aux token's RoPE position off by one.
        shifted = plan.position_ids.clone()
        shifted[plan.patch >= 0] += 1
        dense = _dense_aux_reference(module, spliced, plan, plan.position_ids)
        buggy = _dense_aux_reference(module, spliced, plan, shifted)
        ok = _relative_errors((flex, flex_grad), dense)
        bug = _relative_errors(buggy, dense)
        assert count == plan.num_targets > 0
        assert ok[0] * 10 < bug[0] and ok[1] * 10 < bug[1], (ok, bug)


def _dense_aux_reference(module, spliced, plan, position_ids):
    """The aux loss and its gradient to `spliced` through the dense mask."""
    from cot_compression.encoder.next_patch import (
        dense_additive_mask,
        dense_next_patch_mask,
    )

    loss = module.backbone.next_patch_ce(
        module._aux_embeds(spliced, plan),
        dense_additive_mask(dense_next_patch_mask(plan), spliced.dtype),
        plan.predictor,
        plan.targets,
        position_ids,
    )
    return loss, torch.autograd.grad(loss, spliced)[0]


def _relative_errors(candidate, reference):
    """(loss error relative to the loss, gradient error relative to its max)."""
    (loss, grad), (ref_loss, ref_grad) = candidate, reference
    loss_error = abs(float(loss.detach() - ref_loss.detach())) / abs(
        float(ref_loss.detach())
    )
    return loss_error, float((grad - ref_grad).abs().max() / ref_grad.abs().max())


def flex_width(aux_patch, prefix_len):
    from cot_compression.encoder.next_patch import bucket_width, natural_width

    return bucket_width(natural_width(aux_patch, prefix_len), 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs CUDA")
def test_new_data_at_a_seen_shape_does_not_recompile():
    """Every micro-batch builds a NEW `mask_mod` closure over new tensors. If Dynamo
    guarded on that closure's identity, each step would recompile until the limit --
    the failure mode that became an eager OOM on Sep 11. With the limit at 1 and the
    fatal flag set, a second batch of the same shape must run from the cache.
    """
    import torch._dynamo

    device = torch.device("cuda")
    module = _aux_training_module(width_multiple=64)
    module.backbone.model.to(device)
    dim = module.backbone.hidden_size
    torch._dynamo.reset()
    with torch._dynamo.config.patch(
        recompile_limit=1, fail_on_recompile_limit_hit=True
    ):
        for seed in (0, 1):
            generator = torch.Generator().manual_seed(seed)
            prefix_len = torch.randint(20, 30, (3,), generator=generator)
            slots = torch.stack([torch.arange(p - 10, p) for p in prefix_len.tolist()])
            aux_patch = torch.arange(1, 10).repeat_interleave(3).expand(3, -1).clone()
            aux_ids = torch.randint(100, 200, aux_patch.shape, generator=generator)
            spliced = (torch.randn(3, 30, dim, device=device) * 0.03).requires_grad_()
            loss, _ = module._next_patch_loss(
                spliced,
                aux_ids.to(device),
                aux_patch.to(device),
                prefix_len.to(device),
                slots.to(device),
            )
            loss.backward()


# --------------------------------------------------------------------------- #
# Per-row objective
# --------------------------------------------------------------------------- #


def _prepared_batch(tokenizer, rows, indices, step=False):
    """Collate real prepared rows, exactly as the training loader does."""
    from cot_compression.patching import ParagraphStepPatchingMethod

    prep = EncoderPrepDataset(
        rows,
        tokenizer,
        ParagraphStepPatchingMethod(compression_ratio=4.0)
        if step
        else UniformPatchingMethod(compression_ratio=4.0),
        seed=7,
        max_length=None,
        vocab_bound=1000,
        latent_init="step_mean" if step else "random",
    )
    samples = []
    for index in indices:
        sample = prep[index]
        assert sample is not None, index
        samples.append(sample)
    return collate_encoder_batch(samples, pad_token_id=int(tokenizer.pad_token_id))


@pytest.mark.parametrize("step", [False, True])
def test_objective_is_invariant_to_micro_batch_grouping(tokenizer, step):
    """THE property per-row normalization exists for.

    The same four rows must give the same objective and the same gradient whether
    they arrive as one micro-batch or as two, because every denominator is a global
    count for the step rather than a per-micro-batch one. Per-micro-batch means
    would fail here, and would silently weight rows by how the batcher happened to
    group them -- which changes with world_size, the token budget and the length
    distribution.

    Run over BOTH arms. On the step arm it is also the strongest check on
    `cond_slot` padding: two groupings pad it to different widths, so an
    out-of-range or mis-indexed pad shows up as a changed objective here rather
    than as a plausible number in a run.
    """
    from cot_compression.encoder.training import step_denominators

    torch.manual_seed(0)
    module = _aux_training_module(None, "step_mean" if step else "random")
    rows = _step_rows(4, seed=5) if step else _rows(4, seed=5)
    whole = _prepared_batch(tokenizer, rows, [0, 1, 2, 3], step=step)
    halves = [
        _prepared_batch(tokenizer, rows, pair, step=step) for pair in ([0, 1], [2, 3])
    ]
    if step:
        # The groupings must genuinely disagree on the padded widths, or this is
        # not testing what it claims.
        assert whole["cond_slot"].shape[1] != min(
            h["cond_slot"].shape[1] for h in halves
        ) or whole["slot_mask"].shape[1] != min(h["slot_mask"].shape[1] for h in halves)

    denominators = step_denominators(torch.device("cpu"), [whole])
    assert denominators.tolist() == [4.0, 4.0], "4 rows, all with supervised patches"
    assert (
        step_denominators(torch.device("cpu"), halves).tolist() == denominators.tolist()
    ), "the split must not change what the step divides by"

    def objective(batches):
        module.zero_grad(set_to_none=True)
        total = sum((module(**batch) / denominators).sum() for batch in batches)
        total.backward()
        grads = torch.cat(
            [
                p.grad.flatten()
                for p in module.encoder.parameters()
                if p.grad is not None
            ]
        )
        return float(total.detach()), grads

    one_loss, one_grad = objective([whole])
    two_loss, two_grad = objective(halves)
    assert one_loss == pytest.approx(two_loss, rel=1e-5)
    assert (
        float((one_grad - two_grad).abs().max())
        < 1e-5 * float(one_grad.abs().max()) + 1e-7
    )


def test_step_denominators_exclude_rows_without_a_supervised_patch(tokenizer):
    """A row with no patch to predict has no next-patch loss to average in."""
    from cot_compression.encoder.training import step_denominators

    batch = _prepared_batch(tokenizer, _rows(3, seed=6), [0, 1, 2])
    batch["aux_patch"] = batch["aux_patch"].clone()
    batch["aux_patch"][1] = -1  # row 1 loses every supervised patch
    counts = step_denominators(torch.device("cpu"), [batch])
    assert counts.tolist() == [3.0, 2.0]


def test_kmeans_pool_accumulates_across_micro_batches(tokenizer):
    """The seeding pool must not be whatever one shuffled micro-batch happens to hold.

    With the batch order shuffled, the first micro-batch is a random length rather
    than the longest rows, so a single one can carry far fewer slots than there are
    codes. The pass keeps consuming micro-batches until it has enough.
    """
    from cot_compression.encoder.training import kmeans_init_codebook

    torch.manual_seed(0)
    module = _aux_training_module(None)
    rows = _rows(6, seed=8)
    batches = [_prepared_batch(tokenizer, rows, [i]) for i in range(6)]
    messages: list[str] = []
    logger = SimpleNamespace(info=messages.append)

    kmeans_init_codebook(
        module,
        batches,
        torch.device("cpu"),
        iters=2,
        seed=3,
        world_size=1,
        logger=logger,
        max_batches=4,
    )
    assert "from 4 micro-batch(es)" in messages[0], messages
    # The codebook is seeded from those outputs, not left at its placeholder.
    codebook = module.encoder.quantizer.codebook.detach()
    assert torch.isfinite(codebook).all() and float(codebook.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# Paragraph-step segmentation
# --------------------------------------------------------------------------- #

# A step fixture whose conditioning slot is NOT `m - 1` anywhere it matters -- the
# whole difference between next-patch and next-step supervision. Five slots a row:
#   row 0  steps {0,1,2} {3,4}        cond_slot [_, 2]      (m-1 would say 0)
#   row 1  steps {0} {1,2} {3,4}      cond_slot [_, 0, 2]   (m-1 would say 1)
#   row 2  same, but step 1 subsampled away: step 2 must STILL condition on the last
#          latent of step 1, which is the case the run-keyed weighting can get wrong
#   row 3  a single-step row with nothing supervised -- 1.9% of real traces
_STEP_SLOT_POSITIONS = [
    [2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5],
    [3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5],
]
_STEP_PREFIX_LEN = [7, 6, 8, 6]
_STEP_COND_SLOT = [[0, 2, 0], [0, 0, 2], [0, 0, 2], [0, 0, 0]]
_STEP_AUX_PATCH = [
    [1, 1, -1, -1, -1],
    [1, 1, 2, -1, -1],
    [2, 2, 2, -1, -1],
    [-1, -1, -1, -1, -1],
]
_STEP_AUX_IDS = [
    [12, 13, -100, -100, -100],
    [21, 22, 23, -100, -100],
    [31, 32, 33, -100, -100],
    [-100] * 5,
]


def _step_rows(count: int, seed: int) -> Dataset:
    """Rollouts whose think block has real `\\n\\n` paragraphs.

    `_rows` has none, so under paragraph patching every one of its rows is a single
    step and the next-step loss would silently supervise nothing -- the tests would
    pass while measuring an empty objective. Kept separate so the existing arm's
    assertions stay byte-stable.
    """
    rng = np.random.default_rng(seed)
    records = []
    for index in range(count):
        paragraphs = [
            " ".join(f"step{int(v)}" for v in rng.integers(0, 50, size=int(size)))
            for size in rng.integers(4, 14, size=5)
        ]
        cot = "\n\n".join(paragraphs)
        records.append(
            {
                "messages": [
                    {"role": "user", "content": f"Question {index}? Reason it out."},
                    {
                        "role": "assistant",
                        "content": f"<think>\n{cot}\n</think>\n\nThe answer is {index}.",
                    },
                ],
                "domain": "math",
                "rollout_id": f"p{index:04d}:r0",
            }
        )
    return Dataset.from_list(records)


def _step_prep(rows, tokenizer, **overrides):
    from cot_compression.patching import ParagraphStepPatchingMethod

    kwargs = dict(
        seed=1337,
        max_length=None,
        vocab_bound=1000,
        latent_init="step_mean",
    )
    kwargs.update(overrides)
    return EncoderPrepDataset(
        rows, tokenizer, ParagraphStepPatchingMethod(compression_ratio=4.0), **kwargs
    )


def test_prep_emits_consistent_step_geometry(tokenizer):
    """Every invariant the device code indexes with, checked on a real tokenization."""
    rows = _step_rows(4, seed=5)
    prep = _step_prep(rows, tokenizer)
    for index in range(len(rows)):
        sample = prep[index]
        assert sample is not None
        assert len(sample.cross_limit) == sample.num_slots
        assert len(sample.step_of_latent) == sample.num_slots
        assert sample.query_limit is None, "the default anchor ships no query_limit"

        # cross_limit is constant WITHIN a step and strictly increases between them:
        # that is what "each latent sees its whole step, and nothing after it" means.
        by_step: dict[int, set[int]] = {}
        for step, limit in zip(sample.step_of_latent, sample.cross_limit, strict=True):
            by_step.setdefault(step, set()).add(limit)
        assert all(len(v) == 1 for v in by_step.values())
        limits = [next(iter(by_step[s])) for s in sorted(by_step)]
        assert limits == sorted(set(limits))

        # step_of_latent is non-decreasing, and cond_slot[m] is the last slot of m-1.
        assert sample.step_of_latent == sorted(sample.step_of_latent)
        num_steps = len(by_step)
        assert len(sample.cond_slot) == num_steps
        for step in range(1, num_steps):
            previous = [
                slot
                for slot, owner in enumerate(sample.step_of_latent)
                if owner == step - 1
            ]
            assert sample.cond_slot[step] == previous[-1]

        # Multi-step, or the fixture is not testing what it claims.
        assert num_steps > 1
        assert min(sample.aux_patch) >= 1, "step 0 is never a target"
        assert sample.aux_patch == sorted(sample.aux_patch)


def test_substep_anchor_is_distinct_and_within_the_step(tokenizer):
    rows = _step_rows(3, seed=6)
    sample = _step_prep(rows, tokenizer, query_anchor="substep")[0]
    assert sample is not None and sample.query_limit is not None
    assert len(sample.query_limit) == sample.num_slots
    # Strictly increasing overall, and never past what the latent may attend to --
    # the `query_limit <= cross_limit` guarantee, checked on the CPU so the device
    # never needs a syncing assert.
    assert all(
        a < b for a, b in zip(sample.query_limit, sample.query_limit[1:], strict=False)
    )
    assert all(
        q <= c for q, c in zip(sample.query_limit, sample.cross_limit, strict=True)
    )
    # A step's LAST latent anchors exactly at the step end.
    for step in set(sample.step_of_latent):
        slots = [i for i, s in enumerate(sample.step_of_latent) if s == step]
        assert sample.query_limit[slots[-1]] == sample.cross_limit[slots[-1]]


def test_uniform_patching_geometry_is_unchanged(tokenizer):
    """The degeneracy guarantee: token patching must produce its historical tensors.

    One step per span means `cross_limit` is `prompt_len + span_end` and `cond_slot`
    is `m - 1`, so the pre-step arm computes exactly what it did before -- which is
    what lets the existing suite serve as this change's regression test.
    """
    rows = _rows(3, seed=7)
    prep = EncoderPrepDataset(
        rows,
        tokenizer,
        UniformPatchingMethod(compression_ratio=4.0),
        seed=1337,
        max_length=None,
        vocab_bound=1000,
        latent_init="random",
    )
    for index in range(len(rows)):
        sample = prep[index]
        assert sample is not None
        assert sample.query_limit is None
        assert sample.step_of_latent == list(range(sample.num_slots))
        # One latent per step, so the conditioning slot for unit m is m-1 -- exactly
        # the `(aux_patch - 1)` gather this path used before `cond_slot` existed.
        # Entry 0 is a placeholder; unit 0 is never a target.
        assert sample.cond_slot == [0, *range(sample.num_slots - 1)]
        # One slot per step, so cross_limit is strictly increasing by the patch size.
        assert sample.cross_limit == sorted(set(sample.cross_limit))
        assert len(sample.cross_limit) == sample.num_slots


def test_collate_pads_the_step_fields(tokenizer):
    rows = _step_rows(3, seed=8)
    prep = _step_prep(rows, tokenizer)
    samples = [prep[i] for i in range(3)]
    batch = collate_encoder_batch([s for s in samples if s is not None], pad_token_id=0)
    slots = int(batch["slot_mask"].shape[1])
    assert batch["cond_slot"].shape[0] == batch["step_of_latent"].shape[0]
    assert batch["step_of_latent"].shape[1] == slots
    assert "query_limit" not in batch, "omitted, not None: run_training .to()s values"
    # Every padded gather index must still be in range -- the gather runs before the
    # mask, so an out-of-range pad is a crash, not a masked-out value.
    assert int(batch["cond_slot"].max()) < slots
    assert int(batch["cond_slot"].min()) >= 0
    # The pooled initializer's bags.
    assert int(batch["init_bag_of_slot"].max()) < int(batch["init_bag_offsets"].numel())
    assert batch["init_bag_offsets"][0] == 0


def test_cond_slot_defaults_to_the_previous_slot():
    """`cond_slot=None` must reproduce the pre-step gather exactly."""
    from cot_compression.encoder.next_patch import build_next_patch_batch

    common = dict(
        slot_positions=torch.tensor(_ROWS_SLOT_POSITIONS),
        aux_ids=torch.tensor(_ROWS_AUX_IDS),
        aux_patch=torch.tensor(_ROWS_AUX_PATCH),
        prefix_len=torch.tensor(_ROWS_PREFIX_LEN),
    )
    default = build_next_patch_batch(**common)
    units = max(max(row) for row in _ROWS_AUX_PATCH) + 1
    explicit = build_next_patch_batch(
        **common,
        cond_slot=torch.tensor([list(range(-1, units - 1))] * 4).clamp_min(0),
    )
    assert torch.equal(default.code_limit, explicit.code_limit)
    assert torch.equal(default.predictor, explicit.predictor)


@pytest.mark.parametrize("width_multiple", [None, 7])
def test_next_step_loss_matches_the_naive_per_step_loop(width_multiple):
    """THE correctness test for step supervision, against the specification run literally.

    Same oracle as the per-patch case, with one thing changed: a step is conditioned
    on the LAST latent of the step before it, not on slot `m - 1`. The fixture makes
    those differ in every supervised row, includes a row whose step 1 was subsampled
    away (step 2 must still reach back to step 1's last latent) and a single-step row
    with nothing supervised at all.
    """
    from cot_compression.encoder.probe import naive_next_patch_loss

    torch.manual_seed(0)
    module = _aux_training_module(width_multiple)
    spliced = (torch.randn(4, 9, module.backbone.hidden_size) * 0.03).requires_grad_()
    aux_ids = torch.tensor(_STEP_AUX_IDS)
    aux_patch = torch.tensor(_STEP_AUX_PATCH)
    slot_positions = torch.tensor(_STEP_SLOT_POSITIONS)
    cond_slot = torch.tensor(_STEP_COND_SLOT)

    # The fixture must actually exercise the difference, or this test is vacuous.
    assert any(
        cond_slot[r, m] != m - 1
        for r in range(4)
        for m in set(_STEP_AUX_PATCH[r]) - {-1}
    )

    expected_pair = naive_next_patch_loss(
        module.backbone, spliced, aux_ids, aux_patch, slot_positions, cond_slot
    )
    actual_pair, count, rows_with_steps = module._next_patch_loss(
        spliced,
        aux_ids,
        aux_patch,
        torch.tensor(_STEP_PREFIX_LEN),
        slot_positions,
        cond_slot,
    )
    expected, actual = expected_pair[0], actual_pair[0]
    (got,) = torch.autograd.grad(actual, spliced, retain_graph=True)
    (want,) = torch.autograd.grad(expected, spliced, retain_graph=True)
    loss_error, grad_error = _relative_errors((actual, got), (expected, want))

    token_error = float(
        (actual_pair[1] - expected_pair[1]).abs() / expected_pair[1].abs()
    )
    assert token_error < 1e-6, token_error
    rows_with_any = int((aux_patch >= 0).any(dim=1).sum())
    assert int(rows_with_steps) == rows_with_any < aux_patch.shape[0]
    assert count == int((aux_patch >= 0).sum())
    # Same calibrated tolerances as the per-patch oracle: the two paths differ only
    # in which slot they gather, so the floating-point noise floor is unchanged.
    assert loss_error < 1e-6, loss_error
    assert grad_error < 1e-3, grad_error


def test_next_step_weights_average_within_step_then_over_steps():
    """Spec item 6, asserted on the weights themselves rather than through a model."""
    from cot_compression.encoder.next_patch import patch_weights

    aux_patch = torch.tensor(_STEP_AUX_PATCH)
    weights, steps_per_row = patch_weights(aux_patch)
    assert steps_per_row.tolist() == [1, 2, 1, 0]

    # Row 1 has two steps: one of 2 tokens and one of 1, so 1/(2*2), 1/(2*2), 1/(1*2).
    offsets = torch.tensor([0, 2, 5, 8])
    row1 = weights[int(offsets[1]) : int(offsets[2])]
    assert torch.allclose(row1, torch.tensor([0.25, 0.25, 0.5]))
    # And every supervised row's weights sum to exactly 1.
    for start, end in ((0, 2), (2, 5), (5, 8)):
        assert abs(float(weights[start:end].sum()) - 1.0) < 1e-6


def test_step_mean_init_averages_the_step_embeddings(tokenizer):
    """The pooled initializer, against a literal per-step mean of input embeddings.

    Pooling per STEP and indexing is the whole trick: a step's latents share one
    vector, so the flat bag list stays the CoT length instead of the sum of each
    step's length times its latent count.
    """
    from cot_compression.encoder.frozen import FrozenBackbone
    from cot_compression.encoder.training import EncoderTrainingModule

    rows = _step_rows(2, seed=11)
    samples = [s for s in (_step_prep(rows, tokenizer)[i] for i in range(2)) if s]
    batch = collate_encoder_batch(samples, pad_token_id=0)

    decoder = _tiny_decoder().float()
    backbone = FrozenBackbone(decoder, ce_chunk_tokens=64, gradient_checkpointing=False)
    encoder = CoTEncoder(
        EncoderConfig(
            d_llm=decoder.config.hidden_size,
            n_blocks=1,
            n_heads=4,
            latent_init="step_mean",
        )
    )
    module = EncoderTrainingModule(encoder, backbone, commit_weight=0.25)
    latent = module._latent_init(
        batch["init_ids"],
        batch["init_bag_ids"],
        batch["init_bag_offsets"],
        batch["init_bag_of_slot"],
    )

    table = backbone.embedding_weight
    for row, sample in enumerate(samples):
        bounds = [*sample.init_bag_offsets, len(sample.init_bag_ids)]
        for slot, step in enumerate(sample.step_of_latent):
            ids = sample.init_bag_ids[bounds[step] : bounds[step + 1]]
            want = table[torch.tensor(ids)].float().mean(0)
            assert torch.allclose(latent[row, slot].float(), want, atol=1e-5)
        # Every latent of a step gets the SAME vector -- which is exactly why the
        # within-step cosine diagnostic exists.
        for step in set(sample.step_of_latent):
            slots = [i for i, s in enumerate(sample.step_of_latent) if s == step]
            for slot in slots[1:]:
                assert torch.equal(latent[row, slots[0]], latent[row, slot])


def test_within_step_cosine_reports_collapse_and_diversity():
    """The metric the whole symmetry question is decided on -- so it must not lie."""
    from cot_compression.encoder.frozen import FrozenBackbone
    from cot_compression.encoder.training import EncoderTrainingModule

    decoder = _tiny_decoder().float()
    backbone = FrozenBackbone(decoder, ce_chunk_tokens=64, gradient_checkpointing=False)
    module = EncoderTrainingModule(
        CoTEncoder(EncoderConfig(d_llm=4, n_blocks=1, n_heads=2)),
        backbone,
        commit_weight=0.25,
    )
    slot_mask = torch.ones(1, 4, dtype=torch.bool)
    step_of_latent = torch.tensor([[0, 0, 1, 1]])

    # Duplicated inside each step, orthogonal across them.
    collapsed = torch.tensor(
        [[[1.0, 0, 0, 0], [1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 1.0, 0, 0]]]
    )
    module._measure_step_cosines(collapsed, slot_mask, step_of_latent)
    assert abs(float(module.last_within_step_cos) - 1.0) < 1e-5
    assert abs(float(module.last_across_step_cos)) < 1e-5

    # Orthogonal inside each step instead.
    diverse = torch.eye(4).unsqueeze(0)
    module._measure_step_cosines(diverse, slot_mask, step_of_latent)
    assert abs(float(module.last_within_step_cos)) < 1e-5

    # Absent rather than zero when nothing said which step a latent belongs to.
    fresh = EncoderTrainingModule(
        CoTEncoder(EncoderConfig(d_llm=4, n_blocks=1, n_heads=2)),
        backbone,
        commit_weight=0.25,
    )
    fresh._measure_step_cosines(diverse, slot_mask, None)
    assert fresh.last_within_step_cos is None


def test_training_runs_end_to_end_with_steps_and_no_quantizer(monkeypatch, tmp_path):
    """The step arm through `train_encoder`: paragraphs, pooled init, no codebook."""
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.setattr(
        "cot_compression.encoder.training.AutoModelForCausalLM",
        type("M", (), {"from_pretrained": staticmethod(_tiny_decoder)}),
    )
    tok = AutoTokenizer.from_pretrained(TINY)
    data = DatasetDict(
        {
            "train": _measured(_step_rows(12, seed=0), tok),
            "val": _measured(_step_rows(4, seed=1), tok),
        }
    )
    monkeypatch.setattr(
        "cot_compression.encoder.training.load_trace_lengths", lambda cfg: data
    )

    cfg = _cfg(tmp_path, max_steps=2, log_interval=1)
    cfg.encoder.patching = "paragraph"
    cfg.encoder.use_vq = False
    cfg.encoder.latent_init = "step_mean"
    cfg.encoder.query_anchor = "step"
    cfg.encoder.next_patch_weight = 1.0
    # `step_mean` as a control is the encoder's own starting point, scored with no
    # encoder -- which exercises the pooled path through `eval_baselines` too.
    cfg.encoder.eval_controls = ["random", "step_mean"]
    train_encoder(cfg)

    root = tmp_path / "enc" / "checkpoints"
    pointer = (root / "LATEST").read_text().strip()
    assert (root / pointer / ENCODER_WEIGHTS).exists()
    manifest = json.loads(
        (root / pointer / "encoder_config.json").read_text(encoding="utf-8")
    )
    assert manifest["use_vq"] is False and manifest["query_anchor"] == "step"
    # No codebook was built, so none was saved -- the DDP and checkpoint-hygiene
    # reason `use_vq=False` must not construct the quantizer at all.
    weights = torch.load(root / pointer / ENCODER_WEIGHTS, map_location="cpu")
    assert not [k for k in weights if k.startswith("quantizer")]


def test_architecture_guard_names_use_vq(tmp_path):
    """A VQ checkpoint and a continuous one are different encoders, and must say so."""
    from cot_compression.encoder.training import check_encoder_architecture

    directory = tmp_path / "ckpt"
    directory.mkdir()
    quantized = EncoderConfig(d_llm=16, codebook_size=8, use_vq=True)
    (directory / "encoder_config.json").write_text(
        json.dumps(asdict(quantized)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="use_vq"):
        check_encoder_architecture(
            EncoderConfig(d_llm=16, codebook_size=8, use_vq=False), directory
        )

    # But two continuous encoders differing only in a codebook size neither has are
    # the same encoder -- refusing there would be a false alarm on a dead field.
    continuous = EncoderConfig(d_llm=16, codebook_size=8, use_vq=False)
    (directory / "encoder_config.json").write_text(
        json.dumps(asdict(continuous)), encoding="utf-8"
    )
    check_encoder_architecture(
        EncoderConfig(d_llm=16, codebook_size=64, use_vq=False), directory
    )
    # And query_anchor is compared, since it moves every cross-attention query.
    with pytest.raises(ValueError, match="query_anchor"):
        check_encoder_architecture(
            EncoderConfig(
                d_llm=16, codebook_size=8, use_vq=False, query_anchor="substep"
            ),
            directory,
        )


def test_a_trace_without_paragraph_breaks_has_no_supervised_step(tokenizer):
    """The 1.9% of real traces with no `\\n\\n`, which subsampling used to be needed for.

    They become one step, step 0 is never a target, so they carry no next-step loss
    at all. Excluding them from the aux denominator rather than counting them as zero
    is what keeps the term a mean over rows that actually have one -- and it is now
    reachable with `next_patch_subsample=1.0`, which `step_denominators`' comment
    used to deny.
    """
    from cot_compression.encoder.training import step_denominators

    flat = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "Q?"},
                    {
                        "role": "assistant",
                        "content": "<think>\nall one paragraph here</think>\n\nAns 1.",
                    },
                ],
                "domain": "math",
                "rollout_id": "p0:r0",
            }
        ]
    )
    sample = _step_prep(flat, tokenizer)[0]
    assert sample is not None
    assert len(sample.cond_slot) == 1, "one step"
    assert sample.aux_patch == [], "step 0 is never a target"

    others = _step_prep(_step_rows(2, seed=9), tokenizer)
    mixed = [sample, *(s for s in (others[i] for i in range(2)) if s)]
    batch = collate_encoder_batch(mixed, pad_token_id=int(tokenizer.pad_token_id))
    counts = step_denominators(torch.device("cpu"), [batch])
    assert counts.tolist() == [3.0, 2.0], "3 rows, 2 with a supervised step"


# --------------------------------------------------------------------------- #
# Train/eval agreement for the pooled seeds, and the learned method on steps
# --------------------------------------------------------------------------- #


def _seed_from_prep(prep, tokenizer, index, decoder):
    """The training seed for one prepared row, via `latent_seed`."""
    from cot_compression.encoder.training import latent_seed

    sample = prep[index]
    assert sample is not None
    batch = collate_encoder_batch([sample], pad_token_id=int(tokenizer.pad_token_id))
    return latent_seed(
        decoder.get_input_embeddings().weight,
        prep.latent_init,
        batch["init_ids"],
        batch.get("init_bag_ids"),
        batch.get("init_bag_offsets"),
        batch.get("init_bag_of_slot"),
    )[0]


@pytest.mark.parametrize("init", ["simple_mean", "step_mean"])
def test_pooled_seed_matches_its_eval_twin(tokenizer, init):
    """Training and `build_latent_init` must build the SAME seed, formula for formula.

    They are two code paths -- a batched `embedding_bag` in training, the eval
    harness's `CompressionMethod.materialize` at scoring time -- and nothing else
    ties them together. They once disagreed: training averaged (`sum / c`) while
    `simple_mean` in the harness is `sum / sqrt(c)`, so a checkpoint would have been
    scored from a seed it never trained from.
    """
    from cot_compression.data.answers import (
        cot_token_ids_and_offsets,
        extract_answer_trace,
    )
    from cot_compression.encoder.frozen import build_latent_init
    from cot_compression.patching import ParagraphStepPatchingMethod, SplitContext

    decoder = _tiny_decoder().float()
    rows = _step_rows(2, seed=21)
    patching = ParagraphStepPatchingMethod(compression_ratio=4.0)
    prep = EncoderPrepDataset(
        rows,
        tokenizer,
        patching,
        seed=7,
        max_length=None,
        vocab_bound=1000,
        latent_init=init,
    )
    trained = _seed_from_prep(prep, tokenizer, 0, decoder)

    trace = extract_answer_trace(rows[0]["messages"])
    assert trace is not None
    cot_ids, offsets = cot_token_ids_and_offsets(trace, tokenizer)
    scored = build_latent_init(init, patching).materialize(
        cot_ids,
        0,
        7,
        tokenizer,
        decoder,
        torch.device("cpu"),
        None,
        context=SplitContext(text=trace.trace, offsets=offsets),
    )
    assert scored is not None
    assert scored.shape == trained.shape
    assert torch.allclose(scored, trained, atol=1e-6), float(
        (scored - trained).abs().max()
    )


def test_latent_seed_refuses_a_batch_that_disagrees_with_the_init():
    from cot_compression.encoder.training import latent_seed

    weight = torch.randn(10, 4)
    ids = torch.zeros(1, 2, dtype=torch.long)
    with pytest.raises(RuntimeError, match="no bags"):
        latent_seed(weight, "step_mean", ids, None, None, None)
    bags = (torch.tensor([1, 2, 3]), torch.tensor([0]), torch.tensor([[0, 0]]))
    with pytest.raises(RuntimeError, match="one token per slot"):
        latent_seed(weight, "random", ids, *bags)


def test_learned_method_on_steps_needs_the_layout_and_agrees_on_k(tokenizer):
    """Caveat K1: the eval path must not score a step checkpoint from the wrong K.

    Without the worker's layout, `materialize` must RAISE -- a RuntimeError, since
    the eval loop reads a ValueError as a data skip and would silently score
    nothing. With it, `plan` and `materialize` agree on K by construction.
    """
    from cot_compression.data.answers import (
        cot_token_ids_and_offsets,
        extract_answer_trace,
        prefix_token_ids,
    )
    from cot_compression.encoder.frozen import build_latent_init
    from cot_compression.encoder.method import LearnedCompressionMethod
    from cot_compression.patching import ParagraphStepPatchingMethod, SplitContext

    decoder = _tiny_decoder().eval()
    encoder = CoTEncoder(
        EncoderConfig(
            d_llm=decoder.config.hidden_size,
            n_blocks=1,
            n_heads=4,
            use_vq=False,
            latent_init="step_mean",
            position_encoding="rope",
        )
    ).eval()
    patching = ParagraphStepPatchingMethod(compression_ratio=4.0)
    method = LearnedCompressionMethod(
        encoder=encoder,
        patching=patching,
        latent_init=build_latent_init("step_mean", patching),
    )
    # A new join key: no codebook to name, and the patching carries the rest.
    assert method.name == "learned_enc_L1cont_rope_paragraph_cr4"

    trace = extract_answer_trace(_step_rows(1, seed=23)[0]["messages"])
    assert trace is not None
    cot_ids, offsets = cot_token_ids_and_offsets(trace, tokenizer)
    prefix = prefix_token_ids(trace, tokenizer)
    device = torch.device("cpu")
    with pytest.raises(RuntimeError, match="SplitContext"):
        method.materialize(cot_ids, 0, 7, tokenizer, decoder, device, None, prefix)

    # Exactly what the eval worker ships: the resolved layout, not the text.
    layout = patching.layout(
        len(cot_ids),
        0,
        7,
        None,
        context=SplitContext(text=trace.trace, offsets=offsets),
    )
    shipped = SplitContext(layout=layout)
    plan = method.plan(len(cot_ids), 0, 7, None, device, shipped)
    slots = method.materialize(
        cot_ids, 0, 7, tokenizer, decoder, device, None, prefix, shipped
    )
    assert slots is not None
    assert layout.num_steps > 1, "multi-step, or this is not testing steps"
    assert slots.shape == (plan.num_slots, decoder.config.hidden_size)
    assert plan.num_slots == layout.num_latents
    assert torch.isfinite(slots).all()


def test_within_step_metric_is_absent_under_token_patching():
    """One latent per step leaves no within-step pair: undefined, not 0.0.

    A 0.0 would read as "perfectly diverse" on the existing token-patching arm's
    dashboard, so the metric must be omitted there rather than reported.
    """
    from cot_compression.encoder.frozen import FrozenBackbone
    from cot_compression.encoder.training import EncoderTrainingModule, codebook_metrics

    backbone = FrozenBackbone(
        _tiny_decoder().float(), ce_chunk_tokens=64, gradient_checkpointing=False
    )
    module = EncoderTrainingModule(
        CoTEncoder(EncoderConfig(d_llm=4, n_blocks=1, n_heads=2, use_vq=False)),
        backbone,
        commit_weight=0.25,
    )
    module.pop_totals = lambda train: {}  # type: ignore[method-assign]
    slot_mask = torch.ones(1, 4, dtype=torch.bool)
    state = SimpleNamespace(module=module)

    module._measure_step_cosines(torch.randn(1, 4, 4), slot_mask, torch.arange(4)[None])
    assert float(module.last_within_step_pairs) == 0.0
    assert "enc/within_step_cos" not in codebook_metrics(state)

    module._measure_step_cosines(
        torch.randn(1, 4, 4), slot_mask, torch.tensor([[0, 0, 1, 1]])
    )
    report = codebook_metrics(state)
    assert "enc/within_step_cos" in report and "enc/across_step_cos" in report
    # And a continuous encoder reports no codebook metrics at all.
    assert not [key for key in report if key.startswith("vq/")]
