from __future__ import annotations

import math
import random

import pytest
import torch

import cot_compression.compression as compression
from cot_compression.compression import (
    NoCotCompressionMethod,
    RandomCompressionMethod,
    SignalWeightedMeanCompressionMethod,
    SimpleMeanCompressionMethod,
)
from cot_compression.data.answers import cot_token_ids, extract_answer_trace
from cot_compression.patching import (
    SignalSumPatchingMethod,
    SignalThresholdPatchingMethod,
    UniformPatchingMethod,
)

# Regular tokens: ids 2..(SPECIAL_MIN-1); special/added tokens at SPECIAL_MIN+.
SPECIAL_MIN = 200


class MiniTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.unk_token_id = 1
        self._added = {
            "<|vision_pad|>": SPECIAL_MIN,
            "<think>": SPECIAL_MIN + 1,
            "</think>": SPECIAL_MIN + 2,
        }

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._added.get(token, self.unk_token_id)

    def __call__(self, text, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        ids: list[int] = []
        index = 0
        specials = sorted(self._added, key=len, reverse=True)
        while index < len(text):
            match = next((t for t in specials if text.startswith(t, index)), None)
            if match is not None:
                ids.append(self._added[match])
                index += len(match)
            else:
                ids.append((ord(text[index]) % (SPECIAL_MIN - 2)) + 2)
                index += 1
        return {"input_ids": ids}

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        del tokenize, add_generation_prompt
        return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages)


class MiniModel(torch.nn.Module):
    def __init__(self, vocab: int = 256, hidden: int = 8) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.emb


def _trace(cot: str = "<think>aaaa bbbb cccc</think>", answer: str = "Ans"):
    trace = extract_answer_trace(
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": f"{cot} {answer}"},
        ]
    )
    assert trace is not None
    return trace


DEVICE = torch.device("cpu")


def test_no_vocab_extension_machinery() -> None:
    assert not hasattr(compression, "extend_tokenizer_and_model")


def test_uniform_token_counts_and_slot_shape() -> None:
    tokenizer, model = MiniTokenizer(), MiniModel()
    trace = _trace()
    t = len(cot_token_ids(trace, tokenizer))
    method = SimpleMeanCompressionMethod(UniformPatchingMethod(compression_ratio=4))
    result = method.compress(trace, 0, 7, tokenizer, model, DEVICE, None)

    expected_patches = math.ceil(t / 4)
    assert result.original_cot_tokens == t
    assert result.compressed_cot_tokens == expected_patches
    assert result.slot_embeddings.shape == (expected_patches, 8)
    # Placeholder replaced the CoT text; joined without spaces.
    assert (
        compression.PLACEHOLDER_TOKEN * expected_patches
        in result.messages[-1]["content"]
    )
    assert "aaaa" not in result.messages[-1]["content"]


def test_random_is_deterministic_and_uses_regular_vocab() -> None:
    tokenizer, model = MiniTokenizer(), MiniModel()
    trace = _trace()
    method = RandomCompressionMethod(UniformPatchingMethod(compression_ratio=4))

    first = method.compress(trace, 5, 13, tokenizer, model, DEVICE, None)
    second = method.compress(trace, 5, 13, tokenizer, model, DEVICE, None)
    third = method.compress(trace, 6, 13, tokenizer, model, DEVICE, None)

    assert torch.equal(first.slot_embeddings, second.slot_embeddings)
    assert not torch.equal(first.slot_embeddings, third.slot_embeddings)

    # Slots are exactly the embedding rows of ids drawn from the regular vocab.
    k = first.slot_embeddings.shape[0]
    rng = random.Random(13 + 5)
    ids = [rng.randrange(SPECIAL_MIN) for _ in range(k)]
    assert all(i < SPECIAL_MIN for i in ids)
    assert torch.equal(first.slot_embeddings, model.emb.weight[ids])


def test_signal_weighted_mean_requires_its_signal() -> None:
    tokenizer, model = MiniTokenizer(), MiniModel()
    trace = _trace()
    # A signal-weighted mean always needs its signal, even without signal patching.
    method = SignalWeightedMeanCompressionMethod(
        UniformPatchingMethod(compression_ratio=4)
    )
    assert method.required_signals() == frozenset({"entropy"})
    with pytest.raises(ValueError):
        method.compress(trace, 0, 0, tokenizer, model, DEVICE, None)


def test_mixed_signals_are_both_required() -> None:
    # Weight by surprisal, patch by entropy: the method must declare both, and
    # both must be present or the sample is skipped (ValueError).
    tokenizer, model = MiniTokenizer(), MiniModel()
    trace = _trace()
    t = len(cot_token_ids(trace, tokenizer))
    method = SignalWeightedMeanCompressionMethod(
        SignalThresholdPatchingMethod(compression_ratio=2.0, signal="entropy"),
        signal="surprisal",
    )
    assert method.required_signals() == frozenset({"entropy", "surprisal"})
    with pytest.raises(ValueError):
        # Only entropy supplied; surprisal (the pooling signal) is missing.
        method.compress(
            trace, 0, 0, tokenizer, model, DEVICE, {"entropy": torch.rand(t)}
        )


def test_cache_fed_signal_reproduces_spans_and_slots() -> None:
    tokenizer, model = MiniTokenizer(), MiniModel()
    trace = _trace()
    t = len(cot_token_ids(trace, tokenizer))
    entropy = torch.linspace(0.1, 1.0, t)  # stand-in cached CoT entropies

    method = SignalWeightedMeanCompressionMethod(
        SignalThresholdPatchingMethod(compression_ratio=2.0)
    )
    a = method.compress(
        trace, 0, 0, tokenizer, model, DEVICE, {"entropy": entropy.clone()}
    )
    b = method.compress(
        trace, 0, 0, tokenizer, model, DEVICE, {"entropy": entropy.clone()}
    )

    assert a.compressed_cot_tokens == b.compressed_cot_tokens
    assert torch.equal(a.slot_embeddings, b.slot_embeddings)
    # Signal patching actually split into more than one patch here.
    assert a.compressed_cot_tokens > 1


def test_no_cot_produces_empty_think_block() -> None:
    tokenizer, model = MiniTokenizer(), MiniModel()
    trace = _trace()
    t = len(cot_token_ids(trace, tokenizer))
    method = NoCotCompressionMethod()

    assert method.required_signals() == frozenset()
    result = method.compress(trace, 0, 0, tokenizer, model, DEVICE, None)

    assert result.slot_embeddings is None  # text path, no splicing
    assert result.original_cot_tokens == t
    assert result.compressed_cot_tokens == 0
    content = result.messages[-1]["content"]
    assert "<think></think>" in content  # empty interior, tags kept
    assert "aaaa" not in content


def test_surprisal_sum_patching_with_zero_temperature_compress_path() -> None:
    # End-to-end compress() over surprisal_sum patching + surprisal T=0 pooling:
    # spans partition the CoT, one slot per patch, and each slot is exactly the
    # embedding of its patch's highest-surprisal token (one-hot at T=0).
    tokenizer, model = MiniTokenizer(), MiniModel()
    trace = _trace()
    cot_ids = cot_token_ids(trace, tokenizer)
    t = len(cot_ids)
    surprisal = torch.linspace(0.1, 1.0, t)

    patching = SignalSumPatchingMethod(compression_ratio=2.0, signal="surprisal")
    method = SignalWeightedMeanCompressionMethod(
        patching, temperature=0.0, signal="surprisal"
    )
    result = method.compress(
        trace, 0, 0, tokenizer, model, DEVICE, {"surprisal": surprisal.clone()}
    )

    spans = patching.split(t, 0, 0, surprisal)
    assert result.compressed_cot_tokens == len(spans)
    assert result.slot_embeddings.shape == (len(spans), 8)
    embeds = model.emb(torch.tensor(cot_ids))
    for row, (start, end) in enumerate(spans):
        argmax = start + int(surprisal[start:end].argmax())
        assert torch.allclose(result.slot_embeddings[row], embeds[argmax], atol=1e-6)


def test_method_naming_encodes_patching_param() -> None:
    assert (
        SimpleMeanCompressionMethod(UniformPatchingMethod(8)).name
        == "simple_mean_uniform_cr8"
    )
    assert (
        SignalWeightedMeanCompressionMethod(SignalThresholdPatchingMethod(2.0)).name
        == "entropy_weighted_mean_t1_entropy_threshold_cr2"
    )
    # Surprisal is a parallel family: same composition, "surprisal" in place of
    # "entropy" for both the pooling family and the patching name.
    assert (
        SignalWeightedMeanCompressionMethod(
            SignalSumPatchingMethod(4.0, signal="surprisal"), signal="surprisal"
        ).name
        == "surprisal_weighted_mean_t1_surprisal_sum_cr4"
    )
    assert RandomCompressionMethod().name == "random"
    assert NoCotCompressionMethod().name == "no_cot"


def test_method_naming_encodes_temperature() -> None:
    # Distinct temperatures must yield distinct method names so sweep cells that
    # share a (family, patching, ratio) do not collapse in aggregation.
    patching = SignalThresholdPatchingMethod(4.0)
    names = {
        SignalWeightedMeanCompressionMethod(patching, temperature=t).name
        for t in (0.0, 0.5, 1.0)
    }
    assert names == {
        "entropy_weighted_mean_t0_entropy_threshold_cr4",
        "entropy_weighted_mean_t0.5_entropy_threshold_cr4",
        "entropy_weighted_mean_t1_entropy_threshold_cr4",
    }
    method = SignalWeightedMeanCompressionMethod(patching, temperature=0.5)
    assert method.compression_param == "t0.5"


def _weighted_reduce(entropy_rows: torch.Tensor, temperature: float) -> torch.Tensor:
    """Run reduce_patches on a single patch of one-dimensional embeddings.

    Embeds are set to the identity so the pooled output reads back the weights
    (up to the variance rescale), letting us assert the weighting directly.
    """
    num_tokens = entropy_rows.shape[1]
    embeds = torch.eye(num_tokens).unsqueeze(0)  # [1, n, n]
    method = SignalWeightedMeanCompressionMethod(temperature=temperature)
    return method.reduce_patches(embeds, entropy_rows)[0]


def test_temperature_one_matches_plain_entropy_weighting() -> None:
    entropy = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    pooled = _weighted_reduce(entropy, temperature=1.0)
    weights = entropy[0] / entropy[0].sum()
    expected = weights / weights.pow(2).sum().sqrt()
    assert torch.allclose(pooled, expected, atol=1e-6)


def test_temperature_zero_keeps_only_highest_entropy_token() -> None:
    entropy = torch.tensor([[1.0, 2.0, 9.0, 3.0]])
    pooled = _weighted_reduce(entropy, temperature=0.0)
    # One-hot on argmax (index 2); variance rescale by sqrt(sum w^2)=1 is a no-op.
    expected = torch.tensor([0.0, 0.0, 1.0, 0.0])
    assert torch.allclose(pooled, expected, atol=1e-6)


def test_temperature_between_is_sharper_than_mean_weighting() -> None:
    # Lower T concentrates more weight on the highest-entropy token than T=1.
    entropy = torch.tensor([[1.0, 2.0, 9.0, 3.0]])
    w_half = _weighted_reduce(entropy, temperature=0.5)
    w_one = _weighted_reduce(entropy, temperature=1.0)
    assert w_half[2] > w_one[2]


def test_negative_temperature_rejected() -> None:
    with pytest.raises(ValueError):
        SignalWeightedMeanCompressionMethod(temperature=-1.0)
