from __future__ import annotations

import math
import random

import pytest
import torch

import cot_compression.compression as compression
from cot_compression.compression import (
    EntropyWeightedMeanCompressionMethod,
    RandomCompressionMethod,
    SimpleMeanCompressionMethod,
)
from cot_compression.data.answers import cot_token_ids, extract_answer_trace
from cot_compression.patching import (
    EntropyPatchingMethod,
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

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
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
        [{"role": "user", "content": "Q"}, {"role": "assistant", "content": f"{cot} {answer}"}]
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
    method = SimpleMeanCompressionMethod(UniformPatchingMethod(patch_size=4))
    result = method.compress(trace, 0, 7, tokenizer, model, DEVICE, None)

    expected_patches = math.ceil(t / 4)
    assert result.original_cot_tokens == t
    assert result.compressed_cot_tokens == expected_patches
    assert result.slot_embeddings.shape == (expected_patches, 8)
    # Placeholder replaced the CoT text; joined without spaces.
    assert compression.PLACEHOLDER_TOKEN * expected_patches in result.messages[-1]["content"]
    assert "aaaa" not in result.messages[-1]["content"]


def test_random_is_deterministic_and_uses_regular_vocab() -> None:
    tokenizer, model = MiniTokenizer(), MiniModel()
    trace = _trace()
    method = RandomCompressionMethod(UniformPatchingMethod(patch_size=4))

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


def test_entropy_methods_require_entropy() -> None:
    tokenizer, model = MiniTokenizer(), MiniModel()
    trace = _trace()
    # entropy_weighted_mean always needs entropies, even without entropy patching.
    method = EntropyWeightedMeanCompressionMethod(UniformPatchingMethod(patch_size=4))
    assert method.needs_entropies()
    with pytest.raises(ValueError):
        method.compress(trace, 0, 0, tokenizer, model, DEVICE, None)


def test_cache_fed_entropy_reproduces_spans_and_slots() -> None:
    tokenizer, model = MiniTokenizer(), MiniModel()
    trace = _trace()
    t = len(cot_token_ids(trace, tokenizer))
    entropy = torch.linspace(0.1, 1.0, t)  # stand-in cached CoT entropies

    method = EntropyWeightedMeanCompressionMethod(EntropyPatchingMethod(percentile=50.0))
    a = method.compress(trace, 0, 0, tokenizer, model, DEVICE, entropy.clone())
    b = method.compress(trace, 0, 0, tokenizer, model, DEVICE, entropy.clone())

    assert a.compressed_cot_tokens == b.compressed_cot_tokens
    assert torch.equal(a.slot_embeddings, b.slot_embeddings)
    # Entropy patching actually split into more than one patch here.
    assert a.compressed_cot_tokens > 1


def test_method_naming_encodes_patching_param() -> None:
    assert SimpleMeanCompressionMethod(UniformPatchingMethod(8)).name == "simple_mean_uniform_ps8"
    assert (
        EntropyWeightedMeanCompressionMethod(EntropyPatchingMethod(80.0)).name
        == "entropy_weighted_mean_entropy_p80"
    )
    assert RandomCompressionMethod().name == "random"
