from __future__ import annotations

from types import SimpleNamespace

import torch

from cot_compression.entropy import (
    compute_cot_entropies,
    entropy_cache_path,
    load_entropy_cache,
    save_entropy_cache,
)

SPECIAL_MIN = 200


class MiniTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self._added = {"<think>": SPECIAL_MIN + 1, "</think>": SPECIAL_MIN + 2}

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


class PerPositionModel(torch.nn.Module):
    """Independent per-position logits (no attention) -> batching is exact."""

    def __init__(self, vocab: int = 256, hidden: int = 8) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.emb

    def forward(self, input_ids=None, attention_mask=None):
        del attention_mask
        return SimpleNamespace(logits=self.head(self.emb(input_ids)))


def _examples() -> list[dict]:
    contents = [
        "<think>alpha beta gamma delta</think> A1",
        "<think>x y</think> A2",
        "<think>lorem ipsum dolor sit amet</think> A3",
    ]
    return [
        {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": c}]}
        for c in contents
    ]


def test_batched_equals_single_and_right_pad_invariant() -> None:
    tok, model = MiniTokenizer(), PerPositionModel()
    examples = _examples()
    device = torch.device("cpu")
    args = dict(
        model=model,
        tokenizer=tok,
        examples=examples,
        device=device,
    )
    single = compute_cot_entropies(sample_indices=[0, 1, 2], batch_size=1, max_batch_tokens=None, **args)
    batched = compute_cot_entropies(sample_indices=[0, 1, 2], batch_size=4, max_batch_tokens=None, **args)

    assert set(single) == {0, 1, 2} == set(batched)
    for i in (0, 1, 2):
        # Padding shorter sequences must not change their CoT entropies.
        assert torch.allclose(single[i], batched[i], atol=1e-5)


def test_entropy_length_equals_cot_tokens_and_prefix_always_included() -> None:
    tok, model = MiniTokenizer(), PerPositionModel()
    examples = _examples()
    device = torch.device("cpu")
    ent = compute_cot_entropies(
        model=model,
        tokenizer=tok,
        examples=examples,
        sample_indices=[0],
        batch_size=2,
        max_batch_tokens=None,
        device=device,
    )
    # One entropy per CoT token (the producing-distribution entropy of each).
    assert ent[0].shape[0] == len(tok("<think>alpha beta gamma delta</think>")["input_ids"])


def test_entropy_is_producing_distribution_shifted_back_one() -> None:
    # CoT token j must get H(logits[prefix_len + j - 1]) (the distribution that
    # produced it), NOT H(logits[prefix_len + j]) (which predicts token j+1).
    from cot_compression.data.answers import (
        cot_token_ids,
        extract_answer_trace,
        prefix_token_ids,
    )
    from cot_compression.entropy import _sequence_entropies

    tok, model = MiniTokenizer(), PerPositionModel()
    examples = _examples()
    trace = extract_answer_trace(examples[0]["messages"])
    prefix = prefix_token_ids(trace, tok)
    cot = cot_token_ids(trace, tok)
    full = torch.tensor([prefix + cot])
    full_h = _sequence_entropies(
        model(input_ids=full, attention_mask=torch.ones_like(full)).logits
    )[0]

    got = compute_cot_entropies(
        model=model,
        tokenizer=tok,
        examples=examples,
        sample_indices=[0],
        batch_size=1,
        max_batch_tokens=None,
        device=torch.device("cpu"),
    )[0]

    p, t = len(prefix), len(cot)
    assert torch.allclose(got, full_h[p - 1 : p + t - 1], atol=1e-6)  # producing
    assert not torch.allclose(got, full_h[p : p + t], atol=1e-6)  # not next-token


def test_cache_roundtrip(tmp_path) -> None:
    entropies = {
        2: torch.tensor([0.1, 0.2, 0.3]),
        5: torch.tensor([1.0]),
        9: torch.tensor([0.5, 0.5]),
    }
    path = entropy_cache_path(tmp_path, "Qwen/Qwen3-0.6B")
    assert path.name == "cot_entropies__Qwen__Qwen3-0.6B.npz"
    save_entropy_cache(path, entropies)
    loaded = load_entropy_cache(path)

    assert set(loaded) == set(entropies)
    for key, value in entropies.items():
        assert torch.allclose(loaded[key], value.to(torch.float32))
