from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from cot_compression.signals import (
    compute_cot_signals,
    load_signal_cache,
    save_signal_cache,
    signal_cache_path,
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

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
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
        {
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": c},
            ]
        }
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
    single = compute_cot_signals(
        sample_indices=[0, 1, 2], batch_size=1, max_batch_tokens=None, **args
    )
    batched = compute_cot_signals(
        sample_indices=[0, 1, 2], batch_size=4, max_batch_tokens=None, **args
    )

    for signal in ("entropy", "surprisal"):
        assert set(single[signal]) == {0, 1, 2} == set(batched[signal])
        for i in (0, 1, 2):
            # Padding shorter sequences must not change their CoT values.
            assert torch.allclose(single[signal][i], batched[signal][i], atol=1e-5)


def test_length_equals_cot_tokens_and_prefix_always_included() -> None:
    tok, model = MiniTokenizer(), PerPositionModel()
    examples = _examples()
    device = torch.device("cpu")
    signals = compute_cot_signals(
        model=model,
        tokenizer=tok,
        examples=examples,
        sample_indices=[0],
        batch_size=2,
        max_batch_tokens=None,
        device=device,
    )
    expected_len = len(tok("<think>alpha beta gamma delta</think>")["input_ids"])
    # One value per CoT token (the producing-distribution value of each), for
    # every signal.
    for signal in ("entropy", "surprisal"):
        assert signals[signal][0].shape[0] == expected_len


def test_signals_are_producing_distribution_shifted_back_one() -> None:
    # CoT token j must get the value at logits[prefix_len + j - 1] (the
    # distribution that produced it), NOT at logits[prefix_len + j] (which
    # predicts token j+1). Checked for entropy and surprisal together.
    from torch.nn import functional as F

    from cot_compression.data.answers import (
        cot_token_ids,
        extract_answer_trace,
        prefix_token_ids,
    )
    from cot_compression.signals import _sequence_signals

    tok, model = MiniTokenizer(), PerPositionModel()
    examples = _examples()
    trace = extract_answer_trace(examples[0]["messages"])
    assert trace is not None
    prefix = prefix_token_ids(trace, tok)
    cot = cot_token_ids(trace, tok)
    full = torch.tensor([prefix + cot])
    logits = model(input_ids=full, attention_mask=torch.ones_like(full)).logits
    # target[p] = token predicted by position p = full[p + 1]; last column padded.
    targets = torch.roll(full, shifts=-1, dims=1)
    targets[:, -1] = tok.pad_token_id
    ref = _sequence_signals(logits, targets, ("entropy", "surprisal"))

    got = compute_cot_signals(
        model=model,
        tokenizer=tok,
        examples=examples,
        sample_indices=[0],
        batch_size=1,
        max_batch_tokens=None,
        device=torch.device("cpu"),
    )

    # Independent check that surprisal is exactly -log p of the realized token.
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    manual_surprisal = -log_probs[torch.arange(full.shape[1]), targets[0]]

    p, t = len(prefix), len(cot)
    for signal in ("entropy", "surprisal"):
        producing = ref[signal][0][p - 1 : p + t - 1]
        next_token = ref[signal][0][p : p + t]
        assert torch.allclose(got[signal][0], producing, atol=1e-6)  # producing
        assert not torch.allclose(got[signal][0], next_token, atol=1e-6)  # not next
    assert torch.allclose(
        got["surprisal"][0], manual_surprisal[p - 1 : p + t - 1], atol=1e-6
    )


def test_cache_path_keyed_by_signal(tmp_path) -> None:
    entropy_path = signal_cache_path(tmp_path, "Qwen/Qwen3-0.6B", "entropy")
    surprisal_path = signal_cache_path(tmp_path, "Qwen/Qwen3-0.6B", "surprisal")
    # entropy keeps its historical stem so a pre-refactor cache still resolves.
    assert entropy_path.name == "cot_entropies__Qwen__Qwen3-0.6B.npz"
    assert surprisal_path.name == "cot_surprisals__Qwen__Qwen3-0.6B.npz"


def test_cache_roundtrip(tmp_path) -> None:
    values = {
        2: torch.tensor([0.1, 0.2, 0.3]),
        5: torch.tensor([1.0]),
        9: torch.tensor([0.5, 0.5]),
    }
    path = signal_cache_path(tmp_path, "Qwen/Qwen3-0.6B", "surprisal")
    assert path.name == "cot_surprisals__Qwen__Qwen3-0.6B.npz"
    save_signal_cache(path, values)
    loaded = load_signal_cache(path)

    assert set(loaded) == set(values)
    for key, value in values.items():
        assert torch.allclose(loaded[key], value.to(torch.float32))


def test_legacy_entropy_cache_still_loads(tmp_path) -> None:
    # A cache written before the signal refactor stored values under "entropies";
    # load_signal_cache must still read it via the key fallback.
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        sample_index=np.array([0, 3], dtype=np.int64),
        offsets=np.array([0, 2, 3], dtype=np.int64),
        entropies=np.array([0.1, 0.2, 0.9], dtype=np.float32),
        cot_lengths=np.array([2, 1], dtype=np.int32),
    )
    loaded = load_signal_cache(path)
    assert set(loaded) == {0, 3}
    assert torch.allclose(loaded[0], torch.tensor([0.1, 0.2]))
    assert torch.allclose(loaded[3], torch.tensor([0.9]))


def test_sharded_precompute_reconstructs_the_full_cache(tmp_path):
    """Sharding must be lossless: disjoint, exhaustive, and merge-identical.

    A missing shard yields a cache that is *short* rather than malformed, which
    nothing downstream would notice -- so the merge refuses an incomplete set
    rather than quietly producing one.
    """
    import numpy as np
    import pytest

    from cot_compression.signals import (
        load_signal_cache,
        merge_signal_shards,
        save_signal_cache,
        signal_cache_path,
    )

    rng = np.random.default_rng(0)
    n_rows, n_shards = 37, 4
    full = {
        i: torch.tensor(rng.random(3 + i % 5), dtype=torch.float32)
        for i in range(n_rows)
    }

    edges = np.linspace(0, n_rows, n_shards + 1).round().astype(int)
    assert edges[0] == 0 and edges[-1] == n_rows, "ranges must be exhaustive"
    covered = [i for s in range(n_shards) for i in range(edges[s], edges[s + 1])]
    assert covered == list(range(n_rows)), "ranges must be disjoint and ordered"

    shard_dir = tmp_path / "shards"
    for s in range(n_shards):
        rows = range(int(edges[s]), int(edges[s + 1]))
        save_signal_cache(
            signal_cache_path(shard_dir, "m/x", "surprisal").with_suffix(
                f".shard{s:02d}of{n_shards:02d}.npz"
            ),
            {i: full[i] for i in rows},
        )

    merged = load_signal_cache(
        merge_signal_shards(tmp_path, "m/x", "surprisal", n_shards)
    )
    assert merged.keys() == full.keys()
    for i in full:
        assert torch.allclose(merged[i], full[i]), i

    # An incomplete set must raise, not silently produce a short cache.
    signal_cache_path(shard_dir, "m/x", "surprisal").with_suffix(
        f".shard01of{n_shards:02d}.npz"
    ).unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        merge_signal_shards(tmp_path, "m/x", "surprisal", n_shards)
