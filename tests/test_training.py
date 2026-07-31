from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from cot_compression.data.dolci import DolciSFTData
from cot_compression.training.evaluate import (
    SampleScore,
    TokenLogprobWriter,
    TokenScore,
    evaluate_methods,
    summarize_method,
)
from cot_compression.training.sft import ChunkedCELM, plan_epoch, train_sft

_PATCHING = {
    "compression_ratio": 4.0,
    "random": {"max_exponent": 6},
}


def _eval_cfg(tmp_path, methods, **evaluation_overrides):
    evaluation = {
        "seed": 7,
        "device": "cpu",
        "torch_dtype": "float32",
        "max_length": None,
        "batch_size": 2,
        "max_batch_tokens": 512,
        "num_workers": 0,
        "prefetch_factor": 2,
        "max_examples": None,
        "metric": "answer_logprob",
        "normalize_by_length": True,
        "save_token_logprobs": True,
        "entropy_cache_dir": None,
        "save_entropies": False,
        "methods": methods,
    }
    evaluation.update(evaluation_overrides)
    return OmegaConf.create(
        {
            "paths": {"run_dir": str(tmp_path / "eval")},
            "data": {"prepared_dir": str(tmp_path / "data")},
            "method": {
                "model_name": "tiny",
                "trust_remote_code": False,
                "use_fast_tokenizer": True,
            },
            "evaluation": evaluation,
            "logging": {
                "enabled": False,
                "mode": "offline",
                "project": "tests",
                "entity": None,
                "group": None,
                "name": None,
                "tags": [],
                "log_file_name": "eval.log",
            },
        }
    )


class FakeSFTTokenizer:
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.pad_token = "<pad>"
        self.truncation_side = "right"
        # Pre-registered special tokens (highest ids) so the regular/special
        # boundary is 120; the placeholder is a real single token, no resize.
        self._token_to_id: dict[str, int] = {
            "<|vision_pad|>": 120,
            "<think>": 121,
            "</think>": 122,
        }
        self._vocab_size = 128

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __len__(self) -> int:
        return self._vocab_size

    def add_tokens(self, tokens: list[str]) -> int:
        added = 0
        for token in tokens:
            if token in self._token_to_id:
                continue
            self._token_to_id[token] = self._vocab_size
            self._vocab_size += 1
            added += 1
        return added

    def apply_chat_template(
        self,
        messages,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        del tokenize, add_generation_prompt
        return "".join(
            f"<|{message['role']}|>\n{message['content']}\n" for message in messages
        )

    def __call__(
        self,
        text: str,
        add_special_tokens: bool,
        max_length: int | None = None,
        truncation: bool = False,
        return_offsets_mapping: bool = True,
    ):
        del add_special_tokens, truncation, return_offsets_mapping
        if max_length is None:
            start = 0
        else:
            start = (
                max(0, len(text) - max_length) if self.truncation_side == "left" else 0
            )
            text = text[start : start + max_length]
        input_ids = []
        offsets = []
        index = 0
        tokens_by_length = sorted(self._token_to_id, key=len, reverse=True)
        while index < len(text):
            matched = next(
                (token for token in tokens_by_length if text.startswith(token, index)),
                None,
            )
            if matched is not None:
                input_ids.append(self._token_to_id[matched])
                offsets.append((index + start, index + start + len(matched)))
                index += len(matched)
                continue
            input_ids.append((ord(text[index]) % 64) + 1)
            offsets.append((index + start, index + start + 1))
            index += 1
        return {
            "input_ids": input_ids,
            "attention_mask": [1 for _ in input_ids],
            "offset_mapping": offsets,
        }

    def save_pretrained(self, path) -> None:
        Path(path, "tokenizer_config.json").write_text("{}", encoding="utf-8")

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_token_id)


class TinyBackbone(nn.Module):
    """Stands in for `Qwen3Model`: returns `last_hidden_state`, no head."""

    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)

    def forward(self, input_ids=None, attention_mask=None):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embed_tokens(input_ids))


class TinySFTModel(nn.Module):
    """Mirrors the HF causal-LM shape the two paths rely on: `.model` + `.lm_head`
    for the SFT chunked-CE wrapper, and a `forward` taking `inputs_embeds` for the
    embedding-splicing eval path."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.model = TinyBackbone(512, 8)
        self.lm_head = nn.Linear(8, 128)

    @classmethod
    def from_pretrained(cls, name_or_path=None, *args, **kwargs):
        model = cls()
        # Actually restore when handed a checkpoint directory. A fake that always
        # returns fresh weights would let the resume test pass while resume was
        # silently broken -- and would make it fail while resume was correct.
        weights = Path(str(name_or_path)) / "model.pt" if name_or_path else None
        if weights is not None and weights.exists():
            model.load_state_dict(torch.load(weights, map_location="cpu"))
        return model

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        del gradient_checkpointing_kwargs
        return None

    def resize_token_embeddings(self, size: int, mean_resizing: bool = False) -> None:
        del mean_resizing
        self.lm_head = nn.Linear(8, size)

    def save_pretrained(self, path, state_dict=None) -> None:
        torch.save(
            self.state_dict() if state_dict is None else state_dict,
            Path(path) / "model.pt",
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(
        self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None
    ):
        del attention_mask
        embeds = (
            self.model.embed_tokens(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        logits = self.lm_head(embeds)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


def _tokenized_split(rows: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    records = []
    for index in range(rows):
        length = int(rng.integers(8, 24))
        records.append(
            {
                "input_ids": rng.integers(1, 64, length).tolist(),
                "label_start": int(rng.integers(2, length - 2)),
                "length": length,
                "id": str(index),
            }
        )
    return Dataset.from_list(records)


def _sft_cfg(tmp_path, **training_overrides):
    training = {
        "seed": 7,
        "device": "cpu",
        "deterministic": True,
        "torch_dtype": "float32",
        "autocast_dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "num_train_epochs": 1,
        "max_train_examples": None,
        "max_length": 64,
        "max_batch_tokens": 64,
        "length_group_size": 8,
        "micro_batch_max_sequences": 2,
        "target_global_batch": 4,
        "ce_chunk_tokens": 8,
        "num_workers": 0,
        "prefetch_factor": 2,
        "warmup_ratio": 0.0,
        "min_warmup_steps": 0,
        "schedule_horizon_steps": None,
        "gradient_clip": 1.0,
        "gradient_checkpointing": False,
        "compile": False,
        "max_steps": None,
        "eval_interval": 1000,
        "eval_examples": 8,
        "log_interval": 1000,
        "checkpoint_minutes": 1e9,
        "keep_checkpoints": 2,
        "resume_from_checkpoint": "auto",
    }
    training.update(training_overrides)
    return OmegaConf.create(
        {
            "paths": {"run_dir": str(tmp_path / "sft")},
            "data": {"name": "tiny"},
            "method": {
                "model_name": "tiny",
                "trust_remote_code": False,
                "use_fast_tokenizer": True,
            },
            "training": training,
            "optim": {
                "lr": 0.01,
                "weight_decay": 0.0,
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-8,
            },
            "logging": {
                "enabled": False,
                "mode": "offline",
                "project": "tests",
                "entity": None,
                "group": None,
                "name": None,
                "tags": [],
                "log_file_name": "train.log",
                "log_artifacts": False,
                "max_artifact_mb": 32,
            },
        }
    )


def _patch_sft(monkeypatch, data):
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.setattr("cot_compression.training.sft.AutoTokenizer", FakeSFTTokenizer)
    monkeypatch.setattr(
        "cot_compression.training.sft.AutoModelForCausalLM", TinySFTModel
    )
    monkeypatch.setattr(
        "cot_compression.training.sft.load_tokenized_sft_data", lambda cfg: data
    )


def test_sft_training_runs_and_publishes_a_resumable_checkpoint(
    monkeypatch, tmp_path
) -> None:
    data = DatasetDict(
        {"train": _tokenized_split(24, seed=0), "eval": _tokenized_split(8, seed=1)}
    )
    _patch_sft(monkeypatch, data)

    train_sft(_sft_cfg(tmp_path))

    root = tmp_path / "sft" / "checkpoints"
    pointer = (root / "LATEST").read_text().strip()
    assert (root / pointer / "training_state.pt").exists()
    assert (root / pointer / "model.pt").exists()
    assert (root / "best" / "model.pt").exists()
    # best/ is eval-only, so it must NOT carry optimizer state.
    assert not (root / "best" / "training_state.pt").exists()


def _latest_weights(run_root: Path):
    root = run_root / "sft" / "checkpoints"
    pointer = (root / "LATEST").read_text().strip()
    return torch.load(root / pointer / "model.pt", map_location="cpu"), pointer


def test_resume_reproduces_an_uninterrupted_run(monkeypatch, tmp_path) -> None:
    """The point of the plan+pointer design: stopping and resuming must land on
    bit-identical weights, not merely a plausible continuation."""
    data = DatasetDict(
        {"train": _tokenized_split(24, seed=0), "eval": _tokenized_split(8, seed=1)}
    )

    _patch_sft(monkeypatch, data)
    train_sft(_sft_cfg(tmp_path / "full"))
    reference, final_pointer = _latest_weights(tmp_path / "full")

    # Same configuration, cut off after one step, then resumed via the pointer.
    _patch_sft(monkeypatch, data)
    train_sft(_sft_cfg(tmp_path / "split", max_steps=1))
    _patch_sft(monkeypatch, data)
    train_sft(_sft_cfg(tmp_path / "split"))
    resumed, resumed_pointer = _latest_weights(tmp_path / "split")

    assert resumed_pointer == final_pointer, "resume ended on a different step"
    assert set(reference) == set(resumed)
    for key, value in reference.items():
        torch.testing.assert_close(value, resumed[key], rtol=0, atol=0)


@pytest.mark.parametrize("chunk", [1, 5, 37, 4096])
def test_chunked_ce_matches_reference_loss_and_gradients(chunk: int) -> None:
    torch.manual_seed(0)
    model = TinySFTModel()
    wrapper = ChunkedCELM(model, chunk_tokens=chunk)

    input_ids = torch.randint(1, 64, (2, 19))
    labels = input_ids.clone()
    labels[:, :4] = -100  # prompt region
    labels[0, 10:13] = -100  # an ignored run inside the supervised region
    attention_mask = torch.ones_like(input_ids)

    total = wrapper(input_ids, attention_mask, labels)
    total.backward()
    chunked_grad = model.lm_head.weight.grad.clone()

    model.zero_grad()
    hidden = model.model.embed_tokens(input_ids)
    logits = model.lm_head(hidden[:, :-1])
    expected = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    expected.backward()

    torch.testing.assert_close(total, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        chunked_grad, model.lm_head.weight.grad, rtol=1e-5, atol=1e-5
    )


def test_chunked_ce_handles_a_fully_ignored_chunk() -> None:
    model = TinySFTModel()
    wrapper = ChunkedCELM(model, chunk_tokens=4)
    input_ids = torch.randint(1, 64, (1, 12))
    labels = torch.full_like(input_ids, -100)
    labels[0, -2:] = input_ids[0, -2:]

    total = wrapper(input_ids, torch.ones_like(input_ids), labels)
    total.backward()

    assert torch.isfinite(total)
    assert torch.isfinite(model.lm_head.weight.grad).all()


def test_accumulation_is_token_weighted_not_mean_of_means() -> None:
    """Micro-batches here are token-budgeted, so they hold different numbers of
    supervised tokens. Averaging per-micro-batch means (what the loop used to do)
    silently over-weights short batches."""
    torch.manual_seed(0)
    model = TinySFTModel()
    wrapper = ChunkedCELM(model, chunk_tokens=1024)

    batches = []
    for supervised in (2, 7, 11):
        ids = torch.randint(1, 64, (1, 16))
        labels = torch.full_like(ids, -100)
        labels[0, -supervised:] = ids[0, -supervised:]
        batches.append((ids, labels))
    denom = sum(int((labels[:, 1:] != -100).sum()) for _, labels in batches)

    def grad_of(fn) -> torch.Tensor:
        model.zero_grad()
        fn()
        return model.lm_head.weight.grad.clone()

    accumulated = grad_of(
        lambda: [
            (wrapper(ids, torch.ones_like(ids), labels) / denom).backward()
            for ids, labels in batches
        ]
    )
    joint_ids = torch.cat([ids for ids, _ in batches])
    joint_labels = torch.cat([labels for _, labels in batches])
    reference = grad_of(
        lambda: (
            wrapper(joint_ids, torch.ones_like(joint_ids), joint_labels) / denom
        ).backward()
    )
    naive = grad_of(
        lambda: [
            (
                wrapper(ids, torch.ones_like(ids), labels)
                / int((labels[:, 1:] != -100).sum())
                / len(batches)
            ).backward()
            for ids, labels in batches
        ]
    )

    torch.testing.assert_close(accumulated, reference, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(naive, reference, rtol=1e-3, atol=1e-6), (
        "the naive scheme matched the correct one, so this test proves nothing"
    )


def _plan_cfg(**overrides):
    return _sft_cfg(Path("/tmp"), **overrides)


def test_plan_epoch_is_deterministic_and_ddp_safe() -> None:
    rng = np.random.default_rng(0)
    lengths = rng.integers(4, 80, 200)
    starts = (lengths // 3).astype(np.int64)
    cfg = _plan_cfg()

    first = plan_epoch(lengths, starts, cfg, epoch=0, world_size=4)
    again = plan_epoch(lengths, starts, cfg, epoch=0, world_size=4)
    other = plan_epoch(lengths, starts, cfg, epoch=1, world_size=4)

    assert first.micro == again.micro and first.signature == again.signature
    assert first.micro != other.micro

    # DDP deadlocks unless every rank runs the same number of micro-batches.
    assert len(first.micro) % 4 == 0
    assert all((stop - start) % 4 == 0 for start, stop in first.steps)

    flat = [index for batch in first.micro for index in batch]
    assert len(flat) == len(set(flat)), "an example appeared twice in one epoch"
    assert all(lengths[index] <= cfg.training.max_length for index in flat)
    assert first.dropped == int((lengths > cfg.training.max_length).sum())


def test_plan_denominator_is_the_exact_supervised_token_count() -> None:
    rng = np.random.default_rng(3)
    lengths = rng.integers(6, 40, 120)
    starts = (lengths // 4).astype(np.int64)
    plan = plan_epoch(lengths, starts, _plan_cfg(), epoch=0, world_size=2)

    for (start, stop), denom in zip(plan.steps, plan.denom, strict=True):
        expected = sum(
            int(lengths[i] - starts[i]) for b in plan.micro[start:stop] for i in b
        )
        assert denom == expected


def test_plan_signature_changes_with_the_batching_knobs() -> None:
    lengths = np.full(64, 10, dtype=np.int64)
    starts = np.full(64, 2, dtype=np.int64)
    base = plan_epoch(lengths, starts, _plan_cfg(), epoch=0, world_size=1)
    wider = plan_epoch(
        lengths, starts, _plan_cfg(max_batch_tokens=128), epoch=0, world_size=1
    )
    assert base.signature != wider.signature


def test_answer_loss_evaluation_smoke(monkeypatch, tmp_path) -> None:
    examples = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "<think>Trace</think> Answer"},
                ],
                "dataset_source": "valid",
                "id": "ok",
            },
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer only"},
                ],
                "dataset_source": "invalid",
                "id": "skip",
            },
        ]
    )

    monkeypatch.setattr(
        "cot_compression.training.evaluate.AutoTokenizer",
        FakeSFTTokenizer,
    )
    monkeypatch.setattr(
        "cot_compression.training.evaluate.AutoModelForCausalLM",
        TinySFTModel,
    )
    monkeypatch.setattr(
        "cot_compression.training.evaluate.load_dolci_sft_data",
        lambda cfg: DolciSFTData(train=examples, eval=examples, test=examples),
    )

    cfg = _eval_cfg(
        tmp_path,
        {
            "enabled": ["base", "random"],
            "patching": _PATCHING,
            "random": {"patching": "uniform"},
            "simple_mean": {"patching": None},
            "entropy_weighted_mean": {"patching": None},
        },
    )

    summary_path = evaluate_methods(cfg)
    samples_path = summary_path.with_name("samples.jsonl")
    tokens_path = summary_path.with_name("tokens.jsonl")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    methods = {method["method"]: method for method in summary["methods"]}
    sample_rows = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
    ]
    token_rows = [
        json.loads(line)
        for line in tokens_path.read_text(encoding="utf-8").splitlines()
    ]

    assert set(methods) == {"base", "random_uniform_cr4"}
    assert summary["metric"] == "answer_logprob"
    assert methods["base"]["samples"] == 1
    assert methods["base"]["skipped"] == 1
    assert methods["base"]["method_family"] == "base"
    # base = full CoT, ratio 1.0; random compresses below 1.
    assert methods["base"]["mean_compression_ratio"] == 1.0
    assert methods["random_uniform_cr4"]["samples"] == 1
    assert methods["random_uniform_cr4"]["skipped"] == 1
    assert methods["random_uniform_cr4"]["mean_compression_ratio"] < 1.0
    assert {row["method"] for row in sample_rows} == {"base", "random_uniform_cr4"}
    assert all(row["answer_tokens"] > 0 for row in sample_rows)
    assert all("compression_ratio" in row for row in sample_rows)
    assert {row["method"] for row in token_rows} == {"base", "random_uniform_cr4"}
    assert all("logprob" in row for row in token_rows)
    # The writer runs on a background thread, so assert it neither drops nor
    # duplicates rows: every scored answer token must appear exactly once.
    assert len(token_rows) == sum(row["answer_tokens"] for row in sample_rows)
    # reporting/sweep_stats.py derives answer-relative position from the running
    # row count within each (method, sample_index) group, so groups must stay
    # contiguous and ascending in token_index.
    groups: list[tuple[str, int]] = []
    for row in token_rows:
        key = (row["method"], row["sample_index"])
        if not groups or groups[-1] != key:
            groups.append(key)
    assert len(groups) == len(set(groups)), "a (method, sample) group was split up"
    for key in groups:
        indices = [
            row["token_index"]
            for row in token_rows
            if (row["method"], row["sample_index"]) == key
        ]
        assert indices == sorted(indices)


def test_answer_loss_evaluation_embedding_methods_smoke(monkeypatch, tmp_path) -> None:
    examples = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {
                        "role": "assistant",
                        "content": "<think>Trace here</think> Answer",
                    },
                ],
                "dataset_source": "valid",
                "id": "ok",
            },
        ]
    )

    monkeypatch.setattr(
        "cot_compression.training.evaluate.AutoTokenizer",
        FakeSFTTokenizer,
    )
    monkeypatch.setattr(
        "cot_compression.training.evaluate.AutoModelForCausalLM",
        TinySFTModel,
    )
    monkeypatch.setattr(
        "cot_compression.training.evaluate.load_dolci_sft_data",
        lambda cfg: DolciSFTData(train=examples, eval=examples, test=examples),
    )

    # Exercise both directions of the needs-entropies OR: simple_mean's own
    # reduction never needs entropies but its "entropy" patching does; the
    # reverse for entropy_weighted_mean with "uniform" patching. Entropies are
    # computed inline (no cache dir). save_entropies exercises the byproduct.
    cfg = _eval_cfg(
        tmp_path,
        {
            "enabled": ["simple_mean", "entropy_weighted_mean"],
            "patching": _PATCHING,
            "random": {"patching": None},
            "simple_mean": {"patching": "entropy_threshold"},
            "entropy_weighted_mean": {"patching": "uniform"},
        },
        save_entropies=True,
    )

    summary_path = evaluate_methods(cfg)
    samples_path = summary_path.with_name("samples.jsonl")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    methods = {method["method"]: method for method in summary["methods"]}
    sample_rows = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
    ]

    # patching (strategy + param) folds into the method name so results from
    # different rates never collide when merged across sweep jobs.
    assert set(methods) == {
        "simple_mean_entropy_threshold_cr4",
        "entropy_weighted_mean_t1_uniform_cr4",
    }
    assert methods["simple_mean_entropy_threshold_cr4"]["samples"] == 1
    assert methods["entropy_weighted_mean_t1_uniform_cr4"]["samples"] == 1
    assert (
        methods["simple_mean_entropy_threshold_cr4"]["patching"] == "entropy_threshold"
    )
    assert {row["method"] for row in sample_rows} == {
        "simple_mean_entropy_threshold_cr4",
        "entropy_weighted_mean_t1_uniform_cr4",
    }
    assert all(row["answer_tokens"] > 0 for row in sample_rows)
    assert all(row["compression_ratio"] is not None for row in sample_rows)
    # One answer-entropy artifact per method (no cross-method collision).
    assert (
        summary_path.with_name(
            "answer_entropies__simple_mean_entropy_threshold_cr4.npz"
        )
    ).exists()
    assert (
        summary_path.with_name(
            "answer_entropies__entropy_weighted_mean_t1_uniform_cr4.npz"
        )
    ).exists()


def _run_eval_artifacts(monkeypatch, tmp_path, examples, num_workers):
    monkeypatch.setattr(
        "cot_compression.training.evaluate.AutoTokenizer", FakeSFTTokenizer
    )
    monkeypatch.setattr(
        "cot_compression.training.evaluate.AutoModelForCausalLM", TinySFTModel
    )
    monkeypatch.setattr(
        "cot_compression.training.evaluate.load_dolci_sft_data",
        lambda cfg: DolciSFTData(train=examples, eval=examples, test=examples),
    )
    run_dir = tmp_path / f"workers_{num_workers}"
    cfg = _eval_cfg(
        run_dir,
        {
            # Covers every prep path: base (text), random (non-entropy patching),
            # simple_mean (entropy patching -> span_counts), entropy_weighted_mean
            # (uniform patching, entropy pooling in materialize).
            "enabled": ["base", "random", "simple_mean", "entropy_weighted_mean"],
            "patching": _PATCHING,
            "random": {"patching": "uniform"},
            "simple_mean": {"patching": "entropy_threshold"},
            "entropy_weighted_mean": {"patching": "uniform"},
        },
        num_workers=num_workers,
    )
    summary_path = evaluate_methods(cfg)
    return {
        name: summary_path.with_name(name).read_text(encoding="utf-8")
        for name in ("summary.json", "samples.jsonl", "tokens.jsonl")
    }


def test_answer_logprobs_from_logits_handles_bfloat16() -> None:
    """The token-median histc and the sum-of-squares must work on bf16 logits
    (the model's usual eval dtype) -- torch.histc has no bf16 kernel, so the
    code casts to float32. Regression for a crash that only bf16 surfaced."""
    from cot_compression.training.evaluate import _answer_logprobs_from_logits

    torch.manual_seed(0)
    logits = torch.randn(2, 6, 16, dtype=torch.bfloat16)
    labels = torch.full((2, 6), -100)
    labels[:, 3:] = torch.randint(0, 16, (2, 3))  # last 3 positions are "answer"
    results, hist, under = _answer_logprobs_from_logits(
        logits,
        labels,
        torch.device("cpu"),
        save_entropies=False,
        save_token_logprobs=True,
    )
    assert len(results) == 2
    assert hist.shape[0] == 3000 and hist.sum() + under > 0
    # sum-of-squares (5th field) is finite and non-negative for every row.
    assert all(r[4] >= 0 and r[4] == r[4] for r in results)


def _sample(mean_and_sumsq_tokens: list[float], index: int) -> SampleScore:
    """A SampleScore whose answer tokens have the given per-token log-probs."""
    return SampleScore(
        method="m",
        sample_index=index,
        sample_id=None,
        dataset_source=None,
        answer_tokens=len(mean_and_sumsq_tokens),
        logprob_sum=sum(mean_and_sumsq_tokens),
        logprob_mean=sum(mean_and_sumsq_tokens) / len(mean_and_sumsq_tokens),
        logprob_sumsq=sum(x * x for x in mean_and_sumsq_tokens),
        compressed_cot_tokens=None,
        compression_ratio=None,
    )


def test_summarize_method_token_and_sample_stats() -> None:
    """Token stats pool every answer token; per-sample stats aggregate the sample
    means. Means, medians and stds are all checked against a hand computation."""
    method = SimpleNamespace(
        name="m",
        method_family="m",
        patching_name="none",
        patching_param="none",
        compression_param="none",
    )
    # tokens per sample; pooled = [-1,-2,-0.5,-0.5,-1,-3] -> N=6, sum=-8, sumsq=15.5
    token_lists = [[-1.0, -2.0], [-0.5, -0.5, -1.0], [-3.0]]
    samples = [_sample(t, i) for i, t in enumerate(token_lists)]
    pooled = [x for t in token_lists for x in t]
    hist = torch.histc(torch.tensor(pooled), bins=3000, min=-30.0, max=0.0).numpy()
    summary = summarize_method(
        method, samples, skipped=0, token_hist=hist, token_under=0
    )

    # Pooled token level.
    assert summary.total_answer_tokens == 6
    assert summary.mean_token_logprob == pytest.approx(-8.0 / 6)
    assert summary.std_token_logprob == pytest.approx(
        (15.5 / 6 - (8.0 / 6) ** 2) ** 0.5
    )
    assert summary.median_token_logprob == pytest.approx(-1.0, abs=0.02)  # mid of 6

    # Per-sample level: median of sample means (-1.5, -2/3, -3) is -1.5, distinct
    # from both the sample-mean mean and the pooled token mean.
    sample_means = [-1.5, -2.0 / 3.0, -3.0]
    assert summary.mean_logprob == pytest.approx(sum(sample_means) / 3)
    assert summary.median_logprob == pytest.approx(-1.5)
    assert summary.median_logprob != pytest.approx(summary.mean_logprob)
    assert summary.median_answer_tokens == pytest.approx(2.0)  # median of {2,3,1}


def test_scratch_dir_routes_tokens_to_scratch_only(monkeypatch, tmp_path) -> None:
    """With evaluation.scratch_dir set, tokens.jsonl lands only in scratch while
    summary.json/samples.jsonl are mirrored to both the run dir and scratch."""
    monkeypatch.setattr(
        "cot_compression.training.evaluate.AutoTokenizer", FakeSFTTokenizer
    )
    monkeypatch.setattr(
        "cot_compression.training.evaluate.AutoModelForCausalLM", TinySFTModel
    )
    examples = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": f"Q{i}"},
                    {
                        "role": "assistant",
                        "content": f"<think>{'reasoning ' * (2 + i)}</think> A{i}",
                    },
                ],
                "dataset_source": "valid",
                "id": f"ok-{i}",
            }
            for i in range(4)
        ]
    )
    monkeypatch.setattr(
        "cot_compression.training.evaluate.load_dolci_sft_data",
        lambda cfg: DolciSFTData(train=examples, eval=examples, test=examples),
    )
    scratch = tmp_path / "scratch"
    cfg = _eval_cfg(
        tmp_path / "home",
        {"enabled": ["base"], "patching": _PATCHING},
        scratch_dir=str(scratch),
    )
    summary_path = evaluate_methods(cfg)

    home = summary_path.parent  # <run_dir>/artifacts
    # run_dir.name is "eval" (paths.run_dir = <home>/eval), so scratch mirrors it.
    scratch_art = scratch / "eval" / "artifacts"
    assert (home / "summary.json").exists() and (home / "samples.jsonl").exists()
    assert not (home / "tokens.jsonl").exists()  # kept off HOME
    assert (scratch_art / "tokens.jsonl").exists()
    assert (scratch_art / "summary.json").exists()
    # The mirrored summary is byte-identical to the run-dir copy.
    assert (scratch_art / "summary.json").read_text() == (
        home / "summary.json"
    ).read_text()


def test_worker_prefetch_matches_serial_prep(monkeypatch, tmp_path) -> None:
    """DataLoader workers must not change any output vs main-thread prep.

    Prep is deterministic in (sample_index, seed) and order-independent, and the
    forward runs in the main process either way, so moving tokenization into
    workers must be bit-for-bit invisible in all three artifacts.
    """
    examples = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {
                        "role": "assistant",
                        "content": f"<think>{'reasoning ' * (3 + i)}</think> Answer {i}",
                    },
                ],
                "dataset_source": "valid",
                "id": f"ok-{i}",
            }
            for i in range(6)
        ]
    )

    serial = _run_eval_artifacts(monkeypatch, tmp_path, examples, num_workers=0)
    workers = _run_eval_artifacts(monkeypatch, tmp_path, examples, num_workers=2)
    assert serial == workers


def test_preflight_length_check_raises_on_over_length_cot() -> None:
    from cot_compression.training.evaluate import preflight_length_check
    from cot_compression.training.logging import RunLogger

    logger = object.__new__(RunLogger)  # bypass wandb/file setup; only .info used
    logger.info = lambda message: None  # type: ignore[method-assign]
    entropies = {0: torch.zeros(10), 4: torch.zeros(50)}

    # Within the window: no raise.
    preflight_length_check(entropies, 64, logger)
    # A CoT past the window names the offending sample and its length.
    with pytest.raises(ValueError, match="sample 4 is 50 tokens"):
        preflight_length_check(entropies, 32, logger)


def test_over_length_guard_drops_same_samples_for_every_method(
    monkeypatch, tmp_path
) -> None:
    """The length guard keys on the CoT count, so base and compressed drop the
    same sample -- the paired comparison stays over one population."""
    from cot_compression.training.evaluate import PrepDataset

    tokenizer = FakeSFTTokenizer()
    examples = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": f"<think>{'x' * length}</think> A",
                    },
                ],
                "dataset_source": "valid",
                "id": f"len-{length}",
            }
            for length in (5, 100)
        ]
    )
    placeholder_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")

    def kept_indices(method_name, patching):
        from cot_compression.compression import build_compression_methods

        cfg = _eval_cfg(
            tmp_path,
            {
                "enabled": [method_name],
                "patching": _PATCHING,
                method_name: {"patching": patching},
            },
        )
        method = build_compression_methods(cfg)[0]
        prep = PrepDataset(
            examples,
            tokenizer,
            method,
            limit=len(examples),
            seed=7,
            max_length=None,
            max_position_embeddings=40,  # the 100-char CoT is over, the 5-char one fits
            span_counts=None,
            cot_ids_cache={},
            placeholder_id=int(placeholder_id),
        )
        return [i for i in range(len(examples)) if prep[i] is not None]

    # Base (uncompressed) and a compressed method drop the identical sample set,
    # even though the compressed rendering is far shorter than the window.
    assert kept_indices("base", None) == [0]
    assert kept_indices("simple_mean", "uniform") == [0]


def test_token_writer_output_matches_json_dumps(tmp_path) -> None:
    """The %-format fast path must be byte-identical to json.dumps(asdict(...)).

    tokens.jsonl is consumed by a byte-slicing parser
    (reporting/sweep_stats.py), so the exact key order, separators and float
    formatting are a contract, not an implementation detail.
    """
    path = tmp_path / "tokens.jsonl"
    method = "entropy_weighted_mean_t0.5_entropy_sum_cr4"
    logprobs = [-0.5, -1.0 / 3.0, -1.2345678901234567e-08, -123.456, 0.0]
    writer = TokenLogprobWriter(path, queue_size=2)
    writer.submit(
        json.dumps(method), 7, [1, 2, 3, 4, 5], [10, 11, 12, 13, 14], logprobs
    )
    # A second sample, to pin down that rows stay grouped in submission order.
    writer.submit(json.dumps(method), 3, [1], [15], [-2.5])
    writer.close()

    expected = [
        json.dumps(
            {
                "method": method,
                "sample_index": 7,
                "token_index": index,
                "token_id": token_id,
                "logprob": logprob,
            }
        )
        for index, token_id, logprob in zip(
            [1, 2, 3, 4, 5], [10, 11, 12, 13, 14], logprobs, strict=True
        )
    ]
    expected.append(json.dumps(asdict(TokenScore(method, 3, 1, 15, -2.5))))
    assert path.read_text(encoding="utf-8").splitlines() == expected


def test_token_writer_keeps_non_finite_logprobs_valid_json(tmp_path) -> None:
    """repr() spells -inf/nan in a way json.loads rejects; dumps must be used."""
    path = tmp_path / "tokens.jsonl"
    writer = TokenLogprobWriter(path)
    writer.submit(json.dumps("base"), 0, [1, 2], [5, 6], [-1.5, float("-inf")])
    writer.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["logprob"] for line in lines] == [-1.5, float("-inf")]


def test_token_writer_reraises_writer_thread_failure(tmp_path) -> None:
    """A failed write (e.g. disk quota) must surface, not deadlock the producer."""
    path = tmp_path / "tokens.jsonl"
    writer = TokenLogprobWriter(path, queue_size=1)
    writer._handle.close()  # simulate the handle dying mid-run

    with pytest.raises(ValueError):
        for index in range(100):
            writer.submit(json.dumps("base"), index, [1], [2], [-1.0])
            time.sleep(0.001)
