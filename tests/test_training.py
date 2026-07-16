from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import Dataset
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from cot_compression.data.dolci import DolciSFTData
from cot_compression.training.evaluate import evaluate_methods
from cot_compression.training.sft import train_sft

_PATCHING = {
    "uniform": {"patch_size": 4},
    "random": {"max_exponent": 6},
    "entropy": {"percentile": 80.0},
}


def _eval_cfg(tmp_path, methods, **evaluation_overrides):
    evaluation = {
        "seed": 7,
        "device": "cpu",
        "torch_dtype": "float32",
        "max_length": None,
        "batch_size": 2,
        "max_batch_tokens": 512,
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


class TinySFTModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.embedding = nn.Embedding(512, 8)
        self.lm_head = nn.Linear(8, 128)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def gradient_checkpointing_enable(self) -> None:
        return None

    def resize_token_embeddings(self, size: int, mean_resizing: bool = False) -> None:
        del mean_resizing
        self.lm_head = nn.Linear(8, size)

    def save_pretrained(self, path) -> None:
        torch.save(self.state_dict(), Path(path) / "model.pt")

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None
    ):
        del attention_mask
        embeds = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        logits = self.lm_head(embeds)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


def test_plain_sft_training_step(monkeypatch, tmp_path) -> None:
    examples = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "<think>Trace</think>\nAnswer"},
                ],
                "dataset_source": "test",
                "id": "0",
            }
        ]
    )

    monkeypatch.setattr(
        "cot_compression.training.sft.AutoTokenizer",
        FakeSFTTokenizer,
    )
    monkeypatch.setattr(
        "cot_compression.training.sft.AutoModelForCausalLM",
        TinySFTModel,
    )
    monkeypatch.setattr(
        "cot_compression.training.sft.load_dolci_sft_data",
        lambda cfg: DolciSFTData(train=examples, eval=examples, test=examples),
    )

    cfg = OmegaConf.create(
        {
            "paths": {"run_dir": str(tmp_path / "sft")},
            "data": {"prepared_dir": str(tmp_path / "data")},
            "method": {
                "model_name": "tiny",
                "trust_remote_code": False,
                "use_fast_tokenizer": True,
            },
            "training": {
                "seed": 7,
                "device": "cpu",
                "deterministic": False,
                "torch_dtype": "float32",
                "num_train_epochs": 1,
                "max_steps": 1,
                "batch_size": 1,
                "eval_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "max_length": 128,
                "warmup_ratio": 0.0,
                "eval_interval": 1,
                "eval_iters": 1,
                "checkpoint_interval": 1,
                "log_interval": 1,
                "gradient_clip": 1.0,
                "gradient_checkpointing": False,
                "compile": False,
                "resume_from_checkpoint": None,
            },
            "optim": {
                "lr": 0.001,
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
            },
        }
    )

    checkpoint = train_sft(cfg)

    assert checkpoint.exists()
    assert (checkpoint / "training_state.pt").exists()
    assert (checkpoint / "model.pt").exists()


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

    assert set(methods) == {"base", "random_uniform_ps4"}
    assert summary["metric"] == "answer_logprob"
    assert methods["base"]["samples"] == 1
    assert methods["base"]["skipped"] == 1
    assert methods["base"]["method_family"] == "base"
    # base = full CoT, ratio 1.0; random compresses below 1.
    assert methods["base"]["mean_compression_ratio"] == 1.0
    assert methods["random_uniform_ps4"]["samples"] == 1
    assert methods["random_uniform_ps4"]["skipped"] == 1
    assert methods["random_uniform_ps4"]["mean_compression_ratio"] < 1.0
    assert {row["method"] for row in sample_rows} == {"base", "random_uniform_ps4"}
    assert all(row["answer_tokens"] > 0 for row in sample_rows)
    assert all("compression_ratio" in row for row in sample_rows)
    assert {row["method"] for row in token_rows} == {"base", "random_uniform_ps4"}
    assert all("logprob" in row for row in token_rows)


def test_answer_loss_evaluation_embedding_methods_smoke(monkeypatch, tmp_path) -> None:
    examples = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "<think>Trace here</think> Answer"},
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
            "simple_mean": {"patching": "entropy"},
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
    assert set(methods) == {"simple_mean_entropy_p80", "entropy_weighted_mean_uniform_ps4"}
    assert methods["simple_mean_entropy_p80"]["samples"] == 1
    assert methods["entropy_weighted_mean_uniform_ps4"]["samples"] == 1
    assert methods["simple_mean_entropy_p80"]["patching"] == "entropy"
    assert {row["method"] for row in sample_rows} == {
        "simple_mean_entropy_p80",
        "entropy_weighted_mean_uniform_ps4",
    }
    assert all(row["answer_tokens"] > 0 for row in sample_rows)
    assert all(row["compression_ratio"] is not None for row in sample_rows)
    # One answer-entropy artifact per method (no cross-method collision).
    assert (summary_path.with_name("answer_entropies__simple_mean_entropy_p80.npz")).exists()
    assert (
        summary_path.with_name("answer_entropies__entropy_weighted_mean_uniform_ps4.npz")
    ).exists()
