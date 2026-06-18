from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import Dataset
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from cot_compression.data.dolci import DolciSFTData
from cot_compression.training.sft import train_sft


class FakeSFTTokenizer:
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.truncation_side = "right"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def apply_chat_template(
        self,
        messages,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        return "".join(
            f"<|{message['role']}|>\n{message['content']}\n" for message in messages
        )

    def __call__(
        self,
        text: str,
        add_special_tokens: bool,
        max_length: int,
        truncation: bool,
        return_offsets_mapping: bool,
    ):
        start = max(0, len(text) - max_length) if self.truncation_side == "left" else 0
        text = text[start : start + max_length]
        return {
            "input_ids": [(ord(char) % 64) + 1 for char in text],
            "attention_mask": [1 for _ in text],
            "offset_mapping": [
                (index + start, index + start + 1) for index, _ in enumerate(text)
            ],
        }

    def save_pretrained(self, path) -> None:
        Path(path, "tokenizer_config.json").write_text("{}", encoding="utf-8")


class TinySFTModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.embedding = nn.Embedding(128, 8)
        self.lm_head = nn.Linear(8, 128)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def gradient_checkpointing_enable(self) -> None:
        return None

    def save_pretrained(self, path) -> None:
        torch.save(self.state_dict(), Path(path) / "model.pt")

    def forward(self, input_ids, attention_mask, labels):
        del attention_mask
        logits = self.lm_head(self.embedding(input_ids))
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
        lambda cfg: DolciSFTData(train=examples, eval=examples),
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
