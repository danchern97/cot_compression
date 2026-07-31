from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_dataset,
    load_from_disk,
)
from omegaconf import DictConfig

Message = dict[str, str]

# int32, not the datasets default int64: the train split is ~6.3e9 tokens, so
# the width choice is the difference between a 25 GB and a 50 GB cache.
TOKENIZED_FEATURES = Features(
    {
        "input_ids": Sequence(Value("int32")),
        "label_start": Value("int32"),
        "length": Value("int32"),
        "id": Value("string"),
    }
)


@dataclass(frozen=True)
class DolciSFTData:
    train: Dataset
    eval: Dataset
    test: Dataset


def validate_messages(messages: object) -> list[Message]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("Dolci examples must contain a non-empty messages list.")

    validated = []
    has_assistant = False
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be a dictionary.")

        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError("Each message must have a non-empty string role.")
        if not isinstance(content, str) or not content:
            raise ValueError("Each message must have non-empty string content.")

        if role == "assistant":
            has_assistant = True
        validated.append({"role": role, "content": content})

    if not has_assistant:
        raise ValueError("Dolci SFT examples must contain an assistant message.")
    return validated


def has_valid_messages(example: dict[str, Any]) -> bool:
    try:
        validate_messages(example.get("messages"))
    except ValueError:
        return False
    return True


def select_deterministic_subset(
    dataset: Dataset,
    train_size: int,
    eval_size: int,
    test_size: int,
    seed: int,
) -> DatasetDict:
    required = train_size + eval_size + test_size
    if len(dataset) < required:
        raise ValueError(
            f"Dataset has {len(dataset)} rows, but {required} are required."
        )

    shuffled = dataset.shuffle(seed=seed)
    subset = shuffled.select(range(required))
    return DatasetDict(
        {
            "train": subset.select(range(train_size)),
            "eval": subset.select(range(train_size, train_size + eval_size)),
            "test": subset.select(range(train_size + eval_size, required)),
        }
    )


def _keep_sft_columns(example: dict[str, Any]) -> dict[str, Any]:
    messages = validate_messages(example["messages"])
    return {
        "messages": messages,
        "dataset_source": example.get("dataset_source"),
        "id": example.get("id"),
    }


def _expected_split_sizes(cfg: DictConfig) -> dict[str, int]:
    return {
        "train": int(cfg.data.train_size),
        "eval": int(cfg.data.eval_size),
        "test": int(cfg.data.test_size),
    }


def _validate_split_sizes(dataset_dict: DatasetDict, cfg: DictConfig) -> None:
    expected = _expected_split_sizes(cfg)
    actual = {split: len(dataset_dict[split]) for split in dataset_dict}
    if actual != expected:
        raise ValueError(
            "Prepared Dolci split sizes do not match config. "
            f"Expected {expected}, found {actual}."
        )


def _prepare_subset(cfg: DictConfig) -> DatasetDict:
    source = load_dataset(
        str(cfg.data.source_name),
        name=cfg.data.get("source_config"),
        split=str(cfg.data.source_split),
    )
    dataset = source.filter(has_valid_messages, desc="Filtering invalid Dolci rows")
    subset = select_deterministic_subset(
        dataset=dataset,
        train_size=int(cfg.data.train_size),
        eval_size=int(cfg.data.eval_size),
        test_size=int(cfg.data.test_size),
        seed=int(cfg.data.seed),
    )
    return subset.map(
        _keep_sft_columns,
        remove_columns=subset["train"].column_names,
        desc="Validating Dolci messages",
    )


def load_dolci_sft_data(cfg: DictConfig) -> DolciSFTData:
    prepared_dir = Path(cfg.data.prepared_dir)
    if prepared_dir.exists():
        loaded = load_from_disk(str(prepared_dir))
        if not isinstance(loaded, DatasetDict):
            raise ValueError(f"Expected a DatasetDict at {prepared_dir}.")
        dataset_dict = loaded
        _validate_split_sizes(dataset_dict, cfg)
    else:
        prepared_dir.parent.mkdir(parents=True, exist_ok=True)
        dataset_dict = _prepare_subset(cfg)
        _validate_split_sizes(dataset_dict, cfg)
        dataset_dict.save_to_disk(str(prepared_dir))

    return DolciSFTData(
        train=dataset_dict["train"],
        eval=dataset_dict["eval"],
        test=dataset_dict["test"],
    )


def tokenized_dir(cfg: DictConfig) -> Path:
    """Cache location, keyed by dataset *and* model.

    The label boundary is produced by the model's chat template, so a cache built
    with one tokenizer is meaningless for another. Keying on the model slug makes
    a mismatch impossible rather than merely unlikely.
    """
    slug = str(cfg.method.model_name).replace("/", "__")
    return Path(cfg.paths.tokenized_dir) / str(cfg.data.name) / slug


def _tokenize_row(example: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    from cot_compression.data.chat import tokenize_chat_for_sft

    try:
        chat = tokenize_chat_for_sft(tokenizer, example["messages"])
    except ValueError:
        # Filtered out below. Rows reach here only if the template boundary check
        # fails or the supervised turn is empty, both of which are data faults.
        return {"input_ids": [], "label_start": 0, "length": 0, "id": example["id"]}
    return {
        "input_ids": chat.input_ids,
        "label_start": chat.label_start,
        "length": len(chat.input_ids),
        "id": example["id"],
    }


def build_tokenized_sft_data(cfg: DictConfig) -> Path:
    """Pre-tokenize train+eval once, to disk.

    Not a throughput optimization -- 37 examples/s/proc already outruns four
    H100s. It exists so every sequence length is known *before* training, which
    is what makes length-bucketed batching, deterministic drop-of-overlong-rows
    and an exact `total_steps` (hence a resumable LR schedule) possible.

    Deliberately a separate workflow rather than lazy work inside the SFT path:
    under `tasks_per_node=4` four ranks would race on the same cache and each
    fork `num_proc` workers on a GPU node.

    No length filtering here -- `training.max_length` stays a training-time knob,
    so changing it does not invalidate the cache.
    """
    from transformers import AutoTokenizer

    out = tokenized_dir(cfg)
    if out.exists():
        return out

    tokenizer = AutoTokenizer.from_pretrained(
        str(cfg.method.model_name),
        use_fast=bool(cfg.method.use_fast_tokenizer),
        trust_remote_code=bool(cfg.method.trust_remote_code),
    )
    raw = load_dolci_sft_data(cfg)
    source = DatasetDict({"train": raw.train, "eval": raw.eval})
    tokenized = source.map(
        _tokenize_row,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=source["train"].column_names,
        features=TOKENIZED_FEATURES,
        num_proc=int(cfg.data.tokenize_num_proc),
        # Rows average ~10.5k tokens and are buffered as Python ints before the
        # arrow flush, so the default batch of 1000 would hold ~300 MB per worker
        # -- times 60 workers, enough to matter.
        writer_batch_size=200,
        desc="Tokenizing Dolci for SFT",
    ).filter(
        lambda example: example["length"] > 0,
        num_proc=int(cfg.data.tokenize_num_proc),
        desc="Dropping rows that failed the template boundary check",
    )

    staging = out.with_name(out.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.parent.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(staging))
    os.replace(staging, out)
    return out


def load_tokenized_sft_data(cfg: DictConfig) -> DatasetDict:
    out = tokenized_dir(cfg)
    if not out.exists():
        raise FileNotFoundError(
            f"No pre-tokenized cache at {out}. Build it once with:\n"
            f"  uv run python scripts/run.py tokenize "
            f"data={cfg.data.name} method={cfg.method.model_name}"
        )
    loaded = load_from_disk(str(out))
    if not isinstance(loaded, DatasetDict):
        raise ValueError(f"Expected a DatasetDict at {out}.")
    return loaded
