from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from omegaconf import DictConfig

Message = dict[str, str]


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
