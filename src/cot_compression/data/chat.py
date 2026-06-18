from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cot_compression.data.dolci import Message, validate_messages

IGNORE_INDEX = -100


@dataclass(frozen=True)
class TokenizedChat:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def find_assistant_spans(
    messages: list[Message], rendered: str
) -> list[tuple[int, int]]:
    spans = []
    cursor = 0
    for message in messages:
        content = message["content"]
        start = rendered.find(content, cursor)
        if start == -1:
            raise ValueError("Could not find message content in rendered chat.")
        end = start + len(content)
        if message["role"] == "assistant":
            spans.append((start, end))
        cursor = end
    return spans


def token_overlaps_spans(
    token_start: int,
    token_end: int,
    spans: list[tuple[int, int]],
) -> bool:
    if token_start == token_end:
        return False
    return any(
        token_start < span_end and token_end > span_start
        for span_start, span_end in spans
    )


def tokenize_chat_for_sft(
    tokenizer: Any,
    messages: object,
    max_length: int,
) -> TokenizedChat:
    validated = validate_messages(messages)
    rendered = tokenizer.apply_chat_template(
        validated,
        tokenize=False,
        add_generation_prompt=False,
    )
    assistant_spans = find_assistant_spans(validated, rendered)
    tokenized = tokenizer(
        rendered,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
    )

    input_ids = list(tokenized["input_ids"])
    attention_mask = list(tokenized["attention_mask"])
    offsets = list(tokenized["offset_mapping"])
    labels = [
        token_id if token_overlaps_spans(start, end, assistant_spans) else IGNORE_INDEX
        for token_id, (start, end) in zip(input_ids, offsets, strict=True)
    ]
    return TokenizedChat(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


class QwenChatSFTCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        tokenized = [
            tokenize_chat_for_sft(
                tokenizer=self.tokenizer,
                messages=example["messages"],
                max_length=self.max_length,
            )
            for example in examples
        ]
        max_length = max(len(example.input_ids) for example in tokenized)
        pad_id = int(self.tokenizer.pad_token_id)

        input_ids = []
        attention_mask = []
        labels = []
        for example in tokenized:
            pad = max_length - len(example.input_ids)
            input_ids.append(example.input_ids + [pad_id] * pad)
            attention_mask.append(example.attention_mask + [0] * pad)
            labels.append(example.labels + [IGNORE_INDEX] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
