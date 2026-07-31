from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cot_compression.data.dolci import Message, validate_messages

IGNORE_INDEX = -100


@dataclass(frozen=True)
class TokenizedChat:
    """A rendered chat, plus the token index where supervision begins.

    Only the boundary is stored, not a full label list: labels are always
    ``[IGNORE_INDEX] * label_start + input_ids[label_start:]``, so a parallel
    list would double the pre-tokenized dataset on disk for no information.
    """

    input_ids: list[int]
    label_start: int


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


def supervised_turn_index(messages: list[Message]) -> int:
    """Index of the assistant turn that follows the final user message.

    Qwen3's template renders ``<think>`` only for assistant turns after the last
    user message; earlier ones are rewritten as ``content.split('</think>')[-1]``,
    i.e. reasoning stripped. Supervising a stripped turn would teach the model to
    answer without thinking, so only this turn is ever labelled. It is also the
    turn ``extract_answer_trace`` scores, keeping SFT and eval on the same target.
    """
    last_user = max(
        (index for index, message in enumerate(messages) if message["role"] == "user"),
        default=-1,
    )
    turn = last_user + 1
    if turn >= len(messages) or messages[turn]["role"] != "assistant":
        raise ValueError("No assistant turn follows the final user message.")
    return turn


def _render_ids(tokenizer: Any, messages: list[Message], generation: bool) -> list[int]:
    return list(
        tokenizer(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=generation,
            ),
            add_special_tokens=False,
        )["input_ids"]
    )


def tokenize_chat_for_sft(tokenizer: Any, messages: object) -> TokenizedChat:
    """Render a chat and locate the first supervised token.

    The label boundary comes from the generation prompt, never from a substring
    search: Qwen3's template rewrites whitespace inside the final ``<think>``
    block (``<think>X</think>`` renders as ``<think>\\nX\\n</think>``), so the raw
    message content is not a substring of the render at all. Rendering
    ``messages[:turn]`` with ``add_generation_prompt=True`` reproduces exactly the
    boundary the template defines, and is an exact *token* prefix of the full
    render -- which is asserted here, so a future template change fails loudly
    instead of silently corrupting labels.

    Supervision therefore runs to the end of the sequence and includes the
    trailing ``<|im_end|>``; masking it, as span-based labelling did, means the
    model is never taught to stop.

    No truncation: an over-length row is dropped by the caller rather than
    silently losing its question (left truncation) or its answer (right).
    """
    validated = validate_messages(messages)
    turn = supervised_turn_index(validated)
    prompt_ids = _render_ids(tokenizer, validated[:turn], generation=True)
    input_ids = _render_ids(tokenizer, validated, generation=False)

    label_start = len(prompt_ids)
    if input_ids[:label_start] != prompt_ids:
        raise ValueError("Generation-prompt boundary is not a token prefix.")
    if label_start >= len(input_ids):
        raise ValueError("Supervised turn produced no tokens.")
    return TokenizedChat(input_ids=input_ids, label_start=label_start)


def build_labels(input_ids: list[int], label_start: int) -> list[int]:
    return [IGNORE_INDEX] * label_start + list(input_ids[label_start:])


def pad_collate(
    examples: list[dict[str, Any]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Right-pad pre-tokenized rows into a batch.

    Rows arrive already tokenized with a ``label_start`` boundary, so this does no
    tokenizer work -- batches are length-grouped upstream, which keeps the padding
    it adds to a couple of percent.
    """
    width = max(len(example["input_ids"]) for example in examples)
    input_ids, attention_mask, labels = [], [], []
    for example in examples:
        ids = list(example["input_ids"])
        pad = width - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append(
            build_labels(ids, int(example["label_start"])) + [IGNORE_INDEX] * pad
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
