from __future__ import annotations

import re

import pytest

from cot_compression.data.answers import (
    extract_answer_trace,
    find_answer_span,
    tokenize_answer,
)
from cot_compression.data.chat import (
    IGNORE_INDEX,
    build_labels,
    pad_collate,
    supervised_turn_index,
    tokenize_chat_for_sft,
)
from cot_compression.data.dolci import (
    has_valid_messages,
    select_deterministic_subset,
    validate_messages,
)

THINK_BLOCK = re.compile(r"<think>\s*(.*?)\s*</think>\s*", re.DOTALL)


class FakeChatTokenizer:
    """Char-level tokenizer whose chat template reproduces Qwen3's two rewrites.

    Both are load-bearing and both broke the previous span-search labelling:

    1. Whitespace inside the supervised turn's ``<think>`` block is normalized,
       so the raw message content is *not* a substring of the render.
    2. Assistant turns at or before the final user message have their reasoning
       stripped entirely (``content.split('</think>')[-1]``), so a re-render of a
       message prefix is not a prefix of the full render for those turns.
    """

    eos_token = "<eos>"

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.truncation_side = "right"
        self.added_tokens: list[str] = []

    def __len__(self) -> int:
        return 100 + len(self.added_tokens)

    def add_tokens(self, tokens) -> int:
        self.added_tokens.extend(tokens)
        return len(tokens)

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        assert not tokenize
        last_user = max(
            (i for i, m in enumerate(messages) if m["role"] == "user"), default=-1
        )
        parts = []
        for index, message in enumerate(messages):
            content = message["content"]
            if message["role"] == "assistant":
                if index > last_user:
                    content = THINK_BLOCK.sub(
                        lambda m: f"<think>\n{m.group(1)}\n</think>\n\n", content
                    )
                else:
                    content = content.split("</think>")[-1]
            parts.append(f"<|im_start|>{message['role']}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: int | None = None,
        truncation: bool = False,
        return_offsets_mapping: bool = False,
    ):
        assert not add_special_tokens
        start = 0
        if truncation and max_length is not None:
            start = (
                max(0, len(text) - max_length) if self.truncation_side == "left" else 0
            )
            text = text[start : start + max_length]
        input_ids = []
        offsets = []
        index = 0
        added = {token: 100 + i for i, token in enumerate(self.added_tokens)}
        while index < len(text):
            match = next(
                (token for token in self.added_tokens if text.startswith(token, index)),
                None,
            )
            if match is not None:
                input_ids.append(added[match])
                offsets.append((start + index, start + index + len(match)))
                index += len(match)
            else:
                input_ids.append((ord(text[index]) % 99) + 1)
                offsets.append((start + index, start + index + 1))
                index += 1
        return {
            "input_ids": input_ids,
            "attention_mask": [1 for _ in input_ids],
            "offset_mapping": offsets,
        }


def test_validate_messages_requires_assistant() -> None:
    messages = validate_messages(
        [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "<think>Trace</think>\nAnswer"},
        ]
    )

    assert messages[1]["content"].startswith("<think>")


def test_select_deterministic_subset() -> None:
    from datasets import Dataset

    dataset = Dataset.from_list(
        [{"messages": [{"role": "assistant", "content": str(i)}]} for i in range(10)]
    )

    first = select_deterministic_subset(
        dataset,
        train_size=4,
        eval_size=2,
        test_size=3,
        seed=7,
    )
    second = select_deterministic_subset(
        dataset,
        train_size=4,
        eval_size=2,
        test_size=3,
        seed=7,
    )

    assert first["train"]["messages"] == second["train"]["messages"]
    assert len(first["train"]) == 4
    assert len(first["eval"]) == 2
    assert len(first["test"]) == 3


def test_invalid_dolci_messages_can_be_filtered() -> None:
    assert has_valid_messages(
        {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        }
    )
    assert not has_valid_messages({"messages": [{"role": "assistant", "content": ""}]})


def test_labels_survive_template_think_rewrite() -> None:
    """The regression that broke 448/500 real rows: the old span search looked for
    the raw content in the render, but the template rewrites the think block."""
    tokenizer = FakeChatTokenizer()
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "<think>Trace</think>\nAnswer"},
    ]
    rendered = tokenizer.apply_chat_template(messages)
    assert messages[1]["content"] not in rendered  # the exact condition that failed
    assert "<think>\nTrace\n</think>" in rendered

    chat = tokenize_chat_for_sft(tokenizer, messages)
    labels = build_labels(chat.input_ids, chat.label_start)

    # The fake emits one token per character, so token indices are char indices.
    prompt = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True)
    assert chat.label_start == len(prompt)
    assert set(labels[: chat.label_start]) == {IGNORE_INDEX}
    assert labels[chat.label_start :] == chat.input_ids[chat.label_start :]

    supervised = rendered[chat.label_start :]
    assert supervised.startswith("<think>\nTrace")
    assert supervised.endswith("<|im_end|>\n"), "im_end must be supervised"


def test_multi_turn_supervises_only_the_final_assistant_turn() -> None:
    """Earlier assistant turns are rendered think-stripped, so supervising them
    would teach the model to answer without reasoning."""
    tokenizer = FakeChatTokenizer()
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "<think>Hidden</think>\nEarly"},
        {"role": "user", "content": "Second"},
        {"role": "assistant", "content": "<think>Shown</think>\nLate"},
    ]
    rendered = tokenizer.apply_chat_template(messages)
    assert "Hidden" not in rendered and "Shown" in rendered

    assert supervised_turn_index(messages) == 3
    chat = tokenize_chat_for_sft(tokenizer, messages)
    supervised = rendered[chat.label_start :]
    assert "Late" in supervised
    assert "Early" not in supervised


def test_tokenize_chat_rejects_a_broken_prefix_boundary() -> None:
    """A template that rewrites *earlier* turns breaks the prefix property; that
    must raise rather than silently misalign every label."""

    class DriftingTokenizer(FakeChatTokenizer):
        """Rewrites an earlier turn in the full render only, so the generation
        prompt is no longer a prefix of it."""

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=False
        ) -> str:
            text = super().apply_chat_template(
                messages, tokenize, add_generation_prompt
            )
            return text if add_generation_prompt else text.replace("Question", "Q")

    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "<think>Trace</think>\nAnswer"},
    ]
    tokenize_chat_for_sft(FakeChatTokenizer(), messages)  # sanity: honest one passes

    with pytest.raises(ValueError, match="token prefix"):
        tokenize_chat_for_sft(DriftingTokenizer(), messages)


def test_supervised_turn_index_requires_a_trailing_assistant_turn() -> None:
    with pytest.raises(ValueError, match="No assistant turn"):
        supervised_turn_index(
            [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Reply"},
                {"role": "user", "content": "Dangling"},
            ]
        )


def test_pad_collate_masks_padding() -> None:
    batch = pad_collate(
        [
            {"input_ids": [5, 6, 7], "label_start": 1},
            {"input_ids": [5, 6, 7, 8, 9], "label_start": 2},
        ],
        pad_token_id=0,
    )

    assert batch["input_ids"].shape == (2, 5)
    assert (batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX).all()
    assert batch["labels"][0].tolist() == [
        IGNORE_INDEX,
        6,
        7,
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]
    assert batch["labels"][1].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 7, 8, 9]


def test_extract_answer_trace() -> None:
    trace = extract_answer_trace(
        [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "<think>Trace</think>\nAnswer"},
        ]
    )

    assert trace is not None
    assert trace.trace == "<think>Trace</think>"
    assert trace.answer == "Answer"
    assert extract_answer_trace([{"role": "assistant", "content": "No trace"}]) is None
    assert (
        extract_answer_trace([{"role": "assistant", "content": "<think>x</think>"}])
        is None
    )


def test_tokenize_answer_masks_trace_and_scores_answer() -> None:
    tokenizer = FakeChatTokenizer()
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "<think>Trace</think>\nAnswer"},
    ]
    trace = extract_answer_trace(messages)
    assert trace is not None

    tokenized = tokenize_answer(
        tokenizer=tokenizer,
        messages=trace.messages,
        answer=trace.answer,
        max_length=128,
    )

    assert tokenized is not None
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    assert tokenized.labels[rendered.index("Trace")] == IGNORE_INDEX
    assert tokenized.labels[rendered.index("Answer")] != IGNORE_INDEX


def test_find_answer_span_skips_an_answer_quoted_inside_the_think_block() -> None:
    """The CoT often quotes the final answer; the span must be the real one, at the end.

    A forward search returned the copy inside the think block, so a full-trace render
    scored reasoning tokens as the answer -- 35 of the 489 encoder eval rows. Covers
    the verbatim path production takes, the template-normalized fallback, and an
    earlier assistant turn that says the same thing.
    """
    verbatim = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "<think>\nI will reply 42.\n</think>\n\n42."},
    ]
    rendered = (
        "<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n"
        + verbatim[1]["content"]
        + "<|im_end|>\n"
    )
    start, end = find_answer_span(verbatim, rendered, "42.")
    assert rendered[start:end] == "42." and start > rendered.index("</think>")

    tokenizer = FakeChatTokenizer()
    for messages in (
        [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "<think>Say Same.</think>\nSame."},
        ],
        [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "<think>t</think>\nSame."},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "<think>Say Same.</think>\nSame."},
        ],
    ):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False)
        start, end = find_answer_span(messages, rendered, "Same.")
        assert rendered[start:end] == "Same."
        assert start > rendered.rindex("</think>"), "the real answer, not a quote"


def test_find_answer_span_handles_template_normalized_think_tags() -> None:
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "<think>Trace</think> Answer"},
    ]
    rendered = (
        "<|im_start|>user\nQuestion<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nTrace\n</think>\n\n Answer<|im_end|>\n"
    )

    start, end = find_answer_span(messages, rendered, "Answer")

    assert rendered[start:end] == "Answer"
