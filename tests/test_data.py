from __future__ import annotations

from cot_compression.data.chat import (
    IGNORE_INDEX,
    QwenChatSFTCollator,
    find_assistant_spans,
    tokenize_chat_for_sft,
)
from cot_compression.data.dolci import (
    select_deterministic_subset,
    validate_messages,
)


class FakeChatTokenizer:
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.truncation_side = "right"

    def apply_chat_template(
        self,
        messages,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert not add_generation_prompt
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
        assert not add_special_tokens
        assert truncation
        assert return_offsets_mapping
        start = max(0, len(text) - max_length) if self.truncation_side == "left" else 0
        text = text[start : start + max_length]
        return {
            "input_ids": [(ord(char) % 100) + 1 for char in text],
            "attention_mask": [1 for _ in text],
            "offset_mapping": [
                (index + start, index + start + 1) for index, _ in enumerate(text)
            ],
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

    first = select_deterministic_subset(dataset, train_size=4, eval_size=2, seed=7)
    second = select_deterministic_subset(dataset, train_size=4, eval_size=2, seed=7)

    assert first["train"]["messages"] == second["train"]["messages"]
    assert len(first["train"]) == 4
    assert len(first["eval"]) == 2


def test_chat_tokenization_masks_non_assistant_tokens() -> None:
    tokenizer = FakeChatTokenizer()
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "<think>Trace</think>\nAnswer"},
    ]

    tokenized = tokenize_chat_for_sft(
        tokenizer=tokenizer,
        messages=messages,
        max_length=128,
    )
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    assistant_start, assistant_end = find_assistant_spans(messages, rendered)[0]

    assert tokenized.labels[rendered.index("Question")] == IGNORE_INDEX
    assert tokenized.labels[assistant_start] == tokenized.input_ids[assistant_start]
    assert tokenized.labels[assistant_end - 1] == tokenized.input_ids[assistant_end - 1]


def test_chat_collator_masks_padding() -> None:
    collator = QwenChatSFTCollator(tokenizer=FakeChatTokenizer(), max_length=128)
    batch = collator(
        [
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Longer question"},
                    {"role": "assistant", "content": "Longer answer"},
                ]
            },
        ]
    )

    assert batch["input_ids"].shape[0] == 2
    assert (batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX).all()
