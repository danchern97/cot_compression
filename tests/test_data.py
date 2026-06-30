from __future__ import annotations

from omegaconf import OmegaConf

from cot_compression.compression import (
    RandomCompressionMethod,
    build_compression_methods,
    extend_tokenizer_and_model,
)
from cot_compression.data.answers import (
    extract_answer_trace,
    find_answer_span,
    tokenize_answer,
)
from cot_compression.data.chat import (
    IGNORE_INDEX,
    QwenChatSFTCollator,
    find_assistant_spans,
    tokenize_chat_for_sft,
)
from cot_compression.data.dolci import (
    has_valid_messages,
    select_deterministic_subset,
    validate_messages,
)


class FakeChatTokenizer:
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


class FakeResizeModel:
    def __init__(self) -> None:
        self.resize_calls: list[tuple[int, bool]] = []

    def resize_token_embeddings(self, size: int, mean_resizing: bool) -> None:
        self.resize_calls.append((size, mean_resizing))


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


def test_random_method_is_deterministic_and_extends_vocab() -> None:
    tokenizer = FakeChatTokenizer()
    model = FakeResizeModel()
    cfg = OmegaConf.create(
        {
            "evaluation": {
                "methods": {
                    "enabled": ["base", "random"],
                    "random": {"abstract_vocab_size": 4, "abstract_length": 3},
                }
            }
        }
    )
    methods = build_compression_methods(cfg)
    extend_tokenizer_and_model(tokenizer=tokenizer, model=model, methods=methods)

    trace = extract_answer_trace(
        [{"role": "assistant", "content": "<think>Trace</think>\nAnswer"}]
    )
    assert trace is not None
    random_method = next(
        method for method in methods if isinstance(method, RandomCompressionMethod)
    )
    first = random_method.transform(trace, sample_index=5, seed=13)
    second = random_method.transform(trace, sample_index=5, seed=13)
    third = random_method.transform(trace, sample_index=6, seed=13)
    content = first[0]["content"]

    assert first == second
    assert first != third
    assert content.startswith("<think>")
    assert "</think>\nAnswer" in content
    assert "Trace" not in content
    assert len(tokenizer.added_tokens) == 4
    assert model.resize_calls == [(104, False)]
