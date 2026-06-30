from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cot_compression.data.chat import IGNORE_INDEX, token_overlaps_spans
from cot_compression.data.dolci import Message, validate_messages

THINK_START = "<think>"
THINK_END = "</think>"


@dataclass(frozen=True)
class AnswerTrace:
    messages: list[Message]
    assistant_index: int
    trace: str
    answer: str


@dataclass(frozen=True)
class TokenizedAnswer:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    answer_token_positions: list[int]
    answer_text: str


def extract_answer_trace(messages: object) -> AnswerTrace | None:
    validated = validate_messages(messages)
    assistant_index = None
    for index in range(len(validated) - 1, -1, -1):
        if validated[index]["role"] == "assistant":
            assistant_index = index
            break
    if assistant_index is None:
        return None

    content = validated[assistant_index]["content"]
    start = content.find(THINK_START)
    end = content.find(THINK_END, start + len(THINK_START))
    if start == -1 or end == -1:
        return None

    answer_start = end + len(THINK_END)
    trace = content[start:answer_start]
    answer = content[answer_start:].lstrip()
    if not answer:
        return None

    return AnswerTrace(
        messages=validated,
        assistant_index=assistant_index,
        trace=trace,
        answer=answer,
    )


def replace_trace(trace: AnswerTrace, replacement: str) -> list[Message]:
    messages = [dict(message) for message in trace.messages]
    assistant = messages[trace.assistant_index]
    content = assistant["content"]
    content_start = content.find(THINK_START) + len(THINK_START)
    content_end = content.find(THINK_END, content_start)
    assistant["content"] = content[:content_start] + replacement + content[content_end:]
    return messages


def find_answer_span(
    messages: list[Message], rendered: str, answer: str
) -> tuple[int, int]:
    cursor = 0
    for message in messages:
        content = message["content"]
        start = rendered.find(content, cursor)
        if start == -1:
            break
        end = start + len(content)
        cursor = end
        if message["role"] != "assistant":
            continue

        answer_start = rendered.find(answer, start, end)
        if answer_start != -1:
            return answer_start, answer_start + len(answer)

    answer_start = rendered.rfind(answer)
    if answer_start != -1:
        return answer_start, answer_start + len(answer)

    raise ValueError("Could not find answer text in rendered chat.")


def tokenize_answer(
    tokenizer: Any,
    messages: list[Message],
    answer: str,
    max_length: int | None,
) -> TokenizedAnswer | None:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    try:
        answer_span = find_answer_span(messages, rendered, answer)
    except ValueError:
        return None
    tokenize_kwargs = {
        "add_special_tokens": False,
        "truncation": max_length is not None,
        "return_offsets_mapping": True,
    }
    if max_length is not None:
        tokenize_kwargs["max_length"] = max_length
    tokenized = tokenizer(rendered, **tokenize_kwargs)

    input_ids = list(tokenized["input_ids"])
    attention_mask = list(tokenized["attention_mask"])
    offsets = list(tokenized["offset_mapping"])
    labels = []
    answer_token_positions = []
    for position, (token_id, (start, end)) in enumerate(
        zip(input_ids, offsets, strict=True)
    ):
        if token_overlaps_spans(start, end, [answer_span]):
            labels.append(token_id)
            answer_token_positions.append(position)
        else:
            labels.append(IGNORE_INDEX)

    if not answer_token_positions:
        return None

    return TokenizedAnswer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        answer_token_positions=answer_token_positions,
        answer_text=answer,
    )
