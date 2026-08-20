from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from .config import DATASET_REVISION, DATASET_SPLIT
from .grading import math_answer_is_verifiable

MATH_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
QA_SUFFIX = (
    "Please reason step by step. End your response with exactly one final line "
    'in the form "Answer: <answer>".'
)

QA_SOURCES = frozenset(
    {
        "hamishivi/virtuoussy_multi_subject_rlvr_filtered",
        "hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_"
        "filtered_qwen2_5_openthoughts2_filtered",
    }
)

FILTER_PIPELINE = (
    "require one dataset label, one non-empty ground truth, and a non-empty prompt",
    "select math or general-quality_ref rows",
    "math: require passrate > 0",
    "math: reject explicit image, figure, or diagram references",
    "math: require math_verify to parse and self-verify the ground truth",
    "qa: require a whitelisted Tulu-rewritten or multi-subject source",
    "qa: require a single-line, non-code-fenced answer of at most 128 characters",
    "strip exactly one case-insensitive leading user: marker",
    "append the domain-specific answer-format instruction",
    "render with the model chat template and thinking enabled",
    "reject prompts over model_context_length - max_output_tokens",
)

_LEADING_USER = re.compile(r"^\s*user\s*:\s*", re.IGNORECASE)
_VISUAL_CONTEXT = re.compile(
    r"(?:!\[[^]]*\]\(|<img\b|<image>|\[image\]|"
    r"as\s+(?:is\s+)?shown\s+(?:above|below)\b|"
    r"as\s+(?:is\s+)?shown\s+(?:in|on)\s+(?:the\s+)?"
    r"(?:figure|diagram|image|graph|chart)\b|"
    r"shown\s+in\s+(?:the\s+)?(?:figure|diagram|image|graph|chart)\b|"
    r"(?:figure|diagram|image|graph|chart)\s+(?:above|below)\b|"
    r"following\s+(?:figure|diagram|image|graph|chart)\b|"
    r"(?:refer\s+to|see)\s+(?:the\s+)?"
    r"(?:figure|diagram|image|graph|chart)\b|\bpictured\b)",
    re.IGNORECASE,
)

_SOURCE_METADATA_FIELDS = (
    "custom_id",
    "id",
    "key",
    "dataset",
    "original_dataset",
    "dataset_source",
    "total_rollouts",
    "total_correct_rollouts",
    "passrate",
    "constraint_type",
    "constraint",
    "conversation_hash",
    "model",
    "predicted_label",
)


@dataclass(frozen=True)
class PreparedPrompt:
    source_row_index: int
    prompt_id: str
    source_id: str | None
    source_key: str | None
    source_metadata_json: str
    domain: str
    dataset_label: str
    dataset_source: str
    original_dataset: str
    original_prompt: str
    normalized_prompt: str
    prepared_prompt: str
    rendered_prompt: str
    ground_truth: str
    prompt_tokens: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def strip_leading_user(prompt: str) -> str:
    """Remove exactly one leading ``user:`` role marker."""
    return _LEADING_USER.sub("", prompt, count=1).strip()


def has_missing_visual_context(prompt: str) -> bool:
    return bool(_VISUAL_CONTEXT.search(prompt))


def _single_string(value: object) -> str | None:
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        return None
    item = value[0]
    if not isinstance(item, str) or not item.strip():
        return None
    return item.strip()


def _qa_gold_is_compact(gold: str) -> bool:
    return (
        len(gold) <= 128 and "\n" not in gold and "\r" not in gold and "```" not in gold
    )


def _dataset_label(row: dict[str, Any]) -> str | None:
    return _single_string(row.get("dataset"))


def classify_row(row: dict[str, Any]) -> tuple[str | None, str]:
    """Return (domain, filter reason) before model-specific prompt rendering."""
    label = _dataset_label(row)
    gold = _single_string(row.get("ground_truth"))
    prompt = row.get("prompt")
    if label is None:
        return None, "invalid_dataset_label"
    if gold is None:
        return None, "invalid_ground_truth"
    if not isinstance(prompt, str) or not strip_leading_user(prompt):
        return None, "invalid_prompt"

    if label == "math":
        passrate = row.get("passrate")
        if not isinstance(passrate, (int, float)) or passrate <= 0:
            return None, "math_no_prior_hit"
        if has_missing_visual_context(prompt):
            return None, "math_missing_visual_context"
        if not math_answer_is_verifiable(gold):
            return None, "math_unverifiable_gold"
        return "math", "eligible"

    if label == "general-quality_ref":
        if row.get("original_dataset") not in QA_SOURCES:
            return None, "qa_source_not_whitelisted"
        if not _qa_gold_is_compact(gold):
            return None, "qa_noncompact_gold"
        return "qa", "eligible"

    return None, "domain_not_selected"


def _prompt_id(dataset_revision: str, split: str, source_row_index: int) -> str:
    raw = f"{dataset_revision}:{split}:{source_row_index}".encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def _source_id(row: dict[str, Any]) -> str | None:
    for key in ("custom_id", "id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _source_metadata(row: dict[str, Any]) -> str:
    """Preserve source metadata without copying upstream traces/token arrays."""
    metadata = {key: row[key] for key in _SOURCE_METADATA_FIELDS if key in row}
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str)


def prepare_row(
    row: dict[str, Any],
    source_row_index: int,
    tokenizer: Any,
    max_prompt_tokens: int,
    dataset_revision: str = DATASET_REVISION,
    split: str = DATASET_SPLIT,
) -> tuple[PreparedPrompt | None, str]:
    domain, reason = classify_row(row)
    if domain is None:
        return None, reason

    original_prompt = str(row["prompt"])
    normalized_prompt = strip_leading_user(original_prompt)
    suffix = MATH_SUFFIX if domain == "math" else QA_SUFFIX
    prepared_prompt = f"{normalized_prompt}\n\n{suffix}"
    messages = [{"role": "user", "content": prepared_prompt}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        encoded = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    except Exception:
        return None, "chat_template_error"
    if len(encoded) > max_prompt_tokens:
        return None, "prompt_over_context_budget"

    return (
        PreparedPrompt(
            source_row_index=source_row_index,
            prompt_id=_prompt_id(dataset_revision, split, source_row_index),
            source_id=_source_id(row),
            source_key=row.get("key") if isinstance(row.get("key"), str) else None,
            source_metadata_json=_source_metadata(row),
            domain=domain,
            dataset_label=str(_dataset_label(row)),
            dataset_source=str(row.get("dataset_source") or ""),
            original_dataset=str(row.get("original_dataset") or ""),
            original_prompt=original_prompt,
            normalized_prompt=normalized_prompt,
            prepared_prompt=prepared_prompt,
            rendered_prompt=str(rendered),
            ground_truth=str(_single_string(row.get("ground_truth"))),
            prompt_tokens=len(encoded),
        ),
        "eligible",
    )


def prepare_prompts(
    rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    max_prompt_tokens: int,
    max_examples: int | None = None,
    dataset_revision: str = DATASET_REVISION,
    split: str = DATASET_SPLIT,
) -> tuple[list[PreparedPrompt], dict[str, int]]:
    prepared: list[PreparedPrompt] = []
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        prompt, reason = prepare_row(
            row,
            index,
            tokenizer,
            max_prompt_tokens,
            dataset_revision,
            split,
        )
        counts[reason] += 1
        if prompt is not None:
            prepared.append(prompt)
            if max_examples is not None and len(prepared) >= max_examples:
                counts["stopped_at_max_examples"] += 1
                break
    counts["eligible_total"] = len(prepared)
    counts["eligible_math"] = sum(p.domain == "math" for p in prepared)
    counts["eligible_qa"] = sum(p.domain == "qa" for p in prepared)
    return prepared, dict(sorted(counts.items()))


def partition_prompts(
    prompts: list[PreparedPrompt], num_shards: int, shard_index: int
) -> list[PreparedPrompt]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    start = len(prompts) * shard_index // num_shards
    stop = len(prompts) * (shard_index + 1) // num_shards
    return prompts[start:stop]
