from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from .config import DATASET_REVISION, DATASET_SPLIT

MATH_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
CODE_SUFFIX = (
    "Reason through the problem, then provide the final Python solution in exactly "
    "one fenced `python` code block."
)

DOMAIN_BY_LABEL = {
    "math": "math",
    "general-quality_ref": "general_quality_ref",
    "ifeval": "ifeval",
    "code": "code",
    "code_stdio": "code_stdio",
    "general-quality": "general_quality",
}

SUFFIX_BY_DOMAIN = {
    "math": MATH_SUFFIX,
    "code": CODE_SUFFIX,
    "code_stdio": CODE_SUFFIX,
}

FILTER_PIPELINE = (
    "require one supported dataset label and a non-empty prompt",
    "select all six dataset families for compression trace generation",
    "preserve every non-empty ground-truth entry without using it for eligibility",
    "strip exactly one case-insensitive leading user: marker",
    "append a boxed-answer instruction to math prompts",
    "append a single-Python-code-block instruction to code prompts",
    "leave QA, IFEval, and general-quality prompt semantics unchanged",
    "render with the model chat template and thinking enabled",
    "reject only prompts over model_context_length - max_output_tokens",
)

_LEADING_USER = re.compile(r"^\s*user\s*:\s*", re.IGNORECASE)
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
    ground_truths: tuple[str, ...]
    prompt_tokens: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ground_truths"] = list(self.ground_truths)
        return payload

    @property
    def ground_truth(self) -> str | None:
        return self.ground_truths[0].strip() if self.ground_truths else None


def strip_leading_user(prompt: str) -> str:
    """Remove exactly one leading ``user:`` role marker."""
    return _LEADING_USER.sub("", prompt, count=1).strip()


def _single_string(value: object) -> str | None:
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        return None
    item = value[0]
    if not isinstance(item, str) or not item.strip():
        return None
    return item.strip()


def _ground_truths(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item.strip())


def _dataset_label(row: dict[str, Any]) -> str | None:
    return _single_string(row.get("dataset"))


def classify_row(row: dict[str, Any]) -> tuple[str | None, str]:
    """Return (domain, filter reason) before model-specific prompt rendering."""
    label = _dataset_label(row)
    prompt = row.get("prompt")
    if label is None:
        return None, "invalid_dataset_label"
    domain = DOMAIN_BY_LABEL.get(label)
    if domain is None:
        return None, "domain_not_selected"
    if not isinstance(prompt, str) or not strip_leading_user(prompt):
        return None, "invalid_prompt"
    return domain, "eligible"


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
    suffix = SUFFIX_BY_DOMAIN.get(domain)
    prepared_prompt = (
        f"{normalized_prompt}\n\n{suffix}" if suffix else normalized_prompt
    )
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

    ground_truths = _ground_truths(row.get("ground_truth"))
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
            ground_truths=ground_truths,
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
    allowed_domains: set[str] | frozenset[str] | None = None,
) -> tuple[list[PreparedPrompt], dict[str, int]]:
    prepared: list[PreparedPrompt] = []
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        domain, _ = classify_row(row)
        if (
            domain is not None
            and allowed_domains is not None
            and domain not in allowed_domains
        ):
            counts["domain_not_requested"] += 1
            continue
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
    for domain in DOMAIN_BY_LABEL.values():
        counts[f"eligible_{domain}"] = sum(p.domain == domain for p in prepared)
    return prepared, dict(sorted(counts.items()))


def partition_prompts(
    prompts: list[PreparedPrompt], num_shards: int, shard_index: int
) -> list[PreparedPrompt]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    # Source rows are grouped by dataset family. Modulo partitioning gives every
    # GPU a comparable domain mix while remaining stable across reruns.
    return [
        prompt
        for prompt in prompts
        if prompt.source_row_index % num_shards == shard_index
    ]
