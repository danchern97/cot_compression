from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from trace_generation.artifacts import (
    TRACE_SCHEMA,
    atomic_write_jsonl,
    build_manifest,
    chunk_path,
    ensure_manifest,
    summarize_run,
    validate_chunk,
    write_prepared_shard,
)
from trace_generation.config import GenerationConfig, rollout_seed
from trace_generation.generation import trace_record
from trace_generation.grading import (
    extract_last_boxed,
    extract_qa_answer,
    grade_completion,
    grade_final_response,
    math_answers_equivalent,
    normalize_qa_answer,
)
from trace_generation.preparation import (
    MATH_SUFFIX,
    QA_SOURCES,
    QA_SUFFIX,
    PreparedPrompt,
    classify_row,
    partition_prompts,
    prepare_row,
    strip_leading_user,
)


class FakeTokenizer:
    chat_template = "fake"

    def __init__(self, token_count: int = 8) -> None:
        self.token_count = token_count
        self.last_kwargs: dict[str, object] = {}

    def apply_chat_template(self, messages, **kwargs):
        self.last_kwargs = kwargs
        return f"<user>{messages[0]['content']}</user><assistant>"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(self.token_count))}


def math_row(**overrides):
    row = {
        "dataset": ["math"],
        "ground_truth": ["2"],
        "prompt": "user: What is 1 + 1?",
        "passrate": 0.5,
        "dataset_source": "math-source",
        "original_dataset": "math-source-original",
        "key": "math-key",
        "custom_id": "math-id",
    }
    row.update(overrides)
    return row


def qa_row(**overrides):
    row = {
        "dataset": ["general-quality_ref"],
        "ground_truth": ["Paris"],
        "prompt": "USER: What is the capital of France?",
        "passrate": None,
        "dataset_source": "qa-source",
        "original_dataset": sorted(QA_SOURCES)[0],
        "key": "qa-key",
        "id": "qa-id",
    }
    row.update(overrides)
    return row


def prepared_prompt(index: int, domain: str = "math") -> PreparedPrompt:
    gold = "2" if domain == "math" else "Paris"
    return PreparedPrompt(
        source_row_index=index,
        prompt_id=f"prompt-{index}",
        source_id=f"source-{index}",
        source_key=f"key-{index}",
        source_metadata_json=json.dumps({"passrate": 0.5}),
        domain=domain,
        dataset_label="math" if domain == "math" else "general-quality_ref",
        dataset_source=f"{domain}-source",
        original_dataset=f"{domain}-original",
        original_prompt="user: question",
        normalized_prompt="question",
        prepared_prompt="question\n\ninstruction",
        rendered_prompt="<user>question</user><assistant>",
        ground_truth=gold,
        prompt_tokens=8,
    )


def test_leading_role_cleanup_strips_exactly_one_marker():
    assert strip_leading_user("  UsEr:  user: keep this") == "user: keep this"
    assert strip_leading_user("a user: marker") == "a user: marker"


@pytest.mark.parametrize(
    "domain,row,suffix",
    [("math", math_row(), MATH_SUFFIX), ("qa", qa_row(), QA_SUFFIX)],
)
def test_prompt_suffixes_and_thinking_template(domain, row, suffix):
    row["outputs"] = ["upstream trace that must not be copied"]
    tokenizer = FakeTokenizer()
    prompt, reason = prepare_row(row, 7, tokenizer, max_prompt_tokens=20)
    assert reason == "eligible"
    assert prompt is not None
    assert prompt.domain == domain
    assert prompt.prepared_prompt.endswith(suffix)
    assert prompt.normalized_prompt == strip_leading_user(row["prompt"])
    assert tokenizer.last_kwargs["enable_thinking"] is True
    metadata = json.loads(prompt.source_metadata_json)
    assert metadata["passrate"] == row["passrate"]
    assert "outputs" not in metadata


def test_source_whitelist_and_wildchat_exclusion():
    for source in QA_SOURCES:
        assert classify_row(qa_row(original_dataset=source)) == ("qa", "eligible")
    domain, reason = classify_row(qa_row(original_dataset="allenai/WildChat"))
    assert domain is None
    assert reason == "qa_source_not_whitelisted"


@pytest.mark.parametrize("passrate", [0, -0.1, None, "1"])
def test_math_passrate_gate(passrate):
    assert classify_row(math_row(passrate=passrate))[1] == "math_no_prior_hit"


@pytest.mark.parametrize(
    "prompt",
    [
        "As shown in the figure, find x.",
        "Use the diagram below to answer.",
        "What is pictured here?",
        "![triangle](triangle.png) Find its area.",
        "Refer to the image and answer.",
        "See the graph below and compute the slope.",
    ],
)
def test_visual_context_rejection(prompt):
    assert classify_row(math_row(prompt=prompt))[1] == "math_missing_visual_context"


def test_nonvisual_cross_reference_is_not_rejected():
    assert classify_row(math_row(prompt="As shown in the proof, x is even.")) == (
        "math",
        "eligible",
    )


@pytest.mark.parametrize("gold", ["x" * 129, "two\nlines", "```code```", ""])
def test_qa_answer_compactness(gold):
    assert classify_row(qa_row(ground_truth=[gold]))[0] is None


def test_model_context_filtering():
    prompt, reason = prepare_row(
        math_row(), 0, FakeTokenizer(token_count=11), max_prompt_tokens=10
    )
    assert prompt is None
    assert reason == "prompt_over_context_budget"


def test_boxed_extraction_nested_multiple_and_malformed():
    assert (
        extract_last_boxed(r"first \boxed{1}; final \boxed{\frac{1}{2}}")
        == r"\frac{1}{2}"
    )
    assert extract_last_boxed(r"\boxed{1 + {2 + 3}}") == "1 + {2 + 3}"
    assert extract_last_boxed(r"\boxed{1") is None
    assert extract_last_boxed(r"\boxed 1") is None


def test_symbolic_math_equivalence_and_visible_only_grading():
    result = grade_completion(
        "math",
        r"<think>the hidden answer is wrong: \boxed{3}</think> Final: \boxed{1/2}",
        r"\frac{1}{2}",
    )
    assert result.correct
    assert result.extracted_answer == "1/2"


def test_plain_upstream_trace_can_be_audited_without_think_tags():
    result = grade_final_response(
        "math", r"Reasoning. The answer is \boxed{\sqrt{2}}.", r"\sqrt{2}"
    )
    assert result.correct
    assert result.thinking is None


@pytest.mark.parametrize(
    "gold,prediction",
    [
        (r"\sqrt{2}", r"\sqrt{2}"),
        ("20(q-1)", "20q-20"),
        ("46 / 3", r"\dfrac{46}{3}"),
        (r"10^{17}+3", "100000000000000003"),
        ("n + 1", "n+1"),
        (r"\pi", r"\pi"),
        ("1:3", r"\dfrac{1}{3}"),
    ],
)
def test_math_equivalence_preserves_latex_structure(gold, prediction):
    assert math_answers_equivalent(prediction, gold)


def test_math_equivalence_does_not_collapse_distinct_variables():
    assert not math_answers_equivalent(r"2^{k-1}", r"2^{n-1}")
    assert not math_answers_equivalent(r"\sqrt{2}", r"x=y=z=\sqrt{2}")
    assert math_answers_equivalent("2y+z=9", "2y+z-9=0")


def test_qa_extraction_is_strictly_the_final_nonempty_line():
    assert extract_qa_answer("Reasoning\nAnswer: Paris\n") == "Paris"
    assert extract_qa_answer("Answer: Paris\nextra") is None
    assert extract_qa_answer("The Answer: Paris") is None
    assert extract_qa_answer("Answer:") is None


def test_qa_unicode_normalization_and_token_f1():
    result = grade_completion(
        "qa",
        "<think>reason</think>\nAnswer: ＴＨＥ Paris…",
        "paris",
    )
    assert normalize_qa_answer("An Café—test!") == "cafétest"
    assert result.correct
    assert result.qa_token_f1 == 1.0


def test_missing_think_end_is_a_format_failure():
    result = grade_completion("qa", "Answer: Paris", "Paris")
    assert not result.correct
    assert result.extraction_status == "missing_think_end"
    assert result.extracted_answer is None


def test_seed_and_rollout_ids_are_stable_and_unique():
    config = GenerationConfig(max_output_tokens=16, max_model_len=64)
    prompt = prepared_prompt(41)
    records = [
        trace_record(
            prompt,
            index,
            config,
            completion="<think>x</think>\\boxed{2}",
            completion_tokens=4,
            finish_reason="stop",
        )
        for index in range(4)
    ]
    assert [row["seed"] for row in records] == [1501, 1502, 1503, 1504]
    assert len({row["rollout_id"] for row in records}) == 4
    assert rollout_seed(1337, 41, 3) == 1504


def test_shards_are_complete_nonoverlapping_and_stable():
    prompts = [prepared_prompt(index) for index in range(11)]
    shards = [partition_prompts(prompts, 3, index) for index in range(3)]
    flattened = [prompt.prompt_id for shard in shards for prompt in shard]
    assert flattened == [prompt.prompt_id for prompt in prompts]
    assert len(flattened) == len(set(flattened))
    assert [[p.prompt_id for p in shard] for shard in shards] == [
        [p.prompt_id for p in partition_prompts(prompts, 3, i)] for i in range(3)
    ]


def test_atomic_resume_validation_and_manifest_mismatch(tmp_path: Path):
    config = GenerationConfig(max_output_tokens=16, max_model_len=64, chunk_size=1)
    prompt = prepared_prompt(0)
    counts = {"eligible": 1, "eligible_total": 1, "eligible_math": 1, "eligible_qa": 0}
    manifest = build_manifest(config, counts)
    ensure_manifest(tmp_path, manifest)
    records = [
        trace_record(
            prompt,
            index,
            config,
            completion="<think>x</think>\\boxed{2}",
            completion_tokens=4,
            finish_reason="stop",
        )
        for index in range(4)
    ]
    path = chunk_path(tmp_path, 0, 0)
    atomic_write_jsonl(path, records)
    assert validate_chunk(path, [prompt], config) == records
    assert not list(path.parent.glob("*.tmp.*"))

    corrupted = [dict(row) for row in records]
    corrupted[0]["seed"] += 1
    atomic_write_jsonl(path, corrupted)
    with pytest.raises(ValueError, match="seed mismatch"):
        validate_chunk(path, [prompt], config)
    atomic_write_jsonl(path, records)

    incompatible = build_manifest(replace(config, base_seed=3), counts)
    with pytest.raises(ValueError, match="config hash"):
        ensure_manifest(tmp_path, incompatible)
    with pytest.raises(ValueError, match="different deterministic filters"):
        ensure_manifest(tmp_path, build_manifest(config, counts | {"extra": 1}))


def test_summary_denominators_accounting_schema_and_determinism(tmp_path: Path):
    config = GenerationConfig(max_output_tokens=16, max_model_len=64, chunk_size=2)
    math = prepared_prompt(0, "math")
    qa = prepared_prompt(1, "qa")
    counts = {"eligible": 2, "eligible_total": 2, "eligible_math": 1, "eligible_qa": 1}
    ensure_manifest(tmp_path, build_manifest(config, counts))
    write_prepared_shard(tmp_path, 0, [math, qa])

    rows = []
    for index in range(4):
        rows.append(
            trace_record(
                math,
                index,
                config,
                completion=(
                    "<think>x</think>\\boxed{2}"
                    if index == 0
                    else "<think>x</think>\\boxed{3}"
                ),
                completion_tokens=4 + index,
                finish_reason="stop",
            )
        )
        rows.append(
            trace_record(
                qa,
                index,
                config,
                completion=(
                    "<think>x</think>\nAnswer: Paris"
                    if index < 2
                    else "<think>x</think>\nAnswer: Lyon"
                ),
                completion_tokens=8 + index,
                finish_reason="length" if index == 2 else "stop",
                generation_error="worker error" if index == 3 else None,
            )
        )
    rows.sort(key=lambda row: (row["source_row_index"], row["rollout_index"]))
    path = chunk_path(tmp_path, 0, 0)
    atomic_write_jsonl(path, rows)

    first = summarize_run(tmp_path)
    first_json = (tmp_path / "summary.json").read_text()
    second = summarize_run(tmp_path)
    assert first == second
    assert first_json == (tmp_path / "summary.json").read_text()
    assert first["global"]["rollouts"] == 8
    assert first["global"]["correct_rollouts"] == 3
    assert first["global"]["rollout_pass_rate"] == pytest.approx(3 / 8)
    assert first["global"]["prompt_pass_at_4"] == 1.0
    assert first["domains"]["math"]["prompt_pass_at_4"] == 1.0
    assert first["domains"]["qa"]["correct_rollouts"] == 2
    assert first["global"]["truncated_rollouts"] == 1
    assert first["global"]["generation_errors"] == 1

    parquet = pq.read_table(tmp_path / "traces.parquet")
    assert parquet.num_rows == 8
    assert parquet.schema.names == TRACE_SCHEMA.names
    csv_text = (tmp_path / "summary_by_domain.csv").read_text()
    assert "math" in csv_text and "qa" in csv_text and "global" not in csv_text
