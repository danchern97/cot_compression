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
from trace_generation.fulfillment import (
    FULFILLMENT_PIPELINE,
    next_rollout_index,
    prepare_fulfillment_prompts,
    source_rollout_indices,
    summarize_fulfillment_coverage,
)
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
    CODE_SUFFIX,
    DOMAIN_BY_LABEL,
    MATH_SUFFIX,
    PreparedPrompt,
    classify_row,
    partition_prompts,
    prepare_prompts,
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
        "original_dataset": "hamishivi/new-wildchat-english-general_filtered",
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
        ground_truths=(gold,),
        prompt_tokens=8,
    )


def test_leading_role_cleanup_strips_exactly_one_marker():
    assert strip_leading_user("  UsEr:  user: keep this") == "user: keep this"
    assert strip_leading_user("a user: marker") == "a user: marker"


@pytest.mark.parametrize(
    "domain,row,suffix",
    [
        ("math", math_row(), MATH_SUFFIX),
        (
            "code",
            math_row(dataset=["code"], prompt="user: Implement f(x)."),
            CODE_SUFFIX,
        ),
        (
            "code_stdio",
            math_row(dataset=["code_stdio"], prompt="user: Solve this program."),
            CODE_SUFFIX,
        ),
    ],
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


@pytest.mark.parametrize(
    "label,domain",
    list(DOMAIN_BY_LABEL.items()),
)
def test_all_dataset_families_are_selected(label, domain):
    assert classify_row(math_row(dataset=[label])) == (domain, "eligible")


@pytest.mark.parametrize(
    "label",
    ["general-quality_ref", "ifeval", "general-quality"],
)
def test_unconstrained_families_do_not_get_an_answer_format_suffix(label):
    row = qa_row(dataset=[label], prompt="USER: Preserve my requested format.")
    prompt, reason = prepare_row(row, 3, FakeTokenizer(), max_prompt_tokens=20)
    assert reason == "eligible"
    assert prompt is not None
    assert prompt.prepared_prompt == "Preserve my requested format."


@pytest.mark.parametrize(
    "passrate,prompt",
    [
        (0, "As shown in the figure, find x."),
        (-0.1, "Use the diagram below to answer."),
        (None, "What is pictured here?"),
        ("1", "![triangle](triangle.png) Find its area."),
    ],
)
def test_passrate_and_missing_visual_context_are_metadata_not_filters(passrate, prompt):
    assert classify_row(math_row(passrate=passrate, prompt=prompt)) == (
        "math",
        "eligible",
    )


def test_wildchat_and_multiline_long_references_are_included():
    row = qa_row(
        original_dataset="hamishivi/new-wildchat-english-general_filtered",
        ground_truth=["A long reference.\n" + "x" * 500],
    )
    assert classify_row(row) == ("general_quality_ref", "eligible")


def test_ground_truths_do_not_control_eligibility_and_are_all_preserved():
    tokenizer = FakeTokenizer()
    row = qa_row(ground_truth=[" first ", "", "second\nline", None])
    prompt, reason = prepare_row(row, 4, tokenizer, max_prompt_tokens=20)
    assert reason == "eligible"
    assert prompt is not None
    assert prompt.ground_truth == "first"
    assert prompt.ground_truths == (" first ", "second\nline")

    no_gold, reason = prepare_row(
        qa_row(ground_truth=[]), 5, tokenizer, max_prompt_tokens=20
    )
    assert reason == "eligible"
    assert no_gold is not None
    assert no_gold.ground_truth is None
    assert no_gold.ground_truths == ()


def test_only_invalid_or_unknown_prompts_are_filtered_before_length_check():
    assert classify_row(math_row(dataset=["unknown"])) == (
        None,
        "domain_not_selected",
    )
    assert classify_row(math_row(prompt="user:   ")) == (None, "invalid_prompt")


def test_filter_counts_cover_every_compression_domain():
    rows = [math_row(dataset=[label]) for label in DOMAIN_BY_LABEL]
    prompts, counts = prepare_prompts(rows, FakeTokenizer(), max_prompt_tokens=20)
    assert len(prompts) == len(DOMAIN_BY_LABEL)
    assert counts["eligible_total"] == len(DOMAIN_BY_LABEL)
    for domain in DOMAIN_BY_LABEL.values():
        assert counts[f"eligible_{domain}"] == 1


def test_generation_domain_selection_is_nonoverlapping():
    rows = [math_row(dataset=[label]) for label in DOMAIN_BY_LABEL]
    math, math_counts = prepare_prompts(
        rows,
        FakeTokenizer(),
        max_prompt_tokens=20,
        allowed_domains={"math"},
    )
    nonmath, nonmath_counts = prepare_prompts(
        rows,
        FakeTokenizer(),
        max_prompt_tokens=20,
        allowed_domains=set(DOMAIN_BY_LABEL.values()) - {"math"},
    )
    assert [prompt.domain for prompt in math] == ["math"]
    assert {prompt.domain for prompt in nonmath} == set(DOMAIN_BY_LABEL.values()) - {
        "math"
    }
    assert math_counts["domain_not_requested"] == len(rows) - 1
    assert nonmath_counts["domain_not_requested"] == 1


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


def test_ungraded_domain_retains_structure_without_inventing_a_score():
    result = grade_completion(
        "ifeval", "<think>follow constraints</think>Visible answer", None
    )
    assert result.extraction_status == "ungraded"
    assert result.correct is None
    assert result.score is None


def test_natural_eos_completeness_and_domain_caps():
    config = GenerationConfig(max_model_len=40_960)
    math = trace_record(
        prepared_prompt(1, "math"),
        0,
        config,
        completion="<think>x</think>\\boxed{2}",
        completion_tokens=4,
        finish_reason="stop",
    )
    code = trace_record(
        prepared_prompt(2, "code"),
        0,
        config,
        completion="<think>x</think>```python\npass\n```",
        completion_tokens=8,
        finish_reason="length",
    )
    assert math["complete"] is True
    assert math["ended_by_eos"] is True
    assert math["requested_max_tokens"] == 32_768
    assert code["complete"] is False
    assert code["ended_by_eos"] is False
    assert code["requested_max_tokens"] == 16_384
    assert code["correct"] is None


def test_seed_and_rollout_ids_are_stable_and_unique():
    config = GenerationConfig(
        max_output_tokens=32,
        code_max_output_tokens=16,
        default_max_output_tokens=8,
        max_model_len=64,
    )
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
        for index in range(2)
    ]
    assert [row["seed"] for row in records] == [43_321, 43_322]
    assert len({row["rollout_id"] for row in records}) == 2
    assert rollout_seed(1337, 41, 3) == 43_324


def test_four_attempt_fulfillment_indices_and_base_hash_compatibility():
    base = GenerationConfig(
        max_output_tokens=32,
        code_max_output_tokens=16,
        default_max_output_tokens=8,
        max_model_len=64,
    )
    fulfillment = replace(
        base,
        num_rollouts=4,
        rollout_index_start=2,
        code_max_output_tokens=24,
    )
    assert "rollout_index_start" not in base.payload()
    assert list(fulfillment.rollout_indices) == [2, 3, 4, 5]
    prompt = prepared_prompt(41)
    records = [
        trace_record(
            prompt,
            index,
            fulfillment,
            completion="<think>x</think>\\boxed{2}",
            completion_tokens=4,
            finish_reason="stop",
        )
        for index in fulfillment.rollout_indices
    ]
    assert [row["rollout_id"] for row in records] == [
        "prompt-41:r2",
        "prompt-41:r3",
        "prompt-41:r4",
        "prompt-41:r5",
    ]
    assert len({row["seed"] for row in records}) == 4


def _write_source_run(run_dir: Path) -> tuple[GenerationConfig, list[PreparedPrompt]]:
    config = GenerationConfig(
        domains=("math",),
        max_output_tokens=32,
        code_max_output_tokens=16,
        default_max_output_tokens=8,
        max_model_len=64,
        chunk_size=2,
    )
    prompts = [prepared_prompt(0), prepared_prompt(1)]
    counts = {"eligible": 2, "eligible_total": 2, "eligible_math": 2}
    ensure_manifest(run_dir, build_manifest(config, counts))
    write_prepared_shard(run_dir, 0, prompts)
    rows = [
        trace_record(
            prompts[0],
            0,
            config,
            completion="<think>unfinished",
            completion_tokens=32,
            finish_reason="length",
        ),
        trace_record(
            prompts[0],
            1,
            config,
            completion="no thinking tags",
            completion_tokens=4,
            finish_reason="stop",
        ),
        trace_record(
            prompts[1],
            0,
            config,
            completion="<think>x</think>\\boxed{2}",
            completion_tokens=4,
            finish_reason="stop",
        ),
        trace_record(
            prompts[1],
            1,
            config,
            completion="<think>unfinished",
            completion_tokens=32,
            finish_reason="length",
        ),
    ]
    rows.sort(key=lambda row: (row["source_row_index"], row["rollout_index"]))
    atomic_write_jsonl(chunk_path(run_dir, 0, 0), rows)
    summarize_run(run_dir)
    return config, prompts


def test_fulfillment_selects_only_zero_complete_and_reports_recovery(
    tmp_path: Path,
):
    source_dir = tmp_path / "source"
    base_config, prompts = _write_source_run(source_dir)
    assert next_rollout_index([source_dir]) == 2
    assert source_rollout_indices([source_dir]) == {0, 1}

    config = replace(base_config, num_rollouts=4, rollout_index_start=2)
    selected, counts, provenance = prepare_fulfillment_prompts(
        [source_dir], FakeTokenizer(), config
    )
    assert [prompt.prompt_id for prompt in selected] == [prompts[0].prompt_id]
    assert counts["source_prompts"] == 2
    assert counts["source_prompts_with_complete"] == 1
    assert counts["source_prompts_without_complete"] == 1
    assert counts["eligible_total"] == 1

    run_dir = tmp_path / "fulfillment"
    manifest = build_manifest(
        config,
        counts,
        filter_pipeline=FULFILLMENT_PIPELINE,
        extra={"fulfillment": provenance},
    )
    ensure_manifest(run_dir, manifest)
    write_prepared_shard(run_dir, 0, selected)
    rows = []
    for index in config.rollout_indices:
        rows.append(
            trace_record(
                selected[0],
                index,
                config,
                completion=(
                    "<think>x</think>\\boxed{2}" if index == 3 else "<think>unfinished"
                ),
                completion_tokens=4 if index == 3 else 32,
                finish_reason="stop" if index == 3 else "length",
            )
        )
    path = chunk_path(run_dir, 0, 0)
    atomic_write_jsonl(path, rows)
    assert validate_chunk(path, selected, config) == rows
    summarize_run(run_dir)
    coverage = summarize_fulfillment_coverage(run_dir)
    assert coverage["global"]["target_prompts"] == 1
    assert coverage["global"]["attempted_prompts"] == 1
    assert coverage["global"]["recovered_prompts"] == 1
    assert coverage["global"]["prompts_still_without_complete"] == 0
    assert coverage["global"]["combined_prompt_completion_rate"] == 1.0
    assert (run_dir / "coverage_by_domain.csv").exists()

    next_config = replace(config, rollout_index_start=6)
    assert next_rollout_index([source_dir, run_dir]) == 6
    remaining, next_counts, _ = prepare_fulfillment_prompts(
        [source_dir, run_dir], FakeTokenizer(), next_config
    )
    assert remaining == []
    assert next_counts["source_prompts"] == 2
    assert next_counts["source_prompts_with_complete"] == 2


def test_shards_are_complete_nonoverlapping_and_stable():
    prompts = [prepared_prompt(index) for index in range(11)]
    shards = [partition_prompts(prompts, 3, index) for index in range(3)]
    flattened = [prompt.prompt_id for shard in shards for prompt in shard]
    assert sorted(flattened) == sorted(prompt.prompt_id for prompt in prompts)
    assert len(flattened) == len(set(flattened))
    assert [[p.prompt_id for p in shard] for shard in shards] == [
        [p.prompt_id for p in partition_prompts(prompts, 3, i)] for i in range(3)
    ]


def test_atomic_resume_validation_and_manifest_mismatch(tmp_path: Path):
    config = GenerationConfig(
        max_output_tokens=32,
        code_max_output_tokens=16,
        default_max_output_tokens=8,
        max_model_len=64,
        chunk_size=1,
    )
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
        for index in range(2)
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
    config = GenerationConfig(
        max_output_tokens=32,
        code_max_output_tokens=16,
        default_max_output_tokens=8,
        max_model_len=64,
        chunk_size=2,
    )
    math = prepared_prompt(0, "math")
    qa = prepared_prompt(1, "qa")
    counts = {"eligible": 2, "eligible_total": 2, "eligible_math": 1, "eligible_qa": 1}
    ensure_manifest(tmp_path, build_manifest(config, counts))
    write_prepared_shard(tmp_path, 0, [math, qa])

    rows = []
    for index in range(2):
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
                    if index == 0
                    else "<think>x</think>\nAnswer: Lyon"
                ),
                completion_tokens=8 + index,
                finish_reason="length" if index == 1 else "stop",
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
    assert first["global"]["rollouts"] == 4
    assert first["global"]["correct_rollouts"] == 2
    assert first["global"]["rollout_pass_rate"] == pytest.approx(2 / 4)
    assert first["global"]["prompt_pass_at_k"] == 1.0
    assert first["domains"]["math"]["prompt_pass_at_k"] == 1.0
    assert first["domains"]["qa"]["correct_rollouts"] == 1
    assert first["global"]["complete_rollouts"] == 3
    assert first["global"]["prompt_complete_at_k"] == 1.0
    assert first["global"]["truncated_rollouts"] == 1
    assert first["global"]["generation_errors"] == 0

    parquet = pq.read_table(tmp_path / "traces.parquet")
    assert parquet.num_rows == 4
    assert parquet.schema.names == TRACE_SCHEMA.names
    csv_text = (tmp_path / "summary_by_domain.csv").read_text()
    assert "math" in csv_text and "qa" in csv_text and "global" not in csv_text
