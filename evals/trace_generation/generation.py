from __future__ import annotations

import json
from typing import Any

from .config import GenerationConfig, rollout_seed
from .grading import grade_completion
from .preparation import PreparedPrompt


def _stop_reason(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return repr(value)


def _finish_reason(value: object) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", value)
    return str(enum_value).lower()


def trace_record(
    prompt: PreparedPrompt,
    rollout_index: int,
    config: GenerationConfig,
    *,
    completion: str,
    completion_tokens: int,
    finish_reason: str | None,
    stop_reason: object = None,
    generation_error: str | None = None,
) -> dict[str, Any]:
    truncated = finish_reason == "length"
    grade = grade_completion(
        prompt.domain,
        completion,
        prompt.ground_truth,
        generation_error=generation_error,
        truncated=truncated,
    )
    return {
        "config_hash": config.config_hash,
        **prompt.to_dict(),
        "rollout_id": f"{prompt.prompt_id}:r{rollout_index}",
        "rollout_index": rollout_index,
        "seed": rollout_seed(config.base_seed, prompt.source_row_index, rollout_index),
        "messages": [
            {"role": "user", "content": prompt.prepared_prompt},
            {"role": "assistant", "content": completion},
        ],
        "completion_raw": completion,
        "thinking": grade.thinking,
        "final_response": grade.final_response,
        "extracted_answer": grade.extracted_answer,
        "score": grade.score,
        "correct": grade.correct,
        "qa_token_f1": grade.qa_token_f1,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "stop_reason": _stop_reason(stop_reason),
        "truncated": truncated,
        "extraction_status": grade.extraction_status,
        "generation_error": generation_error,
    }


def build_engine(config: GenerationConfig) -> Any:
    from vllm import LLM

    return LLM(
        model=config.model,
        revision=config.model_revision,
        dtype=config.dtype,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        trust_remote_code=False,
        seed=config.base_seed,
    )


def generate_chunk(
    engine: Any,
    prompts: list[PreparedPrompt],
    config: GenerationConfig,
    *,
    use_tqdm: bool = True,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    requests: list[tuple[PreparedPrompt, int]] = [
        (prompt, rollout_index)
        for prompt in prompts
        for rollout_index in range(config.num_rollouts)
    ]
    rendered = [prompt.rendered_prompt for prompt, _ in requests]
    params = [
        SamplingParams(
            n=1,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            min_p=config.min_p,
            presence_penalty=config.presence_penalty,
            max_tokens=config.max_output_tokens,
            seed=rollout_seed(config.base_seed, prompt.source_row_index, rollout_index),
        )
        for prompt, rollout_index in requests
    ]
    outputs = engine.generate(rendered, sampling_params=params, use_tqdm=use_tqdm)
    if len(outputs) != len(requests):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} results for {len(requests)} requests"
        )

    records = []
    for (prompt, rollout_index), request_output in zip(requests, outputs, strict=True):
        if not request_output.outputs:
            records.append(
                trace_record(
                    prompt,
                    rollout_index,
                    config,
                    completion="",
                    completion_tokens=0,
                    finish_reason=None,
                    generation_error="vLLM returned no completion",
                )
            )
            continue
        output = request_output.outputs[0]
        records.append(
            trace_record(
                prompt,
                rollout_index,
                config,
                completion=str(output.text),
                completion_tokens=len(output.token_ids),
                finish_reason=_finish_reason(output.finish_reason),
                stop_reason=output.stop_reason,
            )
        )
    return records
