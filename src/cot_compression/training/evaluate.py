from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_compression.compression import (
    CompressionMethod,
    EmbeddingCompressionMethod,
    build_compression_methods,
    extend_tokenizer_and_model,
)
from cot_compression.data.answers import extract_answer_trace, tokenize_answer
from cot_compression.data.chat import IGNORE_INDEX
from cot_compression.data.dolci import load_dolci_sft_data
from cot_compression.training.logging import RunLogger
from cot_compression.training.sft import parse_torch_dtype
from cot_compression.training.utils import (
    get_run_dir,
    optional_int,
    resolve_device,
    save_resolved_config,
    set_seed,
)


@dataclass(frozen=True)
class SampleScore:
    method: str
    sample_index: int
    sample_id: str | None
    dataset_source: str | None
    answer_tokens: int
    logprob_sum: float
    logprob_mean: float


@dataclass(frozen=True)
class TokenScore:
    method: str
    sample_index: int
    token_index: int
    token_id: int
    logprob: float


@dataclass(frozen=True)
class MethodSummary:
    method: str
    samples: int
    skipped: int
    mean_logprob: float
    std_logprob: float
    sem_logprob: float
    mean_logprob_sum: float
    mean_answer_tokens: float


@dataclass(frozen=True)
class PreparedSample:
    sample_index: int
    sample_id: str | None
    dataset_source: str | None
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    slot_positions: list[int] = field(default_factory=list)
    slot_embeddings: torch.Tensor | None = None


def build_eval_model_and_tokenizer(
    cfg: DictConfig, device: torch.device
) -> tuple[Any, Any]:
    tokenizer = cast(
        Any,
        AutoTokenizer.from_pretrained(
            str(cfg.method.model_name),
            use_fast=bool(cfg.method.use_fast_tokenizer),
            trust_remote_code=bool(cfg.method.trust_remote_code),
        ),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            str(cfg.method.model_name),
            torch_dtype=parse_torch_dtype(str(cfg.evaluation.torch_dtype)),
            trust_remote_code=bool(cfg.method.trust_remote_code),
        ),
    )
    return model.to(device).eval(), tokenizer


def batch_would_exceed_limit(
    batch: list[PreparedSample],
    candidate: PreparedSample,
    max_batch_tokens: int | None,
) -> bool:
    if max_batch_tokens is None or not batch:
        return False
    max_length = max(
        max(len(sample.input_ids) for sample in batch),
        len(candidate.input_ids),
    )
    return max_length * (len(batch) + 1) > max_batch_tokens


def _pad_batch(
    batch: list[PreparedSample],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(len(sample.input_ids) for sample in batch)
    input_rows = []
    attention_rows = []
    label_rows = []
    for sample in batch:
        pad = max_length - len(sample.input_ids)
        input_rows.append(sample.input_ids + [pad_token_id] * pad)
        attention_rows.append(sample.attention_mask + [0] * pad)
        label_rows.append(sample.labels + [IGNORE_INDEX] * pad)

    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(attention_rows, dtype=torch.long, device=device),
        torch.tensor(label_rows, dtype=torch.long, device=device),
    )


def _answer_logprobs_from_logits(
    logits: torch.Tensor,
    label_tensor: torch.Tensor,
    device: torch.device,
) -> list[tuple[float, list[tuple[int, int, float]]]]:
    logits = logits[:, :-1, :]
    shifted_labels = label_tensor[:, 1:]
    shifted_positions = torch.arange(1, label_tensor.size(1), device=device)
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shifted_labels)

    results = []
    expanded_positions = shifted_positions.expand_as(shifted_labels)
    for row_index in range(shifted_labels.size(0)):
        row_labels = shifted_labels[row_index]
        row_mask = row_labels != IGNORE_INDEX
        if not bool(row_mask.any().item()):
            raise ValueError("No answer tokens were available for loss computation.")

        logprob_values = -nll[row_index][row_mask]
        token_ids = row_labels[row_mask]
        positions = expanded_positions[row_index][row_mask]
        token_logprobs = [
            (int(position.item()), int(token_id.item()), float(token_logprob.item()))
            for position, token_id, token_logprob in zip(
                positions,
                token_ids,
                logprob_values,
                strict=True,
            )
        ]
        results.append((float(logprob_values.sum().item()), token_logprobs))
    return results


def compute_batch_answer_logprobs(
    model: torch.nn.Module,
    batch: list[PreparedSample],
    pad_token_id: int,
    device: torch.device,
) -> list[tuple[float, list[tuple[int, int, float]]]]:
    input_tensor, attention_tensor, label_tensor = _pad_batch(batch, pad_token_id, device)
    with torch.no_grad():
        outputs = model(input_ids=input_tensor, attention_mask=attention_tensor)
    return _answer_logprobs_from_logits(outputs.logits, label_tensor, device)


def compute_batch_answer_logprobs_from_embeds(
    model: torch.nn.Module,
    batch: list[PreparedSample],
    pad_token_id: int,
    device: torch.device,
) -> list[tuple[float, list[tuple[int, int, float]]]]:
    input_tensor, attention_tensor, label_tensor = _pad_batch(batch, pad_token_id, device)
    with torch.no_grad():
        embeds = cast(Any, model).get_input_embeddings()(input_tensor)
        for row_index, sample in enumerate(batch):
            assert sample.slot_embeddings is not None
            for position, vector in zip(
                sample.slot_positions, sample.slot_embeddings, strict=True
            ):
                embeds[row_index, position] = vector.to(
                    device=embeds.device, dtype=embeds.dtype
                )
        outputs = model(inputs_embeds=embeds, attention_mask=attention_tensor)
    return _answer_logprobs_from_logits(outputs.logits, label_tensor, device)


def summarize_method(
    method: str,
    samples: list[SampleScore],
    skipped: int,
) -> MethodSummary:
    logprobs = [sample.logprob_mean for sample in samples]
    summed_logprobs = [sample.logprob_sum for sample in samples]
    token_counts = [sample.answer_tokens for sample in samples]
    std = statistics.stdev(logprobs) if len(logprobs) > 1 else 0.0
    return MethodSummary(
        method=method,
        samples=len(samples),
        skipped=skipped,
        mean_logprob=sum(logprobs) / len(logprobs) if logprobs else float("nan"),
        std_logprob=std,
        sem_logprob=std / math.sqrt(len(logprobs)) if logprobs else float("nan"),
        mean_logprob_sum=sum(summed_logprobs) / len(summed_logprobs)
        if summed_logprobs
        else float("nan"),
        mean_answer_tokens=sum(token_counts) / len(token_counts)
        if token_counts
        else float("nan"),
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def evaluate_method(
    method: CompressionMethod,
    dataset: Any,
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: DictConfig,
    device: torch.device,
) -> tuple[MethodSummary, list[SampleScore], list[TokenScore]]:
    samples = []
    token_rows = []
    skipped = 0
    max_examples = cfg.evaluation.max_examples
    limit = (
        len(dataset) if max_examples is None else min(len(dataset), int(max_examples))
    )
    max_length = optional_int(cfg.evaluation.max_length)
    batch_size = int(cfg.evaluation.batch_size)
    max_batch_tokens = optional_int(cfg.evaluation.max_batch_tokens)
    pad_token_id = int(tokenizer.pad_token_id)

    score_fn = (
        compute_batch_answer_logprobs_from_embeds
        if method.requires_model_embeddings()
        else compute_batch_answer_logprobs
    )

    def score_batch(batch: list[PreparedSample]) -> None:
        if not batch:
            return
        batch_results = score_fn(
            model=model,
            batch=batch,
            pad_token_id=pad_token_id,
            device=device,
        )
        for prepared, (logprob_sum, token_logprobs) in zip(
            batch,
            batch_results,
            strict=True,
        ):
            answer_tokens = len(token_logprobs)
            sample = SampleScore(
                method=method.name,
                sample_index=prepared.sample_index,
                sample_id=prepared.sample_id,
                dataset_source=prepared.dataset_source,
                answer_tokens=answer_tokens,
                logprob_sum=logprob_sum,
                logprob_mean=logprob_sum / answer_tokens,
            )
            samples.append(sample)

            if bool(cfg.evaluation.save_token_logprobs):
                token_rows.extend(
                    TokenScore(
                        method=method.name,
                        sample_index=prepared.sample_index,
                        token_index=token_index,
                        token_id=token_id,
                        logprob=logprob,
                    )
                    for token_index, token_id, logprob in token_logprobs
                )

    batch = []
    for sample_index in range(limit):
        example = dataset[sample_index]
        trace = extract_answer_trace(example["messages"])
        if trace is None:
            skipped += 1
            continue

        slot_embeddings = None
        if method.requires_model_embeddings():
            embedded = cast(EmbeddingCompressionMethod, method).prepare(
                trace=trace,
                tokenizer=tokenizer,
                model=model,
                device=device,
                sample_index=sample_index,
                seed=int(cfg.evaluation.seed),
            )
            messages = embedded.messages
            slot_embeddings = embedded.slot_embeddings
        else:
            messages = method.transform(
                trace=trace,
                sample_index=sample_index,
                seed=int(cfg.evaluation.seed),
                tokenizer=tokenizer,
                model=model,
                device=device,
            )

        tokenized = tokenize_answer(
            tokenizer=tokenizer,
            messages=messages,
            answer=trace.answer,
            max_length=max_length,
        )
        if tokenized is None:
            skipped += 1
            continue

        slot_positions: list[int] = []
        if slot_embeddings is not None:
            slot_token_id = tokenizer.convert_tokens_to_ids(
                cast(EmbeddingCompressionMethod, method).slot_token
            )
            slot_positions = [
                position
                for position, token_id in enumerate(tokenized.input_ids)
                if token_id == slot_token_id
            ]
            if len(slot_positions) != slot_embeddings.size(0):
                # Truncation by max_length cut off some placeholder slots.
                skipped += 1
                continue

        prepared = PreparedSample(
            sample_index=sample_index,
            sample_id=example.get("id"),
            dataset_source=example.get("dataset_source"),
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask,
            labels=tokenized.labels,
            slot_positions=slot_positions,
            slot_embeddings=slot_embeddings,
        )
        if batch_would_exceed_limit(batch, prepared, max_batch_tokens):
            score_batch(batch)
            batch = []
        batch.append(prepared)
        if len(batch) >= batch_size:
            score_batch(batch)
            batch = []

    score_batch(batch)

    return summarize_method(method.name, samples, skipped), samples, token_rows


def evaluate_methods(cfg: DictConfig) -> Path:
    run_dir = get_run_dir(cfg)
    save_resolved_config(cfg, run_dir)
    logger = RunLogger(cfg=cfg, run_dir=run_dir)

    try:
        set_seed(seed=int(cfg.evaluation.seed), deterministic=False)
        device = resolve_device(cfg.evaluation.device)
        logger.info(f"Using device: {device}")

        methods = build_compression_methods(cfg)
        dataset = load_dolci_sft_data(cfg).eval

        summaries = []
        sample_rows = []
        token_rows = []
        for method in methods:
            logger.info(f"Evaluating method: {method.name}")
            model, tokenizer = build_eval_model_and_tokenizer(cfg, device)
            extend_tokenizer_and_model(
                tokenizer=tokenizer, model=model, methods=[method]
            )
            summary, samples, tokens = evaluate_method(
                method=method,
                dataset=dataset,
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                device=device,
            )
            summaries.append(summary)
            sample_rows.extend(samples)
            token_rows.extend(tokens)
            logger.log_metrics(
                {
                    f"{method.name}/mean_logprob": summary.mean_logprob,
                    f"{method.name}/std_logprob": summary.std_logprob,
                    f"{method.name}/samples": float(summary.samples),
                    f"{method.name}/skipped": float(summary.skipped),
                },
                step=0,
            )
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        artifact_dir = run_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        summary_path = artifact_dir / "summary.json"
        samples_path = artifact_dir / "samples.jsonl"
        tokens_path = artifact_dir / "tokens.jsonl"

        summary_payload = {
            "metric": str(cfg.evaluation.metric),
            "normalize_by_length": bool(cfg.evaluation.normalize_by_length),
            "methods": [asdict(summary) for summary in summaries],
        }
        summary_path.write_text(
            json.dumps(summary_payload, indent=2),
            encoding="utf-8",
        )
        write_jsonl(samples_path, [asdict(sample) for sample in sample_rows])
        paths = [summary_path, samples_path]
        if bool(cfg.evaluation.save_token_logprobs):
            write_jsonl(tokens_path, [asdict(token) for token in token_rows])
            paths.append(tokens_path)

        for path in paths:
            logger.log_artifact(path)
    except Exception:
        logger.finish(exit_code=1)
        raise
    else:
        logger.finish()
        return summary_path
