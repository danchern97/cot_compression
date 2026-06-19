from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_compression.compression import (
    CompressionMethod,
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
    resolve_device,
    save_resolved_config,
    set_seed,
)


@dataclass(frozen=True)
class SampleLoss:
    method: str
    sample_index: int
    sample_id: str | None
    dataset_source: str | None
    answer_tokens: int
    loss_sum: float
    loss_mean: float


@dataclass(frozen=True)
class TokenLoss:
    method: str
    sample_index: int
    token_index: int
    token_id: int
    loss: float


@dataclass(frozen=True)
class MethodSummary:
    method: str
    samples: int
    skipped: int
    mean_loss: float
    std_loss: float
    sem_loss: float
    mean_loss_sum: float
    mean_answer_tokens: float


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


def compute_answer_losses(
    model: torch.nn.Module,
    input_ids: list[int],
    attention_mask: list[int],
    labels: list[int],
    device: torch.device,
) -> tuple[float, list[tuple[int, int, float]]]:
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_tensor = torch.tensor([attention_mask], dtype=torch.long, device=device)
    label_tensor = torch.tensor([labels], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_tensor, attention_mask=attention_tensor)
        logits = outputs.logits[:, :-1, :]
        shifted_labels = label_tensor[:, 1:]
        shifted_positions = torch.arange(1, label_tensor.size(1), device=device)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            shifted_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view_as(shifted_labels)

    mask = shifted_labels != IGNORE_INDEX
    if not bool(mask.any().item()):
        raise ValueError("No answer tokens were available for loss computation.")

    loss_values = loss[mask]
    token_ids = shifted_labels[mask]
    positions = shifted_positions.expand_as(shifted_labels)[mask]
    token_losses = [
        (int(position.item()), int(token_id.item()), float(token_loss.item()))
        for position, token_id, token_loss in zip(
            positions,
            token_ids,
            loss_values,
            strict=True,
        )
    ]
    return float(loss_values.sum().item()), token_losses


def summarize_method(
    method: str,
    samples: list[SampleLoss],
    skipped: int,
) -> MethodSummary:
    losses = [sample.loss_mean for sample in samples]
    summed_losses = [sample.loss_sum for sample in samples]
    token_counts = [sample.answer_tokens for sample in samples]
    std = statistics.stdev(losses) if len(losses) > 1 else 0.0
    return MethodSummary(
        method=method,
        samples=len(samples),
        skipped=skipped,
        mean_loss=sum(losses) / len(losses) if losses else float("nan"),
        std_loss=std,
        sem_loss=std / math.sqrt(len(losses)) if losses else float("nan"),
        mean_loss_sum=sum(summed_losses) / len(summed_losses)
        if summed_losses
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
) -> tuple[MethodSummary, list[SampleLoss], list[TokenLoss]]:
    samples = []
    token_rows = []
    skipped = 0
    max_examples = cfg.evaluation.max_examples
    limit = (
        len(dataset) if max_examples is None else min(len(dataset), int(max_examples))
    )

    for sample_index in range(limit):
        example = dataset[sample_index]
        trace = extract_answer_trace(example["messages"])
        if trace is None:
            skipped += 1
            continue

        messages = method.transform(
            trace=trace,
            sample_index=sample_index,
            seed=int(cfg.evaluation.seed),
        )
        tokenized = tokenize_answer(
            tokenizer=tokenizer,
            messages=messages,
            answer=trace.answer,
            max_length=int(cfg.evaluation.max_length),
        )
        if tokenized is None:
            skipped += 1
            continue

        loss_sum, token_losses = compute_answer_losses(
            model=model,
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask,
            labels=tokenized.labels,
            device=device,
        )
        answer_tokens = len(token_losses)
        sample = SampleLoss(
            method=method.name,
            sample_index=sample_index,
            sample_id=example.get("id"),
            dataset_source=example.get("dataset_source"),
            answer_tokens=answer_tokens,
            loss_sum=loss_sum,
            loss_mean=loss_sum / answer_tokens,
        )
        samples.append(sample)

        if bool(cfg.evaluation.save_token_losses):
            token_rows.extend(
                TokenLoss(
                    method=method.name,
                    sample_index=sample_index,
                    token_index=token_index,
                    token_id=token_id,
                    loss=loss,
                )
                for token_index, token_id, loss in token_losses
            )

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
                    f"{method.name}/mean_loss": summary.mean_loss,
                    f"{method.name}/std_loss": summary.std_loss,
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
        if bool(cfg.evaluation.save_token_losses):
            write_jsonl(tokens_path, [asdict(token) for token in token_rows])
            paths.append(tokens_path)

        for path in paths:
            logger.log_artifact(path)
        return summary_path
    finally:
        logger.finish()
