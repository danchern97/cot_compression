from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO, cast

import numpy as np
import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_compression.compression import (
    PLACEHOLDER_TOKEN,
    CompressionMethod,
    build_compression_methods,
)
from cot_compression.data.answers import (
    cot_token_ids,
    extract_answer_trace,
    tokenize_answer,
)
from cot_compression.data.chat import IGNORE_INDEX
from cot_compression.data.dolci import load_dolci_sft_data
from cot_compression.entropy import (
    compute_cot_entropies,
    entropy_cache_path,
    load_entropy_cache,
    save_entropies_npz,
)
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
    compressed_cot_tokens: int | None
    compression_ratio: float | None


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
    method_family: str
    patching: str
    patching_param: str
    compression_param: str
    samples: int
    skipped: int
    mean_logprob: float
    std_logprob: float
    sem_logprob: float
    mean_logprob_sum: float
    mean_answer_tokens: float
    mean_compression_ratio: float
    std_compression_ratio: float
    mean_compressed_cot_tokens: float


@dataclass(frozen=True)
class PreparedSample:
    sample_index: int
    sample_id: str | None
    dataset_source: str | None
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    compressed_cot_tokens: int | None
    compression_ratio: float | None
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


# Per-row: (answer logprob sum, answer token count,
# [(token_index, token_id, logprob), ...] (empty unless requested),
# answer-token entropies or None).
RowResult = tuple[float, int, list[tuple[int, int, float]], torch.Tensor | None]


def _answer_logprobs_from_logits(
    logits: torch.Tensor,
    label_tensor: torch.Tensor,
    device: torch.device,
    save_entropies: bool,
    save_token_logprobs: bool,
) -> list[RowResult]:
    logits = logits[:, :-1, :]
    shifted_labels = label_tensor[:, 1:]
    shifted_positions = torch.arange(1, label_tensor.size(1), device=device)
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shifted_labels)

    results: list[RowResult] = []
    expanded_positions = shifted_positions.expand_as(shifted_labels)
    for row_index in range(shifted_labels.size(0)):
        row_labels = shifted_labels[row_index]
        row_mask = row_labels != IGNORE_INDEX
        answer_tokens = int(row_mask.sum().item())
        if answer_tokens == 0:
            raise ValueError("No answer tokens were available for loss computation.")

        logprob_values = -nll[row_index][row_mask]
        token_logprobs: list[tuple[int, int, float]] = []
        if save_token_logprobs:
            # One bulk device-to-host transfer per tensor. Reading the same
            # values with a .item() per token instead costs one sync each, which
            # is ~31M syncs per method; the values are identical either way.
            token_logprobs = list(
                zip(
                    expanded_positions[row_index][row_mask].tolist(),
                    row_labels[row_mask].tolist(),
                    logprob_values.tolist(),
                    strict=True,
                )
            )
        answer_entropy = None
        if save_entropies:
            # Gather answer-position logits first, then softmax only those, to
            # avoid materializing a [seq, vocab] distribution over the full row.
            answer_logits = logits[row_index][row_mask]
            log_probs = F.log_softmax(answer_logits.float(), dim=-1)
            answer_entropy = -(log_probs.exp() * log_probs).sum(dim=-1).cpu()
        results.append(
            (
                float(logprob_values.sum().item()),
                answer_tokens,
                token_logprobs,
                answer_entropy,
            )
        )
    return results


def score_batch_logprobs(
    model: torch.nn.Module,
    batch: list[PreparedSample],
    pad_token_id: int,
    device: torch.device,
    save_entropies: bool,
    save_token_logprobs: bool,
) -> list[RowResult]:
    """Answer log-probs for a batch, splicing slot embeddings when present.

    A batch is homogeneous (all samples come from the same method), so either
    all carry slot embeddings (embedding-splice path) or none do (text path).
    """
    input_tensor, attention_tensor, label_tensor = _pad_batch(batch, pad_token_id, device)
    uses_slots = batch[0].slot_embeddings is not None
    with torch.no_grad():
        if uses_slots:
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
        else:
            outputs = model(input_ids=input_tensor, attention_mask=attention_tensor)
    return _answer_logprobs_from_logits(
        outputs.logits, label_tensor, device, save_entropies, save_token_logprobs
    )


def summarize_method(
    method: CompressionMethod,
    samples: list[SampleScore],
    skipped: int,
) -> MethodSummary:
    logprobs = [sample.logprob_mean for sample in samples]
    summed_logprobs = [sample.logprob_sum for sample in samples]
    token_counts = [sample.answer_tokens for sample in samples]
    ratios = [
        sample.compression_ratio
        for sample in samples
        if sample.compression_ratio is not None
    ]
    compressed = [
        sample.compressed_cot_tokens
        for sample in samples
        if sample.compressed_cot_tokens is not None
    ]
    std = statistics.stdev(logprobs) if len(logprobs) > 1 else 0.0
    return MethodSummary(
        method=method.name,
        method_family=method.method_family,
        patching=method.patching_name,
        patching_param=method.patching_param,
        compression_param=method.compression_param,
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
        mean_compression_ratio=sum(ratios) / len(ratios) if ratios else float("nan"),
        std_compression_ratio=statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
        mean_compressed_cot_tokens=sum(compressed) / len(compressed)
        if compressed
        else float("nan"),
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _eval_limit(cfg: DictConfig, dataset: Any) -> int:
    max_examples = cfg.evaluation.max_examples
    if max_examples is None:
        return len(dataset)
    return min(len(dataset), int(max_examples))


def _load_cot_entropies(
    dataset: Any,
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: DictConfig,
    device: torch.device,
    sample_indices: range,
) -> dict[int, torch.Tensor]:
    """CoT entropies for the requested samples, from cache or computed inline.

    Loads the shared cache when configured, then batch-computes any missing
    samples on the clean model so a config always has what it needs. The result
    depends only on the model and the samples, so it is loaded once and shared
    by every method that needs it.
    """
    entropies: dict[int, torch.Tensor] = {}
    cache_dir = cfg.evaluation.entropy_cache_dir
    if cache_dir is not None:
        path = entropy_cache_path(Path(str(cache_dir)), str(cfg.method.model_name))
        if path.exists():
            entropies = load_entropy_cache(path)
    missing = [index for index in sample_indices if index not in entropies]
    if missing:
        entropies.update(
            compute_cot_entropies(
                model=model,
                tokenizer=tokenizer,
                examples=dataset,
                sample_indices=missing,
                batch_size=int(cfg.evaluation.batch_size),
                max_batch_tokens=optional_int(cfg.evaluation.max_batch_tokens),
                device=device,
            )
        )
    return entropies


def evaluate_method(
    method: CompressionMethod,
    dataset: Any,
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: DictConfig,
    device: torch.device,
    cot_entropies: dict[int, torch.Tensor],
    cot_ids_cache: dict[int, np.ndarray],
    token_handle: TextIO | None,
) -> tuple[MethodSummary, list[SampleScore], dict[int, torch.Tensor]]:
    samples: list[SampleScore] = []
    answer_entropies: dict[int, torch.Tensor] = {}
    skipped = 0
    indices = range(_eval_limit(cfg, dataset))
    max_length = optional_int(cfg.evaluation.max_length)
    batch_size = int(cfg.evaluation.batch_size)
    max_batch_tokens = optional_int(cfg.evaluation.max_batch_tokens)
    seed = int(cfg.evaluation.seed)
    save_entropies = bool(cfg.evaluation.save_entropies)
    save_token_logprobs = bool(cfg.evaluation.save_token_logprobs)
    pad_token_id = int(tokenizer.pad_token_id)
    placeholder_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)
    unk_id = tokenizer.unk_token_id

    def score_batch(batch: list[PreparedSample]) -> None:
        if not batch:
            return
        for prepared, (
            logprob_sum,
            answer_tokens,
            token_logprobs,
            answer_entropy,
        ) in zip(
            batch,
            score_batch_logprobs(
                model, batch, pad_token_id, device, save_entropies, save_token_logprobs
            ),
            strict=True,
        ):
            samples.append(
                SampleScore(
                    method=method.name,
                    sample_index=prepared.sample_index,
                    sample_id=prepared.sample_id,
                    dataset_source=prepared.dataset_source,
                    answer_tokens=answer_tokens,
                    logprob_sum=logprob_sum,
                    logprob_mean=logprob_sum / answer_tokens,
                    compressed_cot_tokens=prepared.compressed_cot_tokens,
                    compression_ratio=prepared.compression_ratio,
                )
            )
            if token_handle is not None:
                # Streamed as scored rather than accumulated: holding every
                # TokenScore in memory (~31M per method) is what drove peak RSS
                # to ~20 GB. Rows are written in the same order as before.
                for token_index, token_id, logprob in token_logprobs:
                    token_handle.write(
                        json.dumps(
                            asdict(
                                TokenScore(
                                    method=method.name,
                                    sample_index=prepared.sample_index,
                                    token_index=token_index,
                                    token_id=token_id,
                                    logprob=logprob,
                                )
                            )
                        )
                        + "\n"
                    )
            if answer_entropy is not None:
                answer_entropies[prepared.sample_index] = answer_entropy

    batch: list[PreparedSample] = []
    for sample_index in indices:
        example = dataset[sample_index]
        trace = extract_answer_trace(example["messages"])
        if trace is None:
            skipped += 1
            continue

        # Tokenizing the original CoT depends only on the sample, never on the
        # method, so it is computed once and reused by every later method.
        cached_ids = cot_ids_cache.get(sample_index)
        if cached_ids is None:
            try:
                cot_ids = cot_token_ids(trace, tokenizer)
            except ValueError:
                skipped += 1
                continue
            cot_ids_cache[sample_index] = np.asarray(cot_ids, dtype=np.int32)
        else:
            cot_ids = cached_ids.tolist()

        try:
            result = method.compress(
                trace=trace,
                sample_index=sample_index,
                seed=seed,
                tokenizer=tokenizer,
                model=model,
                device=device,
                cot_entropy=cot_entropies.get(sample_index),
                cot_ids=cot_ids,
            )
        except ValueError:
            skipped += 1
            continue

        tokenized = tokenize_answer(
            tokenizer=tokenizer,
            messages=result.messages,
            answer=trace.answer,
            max_length=max_length,
        )
        if tokenized is None:
            skipped += 1
            continue

        slot_positions: list[int] = []
        if result.slot_embeddings is not None:
            if placeholder_id is None or placeholder_id == unk_id:
                raise ValueError(
                    f"Tokenizer has no usable placeholder token {PLACEHOLDER_TOKEN!r}."
                )
            slot_positions = [
                position
                for position, token_id in enumerate(tokenized.input_ids)
                if token_id == placeholder_id
            ]
            if len(slot_positions) != result.slot_embeddings.size(0):
                # Truncation by max_length cut off some placeholder slots.
                skipped += 1
                continue

        ratio = (
            result.compressed_cot_tokens / result.original_cot_tokens
            if result.original_cot_tokens
            else None
        )
        prepared = PreparedSample(
            sample_index=sample_index,
            sample_id=example.get("id"),
            dataset_source=example.get("dataset_source"),
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask,
            labels=tokenized.labels,
            compressed_cot_tokens=result.compressed_cot_tokens,
            compression_ratio=ratio,
            slot_positions=slot_positions,
            slot_embeddings=result.slot_embeddings,
        )
        if batch_would_exceed_limit(batch, prepared, max_batch_tokens):
            score_batch(batch)
            batch = []
        batch.append(prepared)
        if len(batch) >= batch_size:
            score_batch(batch)
            batch = []

    score_batch(batch)

    return summarize_method(method, samples, skipped), samples, answer_entropies


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

        artifact_dir = run_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        summary_path = artifact_dir / "summary.json"
        samples_path = artifact_dir / "samples.jsonl"
        tokens_path = artifact_dir / "tokens.jsonl"
        save_token_logprobs = bool(cfg.evaluation.save_token_logprobs)

        # Read-only and identical for every method, so built once rather than
        # reloaded from disk per method.
        model, tokenizer = build_eval_model_and_tokenizer(cfg, device)
        cot_entropies: dict[int, torch.Tensor] = {}
        if any(method.needs_entropies() for method in methods):
            cot_entropies = _load_cot_entropies(
                dataset, model, tokenizer, cfg, device, range(_eval_limit(cfg, dataset))
            )
        cot_ids_cache: dict[int, np.ndarray] = {}

        summaries = []
        sample_rows = []
        # Per method, so different methods' answer entropies never collide on
        # the same sample_index (one npz file per method).
        answer_entropies_by_method: dict[str, dict[int, torch.Tensor]] = {}
        token_handle = (
            tokens_path.open("w", encoding="utf-8") if save_token_logprobs else None
        )
        try:
            for method in methods:
                logger.info(f"Evaluating method: {method.name}")
                summary, samples, entropies = evaluate_method(
                    method=method,
                    dataset=dataset,
                    model=model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    device=device,
                    cot_entropies=cot_entropies,
                    cot_ids_cache=cot_ids_cache,
                    token_handle=token_handle,
                )
                summaries.append(summary)
                sample_rows.extend(samples)
                if entropies:
                    answer_entropies_by_method[method.name] = entropies
                logger.log_metrics(
                    {
                        f"{method.name}/mean_logprob": summary.mean_logprob,
                        f"{method.name}/mean_compression_ratio": summary.mean_compression_ratio,
                        f"{method.name}/samples": float(summary.samples),
                        f"{method.name}/skipped": float(summary.skipped),
                    },
                    step=0,
                )
        finally:
            if token_handle is not None:
                token_handle.close()

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        # Logged (small) artifacts; large npz files are written to disk only.
        # tokens.jsonl was streamed during scoring rather than written here.
        paths = [summary_path, samples_path]
        if save_token_logprobs:
            paths.append(tokens_path)
        if bool(cfg.evaluation.save_entropies):
            for name, entropies in answer_entropies_by_method.items():
                save_entropies_npz(
                    artifact_dir / f"answer_entropies__{name}.npz", entropies
                )

        for path in paths:
            logger.log_artifact(path)
    except Exception:
        logger.finish(exit_code=1)
        raise
    else:
        logger.finish()
        return summary_path
