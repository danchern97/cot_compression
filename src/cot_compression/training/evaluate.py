from __future__ import annotations

import json
import math
import queue
import statistics
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_compression.compression import (
    PLACEHOLDER_TOKEN,
    CompressionMethod,
    build_compression_methods,
    compressed_messages,
)
from cot_compression.data.answers import (
    cot_token_ids,
    extract_answer_trace,
    prefix_token_ids,
    tokenize_answer,
)
from cot_compression.data.chat import IGNORE_INDEX
from cot_compression.data.dolci import load_dolci_sft_data
from cot_compression.data.dolci_traces import load_trace_data
from cot_compression.signals import (
    compute_cot_signals,
    load_signal_cache,
    save_entropies_npz,
    signal_cache_path,
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
    # Sum of squared per-token answer log-probs for this sample; summed across
    # samples it yields the method's token-level std without keeping every token.
    logprob_sumsq: float
    compressed_cot_tokens: int | None
    compression_ratio: float | None


@dataclass(frozen=True)
class TokenScore:
    """Schema of one tokens.jsonl row. See TokenLogprobWriter for the writer."""

    method: str
    sample_index: int
    token_index: int
    token_id: int
    logprob: float


# Emits exactly what json.dumps(asdict(TokenScore(...))) does: same key order,
# same ", "/": " separators, and repr() floats (which is what json.dumps uses for
# finite floats). Measured at ~1.4 us/row against ~15.4 us for asdict + dumps --
# roughly 8 minutes of main-thread serialization per 31M-row method.
_TOKEN_ROW = (
    '{"method": %s, "sample_index": %d, "token_index": %d,'
    ' "token_id": %d, "logprob": %r}\n'
)

# (method_json, sample_index, token_indices, token_ids, logprobs) for one sample.
TokenRows = tuple[str, int, list[int], list[int], list[float]]


class TokenLogprobWriter:
    """Writes tokens.jsonl from a background thread, preserving row order.

    Serialization is the single largest main-thread cost in evaluation, and it is
    pure Python work that does not need the GPU. One consumer thread over a
    bounded queue keeps it off the critical path (the forward's device syncs
    release the GIL) while emitting rows in submission order, which
    reporting/sweep_stats.py relies on. The queue is bounded so a slow writer
    applies backpressure instead of accumulating rows in memory -- holding them
    all is what previously drove peak RSS to ~20 GB.
    """

    def __init__(self, path: Path, queue_size: int = 256) -> None:
        self._handle = path.open("w", encoding="utf-8")
        self._queue: queue.Queue[TokenRows | None] = queue.Queue(maxsize=queue_size)
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._consume, name="token-logprob-writer", daemon=True
        )
        self._thread.start()

    def submit(
        self,
        method_json: str,
        sample_index: int,
        token_indices: list[int],
        token_ids: list[int],
        logprobs: list[float],
    ) -> None:
        """Queue one sample's rows, re-raising any error from the writer thread."""
        if self._error is not None:
            raise self._error
        self._queue.put((method_json, sample_index, token_indices, token_ids, logprobs))

    def _consume(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if self._error is not None:
                # Already failed: keep draining so a producer blocked on a full
                # queue makes progress and observes the error via submit/close
                # rather than deadlocking.
                continue
            try:
                self._write(item)
            except BaseException as exc:  # noqa: BLE001 - surfaced on the main thread
                self._error = exc

    def _write(self, item: TokenRows) -> None:
        method_json, sample_index, token_indices, token_ids, logprobs = item
        if all(map(math.isfinite, logprobs)):
            self._handle.write(
                "".join(
                    _TOKEN_ROW % (method_json, sample_index, index, token_id, logprob)
                    for index, token_id, logprob in zip(
                        token_indices, token_ids, logprobs, strict=True
                    )
                )
            )
            return
        # repr() spells non-finite floats "-inf"/"nan", which json.loads rejects,
        # while json.dumps spells them "-Infinity"/"NaN". Rare enough (it needs a
        # token the model assigned zero probability) to pay the slow path for.
        method = json.loads(method_json)
        self._handle.write(
            "".join(
                json.dumps(
                    asdict(
                        TokenScore(
                            method=method,
                            sample_index=sample_index,
                            token_index=index,
                            token_id=token_id,
                            logprob=logprob,
                        )
                    )
                )
                + "\n"
                for index, token_id, logprob in zip(
                    token_indices, token_ids, logprobs, strict=True
                )
            )
        )

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        self._handle.close()
        if self._error is not None:
            raise self._error


@dataclass(frozen=True)
class MethodSummary:
    method: str
    method_family: str
    patching: str
    patching_param: str
    compression_param: str
    samples: int
    skipped: int
    # Per-sample statistics: mean/median over samples of each sample's mean answer
    # log-prob, the across-sample std, and the SEM. The SEM is the error bar for
    # comparing methods -- samples are the independent unit there.
    mean_logprob: float
    median_logprob: float
    std_logprob: float
    sem_logprob: float
    mean_logprob_sum: float
    median_logprob_sum: float
    mean_answer_tokens: float
    median_answer_tokens: float
    # Token-level statistics: pooled over every answer token across all samples
    # (total_answer_tokens is the N behind them). std_token_logprob is the spread
    # of the per-token log-prob distribution -- the right quantity for a per-token
    # density/CI band on a single method. (For a *between-method* significance
    # test use sem_logprob instead: a token-count SEM would treat correlated
    # within-sample tokens as independent.)
    total_answer_tokens: int
    mean_token_logprob: float
    median_token_logprob: float
    std_token_logprob: float
    mean_compression_ratio: float
    median_compression_ratio: float
    std_compression_ratio: float
    mean_compressed_cot_tokens: float
    median_compressed_cot_tokens: float


@dataclass
class PreparedSample:
    """One tokenized sample on its way to the GPU.

    Built in a DataLoader worker and pickled to the main process, which is the
    only side holding the model and so the only side that can fill in
    ``slot_embeddings``. Not frozen for exactly that reason.
    """

    sample_index: int
    sample_id: str | None
    dataset_source: str | None
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    # The tokenized original CoT, kept as the int32 array the cross-method
    # cot_ids cache already stores, so the main process can seed that cache from
    # what the worker returns rather than re-tokenizing for the next method.
    cot_ids: np.ndarray
    compressed_cot_tokens: int | None
    compression_ratio: float | None
    slot_positions: list[int] = field(default_factory=list)
    slot_embeddings: torch.Tensor | None = None
    # Populated only when the method declares requires_prefix(): rendering the
    # prompt through the chat template is not free at eval scale.
    prefix_ids: list[int] | None = None


class PrepDataset(TorchDataset["PreparedSample | None"]):
    """Per-sample CPU prep, run in DataLoader worker processes.

    Holds no model and touches no CUDA state, so it is safe under the ``fork``
    start method the DataLoader uses. It performs the three steps that dominate
    evaluation's CPU time -- tokenizing the CoT, rewriting the think block as
    placeholders, and tokenizing the rendered chat -- which previously ran on the
    main thread between forward passes and left the GPU idle most of the time.

    ``span_counts`` carries the slot count for strategies whose boundaries depend
    on a per-token signal (entropy or surprisal). Those are placed by a quantile
    or cumsum, floating-point reductions whose last bits differ between CPU and
    GPU, so the main process resolves them on the eval device and passes the
    counts down rather than letting each worker recompute them and silently shift
    a boundary.

    ``__getitem__`` returns None for a sample that cannot be prepared; the main
    process counts those exactly where the serial loop used to.
    """

    def __init__(
        self,
        dataset: Any,
        tokenizer: Any,
        method: CompressionMethod,
        *,
        limit: int,
        seed: int,
        max_length: int | None,
        max_position_embeddings: int | None,
        span_counts: dict[int, int] | None,
        cot_ids_cache: dict[int, np.ndarray],
        placeholder_id: int,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.method = method
        self.limit = limit
        self.seed = seed
        self.max_length = max_length
        self.max_position_embeddings = max_position_embeddings
        self.span_counts = span_counts
        self.cot_ids_cache = cot_ids_cache
        self.placeholder_id = placeholder_id

    def __len__(self) -> int:
        return self.limit

    def _num_slots(self, sample_index: int, num_cot_tokens: int) -> int | None:
        """Slot count from the main process, or planned here when independent."""
        if self.span_counts is not None:
            return self.span_counts[sample_index]
        plan = self.method.plan(
            num_cot_tokens, sample_index, self.seed, None, torch.device("cpu")
        )
        return plan.num_slots

    def __getitem__(self, index: int) -> PreparedSample | None:
        # Named `index` to match Dataset.__getitem__; it is a dataset sample index.
        sample_index = index
        example = self.dataset[sample_index]
        trace = extract_answer_trace(example["messages"])
        if trace is None:
            return None

        cached_ids = self.cot_ids_cache.get(sample_index)
        if cached_ids is None:
            try:
                cot_ids = cot_token_ids(trace, self.tokenizer)
            except ValueError:
                return None
            cot_array = np.asarray(cot_ids, dtype=np.int32)
        else:
            cot_array = cached_ids
            cot_ids = cached_ids.tolist()

        # Keyed on the CoT token count, which is the same for every method on a
        # given sample, so an over-length sample is dropped identically for base
        # and each compressed variant -- the paired comparison stays over one
        # population. (The base rendering is the longest, so if its CoT alone
        # overflows the window nothing downstream can fit either.) preflight_...
        # already raised for this on cached samples; this backstop covers the
        # uncached path. Checked before tokenizing to skip the doomed work.
        if (
            self.max_position_embeddings is not None
            and len(cot_ids) > self.max_position_embeddings
        ):
            return None

        if self.span_counts is not None and sample_index not in self.span_counts:
            # No signal values for this sample: the serial path raised inside
            # compress() and skipped it, so skip it here too.
            return None
        num_slots = self._num_slots(sample_index, len(cot_ids))

        tokenized = tokenize_answer(
            tokenizer=self.tokenizer,
            messages=compressed_messages(trace, num_slots),
            answer=trace.answer,
            max_length=self.max_length,
        )
        if tokenized is None:
            return None

        slot_positions: list[int] = []
        if num_slots is not None:
            slot_positions = [
                position
                for position, token_id in enumerate(tokenized.input_ids)
                if token_id == self.placeholder_id
            ]
            if len(slot_positions) != num_slots:
                # Truncation by max_length cut off some placeholder slots.
                return None

        return PreparedSample(
            sample_index=sample_index,
            sample_id=example.get("id"),
            dataset_source=example.get("dataset_source"),
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask,
            labels=tokenized.labels,
            cot_ids=cot_array,
            compressed_cot_tokens=num_slots if num_slots is not None else len(cot_ids),
            compression_ratio=(num_slots if num_slots is not None else len(cot_ids))
            / len(cot_ids),
            slot_positions=slot_positions,
            prefix_ids=(
                prefix_token_ids(trace, self.tokenizer)
                if self.method.requires_prefix()
                else None
            ),
        )


def load_eval_dataset(cfg: DictConfig) -> Any:
    """The rows to score, from whichever corpus `data.loader` names.

    `evaluation.split` is semantic rather than literal -- "validation" maps to the
    SFT corpus's `eval` split and the trace corpus's `val` split -- so callers ask
    for what they mean and neither corpus's naming leaks into config files.

    Phase 1 scores `test`; in-training eval scores `validation`. Scoring the split
    a model selected checkpoints on would report a number that is not held out.
    """
    # `val` is the trace corpus's own split name and must work: insulating configs
    # from corpus naming is not worth an error for spelling the split the way the
    # dataset does.
    split = str(cfg.evaluation.get("split") or "validation")
    split = "validation" if split == "val" else split
    loader = str(cfg.data.get("loader", "dolci_sft"))
    if loader == "traces":
        data = load_trace_data(cfg)
        available = {"train": data.train, "validation": data.val, "test": data.test}
    elif loader == "dolci_sft":
        sft = load_dolci_sft_data(cfg)
        available = {"train": sft.train, "validation": sft.eval, "test": sft.test}
    else:
        raise ValueError(
            f"Unknown data.loader {loader!r}; expected traces or dolci_sft."
        )
    if split not in available:
        raise ValueError(
            f"Unknown evaluation.split {split!r}; expected one of {sorted(available)}."
        )
    return available[split]


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


# Per-row: (answer logprob sum, answer token count, token columns (empty unless
# requested), answer-token entropies or None, answer logprob sum-of-squares).
# The sum-of-squares is accumulated per method so summarize_method can report the
# token-level std (pooled over all answer tokens) without holding every logprob.
# The columns are parallel (token_index, token_id, logprob) lists rather than a
# list of triples: nothing between the device transfer and the writer needs them
# zipped, and at ~31M rows per method the intermediate tuples are allocation churn.
TokenColumns = tuple[list[int], list[int], list[float]]
RowResult = tuple[float, int, TokenColumns, torch.Tensor | None, float]


# Streaming histogram for the pooled token-logprob median (summary needs a
# median for every mean, and the ~5M-token pooled median can't be held exactly).
# Answer log-probs are <= 0 and pile up near 0; [-30, 0] at 0.01-nat bins makes
# the median accurate to +/-0.005 nats, and the rare token below -30 (one the
# model gave ~0 probability) is counted as underflow so it can't distort the grid.
_LP_HIST_LO, _LP_HIST_HI, _LP_HIST_BINS = -30.0, 0.0, 3000
_LP_BIN_WIDTH = (_LP_HIST_HI - _LP_HIST_LO) / _LP_HIST_BINS


def _hist_median(hist: np.ndarray, under: int) -> float:
    """Median of the pooled token log-probs from the streamed histogram.

    ``under`` counts values below ``_LP_HIST_LO`` (all more negative than every
    binned value), so they sit at the low end of the cumulative distribution.
    Linear interpolation within the crossing bin keeps the estimate to bin width.
    """
    total = float(under) + float(hist.sum())
    if total <= 0:
        return float("nan")
    target = total / 2.0
    # counts[0] = underflow (below the grid), counts[1:] = the binned values.
    counts = np.empty(_LP_HIST_BINS + 1)
    counts[0] = under
    counts[1:] = hist
    cum = counts.cumsum()
    i = int(np.searchsorted(cum, target, side="left"))
    if i == 0:  # median falls in the (unbinned) underflow tail
        return _LP_HIST_LO
    prev = cum[i - 1]
    frac = (target - prev) / counts[i] if counts[i] > 0 else 0.0
    return _LP_HIST_LO + (i - 1 + frac) * _LP_BIN_WIDTH


def _answer_logprobs_from_logits(
    logits: torch.Tensor,
    label_tensor: torch.Tensor,
    device: torch.device,
    save_entropies: bool,
    save_token_logprobs: bool,
) -> tuple[list[RowResult], np.ndarray, int]:
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
        token_logprobs: TokenColumns = ([], [], [])
        if save_token_logprobs:
            # One bulk device-to-host transfer per tensor. Reading the same
            # values with a .item() per token instead costs one sync each, which
            # is ~31M syncs per method; the values are identical either way.
            token_logprobs = (
                expanded_positions[row_index][row_mask].tolist(),
                row_labels[row_mask].tolist(),
                logprob_values.tolist(),
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
                # float32 for the squared sum: bf16 squares of ~O(1) log-probs
                # lose precision, and it is what feeds std_token_logprob.
                float(logprob_values.float().pow(2).sum().item()),
            )
        )
    # One histc over all answer-token log-probs in the batch (cheap, always on),
    # accumulated per method to give the pooled token-logprob median. -nll is 0
    # at ignore positions but they are masked out first. Cast to float32: histc
    # has no bfloat16 kernel (the model's usual eval dtype), same as torch.quantile.
    batch_lp = -nll[shifted_labels != IGNORE_INDEX].float()
    under = int((batch_lp < _LP_HIST_LO).sum().item())
    hist = (
        torch.histc(
            batch_lp.clamp(_LP_HIST_LO, _LP_HIST_HI),
            bins=_LP_HIST_BINS,
            min=_LP_HIST_LO,
            max=_LP_HIST_HI,
        )
        .cpu()
        .numpy()
    )
    return results, hist, under


def score_batch_logprobs(
    model: torch.nn.Module,
    batch: list[PreparedSample],
    pad_token_id: int,
    device: torch.device,
    save_entropies: bool,
    save_token_logprobs: bool,
) -> tuple[list[RowResult], np.ndarray, int]:
    """Answer log-probs for a batch, splicing slot embeddings when present.

    A batch is homogeneous (all samples come from the same method), so either
    all carry slot embeddings (embedding-splice path) or none do (text path).
    Returns the per-row results plus this batch's token-logprob histogram and
    underflow count (accumulated per method for the pooled token median).
    """
    input_tensor, attention_tensor, label_tensor = _pad_batch(
        batch, pad_token_id, device
    )
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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def summarize_method(
    method: CompressionMethod,
    samples: list[SampleScore],
    skipped: int,
    token_hist: np.ndarray | None = None,
    token_under: int = 0,
) -> MethodSummary:
    logprobs = [sample.logprob_mean for sample in samples]
    summed_logprobs = [sample.logprob_sum for sample in samples]
    token_counts = [float(sample.answer_tokens) for sample in samples]
    # Pooled token-level moments: total_tokens tokens, sum and sum-of-squares of
    # every answer-token log-prob. mean = sum/N, var = sumsq/N - mean^2 (the
    # population variance over all tokens, clamped at 0 against fp round-off).
    # The median comes from the streamed histogram (the pooled tokens can't be
    # held exactly); it is nan when no histogram was supplied.
    total_tokens = int(sum(token_counts))
    token_sum = sum(summed_logprobs)
    token_sumsq = sum(sample.logprob_sumsq for sample in samples)
    if total_tokens:
        mean_token_logprob = token_sum / total_tokens
        token_var = max(0.0, token_sumsq / total_tokens - mean_token_logprob**2)
        std_token_logprob = math.sqrt(token_var)
    else:
        mean_token_logprob = float("nan")
        std_token_logprob = float("nan")
    median_token_logprob = (
        _hist_median(token_hist, token_under)
        if token_hist is not None
        else float("nan")
    )
    ratios = [
        sample.compression_ratio
        for sample in samples
        if sample.compression_ratio is not None
    ]
    compressed = [
        float(sample.compressed_cot_tokens)
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
        mean_logprob=_mean(logprobs),
        median_logprob=_median(logprobs),
        std_logprob=std,
        sem_logprob=std / math.sqrt(len(logprobs)) if logprobs else float("nan"),
        mean_logprob_sum=_mean(summed_logprobs),
        median_logprob_sum=_median(summed_logprobs),
        mean_answer_tokens=_mean(token_counts),
        median_answer_tokens=_median(token_counts),
        total_answer_tokens=total_tokens,
        mean_token_logprob=mean_token_logprob,
        median_token_logprob=median_token_logprob,
        std_token_logprob=std_token_logprob,
        mean_compression_ratio=_mean(ratios),
        median_compression_ratio=_median(ratios),
        std_compression_ratio=statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
        mean_compressed_cot_tokens=_mean(compressed),
        median_compressed_cot_tokens=_median(compressed),
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


def _load_cot_signals(
    dataset: Any,
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: DictConfig,
    device: torch.device,
    sample_indices: range,
    signals: frozenset[str],
) -> dict[str, dict[int, torch.Tensor]]:
    """CoT ``signals`` for the requested samples, from cache or computed inline.

    Loads each signal's shared cache when configured, then batch-computes any
    samples missing from *any* needed signal — one forward pass yields every
    signal, so a single compute fills them all in. The result depends only on the
    model and the samples, so it is loaded once and shared by every method.
    """
    result: dict[str, dict[int, torch.Tensor]] = {name: {} for name in signals}
    cache_dir = cfg.evaluation.entropy_cache_dir
    if cache_dir is not None:
        for name in signals:
            path = signal_cache_path(
                Path(str(cache_dir)), str(cfg.method.model_name), name
            )
            if path.exists():
                result[name] = load_signal_cache(path)
    missing = [
        index
        for index in sample_indices
        if any(index not in result[name] for name in signals)
    ]
    if missing:
        computed = compute_cot_signals(
            model=model,
            tokenizer=tokenizer,
            examples=dataset,
            sample_indices=missing,
            batch_size=int(cfg.evaluation.batch_size),
            max_batch_tokens=optional_int(cfg.evaluation.max_batch_tokens),
            device=device,
            signals=tuple(signals),
        )
        for name in signals:
            result[name].update(computed[name])
    return result


def _identity(sample: PreparedSample | None) -> PreparedSample | None:
    """collate_fn for batch_size=None: pass each prepared sample through as-is."""
    return sample


def preflight_length_check(
    cot_signal_values: dict[int, torch.Tensor],
    max_position_embeddings: int | None,
    logger: RunLogger,
) -> None:
    """Fail fast if any CoT already overflows the model's context window.

    Uses the cached signal values, whose per-sample length is the CoT token
    count, so the check costs nothing and runs before any GPU work. It covers
    only the samples with cached values (all of them, for a signal sweep); the
    worker's per-sample guard is the backstop for anything uncached. Raising here
    turns a dataset that cannot be scored (e.g. a future 32B split with
    40k-token traces) into an immediate, named error instead of a run that
    silently drops a chunk.
    """
    if max_position_embeddings is None or not cot_signal_values:
        return
    longest_index = max(cot_signal_values, key=lambda i: cot_signal_values[i].numel())
    longest = int(cot_signal_values[longest_index].numel())
    if longest > max_position_embeddings:
        raise ValueError(
            f"CoT of sample {longest_index} is {longest} tokens, over the model's "
            f"{max_position_embeddings}-token context window. Scoring it would "
            "require truncating the answer. Filter the dataset or use a "
            "longer-context model."
        )
    logger.info(
        f"Preflight: longest cached CoT is {longest} tokens "
        f"(sample {longest_index}), within the {max_position_embeddings}-token window."
    )


def signal_span_counts(
    method: CompressionMethod,
    cot_signals: dict[str, dict[int, torch.Tensor]],
    indices: range,
    device: torch.device,
    seed: int,
) -> dict[int, int]:
    """Slot counts for signal-patched methods, planned on the eval device.

    Signal boundaries fall where a quantile or cumsum lands -- floating-point
    reductions whose last bits depend on the device. Resolving them here once, on
    the same device the rest of evaluation uses, keeps the layout identical to
    the pre-worker code; a CPU worker recomputing them could shift a boundary on
    a near-tie. Samples with no values for the patching signal are omitted, and
    the worker skips them.
    """
    assert method.patching is not None, "caller guarantees a patched method"
    signal = method.patching.required_signal()
    assert signal is not None, "caller guarantees a signal-patched method"
    values_by_index = cot_signals.get(signal, {})
    counts: dict[int, int] = {}
    for index in indices:
        values = values_by_index.get(index)
        if values is None:
            continue
        plan = method.plan(values.numel(), index, seed, {signal: values}, device)
        assert plan.num_slots is not None
        counts[index] = plan.num_slots
    return counts


def evaluate_method(
    method: CompressionMethod,
    dataset: Any,
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: DictConfig,
    device: torch.device,
    cot_signals: dict[str, dict[int, torch.Tensor]],
    cot_ids_cache: dict[int, np.ndarray],
    token_writer: TokenLogprobWriter | None,
) -> tuple[MethodSummary, list[SampleScore], dict[int, torch.Tensor]]:
    samples: list[SampleScore] = []
    answer_entropies: dict[int, torch.Tensor] = {}
    skipped = 0
    # Escaped once per method rather than per row. Method names are composed from
    # [a-z0-9_.] so this is a no-op in practice, but it keeps the writer's output
    # identical to json.dumps for any name.
    method_json = json.dumps(method.name)
    indices = range(_eval_limit(cfg, dataset))
    max_length = optional_int(cfg.evaluation.max_length)
    batch_size = int(cfg.evaluation.batch_size)
    max_batch_tokens = optional_int(cfg.evaluation.max_batch_tokens)
    seed = int(cfg.evaluation.seed)
    save_entropies = bool(cfg.evaluation.save_entropies)
    save_token_logprobs = bool(cfg.evaluation.save_token_logprobs)
    required_signals = method.required_signals()
    num_workers = int(cfg.evaluation.num_workers)
    prefetch_factor = int(cfg.evaluation.prefetch_factor)
    pad_token_id = int(tokenizer.pad_token_id)
    placeholder_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)
    if placeholder_id is None or placeholder_id == tokenizer.unk_token_id:
        raise ValueError(
            f"Tokenizer has no usable placeholder token {PLACEHOLDER_TOKEN!r}."
        )
    # Sequences longer than the model's context window cannot be scored; workers
    # drop them (a no-op on today's data -- see preflight_length_check).
    max_position_embeddings = optional_int(
        getattr(model.config, "max_position_embeddings", None)
    )

    # Pooled token-logprob histogram for this method's median (see _hist_median).
    token_hist = np.zeros(_LP_HIST_BINS, dtype=np.float64)
    token_under = 0

    def score_batch(batch: list[PreparedSample]) -> None:
        nonlocal token_hist, token_under
        if not batch:
            return
        row_results, batch_hist, batch_under = score_batch_logprobs(
            model, batch, pad_token_id, device, save_entropies, save_token_logprobs
        )
        token_hist += batch_hist
        token_under += batch_under
        for prepared, (
            logprob_sum,
            answer_tokens,
            token_logprobs,
            answer_entropy,
            logprob_sumsq,
        ) in zip(batch, row_results, strict=True):
            samples.append(
                SampleScore(
                    method=method.name,
                    sample_index=prepared.sample_index,
                    sample_id=prepared.sample_id,
                    dataset_source=prepared.dataset_source,
                    answer_tokens=answer_tokens,
                    logprob_sum=logprob_sum,
                    logprob_mean=logprob_sum / answer_tokens,
                    logprob_sumsq=logprob_sumsq,
                    compressed_cot_tokens=prepared.compressed_cot_tokens,
                    compression_ratio=prepared.compression_ratio,
                )
            )
            if token_writer is not None:
                # Handed off as scored rather than accumulated: holding every row
                # in memory (~31M per method) is what drove peak RSS to ~20 GB.
                # The writer thread emits them in this same order.
                token_writer.submit(method_json, prepared.sample_index, *token_logprobs)
            if answer_entropy is not None:
                answer_entropies[prepared.sample_index] = answer_entropy

    span_counts = None
    if method.patching is not None and method.patching.required_signal() is not None:
        span_counts = signal_span_counts(method, cot_signals, indices, device, seed)

    loader = DataLoader(
        PrepDataset(
            dataset,
            tokenizer,
            method,
            limit=len(indices),
            seed=seed,
            max_length=max_length,
            max_position_embeddings=max_position_embeddings,
            span_counts=span_counts,
            cot_ids_cache=cot_ids_cache,
            placeholder_id=int(placeholder_id),
        ),
        # batch_size=None yields one prepared sample at a time, leaving the
        # greedy token-budget batching below untouched. A batch_sampler could not
        # do that job: the budget needs post-tokenization lengths, which only
        # exist once a worker has run.
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        # batch_size=None hands each sample to collate_fn singly, but torch types
        # collate_fn for the batched (list) path, so cast past the stub.
        collate_fn=cast(Any, _identity),
        **({"prefetch_factor": prefetch_factor} if num_workers > 0 else {}),
    )

    batch: list[PreparedSample] = []
    for prepared in loader:
        if prepared is None:
            skipped += 1
            continue

        # Tokenizing the original CoT depends only on the sample, never on the
        # method, so the worker's copy seeds the cache the next method inherits
        # through fork instead of re-tokenizing.
        cot_ids_cache.setdefault(prepared.sample_index, prepared.cot_ids)

        try:
            prepared.slot_embeddings = method.materialize(
                prepared.cot_ids.tolist(),
                prepared.sample_index,
                seed,
                tokenizer,
                model,
                device,
                {
                    name: cot_signals[name].get(prepared.sample_index)
                    for name in required_signals
                },
                prepared.prefix_ids,
            )
        except ValueError:
            # Patching/pooling wanted a signal this sample has none of, which the
            # serial path also counted as a skip.
            skipped += 1
            continue

        slots = prepared.slot_embeddings
        if slots is not None and len(prepared.slot_positions) != slots.size(0):
            # plan() (in the worker) and materialize() (here) disagreed on the
            # span count, which can only happen if their inputs diverged; drop
            # the sample rather than splice a mismatched set of slots.
            skipped += 1
            continue

        if batch_would_exceed_limit(batch, prepared, max_batch_tokens):
            score_batch(batch)
            batch = []
        batch.append(prepared)
        if len(batch) >= batch_size:
            score_batch(batch)
            batch = []

    score_batch(batch)

    return (
        summarize_method(method, samples, skipped, token_hist, token_under),
        samples,
        answer_entropies,
    )


def evaluate_methods(cfg: DictConfig) -> Path:
    run_dir = get_run_dir(cfg)
    save_resolved_config(cfg, run_dir)
    logger = RunLogger(cfg=cfg, run_dir=run_dir)

    try:
        set_seed(seed=int(cfg.evaluation.seed), deterministic=False)
        device = resolve_device(cfg.evaluation.device)
        logger.info(f"Using device: {device}")

        methods = build_compression_methods(cfg)
        dataset = load_eval_dataset(cfg)

        artifact_dir = run_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        # Optional scratch mirror. When evaluation.scratch_dir is set, the large
        # tokens.jsonl is written ONLY there (keeping it off a quota-limited HOME
        # run dir), while the small summary.json/samples.jsonl are written to both
        # places. run_dir.name keeps each multirun job in its own subdir, so the
        # 27 cells of a sweep never collide on one scratch file.
        scratch_cfg = cfg.evaluation.get("scratch_dir")
        scratch_artifact_dir = (
            Path(str(scratch_cfg)) / run_dir.name / "artifacts"
            if scratch_cfg is not None
            else None
        )
        if scratch_artifact_dir is not None:
            scratch_artifact_dir.mkdir(parents=True, exist_ok=True)
        # summary/samples land in every artifact dir; tokens only in the scratch
        # dir when one is set, else alongside the rest in the run dir.
        artifact_dirs = [artifact_dir] + (
            [scratch_artifact_dir] if scratch_artifact_dir is not None else []
        )
        summary_path = artifact_dir / "summary.json"
        samples_path = artifact_dir / "samples.jsonl"
        tokens_path = (scratch_artifact_dir or artifact_dir) / "tokens.jsonl"
        save_token_logprobs = bool(cfg.evaluation.save_token_logprobs)

        # Read-only and identical for every method, so built once rather than
        # reloaded from disk per method.
        model, tokenizer = build_eval_model_and_tokenizer(cfg, device)
        needed_signals = frozenset().union(
            *(method.required_signals() for method in methods)
        )
        cot_signals: dict[str, dict[int, torch.Tensor]] = {
            name: {} for name in needed_signals
        }
        if needed_signals:
            cot_signals = _load_cot_signals(
                dataset,
                model,
                tokenizer,
                cfg,
                device,
                range(_eval_limit(cfg, dataset)),
                needed_signals,
            )
        # Lengths are identical across signals, so any one signal's values feed
        # the preflight window check.
        representative = next(iter(cot_signals.values()), {})
        preflight_length_check(
            representative,
            optional_int(getattr(model.config, "max_position_embeddings", None)),
            logger,
        )
        cot_ids_cache: dict[int, np.ndarray] = {}

        summaries = []
        sample_rows = []
        # Per method, so different methods' answer entropies never collide on
        # the same sample_index (one npz file per method).
        answer_entropies_by_method: dict[str, dict[int, torch.Tensor]] = {}
        token_writer = TokenLogprobWriter(tokens_path) if save_token_logprobs else None
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
                    cot_signals=cot_signals,
                    cot_ids_cache=cot_ids_cache,
                    token_writer=token_writer,
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
            if token_writer is not None:
                token_writer.close()

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        summary_payload = {
            "metric": str(cfg.evaluation.metric),
            "normalize_by_length": bool(cfg.evaluation.normalize_by_length),
            "methods": [asdict(summary) for summary in summaries],
        }
        summary_json = json.dumps(summary_payload, indent=2)
        sample_dicts = [asdict(sample) for sample in sample_rows]
        # Written to every artifact dir (run dir + scratch mirror). These are tiny
        # relative to tokens.jsonl, so the duplicate write is negligible.
        for target in artifact_dirs:
            (target / "summary.json").write_text(summary_json, encoding="utf-8")
            write_jsonl(target / "samples.jsonl", sample_dicts)
        # Logged (small) artifacts; large npz files are written to disk only.
        # tokens.jsonl was streamed during scoring rather than written here.
        paths = [summary_path, samples_path]
        if save_token_logprobs:
            paths.append(tokens_path)
        if bool(cfg.evaluation.save_entropies):
            # Large, like tokens.jsonl -> scratch when set, else the run dir.
            entropies_dir = scratch_artifact_dir or artifact_dir
            for name, entropies in answer_entropies_by_method.items():
                save_entropies_npz(
                    entropies_dir / f"answer_entropies__{name}.npz", entropies
                )

        for path in paths:
            logger.log_artifact(path)
    except Exception:
        logger.finish(exit_code=1)
        raise
    else:
        logger.finish()
        return summary_path
