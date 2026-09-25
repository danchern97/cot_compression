from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from cot_compression.data.chat import IGNORE_INDEX, pad_collate
from cot_compression.data.dolci import load_tokenized_sft_data
from cot_compression.training.logging import RunLogger
from cot_compression.training.sft_loop import (
    LoopState,
    resolve_resume_dir,
    run_training,
)
from cot_compression.training.utils import (
    get_run_dir,
    optional_int,
    save_resolved_config,
    set_seed,
)


def parse_torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unknown torch dtype: {name}")


# --------------------------------------------------------------------------- #
# Distributed
# --------------------------------------------------------------------------- #


def init_distributed() -> tuple[int, int, int]:
    """Join the process group using SLURM's own rank variables.

    submitit already launches one task per GPU (`tasks_per_node=4` -> `srun
    --ntasks-per-node=4`), so every rank is a real SLURM task and torchrun would
    only add a redundant launcher layer. Outside SLURM this returns world size 1
    and never touches NCCL, which keeps the whole loop runnable on CPU in tests.
    """
    if "SLURM_PROCID" not in os.environ or int(os.environ.get("SLURM_NTASKS", 1)) == 1:
        return 0, 0, 1

    global_rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    world_size = int(os.environ["SLURM_NTASKS"])

    # Both launchers request nodes=1, so every rank is on this host. Prefer the
    # address srun advertises, but do not require it: it is set by srun and not
    # by every submission path, and a missing MASTER_ADDR would fail the job
    # minutes in, after the model has already loaded.
    os.environ.setdefault(
        "MASTER_ADDR", os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1")
    )
    # Per-job port so two jobs sharing a node cannot collide on the rendezvous.
    job_id = int(os.environ.get("SLURM_JOB_ID", "0").split("_")[0])
    os.environ.setdefault("MASTER_PORT", str(20000 + job_id % 20000))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        rank=global_rank,
        world_size=world_size,
        timeout=timedelta(minutes=30),
    )
    return global_rank, local_rank, world_size


# --------------------------------------------------------------------------- #
# Batch planning
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BatchPlan:
    """The whole epoch, decided up front and identically on every rank.

    Planning from the pre-tokenized `length` column rather than discovering
    batches while iterating buys three things at once:

    * every rank derives the same `steps` list, so all ranks run the same number
      of micro-batches per optimizer step and DDP cannot deadlock;
    * resume is a list slice, with no need to replay or fast-forward a sampler;
    * `denom[s]` -- the exact global count of supervised tokens in step `s` -- is
      known before the step runs, so the loss needs no extra all-reduce to be
      correctly token-weighted.
    """

    micro: list[list[int]]
    steps: list[tuple[int, int]]
    denom: list[int]
    dropped: int
    kept: int
    signature: str
    # Global rows per step. `denom` counts supervised tokens; the encoder's
    # objective is a mean over ROWS, so it needs this instead -- and comparing it
    # against the rows that actually arrive is what catches a dropped row.
    rows: list[int] = field(default_factory=list)


def _pack(
    order: list[int],
    lengths: np.ndarray,
    max_batch_tokens: int,
    max_sequences: int,
) -> list[list[int]]:
    """Token-budget micro-batches over `order`, in exactly the order given.

    Same greedy rule as the eval path (`evaluate.batch_would_exceed_limit`): admit a
    row while `max_len * (n + 1)` stays inside the budget. Deliberately does NOT
    sort -- the caller owns that decision, because it sets both the padding and
    *which rows share a micro-batch*, and those two pull in opposite directions.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    widest = 0
    for index in order:
        candidate = max(widest, int(lengths[index]))
        full = current and (
            candidate * (len(current) + 1) > max_batch_tokens
            or len(current) >= max_sequences
        )
        if full:
            batches.append(current)
            current, candidate = [], int(lengths[index])
        current.append(int(index))
        widest = candidate
    if current:
        batches.append(current)
    return batches


def _greedy_micro_batches(
    order: np.ndarray,
    lengths: np.ndarray,
    max_batch_tokens: int,
    max_sequences: int,
    group_size: int,
) -> list[list[int]]:
    """Length-grouped, token-budget micro-batches.

    Sorting within a large group first is what keeps padding at a couple of percent
    -- adjacent rows in a 8192-row sorted group differ in length by well under 1%,
    which is why sequence packing is not worth its complexity here.

    This is HF's `LengthGroupedSampler` scheme, and its known consequence is that
    consecutive steps walk one group from longest to shortest. `plan_epoch` fixes
    that with `shuffle_micro_steps`; the grouping itself is untouched, so the SFT
    campaigns keep the exact batches they trained on.
    """
    batches: list[list[int]] = []
    for start in range(0, order.size, group_size):
        group = order[start : start + group_size]
        group = group[np.argsort(-lengths[group], kind="stable")]
        batches += _pack(group.tolist(), lengths, max_batch_tokens, max_sequences)
    return batches


def _bucketed_micro_batches(
    order: np.ndarray,
    lengths: np.ndarray,
    max_batch_tokens: int,
    max_sequences: int,
    bucket_rows: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Global length sort, then shuffle rows inside each bucket before packing.

    Two measured trades, both on the encoder's 175,928-row plan:

    * sorting **globally** rather than within 8192-row groups drops padding from
      0.3% to 0.0%, because a micro-batch's rows are then the nearest in length in
      the whole epoch rather than the nearest within a random pool;
    * shuffling inside a 1024-row bucket costs 1.1% padding (and ~1% of epoch
      time) and buys the thing the strict sort cannot: a row's micro-batch
      companions stop being a deterministic function of its length.

    Bucket width is the dial between those two. It is *not* a dial for domain
    diversity -- domain is nearly a function of length in this corpus, so the
    bucket's length range fixes its domain mix no matter how the rows inside it are
    permuted. Diversity within an optimizer step comes from accumulating several
    micro-steps (`target_global_batch`), not from here.
    """
    ordered = order[np.argsort(-lengths[order], kind="stable")]
    batches: list[list[int]] = []
    for start in range(0, ordered.size, bucket_rows):
        bucket = ordered[start : start + bucket_rows]
        bucket = bucket[rng.permutation(bucket.size)]
        batches += _pack(bucket.tolist(), lengths, max_batch_tokens, max_sequences)
    return batches


def _shuffle_micro_steps(
    micro: list[list[int]], world_size: int, rng: np.random.Generator
) -> list[list[int]]:
    """Shuffle whole micro-steps: `world_size` consecutive micro-batches at a time.

    Blocks, never individual micro-batches. The `world_size` micro-batches of one
    micro-step run concurrently and synchronize on every micro-batch -- the encoder
    all-reduces code counts inside `forward` -- so every rank waits for the slowest
    one. Keeping them adjacent in the sorted order keeps them the same size:
    measured on the encoder plan with the probe's own per-micro-batch timings,
    shuffling individual micro-batches instead costs 1.6x wall clock (5.4 -> 8.5
    h/rank per epoch) for identical work. This is fairseq's `grouped_shuffling`,
    "shuffle batches in groups of num_shards to enable similar sequence lengths on
    each GPU worker when batches are sorted by length".
    """
    steps = [micro[i : i + world_size] for i in range(0, len(micro), world_size)]
    return [batch for index in rng.permutation(len(steps)) for batch in steps[index]]


def plan_epoch(
    lengths: np.ndarray,
    label_starts: np.ndarray,
    cfg: DictConfig,
    epoch: int,
    world_size: int,
) -> BatchPlan:
    training = cfg.training
    max_length = int(training.max_length)
    rng = np.random.default_rng(int(training.seed) + epoch)

    keep = np.flatnonzero(lengths <= max_length)
    dropped = int(lengths.size - keep.size)
    order = keep[rng.permutation(keep.size)]
    budget = optional_int(training.max_train_examples)
    if budget is not None:
        order = order[:budget]

    # `length_bucket_rows` supersedes `length_group_size`: it sorts globally, so a
    # group size would have nothing left to mean.
    bucket_rows = optional_int(training.get("length_bucket_rows"))
    if bucket_rows is not None:
        micro = _bucketed_micro_batches(
            order=order,
            lengths=lengths,
            max_batch_tokens=int(training.max_batch_tokens),
            max_sequences=int(training.micro_batch_max_sequences),
            bucket_rows=bucket_rows,
            rng=rng,
        )
    else:
        micro = _greedy_micro_batches(
            order=order,
            lengths=lengths,
            max_batch_tokens=int(training.max_batch_tokens),
            max_sequences=int(training.micro_batch_max_sequences),
            group_size=int(training.length_group_size),
        )
    # Ragged tail dropped so every rank has a micro-batch in every micro-step.
    micro = micro[: (len(micro) // world_size) * world_size]
    shuffle_micro_steps = bool(training.get("shuffle_micro_steps", False))
    if shuffle_micro_steps:
        # Its own stream, not `rng`: the bucket shuffle above has already advanced
        # that one by a data-dependent number of draws, and the batch order should
        # not change because the length distribution did.
        micro = _shuffle_micro_steps(
            micro, world_size, np.random.default_rng([int(training.seed), epoch, 1])
        )

    target = int(training.target_global_batch)
    steps: list[tuple[int, int]] = []
    denom: list[int] = []
    rows: list[int] = []
    start, sequences, tokens = 0, 0, 0
    for cursor in range(0, len(micro), world_size):
        for batch in micro[cursor : cursor + world_size]:
            sequences += len(batch)
            tokens += int((lengths[batch] - label_starts[batch]).sum())
        if sequences >= target or cursor + world_size == len(micro):
            steps.append((start, cursor + world_size))
            denom.append(tokens)
            rows.append(sequences)
            start, sequences, tokens = cursor + world_size, 0, 0

    payload = "|".join(
        str(value)
        for value in (
            training.seed,
            epoch,
            world_size,
            max_length,
            training.max_batch_tokens,
            training.micro_batch_max_sequences,
            training.length_group_size,
            target,
            budget,
            lengths.size,
            int(lengths.sum()),
        )
    )
    # Appended only when set, so a plan built by the pre-bucketing code hashes to
    # exactly what it did before and the SFT campaigns still resume.
    if bucket_rows is not None:
        payload += f"|bucket{bucket_rows}"
    if shuffle_micro_steps:
        payload += "|shuffled"
    return BatchPlan(
        micro=micro,
        steps=steps,
        denom=denom,
        dropped=dropped,
        kept=int(order.size),
        signature=hashlib.sha256(payload.encode()).hexdigest()[:16],
        rows=rows,
    )


def plan_eval_batches(
    lengths: np.ndarray,
    cfg: DictConfig,
    world_size: int,
) -> list[list[int]]:
    """A fixed evaluation slice, identical at every step and in every run.

    Shuffled before slicing: the rows are stored in dataset order and the batcher
    sorts by length, so taking a raw prefix would evaluate an unrepresentative
    slice rather than a sample of the split.
    """
    training = cfg.training
    rng = np.random.default_rng(int(training.seed))
    keep = np.flatnonzero(lengths <= int(training.max_length))
    order = keep[rng.permutation(keep.size)][: int(training.eval_examples)]
    micro = _greedy_micro_batches(
        order=order,
        lengths=lengths,
        max_batch_tokens=int(training.max_batch_tokens),
        max_sequences=int(training.micro_batch_max_sequences),
        group_size=int(training.length_group_size),
    )
    return micro[: (len(micro) // world_size) * world_size]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class ChunkedCELM(torch.nn.Module):
    """Causal-LM loss that never materializes `[tokens, vocab]` logits.

    `transformers.loss.loss_utils.ForCausalLMLoss` upcasts logits to fp32, so a
    32k-token forward would need ~20 GB fp32 + 10 GB bf16 + 20 GB of gradient for
    the logits alone -- impossible beside 60 GB of fp32 weights, grads and Adam
    moments. Running `lm_head` + cross-entropy in slices under a checkpoint keeps
    peak loss memory at O(chunk x vocab) (~1.5 GB at chunk 1024) and stores only
    the scalar for backward. Selecting label positions instead would not help:
    ~98% of tokens here are labels.

    The shift happens once, before chunking, so no chunk boundary can introduce
    an off-by-one. The return value is a *sum*, not a mean, which is what lets
    the caller divide by an exact global token count.

    This must be the module DDP wraps. Reaching through to the backbone (e.g.
    `ddp.module.model(...)`) skips `Reducer.prepare_for_backward`, and gradients
    are then silently never all-reduced.
    """

    def __init__(self, model: Any, chunk_tokens: int) -> None:
        super().__init__()
        self.model = model
        self.chunk_tokens = chunk_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        return chunked_ce_from_hidden(
            self.model.lm_head, hidden, labels, self.chunk_tokens
        )


def chunked_ce_from_hidden(
    lm_head: Any,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    chunk_tokens: int,
    select_labels: bool = False,
) -> torch.Tensor:
    """Shifted cross-entropy over `hidden`, in slices, returning a SUM.

    Split out of `ChunkedCELM` so the encoder path -- which reaches the same
    hidden states through `inputs_embeds` rather than `input_ids` -- reuses this
    rather than reimplementing the memory argument. The shift happens once, before
    chunking, so no chunk boundary can introduce an off-by-one.

    `select_labels` drops ignored positions before `lm_head`. Mathematically a
    no-op -- `ignore_index` already contributes zero -- but it decides how much
    work happens, and the right answer differs by caller:

    * SFT supervises the whole assistant turn, ~98% of positions, so selecting
      would save nothing and it stays off by default;
    * the encoder supervises the **answer only**, a measured 21.8% of the
      compressed render, so leaving it off runs `lm_head` and the fp32 logit
      upcast over 4.6x more positions than carry a label.
    """
    hidden = hidden[:, :-1].flatten(0, 1)
    targets = labels[:, 1:].flatten()
    if select_labels:
        keep = (targets != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        hidden = hidden.index_select(0, keep)
        targets = targets.index_select(0, keep)
    return chunked_ce_flat(lm_head, hidden, targets, chunk_tokens)


def chunked_ce_flat(
    lm_head: Any,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    chunk_tokens: int,
) -> torch.Tensor:
    """Cross-entropy over ALREADY-PAIRED `[N, d]` states and `[N]` targets, as a SUM.

    The shift-free core of `chunked_ce_from_hidden`. Split out because the encoder's
    next-patch head pairs states with targets by an explicit *gather* -- the state
    predicting a patch's first token is its preceding code, not the position before
    it -- so the one-position shift baked into the caller above does not apply.

    Chunked and checkpointed for the same reason either way: `lm_head` materializes
    `[chunk, 151936]` fp32 logits, and the encoder's aux path runs it over ~16x more
    positions than the answer does.
    """

    def chunk_loss(part: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            lm_head(part).float(),
            target,
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, hidden.size(0), chunk_tokens):
        stop = start + chunk_tokens
        total = total + checkpoint(
            chunk_loss, hidden[start:stop], targets[start:stop], use_reentrant=False
        )
    return total


def chunked_ce_weighted(
    lm_head: Any,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    chunk_tokens: int,
) -> torch.Tensor:
    """`[sum(w * CE), sum(CE)]` over already-paired `[N, d]` states and `[N]` targets.

    Two reductions of one `reduction="none"` cross-entropy, because the encoder needs
    both and a second pass over `lm_head` is the most expensive thing in the step:

    * the **weighted** sum is the objective. Per-row means are just per-token weights
      (`1/A_i` for answer tokens, `1/(P_i * T_ij)` for next-patch targets), so the
      whole per-row normalization lives in `weights` and nothing here knows about it;
    * the **plain** sum is the token-weighted metric twin, free in this pass.

    Only the first carries a gradient the caller uses, and since both come from the
    same CE tensor they cannot disagree about what was scored. Chunked and
    checkpointed exactly like `chunked_ce_flat`: `lm_head` materializes
    `[chunk, 151936]` fp32 logits.
    """

    def chunk_loss(
        part: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        # `ignore_index` contributes exactly 0 under reduction="none", so an ignored
        # position is absent from both sums without being selected out here.
        losses = F.cross_entropy(
            lm_head(part).float(),
            target,
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )
        return torch.stack([(losses * weight).sum(), losses.sum()])

    total = hidden.new_zeros(2, dtype=torch.float32)
    for start in range(0, hidden.size(0), chunk_tokens):
        stop = start + chunk_tokens
        total = total + checkpoint(
            chunk_loss,
            hidden[start:stop],
            targets[start:stop],
            weights[start:stop],
            use_reentrant=False,
        )
    return total


# The fields that make a checkpoint a *different model* rather than a later step
# of the same one. Compared, not hashed, so the error can name what moved.
_ARCHITECTURE_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
)


def check_checkpoint_architecture(cfg: DictConfig, weights_dir: Path) -> None:
    """Refuse a resume whose checkpoint is a different model than the config asks for.

    `run_name` -- which is both the run directory and the W&B run id -- is built
    from run_tag, lr and global batch, with no model identifier. So two campaigns
    on different models can resolve to the same directory, and
    `resume_from_checkpoint=auto` would then load that directory's checkpoint via
    `from_pretrained`, which reads the *checkpoint's own* config.json. The result
    is a job that silently trains the wrong model.

    `plan_signature` cannot catch this. It hashes the batch plan -- seed, world
    size, batching knobs, row count, total length -- and Qwen3-0.6B and Qwen3-4B
    ship byte-identical tokenizers, so they share one tokenized cache and produce
    an identical hash. This check is the only thing standing between a forgotten
    `run_tag` and a destroyed run.
    """
    from transformers import AutoConfig

    # Only a checkpoint that declares an architecture can be checked against one.
    # `save_pretrained` always writes config.json, so in a real run this is always
    # present; its absence means the directory was not written by the HF path at
    # all (the fake models in the tests), and there is nothing to compare.
    if not (weights_dir / "config.json").exists():
        return

    want = AutoConfig.from_pretrained(
        str(cfg.method.model_name),
        trust_remote_code=bool(cfg.method.trust_remote_code),
    )
    found = AutoConfig.from_pretrained(
        str(weights_dir),
        trust_remote_code=bool(cfg.method.trust_remote_code),
    )
    moved = [
        f"{field}: checkpoint has {getattr(found, field, None)}, "
        f"{cfg.method.model_name} has {getattr(want, field, None)}"
        for field in _ARCHITECTURE_FIELDS
        if getattr(found, field, None) != getattr(want, field, None)
    ]
    if moved:
        raise ValueError(
            f"Refusing to resume: {weights_dir} holds a different model than "
            f"method.model_name={cfg.method.model_name}.\n  "
            + "\n  ".join(moved)
            + "\nThis usually means two campaigns share a run_dir because run_name "
            "(run_tag + lr + global batch) carries no model identifier. Give this "
            "run its own run_tag."
        )


def build_sft_model_and_tokenizer(
    cfg: DictConfig,
    device: torch.device,
    weights_dir: Path | None,
) -> tuple[Any, Any]:
    if weights_dir is not None:
        check_checkpoint_architecture(cfg, weights_dir)

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
            str(weights_dir) if weights_dir is not None else str(cfg.method.model_name),
            dtype=parse_torch_dtype(str(cfg.training.torch_dtype)),
            attn_implementation=str(cfg.training.attn_implementation),
            trust_remote_code=bool(cfg.method.trust_remote_code),
        ),
    )
    model.config.use_cache = False
    if bool(cfg.training.gradient_checkpointing):
        # Non-reentrant is required for checkpointing to compose with DDP.
        model.gradient_checkpointing_enable({"use_reentrant": False})
    return model.to(device), tokenizer


def build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Plain AdamW over fp32 parameters.

    fp32 master weights are not optional here. Loading in bf16 puts the Adam
    moments in bf16 too, and at lr 1e-5 against a weight scale of ~1e-2 the
    relative update (~5e-4) sits below bf16's ~4e-3 mantissa step, so updates
    round away and the model does not move. Compute still runs in bf16 via
    autocast; only the master copy and the moments are fp32.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optim.lr),
        betas=(float(cfg.optim.beta1), float(cfg.optim.beta2)),
        eps=float(cfg.optim.eps),
        weight_decay=float(cfg.optim.weight_decay),
        fused=torch.cuda.is_available(),
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _columns(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(dataset["length"], dtype=np.int64),
        np.asarray(dataset["label_start"], dtype=np.int64),
    )


def _build_loader(
    dataset: Any,
    batches: list[list[int]],
    cfg: DictConfig,
    pad_token_id: int,
) -> DataLoader:
    return DataLoader(
        cast(TorchDataset[Any], dataset),
        batch_sampler=batches,
        collate_fn=partial(pad_collate, pad_token_id=pad_token_id),
        num_workers=int(cfg.training.num_workers),
        prefetch_factor=int(cfg.training.prefetch_factor)
        if int(cfg.training.num_workers) > 0
        else None,
        pin_memory=torch.cuda.is_available(),
    )


def train_sft(cfg: DictConfig) -> Path:
    global_rank, local_rank, world_size = init_distributed()
    is_main = global_rank == 0

    run_dir = get_run_dir(cfg)
    if is_main:
        save_resolved_config(cfg, run_dir)
    logger = RunLogger(cfg=cfg, run_dir=run_dir, is_main=is_main)

    # Rank-offset seeds: nothing in the loop consumes RNG (Qwen3 has no dropout
    # and the batch plan is deterministic), but any future sampling should differ
    # per rank rather than be silently correlated.
    set_seed(
        seed=int(cfg.training.seed) + global_rank,
        deterministic=bool(cfg.training.deterministic),
    )
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f"rank {global_rank}/{world_size} on {device}")

    resume_dir = resolve_resume_dir(run_dir, cfg.training.resume_from_checkpoint)
    if resume_dir is not None:
        logger.info(f"Resuming from {resume_dir}")

    model, tokenizer = build_sft_model_and_tokenizer(cfg, device, resume_dir)
    optimizer = build_optimizer(cfg, model)

    tokenized = load_tokenized_sft_data(cfg)
    train_lengths, train_starts = _columns(tokenized["train"])
    eval_lengths, _ = _columns(tokenized["eval"])

    plan = plan_epoch(train_lengths, train_starts, cfg, epoch=0, world_size=world_size)
    logger.info(
        f"plan {plan.signature}: {len(plan.steps)} steps, {len(plan.micro)} micro-batches, "
        f"{plan.kept} rows kept, {plan.dropped} dropped over max_length="
        f"{int(cfg.training.max_length)}"
    )

    # The sweep truncates the *full-run* schedule instead of compressing a cosine
    # into a short arm, so every arm is compared on the LR trajectory it would
    # actually see. Unset => the schedule spans this run's own steps.
    horizon = optional_int(cfg.training.schedule_horizon_steps) or len(plan.steps)
    warmup = max(
        int(cfg.training.min_warmup_steps),
        int(horizon * float(cfg.training.warmup_ratio)),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup,
        num_training_steps=horizon,
    )

    if bool(cfg.training.get("debug_memory", False)) and torch.cuda.is_available():
        logger.info(
            f"memory after model load: "
            f"{torch.cuda.memory_allocated(device) / (1 << 30):.2f} GiB"
        )

    module: torch.nn.Module = ChunkedCELM(model, int(cfg.training.ce_chunk_tokens))
    if world_size > 1:
        module = DistributedDataParallel(
            module,
            device_ids=[local_rank],
            # Not an optimization: without it DDP allocates flat gradient buckets
            # on top of param.grad, ~15 GB extra that does not fit beside the
            # optimizer state.
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )

    if bool(cfg.training.get("debug_memory", False)) and torch.cuda.is_available():
        logger.info(
            f"memory after DDP wrap:   "
            f"{torch.cuda.memory_allocated(device) / (1 << 30):.2f} GiB"
        )

    state = LoopState(
        model=model,
        module=module,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        cfg=cfg,
        run_dir=run_dir,
        device=device,
        plan=plan,
        rank=global_rank,
        world_size=world_size,
    )

    try:
        return run_training(
            state=state,
            train_dataset=tokenized["train"],
            eval_batches=plan_eval_batches(eval_lengths, cfg, world_size),
            eval_dataset=tokenized["eval"],
            make_loader=partial(
                _build_loader, cfg=cfg, pad_token_id=int(tokenizer.pad_token_id)
            ),
            resume_dir=resume_dir,
        )
    finally:
        logger.finish()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
