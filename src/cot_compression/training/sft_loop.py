from __future__ import annotations

import contextlib
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from cot_compression.training.logging import RunLogger

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for type checkers
    from cot_compression.training.sft import BatchPlan

LATEST = "LATEST"
STATE_FILE = "training_state.pt"


@dataclass
class LoopState:
    model: Any
    module: torch.nn.Module
    tokenizer: Any
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    logger: RunLogger
    cfg: DictConfig
    run_dir: Path
    device: torch.device
    plan: BatchPlan
    rank: int = 0
    world_size: int = 1
    global_step: int = 0
    tokens_seen: int = 0
    best_eval_loss: float = float("inf")
    # Total gradient norm of the step just taken, before clipping.
    last_grad_norm: float = 0.0
    # How to force DDP's first-iteration gradient allocation. None keeps the SFT
    # behaviour below; a caller whose module does not take
    # `(input_ids, attention_mask, labels)` positionally -- the encoder path --
    # supplies its own, or a no-op when the transient is too small to matter.
    prime_batch: Callable[[LoopState], None] | None = None
    # Extra scalars merged into every `log_interval` emission. The encoder path
    # reports codebook health here; without it a collapsed codebook and a healthy
    # one produce identical loss curves.
    extra_metrics: Callable[[LoopState], dict[str, float]] | None = None
    # Extra scalars merged into every `eval/` emission, given the eval loss. The
    # encoder path reports its distance to the `no_cot` floor and `base` ceiling
    # here -- the pair that says whether a run is working at all.
    eval_metrics: Callable[[LoopState, float], dict[str, float]] | None = None
    # What `estimate_loss`'s return is logged as. The encoder's objective carries
    # the quantizer and next-patch terms as well as answer CE, so calling it
    # `eval/loss` next to the pure-CE `eval/base_row` invites comparing two
    # different quantities; that path logs every component separately beside it.
    eval_loss_key: str = "eval/loss"
    # Which emitted metric `best/` is selected on. Empty keeps the historical
    # behaviour of selecting on `estimate_loss`'s return, i.e. the full objective.
    # A caller whose objective includes terms it is not judged on -- the encoder,
    # once next-patch supervision dominates the total -- names the metric instead.
    best_metric_key: str = ""
    # Runs after each optimizer step, with the optimizer's state settled. The
    # encoder path reseeds dead codebook entries here -- it needs the optimizer
    # (to clear stale Adam moments) and the step count (to skip EMA warmup),
    # neither of which the module can see.
    post_step: Callable[[LoopState], None] | None = None
    # What to divide this step's loss terms by, counted from the step's own
    # micro-batches before any of them runs. None keeps the SFT behaviour: one
    # scalar loss over `plan.denom[step]` supervised tokens.
    #
    # When set, the module returns a VECTOR of term sums and this returns a vector
    # of the same length; each term is divided by its own global count (rows, rows
    # with a supervised patch, ...) and the results summed. That is the only way to
    # normalize two terms differently and still have the result be independent of
    # how rows were grouped into micro-batches -- which is exactly what a per-row
    # objective under gradient accumulation and DDP needs.
    step_denominators: (
        Callable[[list[dict[str, torch.Tensor]]], torch.Tensor] | None
    ) = None
    _last_checkpoint_time: float = field(default_factory=time.monotonic)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def checkpoint_root(self) -> Path:
        return self.run_dir / "checkpoints"


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #


def _fsync_tree(path: Path) -> None:
    for child in sorted(path.rglob("*")):
        if child.is_file():
            handle = os.open(child, os.O_RDONLY)
            try:
                os.fsync(handle)
            finally:
                os.close(handle)
    handle = os.open(path, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _write_pointer(root: Path, name: str) -> None:
    """Publish the resume target.

    Replacing a small file is atomic on POSIX; renaming a directory onto an
    existing non-empty directory is not. So the newest checkpoint is named
    uniquely and a pointer file -- not a directory rename -- is what makes it
    current.
    """
    staging = root / f".{LATEST}.tmp"
    staging.write_text(name, encoding="utf-8")
    handle = os.open(staging, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)
    os.replace(staging, root / LATEST)


def _prune(root: Path, keep: int) -> None:
    checkpoints = sorted(p for p in root.glob("step-*") if p.is_dir())
    stale = checkpoints[:-keep] if keep > 0 else checkpoints
    for path in stale:
        shutil.rmtree(path, ignore_errors=True)


def save_checkpoint(state: LoopState, full: bool) -> Path:
    """Write a checkpoint that a kill at any instant cannot corrupt.

    Order matters: stage -> fsync -> rename into a *fresh* name -> update the
    pointer -> only then prune. Until the pointer moves, the previous checkpoint
    is still the resume target, so there is no window in which the only usable
    checkpoint has been removed.

    `full` checkpoints carry fp32 weights plus optimizer moments (~48 GB) and are
    what resume reads. `best/` is bf16 weights and tokenizer only (~8 GB): it is
    consumed by eval and the benchmark harness, never resumed from, so shipping
    optimizer state with it would cost 40 GB for nothing.
    """
    root = state.checkpoint_root
    root.mkdir(parents=True, exist_ok=True)
    target = root / (f"step-{state.global_step:07d}" if full else "best")
    staging = root / f".{target.name}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    state_dict = state.model.state_dict()
    if not full:
        state_dict = {
            key: value.to(torch.bfloat16) for key, value in state_dict.items()
        }
    state.model.save_pretrained(staging, state_dict=state_dict)
    state.tokenizer.save_pretrained(staging)

    if full:
        torch.save(
            {
                "global_step": state.global_step,
                "tokens_seen": state.tokens_seen,
                "best_eval_loss": state.best_eval_loss,
                "total_steps": len(state.plan.steps),
                "plan_signature": state.plan.signature,
                "optimizer": state.optimizer.state_dict(),
                "scheduler": state.scheduler.state_dict(),
                "cfg": OmegaConf.to_container(state.cfg, resolve=True),
            },
            staging / STATE_FILE,
        )

    _fsync_tree(staging)
    shutil.rmtree(target, ignore_errors=True)
    os.replace(staging, target)
    if full:
        _write_pointer(root, target.name)
        _prune(root, keep=int(state.cfg.training.keep_checkpoints))
    return target


def resolve_resume_dir(run_dir: Path, setting: Any) -> Path | None:
    """Map `training.resume_from_checkpoint` to a directory.

    `auto` follows the pointer file and returns None on a cold start, which is
    what makes submitit's requeue-on-timeout work without any bookkeeping: the
    resubmitted job runs the identical command line.
    """
    if setting is None:
        return None
    if str(setting) != "auto":
        return Path(str(setting))

    pointer = run_dir / "checkpoints" / LATEST
    if not pointer.exists():
        return None
    candidate = run_dir / "checkpoints" / pointer.read_text(encoding="utf-8").strip()
    return candidate if (candidate / STATE_FILE).exists() else None


def load_training_state(state: LoopState, checkpoint_dir: Path) -> None:
    payload = torch.load(
        checkpoint_dir / STATE_FILE, map_location="cpu", weights_only=False
    )
    if payload["plan_signature"] != state.plan.signature:
        raise ValueError(
            "Refusing to resume: the batch plan changed since this checkpoint "
            f"({payload['plan_signature']} -> {state.plan.signature}). A resume that "
            "silently re-shuffles the curriculum or moves total_steps would "
            "invalidate the run."
        )
    state.optimizer.load_state_dict(payload["optimizer"])
    state.scheduler.load_state_dict(payload["scheduler"])
    state.global_step = int(payload["global_step"])
    state.tokens_seen = int(payload["tokens_seen"])
    state.best_eval_loss = float(payload["best_eval_loss"])


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _autocast(state: LoopState) -> Any:
    if state.device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=_dtype(str(state.cfg.training.autocast_dtype)))


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


@torch.no_grad()
def estimate_loss(
    state: LoopState,
    loader: DataLoader,
) -> float:
    """The objective over the fixed eval slice.

    Sums of losses over sums of denominators, all-reduced -- not a mean of per-batch
    means, which is biased whenever batches hold different numbers of supervised
    tokens or rows (they always do here, since batches are token-budgeted).

    With `step_denominators` set, each of the module's terms is divided by its own
    total over the whole slice, so this is the same quantity the training objective
    is; without it, the historical token-weighted cross-entropy.
    """
    state.module.eval()
    if state.step_denominators is not None:
        terms: torch.Tensor | None = None
        counts: torch.Tensor | None = None
        for batch in loader:
            raw = cast(dict[str, torch.Tensor], batch)
            batch_counts = state.step_denominators([raw])
            with _autocast(state):
                batch_terms = state.module(**move_batch(raw, state.device)).double()
            terms = batch_terms if terms is None else terms + batch_terms
            counts = batch_counts if counts is None else counts + batch_counts
        state.module.train()
        if terms is None or counts is None:
            raise RuntimeError("Evaluation slice was empty.")
        if state.world_size > 1:
            dist.all_reduce(terms, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        return float((terms / counts.clamp_min(1.0)).sum().item())

    totals = torch.zeros(2, dtype=torch.float64, device=state.device)
    for batch in loader:
        moved = move_batch(cast(dict[str, torch.Tensor], batch), state.device)
        with _autocast(state):
            loss_sum = state.module(**moved)
        totals[0] += loss_sum.double()
        totals[1] += float((moved["labels"][:, 1:] != -100).sum())
    if state.world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    state.module.train()

    if totals[1].item() == 0:
        raise RuntimeError("Evaluation slice contained no supervised tokens.")
    return float(totals[0].item() / totals[1].item())


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def _rank_batches(
    plan: BatchPlan, first_micro: int, rank: int, world: int
) -> list[list[int]]:
    return [plan.micro[i] for i in range(first_micro + rank, len(plan.micro), world)]


def run_training(
    state: LoopState,
    train_dataset: Any,
    eval_dataset: Any,
    eval_batches: list[list[int]],
    make_loader: Callable[..., DataLoader],
    resume_dir: Path | None,
) -> Path:
    cfg = state.cfg
    plan = state.plan
    if resume_dir is not None:
        load_training_state(state, resume_dir)
        state.logger.info(f"Resumed at step {state.global_step}/{len(plan.steps)}")

    # A ceiling on this job only. It never enters plan_epoch, so stopping here and
    # resuming lands on exactly the curriculum an uninterrupted run would have had.
    limit = len(plan.steps)
    if cfg.training.max_steps is not None:
        limit = min(limit, int(cfg.training.max_steps))

    if state.global_step >= limit:
        state.logger.info("Nothing to do: checkpoint is already at the final step.")
        return state.checkpoint_root / "best"

    first_micro = plan.steps[state.global_step][0]
    train_loader = make_loader(
        train_dataset, _rank_batches(plan, first_micro, state.rank, state.world_size)
    )
    eval_loader = make_loader(
        eval_dataset, eval_batches[state.rank :: state.world_size]
    )

    state.module.train()
    _prime_gradient_buckets(state)
    debug_memory = bool(cfg.training.get("debug_memory", False))
    if debug_memory:
        state.logger.info(f"before first step | {_memory_report(state)}")
    batches = iter(train_loader)
    window_start = time.monotonic()
    window_tokens = 0

    for step in range(state.global_step, limit):
        start, stop = plan.steps[step]
        micro_per_rank = (stop - start) // state.world_size
        denom = plan.denom[step]
        running = torch.zeros((), dtype=torch.float64, device=state.device)

        # With per-term denominators the step's counts must be known before its
        # first backward, so the whole step is pulled off the loader first. These
        # are pinned CPU tensors straight from the workers -- a few MB -- and the
        # one all-reduce per step is free beside a ~10 s step.
        prefetched: list[dict[str, torch.Tensor]] = []
        denoms: torch.Tensor | None = None
        if state.step_denominators is not None:
            prefetched = [
                cast(dict[str, torch.Tensor], next(batches))
                for _ in range(micro_per_rank)
            ]
            denoms = state.step_denominators(prefetched)
            if state.world_size > 1:
                dist.all_reduce(denoms, op=dist.ReduceOp.SUM)
            # By convention the first denominator is the step's row count, which the
            # plan already knows. They can only disagree if preparation dropped a
            # row, and a silently smaller denominator would re-weight every other
            # row in the step -- so this is an error, not a correction.
            if plan.rows and int(denoms[0]) != plan.rows[step]:
                raise RuntimeError(
                    f"step {step} received {int(denoms[0])} rows but the plan says "
                    f"{plan.rows[step]}: preparation dropped rows, so the per-row "
                    "objective would be normalized by the wrong count."
                )
            denoms = denoms.clamp_min(1.0)

        for micro in range(micro_per_rank):
            raw = (
                prefetched[micro]
                if prefetched
                else cast(dict[str, torch.Tensor], next(batches))
            )
            batch = move_batch(raw, state.device)
            is_last = micro == micro_per_rank - 1
            sync = contextlib.nullcontext()
            if not is_last and state.world_size > 1:
                sync = cast(Any, state.module).no_sync()
            try:
                with sync:
                    with _autocast(state):
                        loss_sum = state.module(**batch)
                    normalized = (
                        loss_sum / denom
                        if denoms is None
                        else (loss_sum / denoms).sum()
                    )
                    # DDP all-reduces gradients as a mean over ranks, so scaling
                    # by world_size recovers the gradient of the global objective.
                    (normalized * state.world_size).backward()
            except torch.OutOfMemoryError as error:
                # A bare OOM traceback says nothing about which shape caused it,
                # and micro-batch shapes vary every step here. Without this you
                # cannot tell a too-long single row from a badly packed batch.
                raise torch.OutOfMemoryError(
                    f"OOM at step {step} micro {micro}/{micro_per_rank} "
                    f"shape={tuple(batch['input_ids'].shape)} "
                    f"({batch['input_ids'].numel()} padded tokens) | "
                    f"{_memory_report(state)}"
                ) from error
            # The NORMALIZED contribution, so `train/loss` is the objective itself
            # whichever denominators are in play.
            running += normalized.detach().double()
            if debug_memory and state.is_main:
                rows, width = batch["input_ids"].shape
                state.logger.info(
                    f"  micro {step}.{micro} rows={rows} width={width} "
                    f"padded_tokens={rows * width} | {_memory_report(state)}"
                )

        # The return was previously discarded. It is the single most useful number
        # for telling a gradient explosion from a forward-pass NaN, and both
        # encoder arms died at step ~780/~990 with no way to distinguish them.
        # Note clipping does NOT contain a NaN: clip_coef = max_norm/(NaN+1e-6) is
        # NaN, so every gradient becomes NaN. An *Inf* norm instead gives
        # clip_coef = 0 and zeroes the gradients. So a NaN here means a genuine
        # 0/0, inf-inf or 0*inf upstream, not a magnitude overflow.
        state.last_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                state.module.parameters(), max_norm=float(cfg.training.gradient_clip)
            )
        )
        state.optimizer.step()
        state.scheduler.step()
        # set_to_none=False, deliberately. DDP was built with
        # gradient_as_bucket_view=True so that .grad aliases the reduction
        # buckets instead of being a second 15 GiB copy. Setting .grad to None
        # destroys that aliasing, and the next backward allocates fresh gradient
        # tensors alongside the buckets -- measured at 45 GiB resident where 30
        # was expected. Zeroing in place keeps the views intact.
        state.optimizer.zero_grad(set_to_none=False)
        state.global_step = step + 1
        if state.post_step is not None:
            state.post_step(state)
        state.tokens_seen += denom
        window_tokens += denom

        if state.global_step % int(cfg.training.log_interval) == 0:
            if state.world_size > 1:
                dist.all_reduce(running, op=dist.ReduceOp.SUM)
            elapsed = max(time.monotonic() - window_start, 1e-9)
            state.logger.log_metrics(
                {
                    "train/loss": float(running.item()),
                    "train/lr": state.scheduler.get_last_lr()[0],
                    # plan.denom is already summed over ranks, so these are global
                    # counts. Scaling by world_size here would report 4x the real
                    # throughput and make a 30% MFU run look like 120%.
                    "train/tokens_per_sec": window_tokens / elapsed,
                    "train/tokens_seen": float(state.tokens_seen),
                    "train/grad_norm": state.last_grad_norm,
                    "train/gpu_mem_gb": (
                        torch.cuda.max_memory_allocated() / 2**30
                        if state.device.type == "cuda"
                        else 0.0
                    ),
                    **(state.extra_metrics(state) if state.extra_metrics else {}),
                },
                step=state.global_step,
            )
            window_start, window_tokens = time.monotonic(), 0

        if state.global_step % int(cfg.training.eval_interval) == 0:
            _evaluate(state, eval_loader)

        if _due_for_checkpoint(state):
            if state.is_main:
                save_checkpoint(state, full=True)
            _barrier(state)
            state._last_checkpoint_time = time.monotonic()

    _evaluate(state, eval_loader)
    # keep_checkpoints=0 means "this run produces no resumable state": sweep arms
    # exist only for their eval-loss curve, and a 48 GB write per arm costs real
    # GPU time and scratch for something that is deleted immediately. `best/` is
    # still written, so the arm remains inspectable.
    if state.is_main and int(cfg.training.keep_checkpoints) > 0:
        save_checkpoint(state, full=True)
    _barrier(state)
    return state.checkpoint_root / "best"


def _evaluate(state: LoopState, eval_loader: DataLoader) -> float:
    """Score the fixed slice; keep `best/` pointing at the lowest loss so far."""
    loss = estimate_loss(state, eval_loader)
    metrics = {state.eval_loss_key: loss}
    if state.eval_metrics is not None:
        metrics.update(state.eval_metrics(state, loss))
    state.logger.log_metrics(metrics, step=state.global_step)
    selected = (
        metrics.get(state.best_metric_key, loss) if state.best_metric_key else loss
    )
    # Rank 0 decides, and every rank then runs the same branch. The metrics are
    # all-reduced, so ranks SHOULD agree -- but the barrier below is a collective,
    # and a float compare that differed on one rank would leave it alone in a
    # barrier the others never enter.
    if _decided_on_rank0(state, selected < state.best_eval_loss):
        state.best_eval_loss = selected
        if state.is_main:
            save_checkpoint(state, full=False)
        _barrier(state)
    return loss


def _prime_gradient_buckets(state: LoopState) -> None:
    """Force DDP's first-iteration gradient allocation while activations are tiny.

    On the first backward `param.grad` does not yet alias DDP's reduction buckets
    -- DDP re-points them during `_rebuild_buckets` at the end of that iteration.
    So the first backward transiently allocates a *second* full set of gradients:
    15 GiB for this model, on top of params and buckets.

    Cold-starting that is harmless, because the AdamW moments do not exist yet
    (measured: 15 + 15 + 15 = 45 GiB, dropping to 30 once buckets are rebuilt).
    Resuming it is fatal: the moments are restored *before* the first backward, so
    the same transient lands on 60 GiB and a real 32k-token micro-batch then needs
    ~84 GiB of a 93 GiB card. That is an OOM on the first step after every
    requeue -- i.e. precisely when the run is supposed to recover.

    Running one negligible batch first pays that transient when activations are
    ~0, after which grads alias the buckets and steady state is back to 60 GiB.
    Gradients are zeroed afterwards, and neither the optimizer nor the scheduler
    is stepped, so training is unaffected.
    """
    if state.world_size <= 1:
        return
    if state.prime_batch is not None:
        state.prime_batch(state)
        return

    ids = torch.full((1, 8), 1, dtype=torch.long, device=state.device)
    labels = ids.clone()
    labels[:, :4] = -100
    with _autocast(state):
        loss = state.module(ids, torch.ones_like(ids), labels)
    (loss * 0.0).backward()
    state.optimizer.zero_grad(set_to_none=False)
    if state.is_main:
        state.logger.info(f"primed DDP buckets | {_memory_report(state)}")


def _memory_report(state: LoopState) -> str:
    if state.device.type != "cuda":
        return "cuda unavailable"
    gib = 1 << 30
    free, total = torch.cuda.mem_get_info(state.device)
    return (
        f"alloc={torch.cuda.memory_allocated(state.device) / gib:.2f} "
        f"reserved={torch.cuda.memory_reserved(state.device) / gib:.2f} "
        f"peak={torch.cuda.max_memory_allocated(state.device) / gib:.2f} "
        f"free={free / gib:.2f}/{total / gib:.2f} GiB"
    )


def _due_for_checkpoint(state: LoopState) -> bool:
    """Decide on rank 0 and broadcast, never per rank.

    The interval is wall-clock, and wall clocks drift between ranks. Evaluating
    it independently lets rank 0 cross the threshold at step N while rank 1
    crosses it at N+1: rank 0 then enters the save barrier alone and the job
    hangs until the walltime kills it. One broadcast byte per step is a rounding
    error against a ~27 s step; a deadlock 40 hours into a 45,000 SBU run is not.
    """
    elapsed = (time.monotonic() - state._last_checkpoint_time) / 60.0
    return _decided_on_rank0(
        state, elapsed >= float(state.cfg.training.checkpoint_minutes)
    )


def _decided_on_rank0(state: LoopState, local: bool) -> bool:
    """Rank 0's value of `local`, broadcast, for any decision that gates a collective."""
    if state.world_size <= 1:
        return local
    flag = torch.tensor([int(local)], dtype=torch.uint8, device=state.device)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def _barrier(state: LoopState) -> None:
    if state.world_size > 1 and dist.is_initialized():
        dist.barrier()
