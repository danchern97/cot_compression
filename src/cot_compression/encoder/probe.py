"""Measure the aux pass on the shapes a run will ACTUALLY see, before paying for it.

Every production failure of the next-patch objective came from shapes no test
exercised: a width that drove Dynamo into symbolic lowering, a padded width whose
score matrix could not fit, a row long enough that building its mask alone ran out
of memory. So this module does not construct representative batches. It builds the
run's own batch plan with `build_encoder_data` -- the code `train_encoder` uses --
and reports on every micro-batch in it.

    uv run python -m cot_compression.encoder.probe census [hydra overrides...]

writes `<paths.output_dir>/probe/census_<plan signature>.json`: per-micro-batch
geometry, summary distributions, the distinct compiled shapes each candidate width
ladder would produce, and the worst micro-batches by width, by `B*T^2` (attention),
and by `B*T` (activations), which the GPU gates then run.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest import mock

import numpy as np
import torch
import torch._dynamo
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from cot_compression.compression import _regular_vocab_bound as regular_vocab_bound
from cot_compression.encoder.frozen import FrozenBackbone, configure_aux_compilation
from cot_compression.encoder.model import CoTEncoder
from cot_compression.encoder.next_patch import bucket_width
from cot_compression.encoder.training import (
    EncoderData,
    EncoderTrainingModule,
    PreparedEncoderSample,
    _build_loader,
    build_encoder_config,
    build_encoder_data,
    drop_unprepared,
    embedding_rms,
    eval_baselines,
)
from cot_compression.training.sft import (
    build_optimizer,
    chunked_ce_flat,
    parse_torch_dtype,
)
from cot_compression.training.utils import disable_cudnn_sdpa, optional_int

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

# Candidate width ladders, as round-up multiples. `None` is the natural width. A
# power-of-two ladder is deliberately absent: it nearly doubles T in the worst case,
# and memory grows with T^2 -- that is what OOM'd the Sep 11 relaunch at T=16384.
LADDERS: tuple[int | None, ...] = (None, 256, 512, 1024, 2048)
WORST_K = 8

log = logging.getLogger("cot_compression.probe")


def compose_encoder_cfg(overrides: list[str]) -> DictConfig:
    """The encoder workflow's config, composed exactly as `scripts/run.py` would.

    `return_hydra_config` plus `HydraConfig.set_config` is what lets
    `${hydra:runtime.cwd}` (in `paths.project_root`) resolve outside `@hydra.main`.
    """
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="run",
            overrides=["workflow=encoder_train", *overrides],
            return_hydra_config=True,
        )
    HydraConfig.instance().set_config(cfg)
    return cfg


@dataclass(frozen=True)
class RowShape:
    context: int
    render: int
    prefix: int
    aux: int


def _row_shapes(samples: list[PreparedEncoderSample | None]) -> list[RowShape]:
    """DataLoader collate: keep only lengths, so workers ship a few ints per row."""
    return [
        RowShape(
            context=len(s.context_ids),
            render=len(s.input_ids),
            prefix=s.prefix_len,
            aux=len(s.aux_ids),
        )
        for s in drop_unprepared(samples)
    ]


def micro_geometry(rows: list[RowShape]) -> dict[str, int]:
    """The shapes one micro-batch puts on the GPU.

    `aux_width` is the column-indexed layout, where each row's aux block starts at its
    OWN prefix length, so the width is `max(prefix + aux)`. `aux_width_concat` is the
    old layout (`max(prefix) + max(aux)`), kept to show what that layout cost.
    """
    return {
        "rows": len(rows),
        "pass_a_width": max(r.context for r in rows),
        "pass_b_width": max(r.render for r in rows),
        "aux_width": max(r.prefix + r.aux for r in rows),
        "aux_width_concat": max(r.prefix for r in rows) + max(r.aux for r in rows),
        "aux_tokens": sum(r.aux for r in rows),
    }


def ladder_report(micros: list[dict[str, int]]) -> dict[str, dict[str, float]]:
    """What each candidate ladder costs: distinct compiled shapes vs padding waste."""
    rows = np.array([m["rows"] for m in micros], dtype=np.float64)
    width = np.array([m["aux_width"] for m in micros], dtype=np.float64)
    natural_tokens = float((rows * width).sum())
    natural_attention = float((rows * width**2).sum())
    report = {}
    for multiple in LADDERS:
        padded = np.array(
            [bucket_width(int(w), multiple) for w in width], dtype=np.float64
        )
        report[str(multiple)] = {
            "distinct_shapes": len(
                {(int(b), int(w)) for b, w in zip(rows, padded, strict=True)}
            ),
            "distinct_widths": len(set(padded.tolist())),
            "token_overhead": float((rows * padded).sum()) / natural_tokens,
            "attention_overhead": float((rows * padded**2).sum()) / natural_attention,
            "max_width": int(padded.max()),
        }
    return report


def worst(
    micros: list[dict[str, int]], steps: list[tuple[int, int]] | None, world_size: int
) -> dict[str, list[dict]]:
    """Top micro-batches per pressure, with the rank and optimizer step running each.

    `steps` is `plan.steps` for train (`None` for eval, which has no optimizer step);
    a DDP gate needs the whole step, since every rank's micro-batch runs together.
    """
    step_of = {}
    for step, (start, stop) in enumerate(steps or []):
        for micro in range(start, stop):
            step_of[micro] = step
    keys = {
        "width": lambda m: m["aux_width"],
        "attention": lambda m: m["rows"] * m["aux_width"] ** 2,
        "activations": lambda m: m["rows"] * m["aux_width"],
    }
    out = {}
    for name, key in keys.items():
        order = sorted(range(len(micros)), key=lambda i: key(micros[i]), reverse=True)
        out[name] = [
            {
                "micro": i,
                "rank": i % world_size,
                "step": step_of.get(i),
                **micros[i],
            }
            for i in order[:WORST_K]
        ]
    return out


def distribution(values: list[int]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "max": int(array.max()),
        **{f"p{q}": float(np.percentile(array, q)) for q in (50, 90, 99, 99.9)},
    }


def measure(batches: list[list[int]], dataset: Any) -> list[dict]:
    loader = DataLoader(
        dataset,
        batch_sampler=batches,
        collate_fn=_row_shapes,
        num_workers=max(1, (os.cpu_count() or 2) - 1),
    )
    micros = []
    for done, rows in enumerate(loader):
        if not rows:
            raise RuntimeError(f"micro-batch {done} prepared no rows")
        micros.append(micro_geometry(rows))
        if done % 2000 == 0:
            log.info("  %d/%d micro-batches", done, len(batches))
    return micros


def _tokenizer_and_data(cfg: DictConfig, world_size: int) -> tuple[Any, EncoderData]:
    """The tokenizer and `build_encoder_data`, exactly as `train_encoder` builds them."""
    model_name = str(cfg.method.model_name)
    tokenizer = cast(
        Any,
        AutoTokenizer.from_pretrained(
            model_name, use_fast=bool(cfg.method.use_fast_tokenizer)
        ),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_rows = int(AutoConfig.from_pretrained(model_name).vocab_size)
    data = build_encoder_data(
        cfg,
        tokenizer,
        vocab_bound=regular_vocab_bound(tokenizer, vocab_rows),
        latent_init=str(cfg.encoder.latent_init),
        next_patch_subsample=float(cfg.encoder.next_patch_subsample),
        world_size=world_size,
    )
    return tokenizer, data


def _output_dir(cfg: DictConfig) -> Path:
    out = Path(str(cfg.paths.output_dir)) / "probe"
    out.mkdir(parents=True, exist_ok=True)
    return out


def census(overrides: list[str], world_size: int = 4, limit: int | None = None) -> Path:
    cfg = compose_encoder_cfg(overrides)
    _, data = _tokenizer_and_data(cfg, world_size)
    splits = {
        "train": (data.plan.micro[:limit], "train", data.plan.steps),
        "eval": (data.eval_batches[:limit], "val", None),
    }
    result: dict[str, Any] = {
        "plan_signature": data.plan.signature,
        "world_size": world_size,
        "overrides": overrides,
    }
    for name, (batches, split, steps) in splits.items():
        log.info("%s: %d micro-batches", name, len(batches))
        dataset = data.prep(
            dataset=data.measured[split], signals=data.signals(split, log)
        )
        micros = measure(batches, dataset)
        result[name] = {
            "micro_batches": len(micros),
            "distributions": {
                key: distribution([m[key] for m in micros]) for key in micros[0]
            },
            "ladders": ladder_report(micros),
            "worst": worst(micros, steps, world_size),
            "micros": micros,
        }

    suffix = "" if limit is None else f"_limit{limit}"
    path = _output_dir(cfg) / f"census_{data.plan.signature}{suffix}.json"
    path.write_text(json.dumps(result, indent=1))
    summary = {
        name: {key: value for key, value in body.items() if key != "micros"}
        for name, body in result.items()
        if isinstance(body, dict)
    }
    log.info("%s", json.dumps(summary, indent=1))
    log.info("wrote %s", path)
    return path


# --------------------------------------------------------------------------- #
# GPU gate: the real model on the census's worst micro-batches
# --------------------------------------------------------------------------- #


def naive_next_patch_loss(
    backbone: FrozenBackbone,
    spliced: Tensor,
    aux_ids: Tensor,
    aux_patch: Tensor,
    slot_positions: Tensor,
) -> Tensor:
    """The specification executed literally: one sequence per (row, patch).

    `[row b's prefix through z_{m-1}; patch m's tokens]` with an ordinary causal mask
    and default positions, first token predicted from the code, the rest from their
    predecessor. Quadratic in the number of patches, so only for small checks -- it
    is the oracle both the unit test and the real-model gate compare against.

    Returns `[sum_i mean_j mean_t CE, sum_t CE]`, the same pair the production path
    returns: the per-row/per-patch means built here by literally averaging, against
    a single weighted sum there.
    """
    table, lm_head = backbone.embedding_weight, backbone.model.lm_head
    row_total = spliced.new_zeros((), dtype=torch.float32)
    token_total = spliced.new_zeros((), dtype=torch.float32)
    for row in range(aux_patch.shape[0]):
        valid = aux_patch[row] >= 0
        patches, ids = aux_patch[row][valid], aux_ids[row][valid]
        unique = torch.unique(patches).tolist()
        patch_means = spliced.new_zeros((), dtype=torch.float32)
        for patch in unique:
            tokens = ids[patches == patch]
            limit = int(slot_positions[row, patch - 1])
            seq = torch.cat(
                [
                    spliced[row : row + 1, : limit + 1],
                    table[tokens].detach().to(spliced.dtype).unsqueeze(0),
                ],
                dim=1,
            )
            ones = torch.ones(seq.shape[:2], dtype=torch.long, device=seq.device)
            hidden = backbone._decode(seq, ones)
            start = seq.shape[1] - tokens.numel()
            patch_sum = chunked_ce_flat(
                lm_head, hidden[0, start - 1 : -1], tokens, backbone.ce_chunk_tokens
            )
            token_total = token_total + patch_sum
            patch_means = patch_means + patch_sum / tokens.numel()
        if unique:
            row_total = row_total + patch_means / len(unique)
    return torch.stack([row_total, token_total])


def _flex_cache_entries() -> int:
    """Compiled cache entries for `flex_attention` -- one per distinct static shape."""
    return len(torch._dynamo.eval_frame._debug_get_cache_entry_list(flex_attention))


@dataclass
class _Rig:
    cfg: DictConfig
    data: EncoderData
    module: EncoderTrainingModule
    optimizer: Any
    pad_token_id: int
    device: torch.device


def _build_backbone(cfg: DictConfig, device: torch.device) -> FrozenBackbone:
    """The frozen decoder exactly as `train_encoder` loads and wraps it."""
    disable_cudnn_sdpa(device)
    decoder = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            str(cfg.method.model_name),
            dtype=parse_torch_dtype(str(cfg.training.decoder_dtype)),
            attn_implementation=str(cfg.training.attn_implementation),
        ),
    ).to(device)
    return FrozenBackbone(
        decoder,
        memory_layer=int(cfg.encoder.memory_layer),
        ce_chunk_tokens=int(cfg.training.ce_chunk_tokens),
        gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
    )


def _build_rig(cfg: DictConfig, world_size: int) -> _Rig:
    """Model, encoder, optimizer and data, assembled with `train_encoder`'s own parts."""
    device = torch.device("cuda")
    torch.set_num_threads(1)
    tokenizer, data = _tokenizer_and_data(cfg, world_size)
    backbone = _build_backbone(cfg, device)
    log.info(
        "aux compilation: %s",
        configure_aux_compilation(int(cfg.training.aux_recompile_limit)),
    )
    bound = regular_vocab_bound(tokenizer, backbone.embedding_weight.shape[0])
    encoder_config = build_encoder_config(
        cfg, backbone.hidden_size, embedding_rms(backbone.embedding_weight, bound)
    )
    encoder = CoTEncoder(encoder_config).to(device)
    module = EncoderTrainingModule(
        encoder,
        backbone,
        commit_weight=encoder_config.commit_weight,
        next_patch_weight=encoder_config.next_patch_weight,
        aux_width_multiple=optional_int(cfg.training.aux_width_multiple),
    )
    return _Rig(
        cfg=cfg,
        data=data,
        module=module,
        optimizer=build_optimizer(cfg, encoder),
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
    )


def _batch(rig: _Rig, dataset: Any, rows: list[int]) -> dict[str, Tensor]:
    loader = _build_loader(dataset, [rows], rig.cfg, rig.pad_token_id)
    return {k: v.to(rig.device) for k, v in next(iter(loader)).items()}


def _run_micro(rig: _Rig, batch: dict[str, Tensor], train: bool) -> dict[str, float]:
    """One micro-batch as training runs it: autocast forward, backward, AdamW step."""
    gib = 2**30
    rig.module.train(train)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    entries = _flex_cache_entries()
    start = time.perf_counter()
    with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16):
        # The module returns one sum per objective term; this measures cost and
        # finiteness, so the global denominators that weigh them are irrelevant
        # here and a plain sum is the cheapest scalar to backward from.
        loss = rig.module(**batch).sum()
    torch.cuda.synchronize()
    forward_peak = torch.cuda.max_memory_allocated()
    finite = bool(torch.isfinite(loss))
    if train:
        loss.backward()
        grads = [p.grad for p in rig.module.encoder.parameters() if p.grad is not None]
        finite = finite and all(bool(torch.isfinite(g).all()) for g in grads)
        rig.optimizer.step()
        rig.optimizer.zero_grad(set_to_none=False)
    torch.cuda.synchronize()
    return {
        "seconds": time.perf_counter() - start,
        "forward_peak_gib": (forward_peak - before) / gib,
        "peak_gib": (torch.cuda.max_memory_allocated() - before) / gib,
        "peak_abs_gib": torch.cuda.max_memory_allocated() / gib,
        "reserved_gib": torch.cuda.memory_reserved() / gib,
        "new_flex_entries": _flex_cache_entries() - entries,
        "finite": finite,
    }


def _chosen(census_split: dict, sample: int, seed: int) -> list[int]:
    picked: list[int] = []
    for rows in census_split["worst"].values():
        picked += [r["micro"] for r in rows[:4] if r["micro"] not in picked]
    rng = np.random.default_rng(seed)
    pool = [i for i in range(census_split["micro_batches"]) if i not in picked]
    picked += [
        int(i) for i in rng.choice(pool, size=min(sample, len(pool)), replace=False)
    ]
    return picked


def numeric_check(
    rig: _Rig, batch: dict[str, Tensor], patches: int
) -> dict[str, float]:
    """Production aux pass vs the naive loop on ONE real row, in fp32, self-calibrated.

    The error of the correct path is only meaningful next to the error of a known bug
    on the same row, so the check also re-runs with that row's aux RoPE positions off
    by one -- the failure class the column-indexed geometry exists to prevent -- and
    the gate is that the correct error is at least 10x smaller.
    """
    import dataclasses

    import cot_compression.encoder.training as training_module

    backbone = rig.module.backbone
    backbone.model.float()
    try:
        row = 0
        aux_patch = batch["aux_patch"][row : row + 1].clone()
        aux_patch[aux_patch > patches] = -1
        aux_ids = torch.where(aux_patch >= 0, batch["aux_ids"][row : row + 1], -100)
        slots = batch["slot_positions"][row : row + 1]
        prefix_len = batch["prefix_len"][row : row + 1]
        spliced = (
            backbone.splice(
                batch["input_ids"][row : row + 1],
                torch.randn(1, slots.shape[1], backbone.hidden_size, device=rig.device)
                * 0.03,
                slots,
                batch["slot_mask"][row : row + 1],
            )
            .float()
            .requires_grad_()
        )

        # Index 0 of both: the per-row/per-patch mean, which is the term that
        # carries the gradient to the codes.
        expected = naive_next_patch_loss(backbone, spliced, aux_ids, aux_patch, slots)[
            0
        ]
        (want,) = torch.autograd.grad(expected, spliced)

        def errors(corrupt: bool) -> tuple[float, float]:
            original = training_module.build_next_patch_batch

            def build(**kwargs: Any) -> Any:
                plan = original(**kwargs)
                if not corrupt:
                    return plan
                position = plan.position_ids.clone()
                position[plan.patch >= 0] += 1
                return dataclasses.replace(plan, position_ids=position)

            with mock.patch.object(training_module, "build_next_patch_batch", build):
                terms, _, _ = rig.module._next_patch_loss(
                    spliced, aux_ids, aux_patch, prefix_len, slots
                )
            actual = terms[0]
            (got,) = torch.autograd.grad(actual, spliced)
            loss_error = abs(float(actual.detach() - expected.detach())) / abs(
                float(expected.detach())
            )
            return loss_error, float((got - want).abs().max() / want.abs().max())

        correct, corrupt = errors(False), errors(True)
    finally:
        backbone.model.to(parse_torch_dtype(str(rig.cfg.training.decoder_dtype)))
    return {
        "loss_error": correct[0],
        "grad_error": correct[1],
        "bug_loss_error": corrupt[0],
        "bug_grad_error": corrupt[1],
        "passes": correct[0] * 10 < corrupt[0] and correct[1] * 10 < corrupt[1],
    }


def gpu(overrides: list[str], census_path: Path, sample: int = 12) -> Path:
    cfg = compose_encoder_cfg(["training.num_workers=0", *overrides])
    census_result = json.loads(census_path.read_text())
    rig = _build_rig(cfg, int(census_result["world_size"]))
    if rig.data.plan.signature != census_result["plan_signature"]:
        raise ValueError("census was built for a different batch plan")

    result: dict[str, Any] = {"census": str(census_path), "overrides": overrides}
    for split, batches, source, train in (
        ("train", rig.data.plan.micro, "train", True),
        ("eval", rig.data.eval_batches, "val", False),
    ):
        dataset = rig.data.prep(
            dataset=rig.data.measured[source],
            signals=rig.data.signals(source, log),
        )
        rows = []
        for index in _chosen(census_result[split], sample if train else 4, seed=0):
            batch = _batch(rig, dataset, batches[index])
            geometry = census_result[split]["micros"][index]
            first = _run_micro(rig, batch, train)
            again = _run_micro(rig, batch, train)
            rows.append({"micro": index, **geometry, "first": first, "again": again})
            log.info(
                "%s micro %d rows=%d aux_width=%d | %.1fs then %.1fs | peak %.1f GiB "
                "(abs %.1f) | new flex entries %d then %d | finite %s",
                split,
                index,
                geometry["rows"],
                geometry["aux_width"],
                first["seconds"],
                again["seconds"],
                again["peak_gib"],
                again["peak_abs_gib"],
                first["new_flex_entries"],
                again["new_flex_entries"],
                first["finite"] and again["finite"],
            )
            if split == "train" and "numeric" not in result:
                result["numeric"] = numeric_check(rig, batch, patches=6)
                log.info("numeric check: %s", result["numeric"])
        result[split] = rows

    worst_peak = max(r["again"]["peak_abs_gib"] for r in result["train"])
    result["gate"] = {
        "worst_peak_abs_gib": worst_peak,
        "peak_under_80_gib": worst_peak <= 80.0,
        "no_recompile_on_repeat": all(
            r["again"]["new_flex_entries"] == 0
            for split in ("train", "eval")
            for r in result[split]
        ),
        "all_finite": all(
            r[k]["finite"]
            for split in ("train", "eval")
            for r in result[split]
            for k in ("first", "again")
        ),
        "numeric_passes": result["numeric"]["passes"],
        "flex_cache_entries": _flex_cache_entries(),
    }
    log.info("GATE: %s", result["gate"])
    job = os.environ.get("SLURM_JOB_ID", "local")
    path = _output_dir(cfg) / f"gpu_{census_result['plan_signature']}_{job}.json"
    path.write_text(json.dumps(result, indent=1))
    log.info("wrote %s", path)
    return path


def baselines(overrides: list[str], world_size: int = 4) -> Path:
    """The eval baselines alone -- no encoder, no training -- on a run's exact slice.

    Calls `eval_baselines`, the same function `train_encoder` runs at startup, with
    the batch plan built for `world_size` ranks, so the numbers are the ones that run
    would log. Reproducing an already-logged baseline is the check that the slice is
    the same one.
    """
    cfg = compose_encoder_cfg(overrides)
    device = torch.device("cuda")
    tokenizer, data = _tokenizer_and_data(cfg, world_size)
    backbone = _build_backbone(cfg, device)
    start = time.perf_counter()
    losses = eval_baselines(backbone, tokenizer, data, device, log)
    rows = sum(len(batch) for batch in data.eval_batches)
    result = {
        "plan_signature": data.plan.signature,
        "world_size": world_size,
        "eval_micro_batches": len(data.eval_batches),
        "eval_rows": rows,
        "overrides": overrides,
        # Both normalizations, as the run logs them: "row" is the objective's own
        # (and the eval harness's `mean_logprob`), "tok" the per-answer-token figure.
        "losses": losses,
        "gaps_to_no_cot": {
            unit: {
                name: losses["no_cot"][unit] - values[unit]
                for name, values in losses.items()
            }
            for unit in ("row", "tok")
        },
        "seconds": time.perf_counter() - start,
    }
    log.info("BASELINES: %s", json.dumps(result, indent=1))
    job = os.environ.get("SLURM_JOB_ID", "local")
    path = _output_dir(cfg) / f"baselines_{data.plan.signature}_{job}.json"
    path.write_text(json.dumps(result, indent=1))
    log.info("wrote %s", path)
    return path


def repro_dynamic(
    overrides: list[str], census_path: Path, minutes: float = 10.0
) -> None:
    """Try to reproduce the Sep 8 crash: flex compiled with AUTOMATIC dynamic shapes.

    Runs distinct-width train micro-batches through the unmodified production step
    with `_compiled_flex_attention` swapped for transformers' flags
    (`torch.compile(flex_attention)`, no `dynamic`). Bounded in time; reports either
    the exception or that none occurred -- both are informative.
    """
    import cot_compression.encoder.frozen as frozen_module

    cfg = compose_encoder_cfg(["training.num_workers=0", *overrides])
    census_result = json.loads(census_path.read_text())
    rig = _build_rig(cfg, int(census_result["world_size"]))
    torch._dynamo.reset()
    dynamic = torch.compile(cast(Any, flex_attention))
    with (
        mock.patch.object(frozen_module, "_compiled_flex_attention", lambda: dynamic),
        torch._dynamo.config.patch(
            fail_on_recompile_limit_hit=False, recompile_limit=64
        ),
    ):
        dataset = rig.data.prep(
            dataset=rig.data.measured["train"], signals=rig.data.signals("train", log)
        )
        micros = census_result["train"]["micros"]
        seen, deadline = set(), time.monotonic() + minutes * 60
        for index in np.random.default_rng(1).permutation(len(micros)).tolist():
            width = micros[index]["aux_width"]
            if width in seen or time.monotonic() > deadline:
                continue
            seen.add(width)
            try:
                _run_micro(rig, _batch(rig, dataset, rig.data.plan.micro[index]), True)
                log.info("repro: width %d ok (%d distinct so far)", width, len(seen))
            except Exception as error:  # noqa: BLE001 -- reporting is the point
                log.error(
                    "REPRODUCED at width %d after %d widths: %s: %s",
                    width,
                    len(seen),
                    type(error).__name__,
                    str(error)[:500],
                )
                return
    log.info("NOT reproduced: %d distinct widths in %.0f min", len(seen), minutes)


USAGE = (
    "usage: python -m cot_compression.encoder.probe census [--limit N] [overrides]\n"
    "       python -m cot_compression.encoder.probe gpu CENSUS_JSON [overrides]\n"
    "       python -m cot_compression.encoder.probe repro CENSUS_JSON [overrides]\n"
    "       python -m cot_compression.encoder.probe baselines [overrides]"
)


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not argv:
        raise SystemExit(USAGE)
    mode, args = argv[0], argv[1:]
    if mode == "census":
        limit = None
        if args[:1] == ["--limit"]:
            limit, args = int(args[1]), args[2:]
        census(args, limit=limit)
    elif mode == "gpu" and args:
        gpu(args[1:], Path(args[0]))
    elif mode == "repro" and args:
        repro_dynamic(args[1:], Path(args[0]))
    elif mode == "baselines":
        baselines(args)
    else:
        raise SystemExit(USAGE)


if __name__ == "__main__":
    main(sys.argv[1:])
