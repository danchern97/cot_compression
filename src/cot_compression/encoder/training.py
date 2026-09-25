"""Per-sample CPU prep, batching, and the module DDP wraps.

`EncoderPrepDataset` is a deliberate twin of `evaluate.PrepDataset` rather than a
generalization of it. That one sits on the eval hot path and carries the
paired-comparison skip semantics; making it serve two masters is how those get
broken quietly. The duplication is small and the repo prefers explicit research
code in training and eval loops.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass, field, fields
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
from datasets import DatasetDict
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from cot_compression.compression import (
    PLACEHOLDER_TOKEN,
    build_patching_method,
    compressed_messages,
    random_slot_token_ids,
)
from cot_compression.compression import _regular_vocab_bound as regular_vocab_bound
from cot_compression.data.answers import (
    cot_token_ids,
    extract_answer_trace,
    prefix_token_ids,
    tokenize_answer,
)
from cot_compression.data.chat import IGNORE_INDEX
from cot_compression.data.dolci_traces import load_trace_lengths
from cot_compression.encoder.frozen import FrozenBackbone, configure_aux_compilation
from cot_compression.encoder.model import (
    LATENT_INITS,
    CoTEncoder,
    EncoderConfig,
    LatentInit,
    MaskMode,
    PositionEncoding,
)
from cot_compression.encoder.next_patch import (
    NextPatchBatch,
    aux_attention_mask,
    bucket_width,
    build_next_patch_batch,
    natural_width,
    patch_weights,
)
from cot_compression.patching import PatchingMethod
from cot_compression.signals import load_signal_cache, signal_cache_path
from cot_compression.training.logging import RunLogger
from cot_compression.training.sft import (
    BatchPlan,
    build_optimizer,
    init_distributed,
    parse_torch_dtype,
    plan_epoch,
    plan_eval_batches,
)
from cot_compression.training.sft_loop import (
    LoopState,
    resolve_resume_dir,
    run_training,
)
from cot_compression.training.utils import (
    disable_cudnn_sdpa,
    exit_failed_rank,
    get_run_dir,
    optional_int,
    save_resolved_config,
    set_seed,
)

# The initializers that reduce to ONE token id per slot, and so can be prepared in
# a DataLoader worker with no embedding table. `simple_mean` averages embeddings
# and is excluded.
TOKEN_LATENT_INITS = tuple(n for n in LATENT_INITS if n != "simple_mean")

# Cap on the k-means sample. Bounds the [N, |C|] distance matrix at 268 MB, and is
# also the target: >=64 points per centroid at |C|=1024, ~1000 at |C|=64.
_KMEANS_MAX_SAMPLES = 65536
# Micro-batches the seeding pass may consume before giving up on reaching the
# target. Only binds when the shuffled plan opens with very short rows.
_KMEANS_MAX_BATCHES = 64

ENCODER_WEIGHTS = "encoder.pt"
ENCODER_CONFIG = "encoder_config.json"
# The fields that make a checkpoint a *different encoder* rather than a later step
# of the same one. Compared, not hashed, so the error can name what moved.
_ARCHITECTURE_FIELDS = (
    "d_llm",
    "n_blocks",
    "n_heads",
    "ffn_mult",
    "codebook_size",
    "embed_rms",
    "latent_init",
    "self_attn_mask",
    "cross_attn_mask",
    # RoPE adds no parameters, so a checkpoint from before it existed loads
    # silently and then computes something else entirely. These two make that a
    # refusal instead.
    "position_encoding",
    "rope_theta",
)


@dataclass
class PreparedEncoderSample:
    """One rollout, tokenized and ready for the GPU.

    `context_ids` is Pass A's input (`[prompt; CoT]`, answer excluded);
    `input_ids` is Pass B's compressed render. They are separate sequences, not
    slices of one, which is why both are carried.
    """

    sample_index: int
    context_ids: list[int]
    input_ids: list[int]
    labels: list[int]
    slot_positions: list[int]
    # prompt_len + span_end per slot: the exclusive bound on the memory index a
    # slot may attend to under a causal cross mask.
    cross_limit: list[int]
    init_ids: list[int]
    # Auxiliary next-patch supervision. `aux_ids` are the CoT tokens of the
    # SUPERVISED patches, concatenated in order; `aux_patch` gives each one its
    # patch index. Patch 0 never appears -- it has no preceding code, so "predict
    # it from the codes before it" is undefined. Empty when the aux loss is off.
    aux_ids: list[int]
    aux_patch: list[int]
    prefix_len: int
    num_slots: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_slots = len(self.slot_positions)


class EncoderPrepDataset(TorchDataset["PreparedEncoderSample | None"]):
    """Tokenization and span planning, in DataLoader worker processes.

    Holds no model and touches no CUDA state, so it is fork-safe. Returns None for
    a sample that cannot be prepared; the caller counts those.
    """

    def __init__(
        self,
        dataset: Any,
        tokenizer: Any,
        patching: PatchingMethod,
        *,
        seed: int,
        max_length: int | None,
        vocab_bound: int,
        latent_init: str,
        signals: dict[int, Tensor] | None = None,
        next_patch_subsample: float = 1.0,
    ) -> None:
        if latent_init not in TOKEN_LATENT_INITS:
            raise NotImplementedError(
                f"latent_init={latent_init!r} averages CoT embeddings, which needs "
                "the embedding table and so cannot be prepared worker-side. "
                f"{TOKEN_LATENT_INITS} all reduce to one token id per slot."
            )
        needed = {patching.required_signal()} - {None}
        if latent_init.endswith("_t0"):
            needed.add(latent_init.removesuffix("_t0"))
        if needed and signals is None:
            raise ValueError(
                f"Patching {patching.name!r} / latent_init {latent_init!r} need the "
                f"{sorted(needed)} signal(s); pass a loaded cache. Build one with:\n"
                "  uv run python scripts/run.py precompute data=dolci_compression_traces"
            )
        self.signals = signals
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.patching = patching
        self.seed = seed
        self.max_length = max_length
        self.vocab_bound = vocab_bound
        self.latent_init = latent_init
        self.next_patch_subsample = next_patch_subsample
        self.placeholder_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> PreparedEncoderSample | None:
        example = self.dataset[index]
        trace = extract_answer_trace(example["messages"])
        if trace is None:
            return None
        try:
            cot_ids = cot_token_ids(trace, self.tokenizer)
        except ValueError:
            return None
        prompt_ids = prefix_token_ids(trace, self.tokenizer)

        # Spans are resolved here, on CPU, rather than on the eval device as
        # `evaluate.signal_span_counts` does. That path exists so a *paired*
        # comparison cannot have a boundary shift between methods from CPU/GPU
        # float differences. Training has no such pairing, and the alternative --
        # resolving centrally -- would mean shipping K span ends per row for
        # 176k rows, which is larger than the corpus.
        values = None
        if self.signals is not None:
            values = self.signals.get(index)
            if values is None:
                return None
        spans = self.patching.split(len(cot_ids), index, self.seed, values)
        num_slots = len(spans)
        tokenized = tokenize_answer(
            tokenizer=self.tokenizer,
            messages=compressed_messages(trace, num_slots),
            answer=trace.answer,
            max_length=self.max_length,
        )
        if tokenized is None:
            return None

        slot_positions = [
            position
            for position, token_id in enumerate(tokenized.input_ids)
            if token_id == self.placeholder_id
        ]
        if len(slot_positions) != num_slots:
            # Truncation cut placeholders off; skipping beats misaligning.
            return None
        if not any(label != IGNORE_INDEX for label in tokenized.labels):
            return None

        prompt_len = len(prompt_ids)
        aux_ids, aux_patch = self._aux_targets(cot_ids, spans, index)
        return PreparedEncoderSample(
            sample_index=index,
            context_ids=prompt_ids + cot_ids,
            input_ids=tokenized.input_ids,
            labels=tokenized.labels,
            slot_positions=slot_positions,
            cross_limit=[prompt_len + end for _, end in spans],
            init_ids=self._init_ids(cot_ids, spans, index, values),
            aux_ids=aux_ids,
            aux_patch=aux_patch,
            # Pass B's own prefix, through the last code. Reusing it means the codes
            # sit in exactly the context they occupy when the answer is scored.
            prefix_len=slot_positions[-1] + 1,
        )

    def _aux_targets(
        self, cot_ids: list[int], spans: list[tuple[int, int]], index: int
    ) -> tuple[list[int], list[int]]:
        """CoT tokens of the supervised patches, with their patch indices.

        Patch 0 is excluded: it has no preceding code. Subsampling is keyed on
        `seed + sample_index`, matching every other random choice in this repo --
        and it is *exactly* equivalent to re-sampling each step, because a run of at
        most one epoch visits each row once. Dropping a patch removes its tokens
        from the sequence entirely, so the rate scales the decoder cost, not just
        the `lm_head`.
        """
        if self.next_patch_subsample <= 0.0:
            return [], []
        rng = (
            None
            if self.next_patch_subsample >= 1.0
            else np.random.default_rng(self.seed + index)
        )
        ids: list[int] = []
        patch: list[int] = []
        for position, (start, end) in enumerate(spans):
            if position == 0:
                continue
            if rng is not None and rng.random() >= self.next_patch_subsample:
                continue
            ids.extend(cot_ids[start:end])
            patch.extend([position] * (end - start))
        return ids, patch

    def _init_ids(
        self,
        cot_ids: list[int],
        spans: list[tuple[int, int]],
        index: int,
        values: Tensor | None,
    ) -> list[int]:
        """One token id per slot; the main process turns these into embeddings.

        Both supported initializers reduce to a token choice, which is what lets
        them share one batch field and one vectorized embedding lookup:

        * `random` draws from the regular vocabulary, via the same helper
          `RandomCompressionMethod` uses, so step 0 is the measured `random`
          baseline rather than a lookalike;
        * `surprisal_t0` takes the highest-surprisal token in each span. That is
          exactly what `SignalWeightedMeanCompressionMethod` computes at T=0 -- the
          softmax collapses to a one-hot and the variance rescale divides by 1 --
          so the two agree by construction, not by reimplementation.
        """
        if self.latent_init == "random":
            return random_slot_token_ids(len(spans), index, self.seed, self.vocab_bound)
        assert values is not None, f"{self.latent_init} requires its signal"
        return [
            cot_ids[start + int(values[start:end].argmax())] for start, end in spans
        ]


def _pad(rows: list[list[int]], value: int) -> Tensor:
    width = max(len(row) for row in rows)
    return torch.tensor(
        [row + [value] * (width - len(row)) for row in rows], dtype=torch.long
    )


def collate_encoder_batch(
    samples: list[PreparedEncoderSample], pad_token_id: int
) -> dict[str, Tensor]:
    """Right-pad a batch into the tensors `EncoderTrainingModule.forward` takes.

    Padded slots get `cross_limit = 1` rather than 0 so their attention row is
    never fully masked -- a fully masked row can produce NaN, and one NaN would
    poison the whole batch's gradient even though the loss masks that slot out.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    context = _pad([s.context_ids for s in samples], pad_token_id)
    context_mask = _pad([[1] * len(s.context_ids) for s in samples], 0).bool()
    input_ids = _pad([s.input_ids for s in samples], pad_token_id)
    attention_mask = _pad([[1] * len(s.input_ids) for s in samples], 0)
    return {
        "context_ids": context,
        "context_mask": context_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": _pad([s.labels for s in samples], IGNORE_INDEX),
        "slot_positions": _pad([s.slot_positions for s in samples], 0),
        "slot_mask": _pad([[1] * s.num_slots for s in samples], 0).bool(),
        "cross_limit": _pad([s.cross_limit for s in samples], 1),
        "init_ids": _pad([s.init_ids for s in samples], 0),
        # `aux_patch` pads with -1, the "not a target" sentinel that both the mask
        # and the gather test against. A row with no supervised patches still needs
        # a column, so the width floors at 1.
        "aux_ids": _pad([s.aux_ids or [IGNORE_INDEX] for s in samples], IGNORE_INDEX),
        "aux_patch": _pad([s.aux_patch or [-1] for s in samples], -1),
        "prefix_len": torch.tensor([s.prefix_len for s in samples], dtype=torch.long),
    }


class EncoderTrainingModule(nn.Module):
    """What DDP wraps: the encoder, and only the encoder.

    `backbone` is a plain object, not an `nn.Module`, so the frozen decoder's
    parameters are invisible to `parameters()`, `state_dict()` and DDP's bucket
    scan without anyone having to remember to filter them.
    """

    def __init__(
        self,
        encoder: CoTEncoder,
        backbone: FrozenBackbone,
        *,
        commit_weight: float,
        next_patch_weight: float = 0.0,
        aux_width_multiple: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.commit_weight = commit_weight
        self.next_patch_weight = next_patch_weight
        self.aux_width_multiple = aux_width_multiple
        # Globally-reduced code assignment counts from the last forward, exposed
        # for logging. `usage` (and so perplexity and dead-code detection) is
        # updated from these inside forward.
        self.last_code_counts: Tensor | None = None
        # A bounded sample of this rank's most recent pre-quantization outputs.
        # Dead codes are reseeded from these; capped so holding it cannot grow
        # with K, which reaches 8033 on the longest rows.
        self.code_pool: Tensor | None = None
        # At least two candidates per code, so a reseed of a mostly-dead codebook
        # is not forced to draw the same vector repeatedly. Capped so holding it
        # cannot grow with K, which reaches 8033 on the longest rows.
        self.code_pool_size = max(512, 2 * encoder.config.codebook_size)
        # Numerators and denominators for every reported loss, accumulated since the
        # last `pop_totals`, separately for train and eval because the two are read
        # on different cadences. Each term is carried BOTH ways -- summed over rows
        # of per-row means (what is optimized) and summed over its own units (what
        # is comparable with earlier runs and with `base`/`no_cot`) -- because a
        # total alone cannot say whether a falling loss is the decoder getting the
        # answer or just the quantizer terms shrinking.
        self.train_totals: Tensor | None = None
        self.eval_totals: Tensor | None = None
        # Dead codes seen by the last `reseed_dead_codes`, before it acted.
        self.last_dead_count = 0
        self.last_quant_rel_error: Tensor | None = None

    # Index layout of `train_totals` / `eval_totals`. A named tuple of floats would
    # need a host sync per micro-batch; one stacked tensor stays on the device until
    # the logging cadence reads it.
    TOTALS = (
        "answer_row",
        "answer_tok",
        "answer_tokens",
        "next_patch_row",
        "next_patch_tok",
        "aux_tokens",
        "codebook_row",
        "commit_row",
        "rows",
        "aux_rows",
    )

    def _accumulate(self, parts: dict[str, Tensor]) -> None:
        row = torch.stack(
            [parts[name].detach().double().reshape(()) for name in self.TOTALS]
        )
        key = "train_totals" if self.training else "eval_totals"
        current = getattr(self, key)
        setattr(self, key, row if current is None else current + row)

    def pop_totals(self, train: bool) -> dict[str, float]:
        """Every loss since the last call, one key per distinct quantity.

        Reduced across ranks here rather than at every micro-batch: this runs on a
        logging cadence, where one extra collective is free, and accumulating
        rank-locally keeps the hot path collective-free.

        `loss` is the objective. `*_row` is its own normalization -- a mean over
        rows of each row's mean. The two CE terms are also reported `*_tok`, per
        answer token and per aux token, which is what compares with runs from
        before the per-row objective and with the eval harness's pooled figures.
        The quantizer terms are reported per row only: their per-slot twins agree
        with them to four decimals, and at ~1e-4 neither moves the objective.
        """
        key = "train_totals" if train else "eval_totals"
        totals = getattr(self, key)
        if totals is None:
            return {}
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals)
        setattr(self, key, None)
        values = dict(zip(self.TOTALS, totals.tolist(), strict=True))

        def ratio(numerator: str, denominator: str) -> float:
            total = values[denominator]
            return values[numerator] / total if total > 0 else 0.0

        report = {
            "answer_ce_row": ratio("answer_row", "rows"),
            "answer_ce_tok": ratio("answer_tok", "answer_tokens"),
            "codebook_row": ratio("codebook_row", "rows"),
            "commit_row": ratio("commit_row", "rows"),
        }
        report["loss"] = (
            report["answer_ce_row"]
            + report["codebook_row"]
            + self.commit_weight * report["commit_row"]
            + self.next_patch_weight * ratio("next_patch_row", "aux_rows")
        )
        # Absent rather than 0.0 when nothing was scored -- a weight-0 run skips the
        # pass in training -- so a chart shows a gap instead of a false flat line.
        if values["aux_tokens"] > 0:
            report["next_patch_ce_row"] = ratio("next_patch_row", "aux_rows")
            report["next_patch_ce_tok"] = ratio("next_patch_tok", "aux_tokens")
        return report

    def forward(
        self,
        context_ids: Tensor,
        context_mask: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        slot_positions: Tensor,
        slot_mask: Tensor,
        cross_limit: Tensor,
        init_ids: Tensor,
        aux_ids: Tensor | None = None,
        aux_patch: Tensor | None = None,
        prefix_len: Tensor | None = None,
    ) -> Tensor:
        """The objective's two terms for this micro-batch, each a SUM OVER ROWS.

        Returns `[answer_CE + codebook + beta*commit, lambda * next_patch_CE]`, where
        every constituent is already a per-row mean (over that row's answer tokens,
        its slots, or its patches' means). `run_training` divides entry *k* by the
        step's *global* count for entry *k* -- rows, and rows carrying a supervised
        patch -- then sums. Two entries rather than one because those two
        denominators differ; sums rather than means because only a sum keeps the
        quotient invariant to how rows were grouped into micro-batches and sharded
        across ranks.

        Each row therefore weighs the same, whatever its answer length, its CoT
        length or its patch count -- which is also how the eval harness reports
        (`mean_logprob`, per sample). The previous normalization (everything over
        the step's answer tokens) folded the aux/answer token ratio into
        `next_patch_weight` and the slots-per-answer-token ratio into
        `commit_weight`; both are now genuine per-row weights.
        """
        memory = self.backbone.encode_context(context_ids, context_mask)
        latent = self.backbone.embedding_weight[init_ids].detach()

        out = self.encoder(
            latent,
            memory,
            slot_mask,
            context_mask,
            cross_limit=cross_limit,
        )
        # Reduce before folding into the usage EMA. This is a collective, and it
        # is safe here because every rank runs the same number of forwards:
        # plan_epoch truncates micro-batches to a multiple of world_size, and
        # plan_eval_batches does the same, so no rank can arrive alone.
        counts = out.counts
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
        if self.training:
            # Training statistics only -- folding eval batches in would make
            # utilization reflect a mixture of two different data distributions.
            self.encoder.quantizer.update_usage(counts)
        self.last_code_counts = counts.detach()
        if self.training:
            # ||g - q|| / ||g||, the fraction of the encoder's output the
            # bottleneck discards. Kept as a tensor: `float()` here would be a
            # host sync on every micro-batch. `commit_weight` no longer restrains
            # this -- the quantizer terms fell to ~0.1% of the objective once the
            # code scale dropped to embed_rms -- so it is worth watching directly.
            g = out.encoder_output[slot_mask]
            q = out.z.detach()[slot_mask]
            self.last_quant_rel_error = (g - q).norm(dim=-1).mean() / g.norm(
                dim=-1
            ).mean().clamp_min(1e-12)

            valid = out.encoder_output[slot_mask]
            if valid.shape[0] > self.code_pool_size:
                pick = torch.randperm(valid.shape[0], device=valid.device)
                valid = valid[pick[: self.code_pool_size]]
            self.code_pool = valid

        spliced = self.backbone.splice(input_ids, out.z, slot_positions, slot_mask)
        answer = self.backbone.answer_ce_weighted(spliced, attention_mask, labels)
        aux, aux_tokens, aux_rows = self._next_patch_loss(
            spliced, aux_ids, aux_patch, prefix_len, slot_positions
        )

        quantizer = out.codebook_loss + self.commit_weight * out.commit_loss
        self._accumulate(
            {
                "answer_row": answer[0],
                "answer_tok": answer[1],
                "answer_tokens": (labels[:, 1:] != IGNORE_INDEX).sum(),
                "next_patch_row": aux[0],
                "next_patch_tok": aux[1],
                "aux_tokens": aux_tokens,
                "codebook_row": out.codebook_loss,
                "commit_row": out.commit_loss,
                "rows": torch.tensor(
                    slot_mask.shape[0], device=slot_mask.device, dtype=torch.long
                ),
                "aux_rows": aux_rows,
            }
        )
        return torch.stack([answer[0] + quantizer, self.next_patch_weight * aux[0]])

    def _next_patch_loss(
        self,
        spliced: Tensor,
        aux_ids: Tensor | None,
        aux_patch: Tensor | None,
        prefix_len: Tensor | None,
        slot_positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predict each patch's tokens from the codes before it.

        Returns `([sum_i mean_j mean_t CE, sum_t CE], aux tokens, rows with a patch)`.
        Each row is `[its Pass B prefix through the last code; its supervised
        patches' tokens]`, so the codes are reused *in place* -- `spliced` already
        holds them, and no second render or splice is needed.

        The first entry is a sum over rows of a row's mean per-patch CE, which is
        what `patch_weights` encodes as per-target weights; `run_training` divides it
        by the global count of rows that *have* a supervised patch. Rows without one
        are excluded rather than counted as zero: their per-row loss does not exist,
        and counting them would silently shrink the term as `next_patch_subsample`
        falls. With subsampling off, every row in this corpus has one (the shortest
        CoT is 24 tokens, so K >= 6 at any ratio in use).
        """
        zeros = spliced.new_zeros(2, dtype=torch.float32)
        none = torch.zeros((), device=spliced.device, dtype=torch.long)
        # At weight 0 the pass is skipped in TRAINING, where it is most of the step's
        # cost, but still run at eval, where it is a few seconds under no_grad. That
        # keeps `eval/next_patch_ce_*` reported for an answer-only run -- which is
        # the measurement that says whether answer-trained codes predict the next
        # patch at all, i.e. whether the two objectives agree.
        if aux_ids is None or (self.next_patch_weight == 0.0 and self.training):
            return zeros, none, none
        assert aux_patch is not None and prefix_len is not None
        # Bucketed on every device. Each distinct width is a flex compilation on
        # CUDA; on CPU padding buys nothing, but applying it anyway means the tests
        # run exactly the production geometry, padding included.
        width = bucket_width(
            natural_width(aux_patch, prefix_len), self.aux_width_multiple
        )
        plan = build_next_patch_batch(
            slot_positions=slot_positions,
            aux_ids=aux_ids,
            aux_patch=aux_patch,
            prefix_len=prefix_len,
            width=width,
        )
        if plan.num_targets == 0:
            # Safe to skip on one rank alone: the aux pass runs no collectives and
            # touches no DDP parameter, so ranks cannot desynchronize over it.
            return zeros, none, none
        weights, patches_per_row = patch_weights(aux_patch)
        embeds = self._aux_embeds(spliced, plan)
        loss = self.backbone.next_patch_ce(
            embeds,
            aux_attention_mask(plan, embeds.dtype),
            plan.predictor,
            plan.targets,
            plan.position_ids,
            weights,
        )
        return loss, (aux_patch >= 0).sum(), (patches_per_row > 0).sum()

    def _aux_embeds(self, spliced: Tensor, plan: NextPatchBatch) -> Tensor:
        """`[B, W, D]`: each row's prefix (codes spliced) then its aux token embeddings.

        Prefix columns carry `spliced` -- and so the gradient to the codes; aux
        columns get the frozen embeddings of the aux tokens, which are exactly the
        `targets` in `aux_rows`/`aux_columns` order; everything else is zero. Pass B
        may be longer than `W` (it continues into the answer) or shorter, hence `cut`.
        """
        batch, length, dim = spliced.shape
        cut = min(length, plan.width)
        prefix = spliced[:, :cut] * plan.is_prefix[:, :cut, None].to(spliced.dtype)
        if plan.width > cut:
            prefix = torch.cat(
                [prefix, spliced.new_zeros((batch, plan.width - cut, dim))], dim=1
            )
        tokens = self.backbone.embedding_weight[plan.targets].detach().to(prefix.dtype)
        return prefix.index_put((plan.aux_rows, plan.aux_columns), tokens)


def identity_collate(batch: Any) -> Any:
    """DataLoader collate for a batch_sampler that already yields whole batches."""
    return batch


def drop_unprepared(
    samples: list[PreparedEncoderSample | None],
) -> list[PreparedEncoderSample]:
    return [sample for sample in samples if sample is not None]


def batch_answer_tokens(samples: list[PreparedEncoderSample]) -> int:
    return sum(
        int(np.count_nonzero(np.asarray(s.labels[1:]) != IGNORE_INDEX)) for s in samples
    )


def step_denominators(device: torch.device, batches: list[dict[str, Tensor]]) -> Tensor:
    """What this step's two loss terms are divided by: `[rows, rows with a patch]`.

    Counted from the collated batches on the host, before any of them runs, and
    all-reduced once by `run_training` -- so the objective is a mean over the step's
    *global* rows and cannot depend on how they were split across micro-batches or
    ranks. Per-micro-batch means would: each group would carry a weight its own row
    count decided.

    The second entry excludes rows with no supervised patch, which is what
    `_next_patch_loss` scores. Only reachable with `next_patch_subsample < 1`.
    """
    rows = sum(int(batch["labels"].shape[0]) for batch in batches)
    with_patches = sum(
        int((batch["aux_patch"] >= 0).any(dim=1).sum()) for batch in batches
    )
    return torch.tensor(
        [float(rows), float(with_patches)], dtype=torch.float64, device=device
    )


class SaveableEncoder:
    """Gives `CoTEncoder` the two methods `save_checkpoint` calls on `state.model`.

    A shim rather than a fork of `sft_loop.save_checkpoint`: that function's
    stage -> fsync -> rename -> pointer -> prune ordering is what makes a kill at
    any instant non-corrupting, and duplicating it to change two lines would be
    the wrong trade.
    """

    def __init__(self, encoder: CoTEncoder) -> None:
        self.encoder = encoder

    def state_dict(self) -> dict[str, Tensor]:
        return self.encoder.state_dict()

    def save_pretrained(
        self, directory: Any, state_dict: dict[str, Tensor] | None = None
    ) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            state_dict if state_dict is not None else self.encoder.state_dict(),
            path / ENCODER_WEIGHTS,
        )
        (path / ENCODER_CONFIG).write_text(
            json.dumps(asdict(self.encoder.config), indent=2), encoding="utf-8"
        )


def check_encoder_architecture(config: EncoderConfig, directory: Path) -> None:
    """Refuse a resume whose checkpoint is a different encoder.

    The same footgun `check_checkpoint_architecture` guards on the SFT path, and
    it bites harder here: `d_llm` is 1024 against Qwen3-0.6B and 2560 against
    Qwen3-4B, and the run directory carries no model identifier. `plan_signature`
    cannot catch it -- the two models ship byte-identical tokenizers, so they
    produce the same batch plan.
    """
    manifest = directory / ENCODER_CONFIG
    if not manifest.exists():
        return
    found = json.loads(manifest.read_text(encoding="utf-8"))
    want = asdict(config)
    # A field the manifest predates reads as its default, which is what that
    # checkpoint actually trained with -- `position_encoding` defaults to "none"
    # precisely so a pre-RoPE checkpoint compares as the encoder it is.
    defaults = {spec.name: spec.default for spec in fields(EncoderConfig)}
    moved = [
        f"{key}: checkpoint has {found.get(key, defaults[key])!r}, config has {value!r}"
        for key, value in want.items()
        if key in _ARCHITECTURE_FIELDS and found.get(key, defaults[key]) != value
    ]
    if moved:
        raise ValueError(
            f"Refusing to resume: {directory} holds a different encoder.\n  "
            + "\n  ".join(moved)
            + "\nGive this run its own run_tag; run_name carries no model identifier."
        )


def load_encoder(
    directory: Path, device: torch.device, config: EncoderConfig | None = None
) -> CoTEncoder:
    """Rebuild an encoder from a checkpoint directory.

    `config`, when given, WINS over the manifest. Training passes the config it
    just built from `cfg`, because the manifest also stores *tunables* --
    `dead_code_threshold`, `usage_decay`, `commit_weight`,
    `soft_assign_temperature` -- and rebuilding from it would silently pin a
    resumed run to whatever those were when the checkpoint was written, ignoring
    the command line. The architecture fields are guaranteed to agree because
    `check_encoder_architecture` has already refused the resume otherwise.
    Inference (`method.load_learned_encoder`) passes nothing and gets the
    manifest, which is what it wants: no cfg exists there.

    Unknown keys are dropped rather than raising: `EncoderConfig` fields get
    renamed (`dead_code_frac` -> `dead_code_threshold`), and a checkpoint written
    before such a rename is still a valid set of *weights*. The fields that decide
    whether the weights even fit are guarded separately and strictly by
    `check_encoder_architecture`, which reads the raw manifest, so loosening here
    costs nothing.
    """
    stored = json.loads((directory / ENCODER_CONFIG).read_text(encoding="utf-8"))
    known = {f.name for f in fields(EncoderConfig)}
    dropped = sorted(set(stored) - known)
    if config is None:
        config = EncoderConfig(**{k: v for k, v in stored.items() if k in known})
    encoder = CoTEncoder(config)
    encoder.load_state_dict(
        torch.load(directory / ENCODER_WEIGHTS, map_location="cpu", weights_only=True)
    )
    if dropped:
        print(f"load_encoder: ignoring retired config fields {dropped}", flush=True)
    return encoder.to(device)


def embedding_rms(embedding_weight: Tensor, vocab_bound: int) -> float:
    """Mean over tokens of each token's own RMS, over the REGULAR vocabulary.

    Per-token rather than the global `sqrt(mean(w**2))` because RMSNorm sets the
    per-vector output scale, and a code is one vector. Regular tokens only,
    matching the population `latent_init=random` draws from; special/added ids
    have outlier embeddings. The two statistics differ by 1% on Qwen3-0.6B
    (0.02895 vs 0.02923), so the choice is not delicate -- but it should be
    stated rather than left to whichever line was written first.
    """
    return float(embedding_weight[:vocab_bound].float().pow(2).mean(-1).sqrt().mean())


def build_encoder_config(
    cfg: DictConfig, d_llm: int, embed_rms: float
) -> EncoderConfig:
    """`d_llm` and `embed_rms` come from the loaded decoder, never from config.

    Deriving them removes the two fields a human could set inconsistently with the
    model actually in use, which is exactly the mismatch that would otherwise
    surface only as a shape error -- or, for `embed_rms`, as a silently dead
    codebook -- thousands of steps in.
    """
    encoder_cfg = cfg.encoder
    return EncoderConfig(
        d_llm=d_llm,
        embed_rms=embed_rms,
        n_blocks=int(encoder_cfg.n_blocks),
        n_heads=int(encoder_cfg.n_heads),
        ffn_mult=int(encoder_cfg.ffn_mult),
        codebook_size=int(encoder_cfg.codebook_size),
        # Cast, not validate: EncoderConfig.__post_init__ is what rejects a bad
        # value, and it does so with a message naming the field. A silent
        # narrowing here would only move the failure somewhere less legible.
        self_attn_mask=cast(MaskMode, str(encoder_cfg.self_attn_mask)),
        cross_attn_mask=cast(MaskMode, str(encoder_cfg.cross_attn_mask)),
        latent_init=cast(LatentInit, str(encoder_cfg.latent_init)),
        position_encoding=cast(
            PositionEncoding, str(encoder_cfg.get("position_encoding", "none"))
        ),
        rope_theta=float(encoder_cfg.get("rope_theta", 1_000_000.0)),
        soft_assign_temperature=float(encoder_cfg.soft_assign_temperature),
        commit_weight=float(encoder_cfg.commit_weight),
        dead_code_threshold=float(encoder_cfg.dead_code_threshold),
        next_patch_weight=float(encoder_cfg.get("next_patch_weight", 0.0)),
        next_patch_subsample=float(encoder_cfg.get("next_patch_subsample", 1.0)),
        usage_decay=float(encoder_cfg.usage_decay),
    )


def plan_columns(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    """Batch-planning lengths and the denominator proxy.

    `lengths` is Pass A's sequence (`prompt + CoT`), because that is what drives
    memory: Pass B is `prompt + K + answer`, which at any compression ratio above
    one is strictly shorter.

    The second array is **not** a real `label_start`. `plan_epoch` computes
    `denom = length - label_start`, and the encoder is supervised on answer tokens
    only, so passing `length - answer_length` makes `denom` exactly the answer
    token count -- which is what `run_training` divides by, and what makes runs at
    different compression ratios comparable.

    That proxy goes **negative** for the 2.3% of rows whose answer is longer than
    their prompt plus CoT -- mostly `ifeval` and `general_quality`, where the model
    reasons briefly and then writes at length. Harmless, because `plan_epoch`
    consumes the array only as `lengths - label_starts` and never as an index;
    `test_plan_epoch_denominator_survives_negative_proxies` pins that, so a change
    to `plan_epoch` that started indexing with it would fail loudly rather than
    silently mis-weight the loss.
    """
    prompt = np.asarray(dataset["prompt_length"], dtype=np.int64)
    cot = np.asarray(dataset["cot_length"], dtype=np.int64)
    answer = np.asarray(dataset["answer_length"], dtype=np.int64)
    if (answer <= 0).any():
        # A zero denominator for a step would divide the loss by nothing; the
        # measurement pass already drops these, so reaching here means the cache
        # was built by something else.
        raise ValueError("A row has no answer tokens; rebuild the measured cache.")
    lengths = prompt + cot
    return lengths, lengths - answer


@dataclass
class EncoderData:
    """The data side of a run: measured splits, the batch plan, and row preparation.

    Built by `build_encoder_data` for BOTH `train_encoder` and the shape probe, so
    the probe measures exactly the micro-batches a run will see rather than a
    reconstruction of them.
    """

    cfg: DictConfig
    measured: DatasetDict
    plan: BatchPlan
    eval_batches: list[list[int]]
    patching: PatchingMethod
    latent_init: str
    prep: Any
    _signal_cache: dict[tuple[str, str], Any] = field(default_factory=dict, repr=False)

    def signals(
        self, split: str, logger: Any, latent_init: str | None = None
    ) -> dict[int, Tensor] | None:
        """The cache `split` needs for `latent_init` (the run's own by default).

        Memoized per (split, signal): the eval preparation and every control on a
        split usually read the same cache, and each load re-reads the npz.
        """
        init = self.latent_init if latent_init is None else latent_init
        signal = required_signal(self.patching, init)
        if signal is None:
            return None
        key = (split, signal)
        if key not in self._signal_cache:
            self._signal_cache[key] = _load_signals(
                self.cfg, self.patching, init, split, self.measured[split], logger
            )
        return self._signal_cache[key]


def build_encoder_data(
    cfg: DictConfig,
    tokenizer: Any,
    *,
    vocab_bound: int,
    latent_init: str,
    next_patch_subsample: float,
    world_size: int,
) -> EncoderData:
    measured = load_trace_lengths(cfg)
    train_lengths, train_starts = plan_columns(measured["train"])
    eval_lengths, _ = plan_columns(measured["val"])
    patching = build_patching_method(
        str(cfg.encoder.patching), cfg.evaluation.methods.patching
    )
    if patching is None:
        raise ValueError("encoder.patching must name a patching strategy.")
    return EncoderData(
        cfg=cfg,
        measured=measured,
        plan=plan_epoch(
            train_lengths, train_starts, cfg, epoch=0, world_size=world_size
        ),
        eval_batches=plan_eval_batches(eval_lengths, cfg, world_size),
        patching=patching,
        latent_init=latent_init,
        prep=partial(
            EncoderPrepDataset,
            tokenizer=tokenizer,
            patching=patching,
            seed=int(cfg.training.seed),
            # None, never `training.max_length`: the batch plan already drops
            # over-length rows, and truncating here would cut placeholder slots off
            # and silently desynchronize `slot_positions` from K.
            max_length=None,
            vocab_bound=vocab_bound,
            latent_init=latent_init,
            next_patch_subsample=next_patch_subsample,
        ),
    )


def _build_loader(
    dataset: Any,
    batches: list[list[int]],
    cfg: DictConfig,
    pad_token_id: int,
) -> DataLoader:
    def collate(samples: list[PreparedEncoderSample | None]) -> dict[str, Tensor]:
        kept = drop_unprepared(samples)
        if not kept:
            raise RuntimeError(
                "Every row in a micro-batch failed preparation. The measured cache "
                "should already have dropped these; rebuild it."
            )
        return collate_encoder_batch(kept, pad_token_id)

    return DataLoader(
        dataset,
        batch_sampler=batches,
        collate_fn=collate,
        num_workers=int(cfg.training.num_workers),
        prefetch_factor=int(cfg.training.prefetch_factor)
        if int(cfg.training.num_workers) > 0
        else None,
        pin_memory=torch.cuda.is_available(),
    )


def verify_signal_cache(
    values: dict[int, Tensor], dataset: Any, path: Path, split: str
) -> None:
    """Refuse a cache that was not built from these exact rows.

    The cache is keyed by row index *within a split*, and train and val both start
    at 0 -- so a val cache handed to the train split is not an error anywhere, it
    just silently attaches the wrong surprisal values to the first rows and drops
    the rest. That happened (precompute defaults to the validation split), and it
    would have corrupted two grid arms with no failure.

    Per-row CoT length is an exact fingerprint: it is already stored in the npz and
    already a column of the measured cache, and no two splits agree on it.
    """
    lengths = np.asarray(dataset["cot_length"], dtype=np.int64)
    if len(values) != lengths.size:
        raise ValueError(
            f"Signal cache {path} holds {len(values)} rows but split {split!r} has "
            f"{lengths.size}. It was almost certainly built for a different split; "
            f"rebuild with evaluation.split={split}."
        )
    for index in (0, lengths.size // 2, lengths.size - 1):
        cached = values.get(index)
        if cached is None or cached.numel() != int(lengths[index]):
            got = "missing" if cached is None else str(cached.numel())
            raise ValueError(
                f"Signal cache {path} disagrees with split {split!r} at row {index}: "
                f"cot_length={int(lengths[index])} but cached values={got}. "
                "The cache belongs to different data; rebuild it."
            )


def required_signal(patching: PatchingMethod, latent_init: str) -> str | None:
    """The one per-token signal `patching` and `latent_init` read together, or None.

    One, not a set: `EncoderPrepDataset` takes a single cache and reads it both for
    span boundaries and for the `*_t0` token choice, so a second signal would be
    silently read as the first.
    """
    needed = {patching.required_signal()} - {None}
    if latent_init.endswith("_t0"):
        needed.add(latent_init.removesuffix("_t0"))
    if len(needed) > 1:
        raise NotImplementedError(
            f"Only one signal is supported at a time; got {needed}."
        )
    return next(iter(needed), None)


def _load_signals(
    cfg: DictConfig,
    patching: PatchingMethod,
    latent_init: str,
    split: str,
    dataset: Any,
    logger: Any,
) -> dict[int, Tensor] | None:
    """The per-token signal cache, when the patching or the init needs one.

    Returned as `{sample_index: values}` for the whole corpus. Only `surprisal` is
    reachable today -- both signal-consuming choices in the grid use it -- so a
    single cache is loaded rather than a dict of them.
    """
    signal = required_signal(patching, latent_init)
    if signal is None:
        return None
    cache_dir = cfg.evaluation.entropy_cache_dir
    if cache_dir is None:
        raise ValueError(
            f"Patching/init need the {signal!r} signal but "
            "evaluation.entropy_cache_dir is unset, so there is nowhere to read it."
        )
    # Per split: indices are split-local, so one directory per split is what keeps
    # a train run from reading a val cache.
    path = signal_cache_path(
        Path(str(cache_dir)) / split, str(cfg.method.model_name), signal
    )
    if not path.exists():
        raise FileNotFoundError(
            f"No {signal!r} cache at {path}. Build it once with:\n"
            f"  uv run python scripts/run.py precompute data={cfg.data.name} "
            f"evaluation.entropy_cache_dir={cache_dir}"
        )
    values = load_signal_cache(path)
    verify_signal_cache(values, dataset, path, split)
    logger.info(f"loaded {len(values)} {signal} rows for split {split!r} from {path}")
    return values


@torch.no_grad()
def kmeans_init_codebook(
    module: EncoderTrainingModule,
    batches: Any,
    device: torch.device,
    *,
    iters: int,
    seed: int,
    world_size: int,
    logger: Any,
    max_batches: int = _KMEANS_MAX_BATCHES,
) -> None:
    """Seed the codebook from k-means over real encoder outputs.

    Runs on rank 0 and broadcasts, for the reason `reseed_dead_codes` documents:
    each rank sees different data, and DDP synchronizes gradients rather than
    parameters, so a per-rank k-means would leave every rank a different codebook
    with nothing reporting it.

    Consumes micro-batches until the pool holds `_KMEANS_MAX_SAMPLES` slots, rather
    than trusting one of them. A single micro-batch used to be plenty because the
    plan handed over its longest rows first (~30,000 slots); with the batch order
    shuffled, the first micro-batch is a random length instead and can carry ~1,300
    slots -- barely one point per centroid at |C|=1024. `max_batches` stops a run of
    very short micro-batches from turning startup into an epoch.

    The cap is what bounds the `[N, |C|]` distance matrix (268 MB at the limit).
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    parts: list[Tensor] = []
    collected = 0
    used = 0
    for batch in batches:
        moved = {key: value.to(device) for key, value in batch.items()}
        memory = module.backbone.encode_context(
            moved["context_ids"], moved["context_mask"]
        )
        latent = module.backbone.embedding_weight[moved["init_ids"]].detach()
        hidden = module.encoder.encode(
            latent,
            memory,
            moved["slot_mask"],
            moved["context_mask"],
            cross_limit=moved["cross_limit"],
        )
        parts.append(hidden[moved["slot_mask"]].float())
        collected += int(parts[-1].shape[0])
        used += 1
        if collected >= _KMEANS_MAX_SAMPLES or used >= max_batches:
            break
    if not parts:
        raise RuntimeError("No micro-batches to seed the codebook from.")
    pool = torch.cat(parts)
    if pool.shape[0] > _KMEANS_MAX_SAMPLES:
        pick = torch.randperm(pool.shape[0], generator=generator, device=pool.device)
        pool = pool[pick[:_KMEANS_MAX_SAMPLES]]

    quantizer = module.encoder.quantizer
    quantizer.init_codebook_from_encoder_outputs(pool, generator, iters=iters)
    if world_size > 1:
        dist.broadcast(quantizer.codebook.data, src=0)
        dist.broadcast(quantizer.usage, src=0)
    logger.info(
        f"k-means codebook init: {iters} iters over {pool.shape[0]} encoder outputs "
        f"from {used} micro-batch(es) "
        f"| encoder-output RMS {float(pool.pow(2).mean(-1).sqrt().mean()):.4f} "
        f"| codebook RMS {float(quantizer.codebook.detach().pow(2).mean().sqrt()):.4f}"
    )


def _prime_encoder_buckets(state: Any) -> None:
    """No-op: the encoder's first-backward transient is ~136 MB, not ~15 GB.

    `_prime_gradient_buckets` exists because on the 4B SFT run the first backward
    transiently allocates a second full set of gradients on top of restored Adam
    moments, which OOMs on every requeue. At 34M parameters that transient is
    negligible, so priming would spend a forward pass to avoid nothing.
    """
    return None


def train_encoder(cfg: DictConfig) -> Path:
    """Train the compressor against a frozen decoder. Mirrors `train_sft`.

    Reuses `plan_epoch` and `run_training` wholesale: the batch plan, the
    crash-safe checkpoint ordering, the token-weighted eval and the requeue path
    are all identical problems, and the encoder differs only in what the module
    computes and what the denominator counts.
    """
    global_rank, local_rank, world_size = init_distributed()
    is_main = global_rank == 0
    # Pinned before any DataLoader exists. The pin-memory thread calls
    # `torch.set_num_threads(1)` (torch/utils/data/_utils/pin_memory.py), and Dynamo
    # guards on that global: every flip invalidates compiled flex attention and
    # counts toward the recompile limit. The Sep 11 resume logged exactly that
    # reason ("GLOBAL_STATE changed: num_threads") on its way to the limit. CUDA only:
    # flex attention is only compiled there, and a CPU run would just be slower.
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    run_dir = get_run_dir(cfg)
    if is_main:
        save_resolved_config(cfg, run_dir)
    logger = RunLogger(cfg=cfg, run_dir=run_dir, is_main=is_main)
    set_seed(
        seed=int(cfg.training.seed) + global_rank,
        deterministic=bool(cfg.training.deterministic),
    )
    if bool(cfg.training.get("detect_anomaly", False)):
        # Diagnostic only, and roughly 3x slower: every backward op is checked and
        # the first one producing a NaN raises with the FORWARD traceback that
        # created it. This is the tool for "which op made the NaN", which no
        # amount of metric logging can answer after the fact.
        torch.autograd.set_detect_anomaly(True)
        logger.info(
            "autograd anomaly detection ON -- diagnostic run, expect ~3x slower"
        )
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if disable_cudnn_sdpa(device):
        logger.info("cuDNN SDPA disabled (see disable_cudnn_sdpa); SDPA uses flash")
    logger.info(f"rank {global_rank}/{world_size} on {device}")

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
    decoder = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            str(cfg.method.model_name),
            dtype=parse_torch_dtype(str(cfg.training.decoder_dtype)),
            attn_implementation=str(cfg.training.attn_implementation),
            trust_remote_code=bool(cfg.method.trust_remote_code),
        ),
    ).to(device)
    backbone = FrozenBackbone(
        decoder,
        memory_layer=int(cfg.encoder.memory_layer),
        ce_chunk_tokens=int(cfg.training.ce_chunk_tokens),
        gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
    )
    logger.info(
        "aux compilation: "
        f"{configure_aux_compilation(int(cfg.training.aux_recompile_limit))}"
    )

    encoder_config = build_encoder_config(
        cfg,
        backbone.hidden_size,
        embedding_rms(
            backbone.embedding_weight,
            regular_vocab_bound(tokenizer, backbone.embedding_weight.shape[0]),
        ),
    )
    resume_dir = resolve_resume_dir(run_dir, cfg.training.resume_from_checkpoint)
    if resume_dir is not None:
        check_encoder_architecture(encoder_config, resume_dir)
        encoder = load_encoder(resume_dir, device, config=encoder_config)
        logger.info(f"Resuming encoder weights from {resume_dir}")
    else:
        encoder = CoTEncoder(encoder_config).to(device)
        # The codebook is left at its placeholder here and seeded by k-means over
        # a real batch below, once the datasets exist. Deliberately NOT on resume:
        # the checkpoint's codebook is the trained one.
    logger.info(
        f"encoder: {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M parameters, "
        f"d_llm={encoder_config.d_llm}, |C|={encoder_config.codebook_size}, "
        f"masks={encoder_config.self_attn_mask}/{encoder_config.cross_attn_mask}, "
        f"init={encoder_config.latent_init}"
    )

    optimizer = build_optimizer(cfg, encoder)
    data = build_encoder_data(
        cfg,
        tokenizer,
        vocab_bound=regular_vocab_bound(tokenizer, backbone.embedding_weight.shape[0]),
        latent_init=encoder_config.latent_init,
        next_patch_subsample=encoder_config.next_patch_subsample,
        world_size=world_size,
    )
    measured, plan, prep = data.measured, data.plan, data.prep
    logger.info(
        f"plan {plan.signature}: {len(plan.steps)} steps, {len(plan.micro)} micro-batches, "
        f"{plan.kept} rows kept, {plan.dropped} dropped over max_length="
        f"{int(cfg.training.max_length)}"
    )

    horizon = optional_int(cfg.training.schedule_horizon_steps) or len(plan.steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=max(
            int(cfg.training.min_warmup_steps),
            int(horizon * float(cfg.training.warmup_ratio)),
        ),
        num_training_steps=horizon,
    )

    module: torch.nn.Module = EncoderTrainingModule(
        encoder,
        backbone,
        commit_weight=encoder_config.commit_weight,
        next_patch_weight=encoder_config.next_patch_weight,
        aux_width_multiple=optional_int(cfg.training.aux_width_multiple),
    )
    if world_size > 1:
        module = DistributedDataParallel(
            module,
            device_ids=[local_rank],
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )

    eval_batches = data.eval_batches
    val_signals = data.signals("val", logger)
    # All eval batches, not a prefix: `estimate_loss` scores every one of them, and
    # a baseline computed on a subset would make every gap a difference between
    # two different populations. One forward per baseline over the slice at startup.
    baselines = eval_baselines(backbone, tokenizer, data, device, logger)
    for unit, label in (("row", "per row"), ("tok", "per answer token")):
        logger.info(
            f"eval baselines ({label}): "
            + " ".join(
                f"{name}={values[unit]:.4f}" for name, values in baselines.items()
            )
            + f" headroom={baselines['no_cot'][unit] - baselines['base'][unit]:.4f}"
        )

    state = LoopState(
        model=SaveableEncoder(encoder),
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
        prime_batch=_prime_encoder_buckets,
        extra_metrics=codebook_metrics,
        eval_metrics=partial(eval_report, baselines),
        eval_loss_key="eval/loss",
        step_denominators=partial(step_denominators, device),
        # `best/` tracks the research objective, not the training objective: the
        # next-patch term dominates the total at any useful weight, so selecting on
        # it would keep the best next-patch predictor rather than the best answer
        # predictor. Per row, matching both the objective and `mean_logprob`, the
        # statistic the eval harness and every plot in the repo report.
        best_metric_key="eval/answer_ce_row",
        post_step=with_injected_fault(
            reseed_dead_codes, cfg.training.get("fault_injection"), global_rank
        ),
    )

    train_dataset = prep(
        dataset=measured["train"], signals=data.signals("train", logger)
    )
    eval_dataset = prep(dataset=measured["val"], signals=val_signals)
    make_loader = partial(
        _build_loader, cfg=cfg, pad_token_id=int(tokenizer.pad_token_id)
    )

    if resume_dir is None:
        # After the datasets exist and before the first optimizer step: the
        # codebook must be seeded from encoder outputs the encoder actually
        # produces, which needs real batches. EnCodec's "first training batch",
        # extended over as many of the plan's leading micro-batches as it takes to
        # reach a pool worth running k-means on -- every rank walks the same ones.
        kmeans_init_codebook(
            cast(EncoderTrainingModule, getattr(module, "module", module)),
            make_loader(train_dataset, plan.micro[:_KMEANS_MAX_BATCHES]),
            device,
            iters=int(cfg.encoder.kmeans_iters),
            seed=int(cfg.training.seed),
            world_size=world_size,
            logger=logger,
        )

    try:
        return run_training(
            state=state,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_batches=eval_batches,
            make_loader=make_loader,
            resume_dir=resume_dir,
        )
    except Exception as error:
        # Log the traceback HERE, on the rank that raised, then re-raise it as a
        # plain RuntimeError carrying the text. Hydra packages a failed task into a
        # JobReturn and submitit pickles it; an InductorError holds graph objects
        # with weakrefs, so that dump died with "cannot pickle
        # 'weakref.ReferenceType' object" and REPLACED the real error -- on
        # 2026-09-08 it had to be recovered from a partial result pickle.
        failure = picklable_error(error, global_rank, logger.logger)
        if world_size > 1:
            logger.finish()
            exit_failed_rank(global_rank)
        raise failure from None
    finally:
        logger.finish()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def picklable_error(error: BaseException, rank: int, log: Any) -> RuntimeError:
    """Log `error` with its traceback and return a RuntimeError that pickles.

    The returned exception holds only strings -- no traceback, cause or context --
    so submitit can record it, while the full traceback is already in the run log
    written by the rank that raised.
    """
    text = "".join(traceback.format_exception(error))
    log.error("rank %d raised; training cannot continue:\n%s", rank, text)
    return RuntimeError(f"rank {rank}: {type(error).__name__}: {error}\n{text}")


@torch.no_grad()
def codebook_geometry(codebook: Tensor) -> dict[str, float]:
    """Where the codes sit relative to each other, not just how often each is used.

    Usage statistics can look healthy while the codes themselves converge: a
    codebook whose entries all point the same way still spreads its assignments if
    the encoder's outputs are noisy enough. These watch the geometry directly, from
    one `[|C|, |C|]` cosine matrix (1M entries at the largest size, so it is
    affordable at `log_interval`).

    Read `cos_nn_mean` against its OWN starting value, not against zero. The codes
    live in the decoder's input-embedding space, which is strongly anisotropic, so
    it starts near 1 at k-means initialization (0.996 at |C|=1024) and only a move
    away from that value is signal. `cos_mean` is the absolute measure of spread.
    """
    codes = codebook.float()
    size = codes.shape[0]
    normalized = codes / codes.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cosine = normalized @ normalized.t()
    off_diagonal = ~torch.eye(size, dtype=torch.bool, device=codes.device)
    neighbour = cosine.masked_fill(~off_diagonal, -1.0).max(dim=-1).values
    distances = torch.cdist(codes, codes).masked_fill(~off_diagonal, float("inf"))
    rms = codes.pow(2).mean().sqrt().clamp_min(1e-12)
    return {
        "vq/cos_mean": float(cosine[off_diagonal].mean()),
        "vq/cos_nn_mean": float(neighbour.mean()),
        # In units of the codebook's own RMS, so it survives the scale changes
        # `out_norm_gain` can drag the whole codebook through.
        "vq/dist_nn_mean": float(distances.min(dim=-1).values.mean() / rms),
    }


def codebook_metrics(state: Any) -> dict[str, float]:
    """Every `log_interval` emission beyond the loop's own: losses and codebook health.

    One metric per question, because a workspace of near-duplicates hides the curve
    that matters. The losses are window means over the whole log interval, and
    `train/loss` REPLACES the loop's single-step value of the same objective -- the
    window mean is the same quantity with a tenth of the noise.

    The codebook metrics exist because the failure this baseline is most exposed to
    is silent: a codebook collapsed onto three entries produces the same smooth loss
    curve as a healthy one.
    """
    module = state.module
    inner = getattr(module, "module", module)
    encoder = inner.encoder
    quantizer = encoder.quantizer
    size = quantizer.config.codebook_size
    pool = inner.code_pool
    return {
        **{f"train/{name}": value for name, value in inner.pop_totals(True).items()},
        **codebook_geometry(quantizer.codebook.detach()),
        # The effective fraction of the codebook in use, exp(entropy)/|C|: 1.0 is
        # uniform, 1/|C| is total collapse. Normalized, so comparable across sizes.
        "vq/perplexity_frac": float(quantizer.perplexity()) / size,
        # Dead codes reseeded at the last step, as a fraction of the codebook --
        # the count `reseed_dead_codes` acted on, not a re-query (which would read
        # 0, since reseeding runs before this block). Nonzero late in training is
        # chronic collapse that the restarts are papering over.
        "vq/dead_codes_frac": inner.last_dead_count / size,
        # Fraction of the encoder's output the bottleneck throws away. Rises when
        # the codebook stops tracking the encoder.
        "vq/quant_rel_error": (
            float(inner.last_quant_rel_error)
            if inner.last_quant_rel_error is not None
            else 0.0
        ),
        # The scale contract: the encoder's outputs and the codebook must stay one
        # distribution at the decoder's embedding scale (~0.029). Drifting apart is
        # the failure that killed an earlier campaign.
        "enc/encoder_output_rms": (
            float(pool.detach().pow(2).mean(-1).sqrt().mean())
            if pool is not None
            else 0.0
        ),
        "enc/codebook_rms": float(quantizer.codebook.detach().pow(2).mean().sqrt()),
    }


@torch.no_grad()
def eval_baselines(
    backbone: FrozenBackbone,
    tokenizer: Any,
    data: EncoderData,
    device: torch.device,
    logger: Any,
) -> dict[str, dict[str, float]]:
    """Every constant the encoder is compared against, on its exact eval slice.

    `base` (full trace) and `no_cot` (empty think block) bound the encoder. Each name
    in `encoder.eval_controls` is a training-free CONTROL that goes through the
    encoder's own `EncoderPrepDataset`: the same spans, K and splice positions, with
    each slot holding that latent initializer's token embedding and no encoder --
    `random`, arbitrary regular-vocabulary tokens, and `surprisal_t0`, each span's
    highest-surprisal token, which is `surprisal_weighted_mean` compression at T=0.
    """
    controls = {}
    for name in data.cfg.encoder.get("eval_controls", ["random"]):
        if name in ("base", "no_cot"):
            raise ValueError(f"eval control {name!r} would shadow a bound")
        controls[name] = data.prep(
            dataset=data.measured["val"],
            signals=data.signals("val", logger, latent_init=name),
            latent_init=name,
        )
    return baseline_answer_losses(
        backbone,
        tokenizer,
        data.measured["val"],
        data.eval_batches,
        device,
        controls=controls,
    )


@torch.no_grad()
def baseline_answer_losses(
    backbone: FrozenBackbone,
    tokenizer: Any,
    dataset: Any,
    batches: list[list[int]],
    device: torch.device,
    controls: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Answer CE for `base`, `no_cot` and each control, per row and per token.

    All are **constants** for a given decoder and slice -- none depends on the
    encoder -- so they are computed once at startup and the gaps are reported at
    every eval thereafter. Scored on exactly the rows `estimate_loss` uses.

    `{"row": ..., "tok": ...}` per name: `row` is the mean over rows of each row's
    mean answer CE, which is the objective's normalization and the eval harness's
    `mean_logprob`; `tok` is the pooled per-answer-token figure. They differ
    substantially here -- measured on 19,866 paired rows, `no_cot - base` is 0.39
    per token but 0.57 per row, because short answers depend on the CoT ~3.5x more
    per token and hold only 6% of the tokens.

    The run is working iff the gap to `no_cot` widens; without these a loss curve
    says nothing, because a model that ignores the codes still produces a smooth one.

    Every entry must average over the SAME rows and the SAME answer tokens, or a gap
    would compare two populations. That is checked, not assumed: each render
    tokenizes the answer in its own context and the controls can drop a row the full
    render keeps.
    """
    controls = controls or {}
    sums = {"base": 0.0, "no_cot": 0.0, **dict.fromkeys(controls, 0.0)}
    row_sums = dict.fromkeys(sums, 0.0)
    counts = dict.fromkeys(sums, 0)
    row_counts = dict.fromkeys(sums, 0)
    for rows in batches:
        prepared: dict[str, list[Any]] = {"base": [], "no_cot": []}
        for index in rows:
            trace = extract_answer_trace(dataset[index]["messages"])
            if trace is None:
                continue
            for name, messages in (
                ("base", trace.messages),
                ("no_cot", compressed_messages(trace, 0)),
            ):
                out = tokenize_answer(
                    tokenizer=tokenizer,
                    messages=messages,
                    answer=trace.answer,
                    max_length=None,
                )
                if out is not None:
                    prepared[name].append(out)
        if not prepared["base"] or len(prepared["base"]) != len(prepared["no_cot"]):
            continue
        for name, items in prepared.items():
            width = max(len(item.input_ids) for item in items)
            pad = int(tokenizer.pad_token_id)
            ids = torch.tensor(
                [i.input_ids + [pad] * (width - len(i.input_ids)) for i in items],
                dtype=torch.long,
                device=device,
            )
            mask = torch.tensor(
                [
                    [1] * len(i.input_ids) + [0] * (width - len(i.input_ids))
                    for i in items
                ],
                dtype=torch.long,
                device=device,
            )
            labels = torch.tensor(
                [i.labels + [IGNORE_INDEX] * (width - len(i.labels)) for i in items],
                dtype=torch.long,
                device=device,
            )
            embeds = backbone.model.get_input_embeddings()(ids)
            scored = backbone.answer_ce_weighted(embeds, mask, labels)
            row_sums[name] += float(scored[0])
            sums[name] += float(scored[1])
            counts[name] += int((labels[:, 1:] != IGNORE_INDEX).sum())
            row_counts[name] += int(labels.shape[0])
        for name, prep in controls.items():
            row_sum, token_sum, tokens, scored_rows = _token_init_control_ce(
                backbone, prep, rows, device
            )
            row_sums[name] += row_sum
            sums[name] += token_sum
            counts[name] += tokens
            row_counts[name] += scored_rows
    if counts["base"] == 0:
        raise RuntimeError("Baseline slice contained no answer tokens.")
    for what, tallies in (("answer tokens", counts), ("rows", row_counts)):
        if len(set(tallies.values())) > 1:
            raise RuntimeError(
                f"Baselines were scored on different {what}: {tallies}. Their gaps "
                "would compare different populations."
            )
    return {
        name: {
            "row": row_sums[name] / row_counts[name],
            "tok": sums[name] / counts[name],
        }
        for name in sums
    }


@torch.no_grad()
def _token_init_control_ce(
    backbone: FrozenBackbone, prep: Any, rows: list[int], device: torch.device
) -> tuple[float, float, int, int]:
    """`(per-row CE sum, per-token CE sum, answer tokens, rows)` for a token init.

    Each slot holds `init_ids`' embedding and no encoder runs.

    Reuses `EncoderPrepDataset` rather than reimplementing span planning, so K and
    the spans match the encoder's per row exactly and the control is paired with
    it row by row. The token choice is the preparation's own: `random` draws via
    `random_slot_token_ids` keyed on `seed + sample_index`; `surprisal_t0` takes each
    span's highest-surprisal token, whose embedding is exactly what
    `SignalWeightedMeanCompressionMethod` returns at T=0 (one-hot weights, and the
    variance rescale divides by sqrt(1)).
    """
    samples = [prep[index] for index in rows]
    kept = [sample for sample in samples if sample is not None]
    if not kept:
        return 0.0, 0.0, 0, 0
    batch = collate_encoder_batch(kept, pad_token_id=int(prep.tokenizer.pad_token_id))
    moved = {key: value.to(device) for key, value in batch.items()}
    codes = backbone.embedding_weight[moved["init_ids"]].detach()
    spliced = backbone.splice(
        moved["input_ids"], codes, moved["slot_positions"], moved["slot_mask"]
    )
    scored = backbone.answer_ce_weighted(
        spliced, moved["attention_mask"], moved["labels"]
    )
    return (
        float(scored[0]),
        float(scored[1]),
        int((moved["labels"][:, 1:] != IGNORE_INDEX).sum()),
        len(kept),
    )


def eval_report(
    baselines: dict[str, dict[str, float]], state: Any, loss: float
) -> dict[str, float]:
    """Every eval emission: the model's losses beside the constants they are judged by.

    The bounds (`base`, `no_cot`) and controls (`random`, `surprisal_t0`) are logged
    at every eval as `eval/<name>_row` and `eval/<name>_tok`, even though they never
    change. That is deliberate: it is what lets one chart overlay the model's answer
    CE on all four references, per normalization, which reads at a glance where the
    run sits -- above `no_cot` means the codes carry usable information, below
    `surprisal_t0` means the encoder beats the training-free method it starts from.
    Gap metrics were dropped for the same reason: with constant references, every
    gap curve is the answer-CE curve shifted vertically.

    `eval/loss` is the objective; `estimate_loss` returns the same number, and this
    value (computed from the same accumulators as every component) replaces it.
    """
    del loss
    module = state.module
    losses = getattr(module, "module", module).pop_totals(train=False)
    metrics = {f"eval/{name}": value for name, value in losses.items()}
    for name, values in baselines.items():
        for unit, value in values.items():
            metrics[f"eval/{name}_{unit}"] = value
    return metrics


def with_injected_fault(post_step: Any, fault: Any, rank: int) -> Any:
    """Make exactly one rank raise after one step, to test failure handling end to end.

    Off unless requested with `+training.fault_injection={rank:3,step:5}`. It exists
    because the failure path is only observable through the real launcher: a lone
    rank raising must take the whole job down within seconds (`SLURM_KILL_BAD_EXIT`)
    and leave its real traceback in the log (`picklable_error`) -- the two things
    that failed on 2026-09-08, and that no single-process test can exercise.
    """
    if not fault:
        return post_step
    target_rank, target_step = int(fault.rank), int(fault.step)

    def step(state: Any) -> None:
        post_step(state)
        if rank == target_rank and state.global_step == target_step:
            raise RuntimeError(
                f"injected fault on rank {rank} after step {target_step} "
                "(training.fault_injection)"
            )

    return step


def reseed_dead_codes(state: Any) -> None:
    """Jukebox-style random restarts (arXiv:2005.00341), run after EVERY step.

    No warmup and no interval, matching every reference implementation (Jukebox
    `update_k`, EnCodec `core_vq`, lucidrains `expire_codes_`, vqtorch
    `ReplaceLRU`): they all replace on every forward, so a dead code is caught
    while it is one code rather than after a hundred steps of drift have made it
    a majority. Batching replacement into periodic events is the mechanism Huh et
    al. (arXiv:2305.08842) identify behind restart-induced loss spikes, and it is
    what took this run from 1.11 to 18.0 at step 200.

    Safe here only because `init_codebook_from_encoder_outputs` and `embed_rms`
    put the codebook, the encoder outputs and the decoder's embedding space on one
    scale. With a vocabulary-initialized codebook it would not be.

    One thing remains specific to this setup. `usage` is identical across ranks
    (it is folded from all-reduced counts), so `dead` matches everywhere -- but
    each rank's *pool* of encoder outputs is different data. The replacement
    vectors are therefore drawn on rank 0 and broadcast; drawing per rank would
    silently give every rank a different codebook, since DDP synchronizes
    gradients, not parameters.

    Adam's moments for a replaced row are deliberately left alone. EnCodec has no
    optimizer state to reset, and vqtorch -- a gradient codebook under AdamW, from
    the group that studied replacement policies -- does not reset it either.
    """
    module = getattr(state.module, "module", state.module)
    quantizer = module.encoder.quantizer
    # Measured and recorded unconditionally, BEFORE the enable check. Behind it,
    # a run with restarts disabled reported zero dead codes forever -- the metric
    # would have been blind on exactly the ablation that needs it most.
    dead = quantizer.dead_codes()
    module.last_dead_count = int(dead.numel())

    if not bool(state.cfg.encoder.get("reseed_dead_codes", False)):
        return
    if dead.numel() == 0:
        return
    # Past this point every rank MUST reach the broadcast below. `dead` is derived
    # from `usage`, which is folded from all-reduced counts and so is identical
    # everywhere -- but `code_pool` is rank-local, and returning early on one rank
    # while another blocks in `dist.broadcast` is a silent hang, not an error. It
    # cannot happen with real batches (every rank runs the same number of forwards
    # and no batch has zero valid slots), so raise rather than skip.
    pool = module.code_pool
    if pool is None or pool.shape[0] == 0:
        raise RuntimeError(
            f"{dead.numel()} dead codes but no encoder-output pool to reseed "
            f"from at step {state.global_step}. Skipping would deadlock DDP."
        )

    # Uniform draw from real encoder outputs, unperturbed: EnCodec's `replace_` is
    # `sample_vectors(samples, ...)` with no jitter.
    picks = torch.randint(0, pool.shape[0], (dead.numel(),), device=pool.device)
    replacements = pool[picks].clone().float()
    if state.world_size > 1:
        # Broadcast the vectors, not the indices: the pools differ per rank.
        dist.broadcast(replacements, src=0)

    quantizer.reseed_codes(dead, replacements)
    # Debug, not info: this now runs every step, and at info it would bury the
    # metric lines. `vq/reseeds_total` is the metric that tracks it.
    state.logger.logger.debug(
        f"step={state.global_step} | reseeded {dead.numel()} dead codes "
        f"({quantizer.reseeds} cumulative)"
    )
