"""The trained encoder, as a `CompressionMethod` the eval harness can score.

Wrapping it this way rather than writing a parallel eval loop is the whole point:
results land in the same `summary.json` / `samples.jsonl`, under the same
`_compose_name` join key, alongside `base`, `no_cot` and the training-free pooling
methods on the same rows. Nothing downstream -- plots, reports, the sweep
aggregator -- needs to know a learned method exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from cot_compression.compression import (
    CompressionMethod,
    EmbeddingCompressionMethod,
    _compose_name,
    _patching_values,
    _step_layout,
)
from cot_compression.encoder.frozen import build_latent_init
from cot_compression.encoder.model import CoTEncoder
from cot_compression.patching import PatchingMethod, SplitContext


@dataclass(frozen=True)
class LearnedCompressionMethod(EmbeddingCompressionMethod):
    """Runs the trained compressor to produce one code per patch.

    Subclasses `EmbeddingCompressionMethod` for its `plan`, name composition and
    slot bookkeeping, but overrides `materialize` entirely -- `reduce_patches` is
    never reached, because pooling is exactly what the encoder replaces.
    """

    # Optional only because the parent is a frozen dataclass with defaulted
    # fields, so these need defaults too; `__init__` always sets them.
    # repr/compare off: the generated __repr__ would print the whole encoder into
    # every log line naming a method, and __eq__ would compare nn.Modules.
    encoder: CoTEncoder | None = field(default=None, repr=False, compare=False)
    latent_init: CompressionMethod | None = field(
        default=None, repr=False, compare=False
    )

    def __init__(
        self,
        encoder: CoTEncoder,
        patching: PatchingMethod | None,
        latent_init: CompressionMethod,
    ) -> None:
        config = encoder.config
        # `k{size}` names the codebook; without one there is no size to name, and
        # reporting the dead field would make two different encoders share a join
        # key. Byte-identical to before for every VQ checkpoint.
        tag = (
            f"enc_L{config.n_blocks}k{config.codebook_size}"
            if config.use_vq
            else f"enc_L{config.n_blocks}cont"
        )
        # Appended only when set, so every method name written before positional
        # encodings existed stays byte-identical -- the name is the join key
        # between runs and every plot.
        if config.position_encoding != "none":
            tag = f"{tag}_{config.position_encoding}"
        super().__init__(
            name=_compose_name("learned", patching, tag),
            method_family="learned",
            patching=patching,
            compression_param=tag,
        )
        object.__setattr__(self, "encoder", encoder)
        object.__setattr__(self, "latent_init", latent_init)

    def requires_prefix(self) -> bool:
        """Cross-attention reads `[prompt; CoT]`, so the prompt must come along."""
        return True

    def _parts(self) -> tuple[CoTEncoder, CompressionMethod]:
        assert self.encoder is not None and self.latent_init is not None
        return self.encoder, self.latent_init

    def required_signals(self) -> frozenset[str]:
        """The union of what the patching, the pooling *and the init* consume.

        The init is the easy one to forget: `surprisal_t0` needs the surprisal
        signal even when the patching is signal-free, and omitting it here would
        make every sample raise and be skipped.
        """
        _, latent_init = self._parts()
        return super().required_signals() | latent_init.required_signals()

    def reduce_patches(
        self, embeds: torch.Tensor, weights: torch.Tensor | None
    ) -> torch.Tensor:
        raise AssertionError(
            "LearnedCompressionMethod overrides materialize; reduce_patches is "
            "unreachable. Reaching it means the base class changed."
        )

    @torch.no_grad()
    def materialize(
        self,
        cot_ids: list[int],
        sample_index: int,
        seed: int,
        tokenizer: Any,
        model: Any,
        device: torch.device,
        cot_signals: dict[str, torch.Tensor | None] | None,
        prefix_ids: list[int] | None = None,
        context: SplitContext | None = None,
    ) -> torch.Tensor | None:
        if prefix_ids is None:
            raise ValueError(
                "LearnedCompressionMethod needs prefix_ids; requires_prefix() must "
                "be honoured by the caller."
            )
        # Refuse rather than fall back. Without the layout this would still produce
        # a [K, d] matrix -- from the wrong K, and with every latent of a step
        # anchored at the same position it never trained under -- i.e. a plausible
        # number that is wrong. RuntimeError, not ValueError: `evaluate_method`
        # counts a ValueError from `materialize` as a data skip, which would turn
        # this wiring bug into a run that silently scores nothing.
        if (
            self.patching is not None
            and self.patching.requires_context()
            and context is None
        ):
            raise RuntimeError(
                f"patching {self.patching.name!r} requires a SplitContext; the "
                "eval worker must resolve the layout and pass it along."
            )
        # Pass A. Deliberately NOT through FrozenBackbone: that constructor mutates
        # the model (freezing it, enabling gradient checkpointing), which is right
        # for training and pointless under no_grad where nothing is retained.
        context_ids = torch.tensor([prefix_ids + cot_ids], device=device)
        memory = model.model(
            input_ids=context_ids, attention_mask=torch.ones_like(context_ids)
        ).last_hidden_state

        encoder, latent_init = self._parts()
        # The context goes to the initializer too: it re-splits through
        # `_split_spans`, and without it a paragraph-patched row would get a seed
        # sized for uniform spans against a step-sized render.
        latent = latent_init.materialize(
            cot_ids,
            sample_index,
            seed,
            tokenizer,
            model,
            device,
            cot_signals,
            context=context,
        )
        if latent is None:
            return None

        layout = _step_layout(
            self,
            len(cot_ids),
            _patching_values(self, cot_signals, device),
            sample_index,
            seed,
            context,
        )
        prompt_len = len(prefix_ids)
        # Step end, shared by a step's latents: each sees its whole step and nothing
        # after it. Under one-latent-per-step patching this is the span end, as before.
        cross_limit = torch.tensor(
            [[prompt_len + layout.steps[step][1] for step in layout.step_of_latent]],
            device=device,
        )
        query_limit = (
            torch.tensor(
                [[prompt_len + end for _, end in layout.latents]], device=device
            )
            if encoder.config.query_anchor == "substep"
            else None
        )
        slots = latent.shape[0]
        out = encoder(
            latent.unsqueeze(0).to(memory.dtype),
            memory,
            torch.ones(1, slots, dtype=torch.bool, device=device),
            torch.ones(1, context_ids.shape[1], dtype=torch.bool, device=device),
            cross_limit=cross_limit,
            query_limit=query_limit,
        )
        return out.z[0]


def build_learned_method(
    cfg: DictConfig, patching: PatchingMethod | None, device: torch.device
) -> LearnedCompressionMethod:
    """Load a trained encoder from `evaluation.methods.learned.checkpoint`.

    The checkpoint carries its own `EncoderConfig`, so the latent initializer and
    every architectural choice come from what was trained -- not from whatever the
    eval config happens to say. A mismatch there would score one model while
    reporting another's configuration.
    """
    from cot_compression.encoder.training import load_encoder

    checkpoint = cfg.evaluation.methods.learned.checkpoint
    if checkpoint is None:
        raise ValueError(
            "evaluation.methods.learned.checkpoint must point at a trained "
            "encoder directory (one containing encoder.pt and encoder_config.json)."
        )
    encoder = load_encoder(Path(str(checkpoint)), device).eval()
    return LearnedCompressionMethod(
        encoder=encoder,
        patching=patching,
        latent_init=build_latent_init(encoder.config.latent_init, patching),
    )
