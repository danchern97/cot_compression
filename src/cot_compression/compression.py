from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from cot_compression.data.answers import AnswerTrace, replace_trace
from cot_compression.data.dolci import Message
from cot_compression.patching import (
    EntropyPatchingMethod,
    PatchingMethod,
    RandomPatchingMethod,
    UniformPatchingMethod,
)


def _tokenize_cot(trace: AnswerTrace, tokenizer: Any) -> list[int]:
    cot_ids = tokenizer(trace.trace, add_special_tokens=False)["input_ids"]
    if not cot_ids:
        raise ValueError("CoT trace produced no tokens to compress.")
    return list(cot_ids)


def _prefix_ids(trace: AnswerTrace, tokenizer: Any) -> list[int]:
    """Tokenize everything preceding the CoT trace in the rendered chat.

    Used so the entropy forward pass can see the same prompt context the
    main answer-logprob forward pass sees, instead of scoring the CoT in
    isolation.
    """
    rendered = tokenizer.apply_chat_template(
        trace.messages, tokenize=False, add_generation_prompt=False
    )
    trace_start = rendered.rfind(trace.trace)
    if trace_start == -1:
        raise ValueError("Could not find CoT trace text in rendered chat.")
    prefix_text = rendered[:trace_start]
    if not prefix_text:
        return []
    return list(tokenizer(prefix_text, add_special_tokens=False)["input_ids"])


def _token_entropies(
    cot_embeds: torch.Tensor,
    prefix_ids: list[int],
    model: Any,
    device: torch.device,
) -> torch.Tensor:
    if prefix_ids:
        prefix_tensor = torch.tensor(prefix_ids, device=device)
        prefix_embeds = model.get_input_embeddings()(prefix_tensor)
        full_embeds = torch.cat([prefix_embeds, cot_embeds], dim=0)
    else:
        full_embeds = cot_embeds
    logits = model(inputs_embeds=full_embeds.unsqueeze(0)).logits[0]
    log_probs = F.log_softmax(logits, dim=-1)
    entropies = -(log_probs.exp() * log_probs).sum(dim=-1)
    return entropies[len(prefix_ids) :]


def _cot_embeds_spans_and_entropies(
    trace: AnswerTrace,
    cot_ids: list[int],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    patching: PatchingMethod | None,
    isolate_cot_context: bool,
    needs_entropies: bool,
    sample_index: int,
    seed: int,
) -> tuple[torch.Tensor, list[tuple[int, int]], torch.Tensor | None]:
    """Embed the CoT, optionally compute per-token entropies, and split into
    patch spans. Shared by EmbeddingCompressionMethod.prepare() (which also
    needs the embeddings for reduce_patch) and RandomCompressionMethod's
    patched path (which only needs the spans, and entropies if its patching
    strategy requires them).
    """
    ids_tensor = torch.tensor(cot_ids, device=device)
    embeds = model.get_input_embeddings()(ids_tensor)

    entropies = None
    if needs_entropies:
        prefix_ids = [] if isolate_cot_context else _prefix_ids(trace, tokenizer)
        entropies = _token_entropies(embeds, prefix_ids, model, device)

    spans = (
        patching.split(len(cot_ids), sample_index, seed, entropies)
        if patching is not None
        else [(0, len(cot_ids))]
    )
    return embeds, spans, entropies


@dataclass(frozen=True)
class CompressionMethod:
    name: str
    abstract_tokens: list[str]

    def transform(
        self,
        trace: AnswerTrace,
        sample_index: int,
        seed: int,
        tokenizer: Any = None,
        model: Any = None,
        device: torch.device | None = None,
    ) -> list[Message]:
        raise NotImplementedError

    def requires_model_embeddings(self) -> bool:
        """Whether this method needs prepare() at eval time instead of transform().

        Methods returning True replace the CoT trace with one placeholder
        token per patch; each placeholder's embedding is computed per-sample
        and spliced directly into inputs_embeds, rather than looked up from
        the model's embedding table.
        """
        return False


@dataclass(frozen=True)
class BaseCompressionMethod(CompressionMethod):
    def __init__(self) -> None:
        super().__init__(name="base", abstract_tokens=[])

    def transform(
        self,
        trace: AnswerTrace,
        sample_index: int,
        seed: int,
        tokenizer: Any = None,
        model: Any = None,
        device: torch.device | None = None,
    ) -> list[Message]:
        del sample_index, seed, tokenizer, model, device
        return trace.messages


@dataclass(frozen=True)
class RandomCompressionMethod(CompressionMethod):
    """Replaces the CoT with random abstract-token text.

    With no `patching`, the whole trace becomes `abstract_length` random
    tokens (unchanged, original behavior). With `patching`, each patch
    becomes exactly 1 random abstract token instead -- matching the other
    compression methods, which always emit one unit of compressed
    representation per patch, so compression ratios stay comparable across
    methods at the same patching granularity.
    """

    abstract_length: int
    patching: PatchingMethod | None = None
    isolate_cot_context: bool = False

    def __init__(
        self,
        abstract_vocab_size: int,
        abstract_length: int,
        patching: PatchingMethod | None = None,
        isolate_cot_context: bool = False,
    ) -> None:
        if abstract_vocab_size <= 0:
            raise ValueError("abstract_vocab_size must be positive.")
        if abstract_length <= 0:
            raise ValueError("abstract_length must be positive.")
        tokens = [f"<abs_{index:05d}>" for index in range(abstract_vocab_size)]
        name = "random" if patching is None else f"random_{patching.name}"
        super().__init__(name=name, abstract_tokens=tokens)
        object.__setattr__(self, "abstract_length", abstract_length)
        object.__setattr__(self, "patching", patching)
        object.__setattr__(self, "isolate_cot_context", isolate_cot_context)

    def transform(
        self,
        trace: AnswerTrace,
        sample_index: int,
        seed: int,
        tokenizer: Any = None,
        model: Any = None,
        device: torch.device | None = None,
    ) -> list[Message]:
        rng = random.Random(seed + sample_index)
        if self.patching is None:
            del tokenizer, model, device
            replacement = " ".join(
                rng.choice(self.abstract_tokens) for _ in range(self.abstract_length)
            )
            return replace_trace(trace, replacement)

        cot_ids = _tokenize_cot(trace, tokenizer)
        needs_entropies = self.patching.requires_entropies()
        with torch.no_grad():
            _, spans, _ = _cot_embeds_spans_and_entropies(
                trace,
                cot_ids,
                tokenizer,
                model,
                device,
                self.patching,
                self.isolate_cot_context,
                needs_entropies,
                sample_index,
                seed,
            )
        replacement = " ".join(rng.choice(self.abstract_tokens) for _ in spans)
        return replace_trace(trace, replacement)


@dataclass(frozen=True)
class EmbeddedTrace:
    messages: list[Message]
    slot_embeddings: torch.Tensor  # [num_patches, hidden_size]


@dataclass(frozen=True)
class EmbeddingCompressionMethod(CompressionMethod):
    """Base for methods that compress the CoT into per-patch embedding vectors.

    Each patch (by default, the whole trace as a single patch; see
    `patching`) is reduced to one vector by `reduce_patch`. The trace text is
    replaced by one repeated placeholder token per patch, and at eval time
    those placeholder positions have their embeddings overwritten with the
    computed vectors directly in inputs_embeds (never written into the
    model's embedding table, so different samples in a batch can carry
    different vectors for the same placeholder token).
    """

    slot_token: str
    patching: PatchingMethod | None = None
    isolate_cot_context: bool = False

    def requires_model_embeddings(self) -> bool:
        return True

    def needs_entropies(self) -> bool:
        return False

    def reduce_patch(
        self,
        embeds: torch.Tensor,
        entropies: torch.Tensor | None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def prepare(
        self,
        trace: AnswerTrace,
        tokenizer: Any,
        model: Any,
        device: torch.device,
        sample_index: int,
        seed: int,
    ) -> EmbeddedTrace:
        cot_ids = _tokenize_cot(trace, tokenizer)
        needs_entropies = self.needs_entropies() or (
            self.patching is not None and self.patching.requires_entropies()
        )

        with torch.no_grad():
            embeds, spans, entropies = _cot_embeds_spans_and_entropies(
                trace,
                cot_ids,
                tokenizer,
                model,
                device,
                self.patching,
                self.isolate_cot_context,
                needs_entropies,
                sample_index,
                seed,
            )
            slot_vectors = [
                self.reduce_patch(
                    embeds[start:end],
                    entropies[start:end] if entropies is not None else None,
                )
                for start, end in spans
            ]
            slot_embeddings = torch.stack(slot_vectors, dim=0)

        placeholder = " ".join([self.slot_token] * len(spans))
        messages = replace_trace(trace, placeholder)
        return EmbeddedTrace(messages=messages, slot_embeddings=slot_embeddings)


@dataclass(frozen=True)
class SimpleMeanCompressionMethod(EmbeddingCompressionMethod):
    def __init__(self, patching: PatchingMethod | None = None) -> None:
        name = "simple_mean" if patching is None else f"simple_mean_{patching.name}"
        super().__init__(
            name=name,
            abstract_tokens=["<abs_mean>"],
            slot_token="<abs_mean>",
            patching=patching,
        )

    def reduce_patch(
        self,
        embeds: torch.Tensor,
        entropies: torch.Tensor | None,
    ) -> torch.Tensor:
        del entropies
        return embeds.mean(dim=0)


@dataclass(frozen=True)
class EntropyWeightedMeanCompressionMethod(EmbeddingCompressionMethod):
    def __init__(
        self,
        patching: PatchingMethod | None = None,
        isolate_cot_context: bool = False,
    ) -> None:
        name = (
            "entropy_weighted_mean"
            if patching is None
            else f"entropy_weighted_mean_{patching.name}"
        )
        super().__init__(
            name=name,
            abstract_tokens=["<abs_entropy_mean>"],
            slot_token="<abs_entropy_mean>",
            patching=patching,
            isolate_cot_context=isolate_cot_context,
        )

    def needs_entropies(self) -> bool:
        return True

    def reduce_patch(
        self,
        embeds: torch.Tensor,
        entropies: torch.Tensor | None,
    ) -> torch.Tensor:
        assert entropies is not None
        weights = entropies / entropies.sum()
        return (weights.unsqueeze(-1) * embeds).sum(dim=0)


def build_patching_method(
    strategy: str | None,
    patching_cfg: DictConfig,
) -> PatchingMethod | None:
    if strategy is None:
        return None
    if strategy == "uniform":
        return UniformPatchingMethod(patch_size=int(patching_cfg.uniform.patch_size))
    if strategy == "random":
        return RandomPatchingMethod(
            max_exponent=int(patching_cfg.random.max_exponent)
        )
    if strategy == "entropy":
        return EntropyPatchingMethod(
            percentile=float(patching_cfg.entropy.percentile)
        )
    raise ValueError(f"Unknown patching strategy: {strategy}")


def build_compression_methods(cfg: DictConfig) -> list[CompressionMethod]:
    methods = []
    for name in cfg.evaluation.methods.enabled:
        if name == "base":
            methods.append(BaseCompressionMethod())
        elif name == "random":
            random_cfg = cfg.evaluation.methods.random
            methods.append(
                RandomCompressionMethod(
                    abstract_vocab_size=int(random_cfg.abstract_vocab_size),
                    abstract_length=int(random_cfg.abstract_length),
                    patching=build_patching_method(
                        random_cfg.patching, cfg.evaluation.methods.patching
                    ),
                    isolate_cot_context=bool(random_cfg.isolate_cot_context),
                )
            )
        elif name == "simple_mean":
            method_cfg = cfg.evaluation.methods.simple_mean
            methods.append(
                SimpleMeanCompressionMethod(
                    patching=build_patching_method(
                        method_cfg.patching, cfg.evaluation.methods.patching
                    ),
                )
            )
        elif name == "entropy_weighted_mean":
            method_cfg = cfg.evaluation.methods.entropy_weighted_mean
            methods.append(
                EntropyWeightedMeanCompressionMethod(
                    patching=build_patching_method(
                        method_cfg.patching, cfg.evaluation.methods.patching
                    ),
                    isolate_cot_context=bool(method_cfg.isolate_cot_context),
                )
            )
        else:
            raise ValueError(f"Unknown evaluation method: {name}")
    return methods


def extend_tokenizer_and_model(
    tokenizer: Any,
    model: Any,
    methods: list[CompressionMethod],
) -> None:
    tokens = []
    for method in methods:
        tokens.extend(method.abstract_tokens)
    if not tokens:
        return

    tokenizer.add_tokens(tokens)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
