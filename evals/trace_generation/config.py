from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

DATASET_NAME = "allenai/Dolci-Think-RL-7B"
DATASET_REVISION = "0fb6466d31ef3a9dd16985ef635e6429e05a6491"
DATASET_SPLIT = "train"
MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

FILTER_VERSION = "compression-all-v1"
DOMAIN_NAMES = (
    "math",
    "code",
    "code_stdio",
    "ifeval",
    "general_quality_ref",
    "general_quality",
)
NUM_ROLLOUTS = 2
FULFILLMENT_NUM_ROLLOUTS = 4
BASE_SEED = 1337
MAX_OUTPUT_TOKENS = 32_768
CODE_MAX_OUTPUT_TOKENS = 16_384
DEFAULT_MAX_OUTPUT_TOKENS = 8_192
ROLLOUT_SEED_STRIDE = 1_024
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
PRESENCE_PENALTY = 0.0


@dataclass(frozen=True)
class GenerationConfig:
    dataset: str = DATASET_NAME
    dataset_revision: str = DATASET_REVISION
    split: str = DATASET_SPLIT
    model: str = MODEL_NAME
    model_revision: str = MODEL_REVISION
    domains: tuple[str, ...] = DOMAIN_NAMES
    num_rollouts: int = NUM_ROLLOUTS
    rollout_index_start: int = 0
    base_seed: int = BASE_SEED
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    code_max_output_tokens: int = CODE_MAX_OUTPUT_TOKENS
    default_max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    top_k: int = TOP_K
    min_p: float = MIN_P
    presence_penalty: float = PRESENCE_PENALTY
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.95
    batch_invariant: bool = False
    v1_multiprocessing: bool = True
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 32_768
    enable_prefix_caching: bool = True
    max_model_len: int = 40_960
    max_examples: int | None = None
    num_shards: int = 1
    chunk_size: int = 128
    filter_version: str = FILTER_VERSION

    def __post_init__(self) -> None:
        if self.num_rollouts <= 0:
            raise ValueError("num_rollouts must be positive")
        if self.rollout_index_start < 0:
            raise ValueError("rollout_index_start must be non-negative")
        if self.rollout_index_start + self.num_rollouts > ROLLOUT_SEED_STRIDE:
            raise ValueError(
                "rollout index range exceeds the stable seed stride "
                f"({ROLLOUT_SEED_STRIDE})"
            )
        if not self.domains or not set(self.domains) <= set(DOMAIN_NAMES):
            raise ValueError(f"domains must be selected from {DOMAIN_NAMES}")
        if (
            min(
                self.max_output_tokens,
                self.code_max_output_tokens,
                self.default_max_output_tokens,
            )
            <= 0
        ):
            raise ValueError("all output token caps must be positive")
        if self.max_output_tokens < max(
            self.code_max_output_tokens, self.default_max_output_tokens
        ):
            raise ValueError("max_output_tokens must be the largest domain cap")
        if self.max_model_len <= self.max_output_tokens:
            raise ValueError("max_model_len must be greater than max_output_tokens")
        if self.max_examples is not None and self.max_examples <= 0:
            raise ValueError("max_examples must be positive when set")
        if self.num_shards <= 0:
            raise ValueError("num_shards must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.max_num_seqs <= 0 or self.max_num_batched_tokens <= 0:
            raise ValueError("vLLM batch limits must be positive")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["domains"] = list(self.domains)
        # Preserve hashes of schema-v2 base runs created before fulfillment
        # introduced an explicit rollout offset.
        if self.rollout_index_start == 0:
            payload.pop("rollout_index_start")
        return payload

    @property
    def rollout_indices(self) -> range:
        return range(
            self.rollout_index_start,
            self.rollout_index_start + self.num_rollouts,
        )

    @property
    def config_hash(self) -> str:
        encoded = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]

    @property
    def model_slug(self) -> str:
        return self.model.replace("/", "__").replace(" ", "_")

    def run_dir(self, output_root: Path) -> Path:
        return Path(output_root) / f"{self.model_slug}__{self.config_hash}"

    def max_tokens_for(self, domain: str) -> int:
        if domain == "math":
            return self.max_output_tokens
        if domain in {"code", "code_stdio"}:
            return self.code_max_output_tokens
        return self.default_max_output_tokens


def rollout_seed(base_seed: int, source_row_index: int, rollout_index: int) -> int:
    """Stable, order-independent seed for one rollout."""
    if not 0 <= rollout_index < ROLLOUT_SEED_STRIDE:
        raise ValueError(f"rollout_index must be in [0, {ROLLOUT_SEED_STRIDE})")
    return base_seed + source_row_index * ROLLOUT_SEED_STRIDE + rollout_index
