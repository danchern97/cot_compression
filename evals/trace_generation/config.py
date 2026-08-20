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

FILTER_VERSION = "math-qa-v2"
NUM_ROLLOUTS = 4
BASE_SEED = 1337
MAX_OUTPUT_TOKENS = 32_768
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
    num_rollouts: int = NUM_ROLLOUTS
    base_seed: int = BASE_SEED
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    top_k: int = TOP_K
    min_p: float = MIN_P
    presence_penalty: float = PRESENCE_PENALTY
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 40_960
    max_examples: int | None = None
    num_shards: int = 1
    chunk_size: int = 64
    filter_version: str = FILTER_VERSION

    def __post_init__(self) -> None:
        if self.num_rollouts != NUM_ROLLOUTS:
            raise ValueError(f"num_rollouts must be exactly {NUM_ROLLOUTS}")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.max_model_len <= self.max_output_tokens:
            raise ValueError("max_model_len must be greater than max_output_tokens")
        if self.max_examples is not None and self.max_examples <= 0:
            raise ValueError("max_examples must be positive when set")
        if self.num_shards <= 0:
            raise ValueError("num_shards must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")

    def payload(self) -> dict[str, object]:
        return asdict(self)

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


def rollout_seed(base_seed: int, source_row_index: int, rollout_index: int) -> int:
    """Stable, order-independent seed for one rollout."""
    return base_seed + source_row_index * NUM_ROLLOUTS + rollout_index
