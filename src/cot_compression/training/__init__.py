"""Training workflows."""

from cot_compression.training.evaluate import evaluate_methods
from cot_compression.training.sft import train_sft

__all__ = ["evaluate_methods", "train_sft"]
