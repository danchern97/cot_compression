"""Training, evaluation, and generation workflows."""

from research_project_template.training.evaluate import evaluate
from research_project_template.training.generate import generate
from research_project_template.training.train import train

__all__ = ["evaluate", "generate", "train"]
