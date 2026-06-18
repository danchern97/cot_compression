"""Small research template package."""

from research_project_template.data import build_data_module
from research_project_template.methods import build_method
from research_project_template.training.evaluate import evaluate
from research_project_template.training.generate import generate
from research_project_template.training.train import train

__all__ = [
    "build_data_module",
    "build_method",
    "evaluate",
    "generate",
    "train",
]
