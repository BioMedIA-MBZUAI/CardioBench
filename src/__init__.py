"""Self-contained modular pipeline for Echo/BioMed CLIP workflows."""

from .datasets import DatasetLoader, DatasetItem
from .embeddings import EmbeddingConfig, generate_embeddings, generate_embeddings_for_splits
from .models import ModelConfig, load_model, load_model_by_name, resolve_model_id
from .regression import (
    EjectionFractionConfig,
    LvhRegressionConfig,
    estimate_ejection_fraction,
    estimate_lvh_metrics,
)

__all__ = [
    "DatasetLoader",
    "DatasetItem",
    "EmbeddingConfig",
    "generate_embeddings",
    "generate_embeddings_for_splits",
    "ModelConfig",
    "load_model",
    "load_model_by_name",
    "resolve_model_id",
    "EjectionFractionConfig",
    "LvhRegressionConfig",
    "estimate_ejection_fraction",
    "estimate_lvh_metrics",
]
