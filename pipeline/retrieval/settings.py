"""Configuration resolution for embedding and vector retrieval commands."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .embedder import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_EMBEDDING_MODEL

load_dotenv()


def _env_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def load_retrieval_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Retrieval config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_embedding_settings(
    config: dict,
    *,
    model: str | None = None,
    dimension: int | None = None,
    normalize: bool | None = None,
) -> dict:
    """Resolve CLI > environment > YAML > default settings."""
    embedding = config.get("embedding", {})
    resolved_model = (
        model
        or os.getenv("EMBEDDING_MODEL")
        or embedding.get("model")
        or DEFAULT_EMBEDDING_MODEL
    )
    env_dimension = os.getenv("EMBEDDING_DIMENSION")
    resolved_dimension = int(
        dimension
        if dimension is not None
        else env_dimension
        if env_dimension is not None
        else embedding.get("dimension", DEFAULT_EMBEDDING_DIMENSION)
    )
    env_normalize = _env_bool("EMBEDDING_NORMALIZE")
    resolved_normalize = (
        normalize
        if normalize is not None
        else env_normalize
        if env_normalize is not None
        else bool(embedding.get("normalize", True))
    )
    provider = os.getenv("EMBEDDING_PROVIDER") or embedding.get("provider", "gemini")
    if provider != "gemini":
        raise ValueError(
            f"Unsupported embedding provider {provider!r}; this pipeline requires 'gemini'"
        )
    return {
        "provider": provider,
        "model": resolved_model,
        "dimension": resolved_dimension,
        "normalize": resolved_normalize,
    }
