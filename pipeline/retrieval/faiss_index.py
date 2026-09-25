"""Validated FAISS IndexFlatIP builder for Gemini embeddings."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from .embedder import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_EMBEDDING_MODEL
from .prompts import PROMPT_VERSION
from .taxonomy import TAXONOMY_VERSION


REQUIRED_EMBEDDING_COLUMNS = {
    "scene_id",
    "embedding",
    "embedding_model",
    "embedding_dimension",
    "normalized",
    "pipeline_version",
}
REQUIRED_ID_MAP_COLUMNS = {"faiss_id", "scene_id"}


def _single_value(frame: pd.DataFrame, column: str):
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"embeddings.parquet has inconsistent {column}: {values}")
    return values[0]


def load_embedding_artifacts(
    embeddings_path: str | Path,
    id_map_path: str | Path,
    *,
    expected_model: str | None = None,
    expected_dimension: int | None = None,
    expected_normalized: bool | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    embeddings_path = Path(embeddings_path)
    id_map_path = Path(id_map_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings artifact not found: {embeddings_path}")
    if not id_map_path.exists():
        raise FileNotFoundError(f"Embedding ID map not found: {id_map_path}")

    embeddings = pd.read_parquet(embeddings_path)
    id_map = pd.read_parquet(id_map_path)
    missing = REQUIRED_EMBEDDING_COLUMNS - set(embeddings.columns)
    if missing:
        raise ValueError(
            f"embeddings.parquet has incompatible schema; missing columns: {sorted(missing)}"
        )
    missing = REQUIRED_ID_MAP_COLUMNS - set(id_map.columns)
    if missing:
        raise ValueError(
            f"id_map.parquet has incompatible schema; missing columns: {sorted(missing)}"
        )
    if embeddings.empty:
        raise ValueError("embeddings.parquet contains no vectors")
    if len(embeddings) != len(id_map):
        raise ValueError(
            f"Embedding/id_map row mismatch: {len(embeddings)} != {len(id_map)}"
        )

    model = str(_single_value(embeddings, "embedding_model"))
    dimension = int(_single_value(embeddings, "embedding_dimension"))
    normalized = bool(_single_value(embeddings, "normalized"))
    pipeline_version = str(_single_value(embeddings, "pipeline_version"))
    if expected_model is not None and model != expected_model:
        raise ValueError(
            f"Embedding model mismatch: expected {expected_model}, artifact uses {model}"
        )
    if expected_dimension is not None and dimension != expected_dimension:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dimension}, artifact uses {dimension}"
        )
    if expected_normalized is not None and normalized != expected_normalized:
        raise ValueError(
            f"Embedding normalization mismatch: expected {expected_normalized}, artifact uses {normalized}"
        )

    try:
        vectors = np.stack(
            [np.asarray(value, dtype=np.float32) for value in embeddings["embedding"]]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("embeddings.parquet contains malformed vectors") from exc
    if vectors.ndim != 2 or vectors.shape != (len(embeddings), dimension):
        raise ValueError(
            "Embedding matrix shape mismatch: "
            f"expected ({len(embeddings)}, {dimension}), got {vectors.shape}"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding matrix contains NaN or infinite values")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Embedding matrix contains zero or invalid vectors")

    expected_ids = np.arange(len(id_map), dtype=np.int64)
    actual_ids = id_map["faiss_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_ids, expected_ids):
        raise ValueError("id_map.parquet faiss_id must be contiguous and start at zero")
    if embeddings["scene_id"].astype(str).tolist() != id_map["scene_id"].astype(str).tolist():
        raise ValueError("Scene order differs between embeddings.parquet and id_map.parquet")

    metadata = {
        "embedding_model": model,
        "embedding_dimension": dimension,
        "normalized": normalized,
        "pipeline_version": pipeline_version,
    }
    return np.asarray(vectors, dtype=np.float32), id_map[["faiss_id", "scene_id"]], metadata


def build_faiss(
    embeddings_path: str | Path,
    id_map_path: str | Path,
    output_root: str | Path,
    *,
    embedding_provider: str = "gemini",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    normalize: bool = True,
    index_version: str = "v1",
) -> Path:
    vectors, id_map, metadata = load_embedding_artifacts(
        embeddings_path,
        id_map_path,
        expected_model=embedding_model,
        expected_dimension=dimension,
        expected_normalized=normalize,
    )
    if normalize:
        faiss.normalize_L2(vectors)
    if vectors.dtype != np.float32:
        raise ValueError(f"FAISS vectors must be float32, got {vectors.dtype}")

    index = faiss.IndexFlatIP(dimension)
    if index.d != dimension:
        raise RuntimeError(
            f"FAISS index dimension mismatch: expected {dimension}, got {index.d}"
        )
    index.add(vectors)

    output_dir = Path(output_root) / index_version
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    id_map[["faiss_id", "scene_id"]].sort_values("faiss_id").to_parquet(
        output_dir / "id_map.parquet", index=False
    )
    manifest = {
        "index_version": index_version,
        "embedding_provider": embedding_provider,
        "embedding_model": metadata["embedding_model"],
        "embedding_dimension": metadata["embedding_dimension"],
        "index_type": "IndexFlatIP",
        "normalized": metadata["normalized"],
        "number_of_scenes": int(index.ntotal),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "taxonomy_version": TAXONOMY_VERSION,
        "prompt_version": PROMPT_VERSION,
        "pipeline_version": metadata["pipeline_version"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir
