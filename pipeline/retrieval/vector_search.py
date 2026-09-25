"""Query embedding and FAISS lookup with artifact compatibility checks."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from .embedder import GeminiEmbedder


def search_faiss(
    query: str,
    index_dir: str | Path,
    *,
    top_k: int = 10,
    embedder: GeminiEmbedder | None = None,
) -> list[dict]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must be a non-empty string")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    index_dir = Path(index_dir)
    manifest_path = index_dir / "manifest.json"
    index_path = index_dir / "index.faiss"
    id_map_path = index_dir / "id_map.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Search artifact not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_name = manifest.get("embedding_model")
    dimension = int(manifest.get("embedding_dimension", 0))
    normalized = bool(manifest.get("normalized"))
    if manifest.get("index_type") != "IndexFlatIP":
        raise ValueError(f"Unsupported FAISS index type: {manifest.get('index_type')}")
    if manifest.get("embedding_provider") != "gemini":
        raise ValueError(
            f"Unsupported embedding provider: {manifest.get('embedding_provider')}"
        )

    model = embedder or GeminiEmbedder(
        model=model_name, dimension=dimension, normalize=normalized
    )
    if (
        model.model != model_name
        or model.dimension != dimension
        or model.normalize != normalized
    ):
        raise ValueError(
            "Query embedder does not match index manifest: "
            f"expected {model_name}/{dimension}/normalized={normalized}"
        )

    for path in (index_path, id_map_path):
        if not path.exists():
            raise FileNotFoundError(f"Search artifact not found: {path}")

    index = faiss.read_index(str(index_path))
    if index.d != dimension:
        raise ValueError(
            f"FAISS index dimension mismatch: manifest={dimension}, index={index.d}"
        )
    if index.ntotal < 1:
        raise ValueError("FAISS index is empty")

    id_map = pd.read_parquet(id_map_path)
    if not {"faiss_id", "scene_id"}.issubset(id_map.columns):
        raise ValueError("Index id_map.parquet must contain faiss_id and scene_id")
    mapping = {
        int(row.faiss_id): str(row.scene_id)
        for row in id_map.itertuples(index=False)
    }
    if len(mapping) != index.ntotal:
        raise ValueError(
            f"FAISS/id_map size mismatch: index={index.ntotal}, id_map={len(mapping)}"
        )

    query_vector = np.asarray([model.embed_query(query)], dtype=np.float32)
    if query_vector.shape != (1, dimension):
        raise ValueError(
            f"Query vector shape mismatch: expected (1, {dimension}), got {query_vector.shape}"
        )
    if not np.isfinite(query_vector).all():
        raise ValueError("Query vector contains NaN or infinite values")
    if normalized:
        if float(np.linalg.norm(query_vector)) <= 0:
            raise ValueError("Query vector has zero norm")
        faiss.normalize_L2(query_vector)

    scores, ids = index.search(query_vector, min(top_k, int(index.ntotal)))
    results = []
    for score, faiss_id in zip(scores[0], ids[0], strict=True):
        faiss_id = int(faiss_id)
        if faiss_id < 0:
            continue
        if faiss_id not in mapping:
            raise ValueError(f"FAISS result ID {faiss_id} is missing from id_map")
        results.append(
            {
                "faiss_id": faiss_id,
                "scene_id": mapping[faiss_id],
                "score": float(score),
            }
        )
    return results
