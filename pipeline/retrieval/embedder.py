"""Gemini Embedding 2 adapter with validation, retry, and resumable persistence."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

LOGGER = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSION = 768
DEFAULT_BATCH_SIZE = 16
EMBEDDING_PIPELINE_VERSION = "embedding-pipeline-v2"
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}

EMBEDDING_COLUMNS = (
    "scene_id",
    "embedding",
    "embedding_model",
    "embedding_dimension",
    "normalized",
    "pipeline_version",
    "text_hash",
)


def searchable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validated_vector(values: Any, dimension: int, normalize: bool) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Gemini returned a non-numeric embedding") from exc
    if vector.shape != (dimension,):
        raise ValueError(
            f"Embedding dimension mismatch: expected {dimension}, got shape {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise ValueError("Embedding contains NaN or infinite values")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding has zero or invalid norm")
    if normalize:
        vector = vector / norm
    return np.asarray(vector, dtype=np.float32)


class GeminiEmbedder:
    """Generate validated document and query vectors with Gemini Embedding 2."""

    provider = "gemini"

    def __init__(
        self,
        model: str | None = None,
        dimension: int | None = None,
        normalize: bool = True,
        *,
        api_key: str | None = None,
        client=None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
    ):
        self.model = model or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.dimension = int(
            dimension
            if dimension is not None
            else os.getenv("EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION))
        )
        self.normalize = bool(normalize)
        self.max_retries = int(
            max_retries
            if max_retries is not None
            else os.getenv("EMBEDDING_MAX_ATTEMPTS", os.getenv("GEMINI_MAX_ATTEMPTS", "5"))
        )
        self.retry_delay = float(
            retry_delay
            if retry_delay is not None
            else os.getenv("EMBEDDING_RETRY_DELAY_SECONDS", os.getenv("GEMINI_RETRY_DELAY_SECONDS", "10"))
        )
        if self.dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        if self.max_retries < 1:
            raise ValueError("Embedding max_retries must be positive")
        if not math.isfinite(self.retry_delay) or self.retry_delay <= 0:
            raise ValueError("Embedding retry_delay must be finite and positive")

        key = api_key or os.getenv("GEMINI_API_KEY")
        if client is None and not key:
            raise RuntimeError(
                "GEMINI_API_KEY is required for Gemini embeddings. Set it in the environment or .env."
            )
        self.client = client or genai.Client(api_key=key)

    def _wait_before_retry(self, attempt: int) -> None:
        base = min(60.0, self.retry_delay * (2 ** min(attempt, 10)))
        delay = min(60.0, base + random.uniform(0, base * 0.2))
        LOGGER.warning("Retrying Gemini embedding request in %.1fs", delay)
        time.sleep(delay)

    def _embed_request(self, contents):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self.client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dimension
                    ),
                )
            except (httpx.TransportError, errors.APIError) as exc:
                if isinstance(exc, errors.APIError) and exc.code not in TRANSIENT_STATUS_CODES:
                    raise RuntimeError(
                        f"Gemini embedding request failed permanently with HTTP {exc.code}: {exc}"
                    ) from exc
                last_error = exc
                LOGGER.warning(
                    "Gemini embedding request failed attempt=%d/%d: %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                if attempt + 1 < self.max_retries:
                    self._wait_before_retry(attempt)
        raise RuntimeError(
            f"Gemini embedding request failed after {self.max_retries} attempts"
        ) from last_error

    def _vectors_from_response(self, response, expected_count: int) -> list[np.ndarray]:
        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise RuntimeError("Gemini embedding response contains no embeddings")
        if len(embeddings) != expected_count:
            raise RuntimeError(
                "Gemini embedding response count mismatch: "
                f"expected {expected_count}, got {len(embeddings)}"
            )
        vectors = []
        for position, embedding in enumerate(embeddings):
            values = getattr(embedding, "values", None)
            if values is None:
                raise RuntimeError(
                    f"Gemini embedding response item {position} has no values"
                )
            vectors.append(_validated_vector(values, self.dimension, self.normalize))
        return vectors

    @staticmethod
    def _require_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding input text must be a non-empty string")
        return text

    def embed_text(self, text: str) -> list[float]:
        """Embed one text without adding metadata or serializing its source record."""
        text = self._require_text(text)
        response = self._embed_request(text)
        return self._vectors_from_response(response, 1)[0].tolist()

    def embed_documents(
        self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> list[list[float]]:
        """Embed documents in moderate batches, preserving input order."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        checked = [self._require_text(text) for text in texts]
        if not checked:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(checked), batch_size):
            chunk = checked[start : start + batch_size]
            # Gemini Embedding 2 aggregates a list of raw parts. Wrapping each
            # text in a separate Content requests one vector per document.
            contents = [
                types.Content(parts=[types.Part.from_text(text=text)])
                for text in chunk
            ]
            response = self._embed_request(contents)
            vectors.extend(
                vector.tolist()
                for vector in self._vectors_from_response(response, len(chunk))
            )
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a query with the same model, dimension, and normalization policy."""
        return self.embed_text(query)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Compatibility helper returning an `(n, dimension)` float32 array."""
        return np.asarray(self.embed_documents(texts), dtype=np.float32)


# Backward-compatible import name for code that previously used Embedder.
Embedder = GeminiEmbedder


def _read_corpus(corpus_path: Path) -> list[dict]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Semantic corpus not found: {corpus_path}")
    rows = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Semantic corpus is empty: {corpus_path}")

    seen = set()
    for position, row in enumerate(rows):
        scene_id = row.get("scene_id")
        text = row.get("searchable_text")
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError(f"Corpus row {position} has an invalid scene_id")
        if scene_id in seen:
            raise ValueError(f"Duplicate scene_id in corpus: {scene_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Scene {scene_id} has empty searchable_text")
        seen.add(scene_id)
    return rows


def _load_embedding_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    if not set(EMBEDDING_COLUMNS).issubset(frame.columns):
        LOGGER.warning("Ignoring incompatible embedding cache at %s", path)
        return {}
    return {str(row["scene_id"]): row.to_dict() for _, row in frame.iterrows()}


def _cache_matches(
    record: dict,
    *,
    model: GeminiEmbedder,
    pipeline_version: str,
    text_hash: str,
) -> bool:
    try:
        return (
            record["embedding_model"] == model.model
            and int(record["embedding_dimension"]) == model.dimension
            and bool(record["normalized"]) == model.normalize
            and record["pipeline_version"] == pipeline_version
            and record["text_hash"] == text_hash
        )
    except (KeyError, TypeError, ValueError):
        return False


def _write_embedding_artifacts(
    records: list[dict], embeddings_path: Path, id_map_path: Path
) -> None:
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "scene_id": pa.array([r["scene_id"] for r in records], type=pa.string()),
            "embedding": pa.array(
                [r["embedding"] for r in records], type=pa.list_(pa.float32())
            ),
            "embedding_model": pa.array(
                [r["embedding_model"] for r in records], type=pa.string()
            ),
            "embedding_dimension": pa.array(
                [r["embedding_dimension"] for r in records], type=pa.int32()
            ),
            "normalized": pa.array(
                [r["normalized"] for r in records], type=pa.bool_()
            ),
            "pipeline_version": pa.array(
                [r["pipeline_version"] for r in records], type=pa.string()
            ),
            "text_hash": pa.array([r["text_hash"] for r in records], type=pa.string()),
        }
    )
    embedding_tmp = embeddings_path.with_suffix(".parquet.tmp")
    id_map_tmp = id_map_path.with_suffix(".parquet.tmp")
    pq.write_table(table, embedding_tmp)
    pd.DataFrame(
        {
            "faiss_id": np.arange(len(records), dtype=np.int64),
            "scene_id": [r["scene_id"] for r in records],
        }
    ).to_parquet(id_map_tmp, index=False)
    embedding_tmp.replace(embeddings_path)
    id_map_tmp.replace(id_map_path)


def embed_corpus(
    corpus_path: str | Path,
    output_dir: str | Path,
    embedder: GeminiEmbedder | None = None,
    *,
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pipeline_version: str = EMBEDDING_PIPELINE_VERSION,
) -> tuple[Path, Path]:
    """Embed only searchable_text and persist resumable Parquet artifacts."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    embeddings_path = output_dir / "embeddings.parquet"
    id_map_path = output_dir / "id_map.parquet"
    rows = _read_corpus(corpus_path)
    model = embedder or GeminiEmbedder()
    cache = {} if force else _load_embedding_cache(embeddings_path)

    records_by_scene: dict[str, dict] = {}
    pending: list[dict] = []
    for row in rows:
        scene_id = row["scene_id"]
        text = row["searchable_text"]
        digest = searchable_text_hash(text)
        cached = cache.get(scene_id)
        if cached and _cache_matches(
            cached,
            model=model,
            pipeline_version=pipeline_version,
            text_hash=digest,
        ):
            try:
                vector = _validated_vector(
                    cached["embedding"], model.dimension, model.normalize
                )
            except ValueError:
                pending.append({"scene_id": scene_id, "text": text, "text_hash": digest})
            else:
                records_by_scene[scene_id] = {
                    "scene_id": scene_id,
                    "embedding": vector,
                    "embedding_model": model.model,
                    "embedding_dimension": model.dimension,
                    "normalized": model.normalize,
                    "pipeline_version": pipeline_version,
                    "text_hash": digest,
                }
                continue
        else:
            pending.append({"scene_id": scene_id, "text": text, "text_hash": digest})

    LOGGER.info(
        "Embedding corpus scenes=%d cached=%d pending=%d model=%s dimension=%d",
        len(rows),
        len(records_by_scene),
        len(pending),
        model.model,
        model.dimension,
    )

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = model.embed_documents(
            [item["text"] for item in batch], batch_size=batch_size
        )
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Embedding batch count mismatch: expected {len(batch)}, got {len(vectors)}"
            )
        for item, values in zip(batch, vectors, strict=True):
            vector = _validated_vector(values, model.dimension, model.normalize)
            records_by_scene[item["scene_id"]] = {
                "scene_id": item["scene_id"],
                "embedding": vector,
                "embedding_model": model.model,
                "embedding_dimension": model.dimension,
                "normalized": model.normalize,
                "pipeline_version": pipeline_version,
                "text_hash": item["text_hash"],
            }
        completed = [
            records_by_scene[row["scene_id"]]
            for row in rows
            if row["scene_id"] in records_by_scene
        ]
        _write_embedding_artifacts(completed, embeddings_path, id_map_path)
        LOGGER.info(
            "Embedding progress completed=%d/%d",
            len(completed),
            len(rows),
        )

    final_records = [records_by_scene[row["scene_id"]] for row in rows]
    _write_embedding_artifacts(final_records, embeddings_path, id_map_path)
    LOGGER.info("Embedding artifacts written scenes=%d path=%s", len(rows), embeddings_path)
    return embeddings_path, id_map_path
