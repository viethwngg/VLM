import json
import shutil
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from google.genai import errors

from pipeline.retrieval.embedder import (
    EMBEDDING_PIPELINE_VERSION,
    GeminiEmbedder,
    embed_corpus,
)
from pipeline.retrieval.faiss_index import build_faiss
from pipeline.retrieval.vector_search import search_faiss


@pytest.fixture
def workspace_tmp_path():
    root = Path.cwd() / ".test-work" / uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


class FakeModels:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.calls = []
        self.side_effects = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.side_effects:
            effect = self.side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return effect
        contents = kwargs["contents"]
        texts = (
            [contents]
            if isinstance(contents, str)
            else [content.parts[0].text for content in contents]
        )
        embeddings = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            vector[sum(text.encode("utf-8")) % self.dimension] = 1.0
            embeddings.append(SimpleNamespace(values=vector.tolist()))
        return SimpleNamespace(embeddings=embeddings)


def make_embedder(models=None, **kwargs):
    models = models or FakeModels()
    client = SimpleNamespace(models=models)
    return GeminiEmbedder(
        client=client,
        model="gemini-embedding-2",
        dimension=768,
        max_retries=3,
        retry_delay=1,
        **kwargs,
    ), models


def write_corpus(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_embed_text_returns_768_float_values_and_sets_output_dimension():
    embedder, models = make_embedder()
    text = "A pedestrian crosses the road."
    vector = embedder.embed_text(text)
    assert len(vector) == 768
    assert models.calls[0]["contents"] == text
    assert models.calls[0]["model"] == "gemini-embedding-2"
    assert models.calls[0]["config"].output_dimensionality == 768
    assert np.isclose(np.linalg.norm(np.asarray(vector, dtype=np.float32)), 1.0)


def test_wrong_returned_dimension_fails_clearly():
    embedder, _ = make_embedder(FakeModels(dimension=767))
    with pytest.raises(ValueError, match="expected 768.*767"):
        embedder.embed_text("valid text")


def test_missing_api_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiEmbedder()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_invalid_vector_values_are_rejected(bad_value):
    response = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[bad_value] + [0.0] * 767)]
    )
    models = FakeModels()
    models.side_effects = [response]
    embedder, _ = make_embedder(models)
    with pytest.raises(ValueError, match="NaN or infinite"):
        embedder.embed_text("valid text")


def test_transient_api_error_is_retried(monkeypatch):
    models = FakeModels()
    success = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[1.0] + [0.0] * 767)]
    )
    models.side_effects = [
        errors.ServerError(503, {"error": {"message": "busy"}}),
        success,
    ]
    embedder, _ = make_embedder(models)
    sleep = Mock()
    monkeypatch.setattr("pipeline.retrieval.embedder.time.sleep", sleep)
    assert len(embedder.embed_text("valid text")) == 768
    assert len(models.calls) == 2
    sleep.assert_called_once()


def test_embed_corpus_sends_only_searchable_text_and_persists_contract(workspace_tmp_path):
    corpus = workspace_tmp_path / "semantic_corpus.jsonl"
    output = workspace_tmp_path / "embeddings"
    rows = [
        {
            "scene_id": "scene-1",
            "searchable_text": "pedestrian crossing",
            "frame_uri": "must-not-be-embedded.mp4",
            "provenance": {"model": "must-not-be-embedded"},
        },
        {
            "scene_id": "scene-2",
            "searchable_text": "car turning right",
            "timestamp_s": 12.5,
        },
    ]
    write_corpus(corpus, rows)
    embedder, models = make_embedder()

    embeddings_path, id_map_path = embed_corpus(
        corpus, output, embedder, batch_size=2
    )

    sent_texts = [content.parts[0].text for content in models.calls[0]["contents"]]
    assert sent_texts == [row["searchable_text"] for row in rows]
    frame = pd.read_parquet(embeddings_path)
    assert set(
        [
            "scene_id",
            "embedding",
            "embedding_model",
            "embedding_dimension",
            "normalized",
            "pipeline_version",
            "text_hash",
        ]
    ).issubset(frame.columns)
    vectors = np.stack(frame["embedding"].to_list())
    assert vectors.shape == (2, 768)
    assert vectors.dtype == np.float32
    assert frame["embedding_model"].tolist() == ["gemini-embedding-2"] * 2
    assert frame["embedding_dimension"].tolist() == [768, 768]
    assert frame["pipeline_version"].tolist() == [EMBEDDING_PIPELINE_VERSION] * 2
    assert pd.read_parquet(id_map_path).to_dict("records") == [
        {"faiss_id": 0, "scene_id": "scene-1"},
        {"faiss_id": 1, "scene_id": "scene-2"},
    ]


def test_embedding_resume_skips_unchanged_and_reembeds_changed_text(workspace_tmp_path):
    corpus = workspace_tmp_path / "semantic_corpus.jsonl"
    output = workspace_tmp_path / "embeddings"
    rows = [
        {"scene_id": "scene-1", "searchable_text": "first text"},
        {"scene_id": "scene-2", "searchable_text": "second text"},
    ]
    write_corpus(corpus, rows)
    embedder, models = make_embedder()
    embed_corpus(corpus, output, embedder, batch_size=2)
    assert len(models.calls) == 1

    embed_corpus(corpus, output, embedder, batch_size=2)
    assert len(models.calls) == 1

    rows[1]["searchable_text"] = "second text changed"
    write_corpus(corpus, rows)
    embed_corpus(corpus, output, embedder, batch_size=2)
    assert len(models.calls) == 2
    sent = [content.parts[0].text for content in models.calls[-1]["contents"]]
    assert sent == ["second text changed"]


class StaticEmbedder:
    provider = "gemini"
    model = "gemini-embedding-2"
    dimension = 768
    normalize = True

    def embed_documents(self, texts, batch_size=16):
        vectors = []
        for index, _ in enumerate(texts):
            vector = np.zeros(self.dimension, dtype=np.float32)
            vector[index] = 1.0
            vectors.append(vector.tolist())
        return vectors

    def embed_query(self, query):
        vector = np.zeros(self.dimension, dtype=np.float32)
        vector[1] = 1.0
        return vector.tolist()


def test_faiss_manifest_dimension_and_scene_mapping(workspace_tmp_path):
    corpus = workspace_tmp_path / "semantic_corpus.jsonl"
    embedding_dir = workspace_tmp_path / "embeddings"
    index_root = workspace_tmp_path / "index"
    write_corpus(
        corpus,
        [
            {"scene_id": "scene-a", "searchable_text": "first"},
            {"scene_id": "scene-b", "searchable_text": "second"},
        ],
    )
    embeddings_path, id_map_path = embed_corpus(
        corpus, embedding_dir, StaticEmbedder(), batch_size=2
    )
    index_dir = build_faiss(embeddings_path, id_map_path, index_root)

    import faiss

    index = faiss.read_index(str(index_dir / "index.faiss"))
    assert index.d == 768
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["embedding_provider"] == "gemini"
    assert manifest["embedding_model"] == "gemini-embedding-2"
    assert manifest["embedding_dimension"] == 768
    assert manifest["normalized"] is True

    results = search_faiss(
        "find second scene", index_dir, top_k=2, embedder=StaticEmbedder()
    )
    assert results[0]["faiss_id"] == 1
    assert results[0]["scene_id"] == "scene-b"


def test_query_embedder_must_match_corpus_dimension(workspace_tmp_path):
    index_dir = workspace_tmp_path / "index" / "v1"
    index_dir.mkdir(parents=True)
    (index_dir / "manifest.json").write_text(
        json.dumps(
            {
                "index_type": "IndexFlatIP",
                "embedding_provider": "gemini",
                "embedding_model": "gemini-embedding-2",
                "embedding_dimension": 768,
                "normalized": True,
            }
        ),
        encoding="utf-8",
    )
    wrong = SimpleNamespace(
        model="gemini-embedding-2", dimension=384, normalize=True
    )
    with pytest.raises(ValueError, match="does not match index manifest"):
        search_faiss("query", index_dir, embedder=wrong)
