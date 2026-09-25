"""Live Gemini Embedding 2 smoke test. Never prints the API key."""

import json
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.retrieval.embedder import GeminiEmbedder
from pipeline.retrieval.settings import load_retrieval_config, resolve_embedding_settings


SMOKE_TEXT = "A pedestrian crosses a road while a car turns right at an intersection."
SMOKE_QUERY = "pedestrian crossing near turning vehicle"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config = load_retrieval_config(root / "config/retrieval.yaml")
    settings = resolve_embedding_settings(config)
    embedder = GeminiEmbedder(
        model=settings["model"],
        dimension=settings["dimension"],
        normalize=settings["normalize"],
    )

    vector = np.asarray(embedder.embed_text(SMOKE_TEXT), dtype=np.float32)
    print(f"Model: {embedder.model}")
    print(f"Dimension: {len(vector)}")
    print(f"Vector norm: {np.linalg.norm(vector):.6f}")
    print("Success: true")

    corpus_path = root / "artifacts/semantic/semantic_corpus.jsonl"
    if not corpus_path.exists():
        return 0
    rows = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:5]
    rows = [row for row in rows if str(row.get("searchable_text", "")).strip()]
    if not rows:
        return 0

    vectors = np.asarray(
        embedder.embed_documents([row["searchable_text"] for row in rows]),
        dtype=np.float32,
    )
    query = np.asarray([embedder.embed_query(SMOKE_QUERY)], dtype=np.float32)
    if vectors.shape != (len(rows), embedder.dimension):
        raise RuntimeError(f"Unexpected document matrix shape: {vectors.shape}")
    if query.shape != (1, embedder.dimension):
        raise RuntimeError(f"Unexpected query shape: {query.shape}")
    if embedder.normalize:
        faiss.normalize_L2(vectors)
        faiss.normalize_L2(query)

    index = faiss.IndexFlatIP(embedder.dimension)
    index.add(vectors)
    scores, ids = index.search(query, min(3, len(rows)))
    results = [
        {"scene_id": rows[int(index_id)]["scene_id"], "score": float(score)}
        for score, index_id in zip(scores[0], ids[0], strict=True)
        if index_id >= 0
    ]
    print("Search results:")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
