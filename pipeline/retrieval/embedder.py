"""Configurable embedding adapter and Parquet persistence."""
import hashlib, json, os
from pathlib import Path
import numpy as np

class Embedder:
    def __init__(self, provider=None, model=None, dimension=None, normalize=True):
        self.provider = provider or os.getenv("EMBEDDING_PROVIDER", "local")
        self.model = model or os.getenv("EMBEDDING_MODEL", "hash-384")
        self.dimension = dimension or int(os.getenv("EMBEDDING_DIMENSION", "384"))
        self.normalize = normalize
    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode()).digest(); idx = int.from_bytes(digest[:4], "big") % self.dimension
                vectors[row, idx] += 1.0 if digest[4] % 2 else -1.0
        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            zero_rows = norms[:, 0] == 0
            vectors[zero_rows, 0] = 1.0
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
        return vectors

def embed_corpus(corpus_path: str | Path, output_dir: str | Path, embedder=None):
    import pandas as pd
    rows = [json.loads(line) for line in Path(corpus_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    model = embedder or Embedder(); vectors = model.encode([r["searchable_text"] for r in rows]); out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(vectors).to_parquet(out / "embeddings.parquet", index=False)
    pd.DataFrame({"scene_id": [r["scene_id"] for r in rows], "embedding_model": model.model, "dimension": model.dimension, "normalized": model.normalize, "pipeline_version": "semantic-pipeline-v1"}).to_parquet(out / "id_map.parquet", index=False)
    return out / "embeddings.parquet", out / "id_map.parquet"
