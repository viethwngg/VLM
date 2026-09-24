"""FAISS IndexFlatIP builder."""
import json
from datetime import datetime, timezone
from pathlib import Path
import faiss, numpy as np, pandas as pd
from .taxonomy import TAXONOMY_VERSION
from .prompts import PROMPT_VERSION

def build_faiss(embeddings_path, id_map_path, output_root, embedding_provider="local", embedding_model="hash-384", pipeline_version="semantic-pipeline-v1"):
    vectors = pd.read_parquet(embeddings_path).to_numpy(dtype="float32"); ids = pd.read_parquet(id_map_path)
    index = faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors)
    out = Path(output_root) / "v1"; out.mkdir(parents=True, exist_ok=True); faiss.write_index(index, str(out / "index.faiss")); ids.to_parquet(out / "id_map.parquet", index=False)
    manifest = {"index_version": "v1", "embedding_provider": embedding_provider, "embedding_model": embedding_model, "embedding_dimension": int(vectors.shape[1]), "normalized": True, "index_type": "IndexFlatIP", "number_of_scenes": int(len(ids)), "created_at": datetime.now(timezone.utc).isoformat(), "taxonomy_version": TAXONOMY_VERSION, "prompt_version": PROMPT_VERSION, "pipeline_version": pipeline_version}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8"); return out
