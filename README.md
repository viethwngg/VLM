# RAV-21 semantic retrieval MVP

Activate `.venv`, install dependencies with `pip install -r requirements.txt`, and copy `.env.example` to `.env`. Set `GEMINI_API_KEY` and optionally `GEMINI_MODEL` (default `gemini-2.5-flash`). Secrets stay local and are ignored by Git.

Run one scene or all scenes from the repository root:

```powershell
python scripts/run_semantic_pipeline.py --scene-id scene-0061
python scripts/run_semantic_pipeline.py --all --resume
python scripts/build_embeddings.py
python scripts/build_faiss_index.py
pytest
```

The canonical per-scene record is written to `artifacts/semantic/<scene_id>/semantic_metadata.json`. The batch corpus is `artifacts/semantic/semantic_corpus.jsonl`; vectors and mappings are in `artifacts/embeddings/`, and the `IndexFlatIP` index is in `artifacts/index/v1/`. The taxonomy is versioned as `road-v1`, and the prompt as `road-vlm-v1`. The local hash embedder makes the pipeline testable without an embedding API; replace it through `EMBEDDING_PROVIDER`/`EMBEDDING_MODEL` without changing the schema.
