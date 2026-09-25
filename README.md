# RAV-21 semantic retrieval MVP

Activate `.venv`, install dependencies with `pip install -r requirements.txt`, and copy `.env.example` to `.env`. Set `GEMINI_API_KEY` and optionally `GEMINI_MODEL` (default `gemini-3.8-flash`). If an existing `.env` or shell sets `GEMINI_MODEL=gemini-2.5-flash` and the API reports it unavailable, update that setting to `gemini-3.8-flash`. Secrets stay local and are ignored by Git.

Run one scene or all scenes from the repository root:

Temporary API failures are retried up to `GEMINI_MAX_ATTEMPTS` total attempts (default 5), with exponential backoff starting at `GEMINI_RETRY_DELAY_SECONDS` (default 10 seconds), plus up to 20% jitter and a 60-second cap per wait. SDK retries and request time are additional. When a model returns 404 or 503, the pipeline rotates through the video-capable models in `GEMINI_FALLBACK_MODELS` (default `gemini-3.7-flash,gemini-3.5-flash-lite`) before waiting and trying again. Set this variable to an empty value to disable fallback. Generation retries reuse the uploaded video within the current run, and provenance records the model that produced the result. Permanent API errors such as 400/401/403 fail immediately. These settings can be set in `.env` or the shell.

```powershell
python scripts/run_semantic_pipeline.py --scene-id scene-0061
python scripts/run_semantic_pipeline.py --all --resume
python scripts/build_embeddings.py
python scripts/build_faiss_index.py
pytest
```

The canonical per-scene record is written to `artifacts/semantic/<scene_id>/semantic_metadata.json`. The batch corpus is `artifacts/semantic/semantic_corpus.jsonl`; vectors and mappings are in `artifacts/embeddings/`, and the `IndexFlatIP` index is in `artifacts/index/v1/`. The taxonomy is versioned as `road-v1`, and the structured-output prompt as `road-vlm-v2`. The local hash embedder makes the pipeline testable without an embedding API; replace it through `EMBEDDING_PROVIDER`/`EMBEDDING_MODEL` without changing the schema.
