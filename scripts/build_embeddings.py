import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.retrieval.embedder import DEFAULT_BATCH_SIZE, GeminiEmbedder, embed_corpus
from pipeline.retrieval.settings import load_retrieval_config, resolve_embedding_settings


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build resumable Gemini document embeddings")
    parser.add_argument("--corpus", type=Path, default=root / "artifacts/semantic/semantic_corpus.jsonl")
    parser.add_argument("--output", type=Path, default=root / "artifacts/embeddings")
    parser.add_argument("--config", type=Path, default=root / "config/retrieval.yaml")
    parser.add_argument("--model")
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("EMBEDDING_BATCH_SIZE", DEFAULT_BATCH_SIZE)))
    parser.add_argument("--force", action="store_true")
    normalization = parser.add_mutually_exclusive_group()
    normalization.add_argument("--normalize", dest="normalize", action="store_true")
    normalization.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_retrieval_config(args.config)
    settings = resolve_embedding_settings(
        config, model=args.model, dimension=args.dimension, normalize=args.normalize
    )
    embedder = GeminiEmbedder(
        model=settings["model"],
        dimension=settings["dimension"],
        normalize=settings["normalize"],
    )
    embeddings_path, id_map_path = embed_corpus(
        args.corpus,
        args.output,
        embedder,
        force=args.force,
        batch_size=args.batch_size,
    )
    logging.info("status=success embeddings=%s id_map=%s", embeddings_path, id_map_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
