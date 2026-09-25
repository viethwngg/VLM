import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.retrieval.faiss_index import build_faiss
from pipeline.retrieval.settings import load_retrieval_config, resolve_embedding_settings


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build a validated FAISS IndexFlatIP")
    parser.add_argument("--embeddings", type=Path, default=root / "artifacts/embeddings/embeddings.parquet")
    parser.add_argument("--id-map", type=Path, default=root / "artifacts/embeddings/id_map.parquet")
    parser.add_argument("--output", type=Path, default=root / "artifacts/index")
    parser.add_argument("--config", type=Path, default=root / "config/retrieval.yaml")
    parser.add_argument("--model")
    parser.add_argument("--dimension", type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_retrieval_config(args.config)
    settings = resolve_embedding_settings(
        config, model=args.model, dimension=args.dimension
    )
    index_config = config.get("index", {})
    if index_config.get("type", "IndexFlatIP") != "IndexFlatIP":
        raise ValueError("Only IndexFlatIP is supported")
    index_dimension = int(index_config.get("dimension", settings["dimension"]))
    if index_dimension != settings["dimension"]:
        raise ValueError(
            "Index and embedding dimensions differ: "
            f"{index_dimension} != {settings['dimension']}"
        )
    output_dir = build_faiss(
        args.embeddings,
        args.id_map,
        args.output,
        embedding_provider=settings["provider"],
        embedding_model=settings["model"],
        dimension=settings["dimension"],
        normalize=settings["normalize"],
        index_version=str(index_config.get("version", "v1")),
    )
    logging.info("status=success index=%s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
