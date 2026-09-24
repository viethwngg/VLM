from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.retrieval.faiss_index import build_faiss
root = Path(__file__).resolve().parents[1]
build_faiss(root / "artifacts/embeddings/embeddings.parquet", root / "artifacts/embeddings/id_map.parquet", root / "artifacts/index")
