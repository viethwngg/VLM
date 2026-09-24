from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.retrieval.embedder import embed_corpus
root = Path(__file__).resolve().parents[1]
embed_corpus(root / "artifacts/semantic/semantic_corpus.jsonl", root / "artifacts/embeddings")
