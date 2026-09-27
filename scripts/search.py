import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.retrieval.vector_search import search_faiss


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Search the RAV-21 Gemini/FAISS index")
    parser.add_argument(
        "query",
        nargs="?",
        help="Search text. If omitted, the command prompts for it interactively.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--index-dir", type=Path, default=root / "artifacts/index/v1")
    args = parser.parse_args()

    query = args.query
    if query is None:
        try:
            query = input("Nhập nội dung cần tìm: ").strip()
        except EOFError:
            parser.error("không đọc được nội dung tìm kiếm từ terminal")
    if not query:
        parser.error("nội dung tìm kiếm không được để trống")

    results = search_faiss(query, args.index_dir, top_k=args.top_k)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
