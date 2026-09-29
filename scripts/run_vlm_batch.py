"""Command-line workflow for file-based Gemini VLM batch processing."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from pipeline.retrieval.gemini_batch import GeminiBatchPipeline
from scripts.run_semantic_pipeline import discover_video_jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and manage resumable Gemini VLM Batch API jobs."
    )
    parser.add_argument(
        "command",
        choices=("prepare", "submit", "status", "download", "retry-failed", "run"),
    )
    parser.add_argument("--limit", type=int, help="Prepare at most N uncompleted videos")
    parser.add_argument("--scene-id", help="Filter using the existing scene/video ID rules")
    parser.add_argument("--force", action="store_true", help="Include scenes with valid existing output")
    parser.add_argument("--batch-size", type=int, help="Requests per batch split")
    parser.add_argument("--batch-dir", type=Path, help="Batch artifact directory")
    parser.add_argument("--model", help="Gemini model; defaults to GEMINI_MODEL/current sync model")
    return parser


def _pipeline(args, root: Path, config: dict) -> GeminiBatchPipeline:
    semantic = config["semantic"]
    batch_dir = args.batch_dir or Path(os.getenv("VLM_BATCH_DIR", "artifacts/vlm_batch"))
    if not batch_dir.is_absolute():
        batch_dir = root / batch_dir
    return GeminiBatchPipeline(
        data_root=root / semantic["data_root"],
        output_root=root / semantic["output_root"],
        batch_root=batch_dir,
        camera=semantic["camera"],
        num_frames=semantic["num_frames"],
        model=args.model,
        batch_size=args.batch_size,
    )


def _print_prepare(summary: dict) -> None:
    labels = (
        ("Total videos discovered", "total_videos_discovered"),
        ("Already completed", "already_completed"),
        ("Need processing", "need_processing"),
        ("Upload successful", "upload_successful"),
        ("Upload failed", "upload_failed"),
        ("Batch requests generated", "batch_requests_generated"),
        ("Batch files generated", "batch_count"),
    )
    for label, key in labels:
        print(f"{label}: {summary[key]}")


def _print_jobs(jobs: list[dict]) -> None:
    if not jobs:
        print("No batch jobs.")
        return
    for job in jobs:
        print(f"Batch: {job.get('job_name', job.get('batch_id'))}")
        print(f"State: {job.get('status')}")
        print(f"Requests: {job.get('request_count', 0)}")
        print(f"Succeeded: {job.get('successful')}")
        print(f"Failed: {job.get('failed')}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config/retrieval.yaml").read_text(encoding="utf-8"))
    pipeline = _pipeline(args, root, config)

    if args.command in {"prepare", "run"}:
        jobs = discover_video_jobs(pipeline.data_root, args.scene_id)
        if not jobs:
            parser.error(f"no matching videos found under {pipeline.data_root}")
        summary = pipeline.prepare(jobs, limit=args.limit, force=args.force)
        _print_prepare(summary)
        if args.command == "prepare":
            return 1 if summary["upload_failed"] else 0
        submitted = pipeline.submit()
        _print_jobs(submitted)
        return 1 if summary["upload_failed"] else 0

    if args.command == "submit":
        _print_jobs(pipeline.submit())
    elif args.command == "status":
        _print_jobs(pipeline.status())
    elif args.command == "download":
        summary = pipeline.download()
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        return 1 if summary["results_failed"] else 0
    elif args.command == "retry-failed":
        _print_jobs(pipeline.retry_failed())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

