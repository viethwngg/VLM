"""Primary CLI for the resumable Gemini Batch-only semantic pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from pipeline.retrieval.gemini_batch import GeminiBatchPipeline
from pipeline.retrieval.prompts import PROMPT_VERSION


LOGGER = logging.getLogger(__name__)
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
COMMANDS = (
    "prepare",
    "submit",
    "status",
    "collect",
    "retry",
    "run",
    # Backward-compatible aliases from run_vlm_batch.py.
    "download",
    "retry-failed",
)


def _path_id(path: Path) -> str:
    """Convert a relative path to a stable artifact-directory name."""
    return "__".join(path.parts)


def discover_video_jobs(
    data_root: str | Path, scene_id: str | None = None
) -> list[tuple[str, Path]]:
    """Find every video below data_root and assign a unique scene ID to it."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    videos = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    videos_per_directory = Counter(path.parent for path in videos)
    jobs: list[tuple[str, Path]] = []

    for video in videos:
        relative = video.relative_to(root)
        if relative.parent == Path("."):
            video_scene_id = video.stem
        else:
            parent_id = _path_id(relative.parent)
            video_scene_id = (
                parent_id
                if videos_per_directory[video.parent] == 1
                else f"{parent_id}__{video.stem}"
            )

        if scene_id is None or scene_id in {
            video_scene_id,
            video.parent.name,
            video.stem,
        }:
            jobs.append((video_scene_id, video))

    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and manage resumable Gemini Batch VLM V3 jobs."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=COMMANDS,
        help="Batch lifecycle stage (default: run, which prepares and submits).",
    )
    parser.add_argument("--limit", type=int, help="Prepare at most N uncompleted videos")
    parser.add_argument(
        "--scene-id", help="Filter by scene directory, generated scene ID, or video stem"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all videos (default when --scene-id is omitted)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Compatibility flag; Batch lifecycle commands are always resumable",
    )
    parser.add_argument(
        "--force", action="store_true", help="Include valid current V3 outputs"
    )
    parser.add_argument("--batch-size", type=int, help="Requests per batch split")
    parser.add_argument("--batch-dir", type=Path, help="Batch artifact directory")
    parser.add_argument(
        "--model", help="Gemini model; defaults to GEMINI_MODEL/current configured model"
    )
    return parser


def build_pipeline(args, root: Path, config: dict) -> GeminiBatchPipeline:
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
    config = yaml.safe_load(
        (root / "config/retrieval.yaml").read_text(encoding="utf-8")
    )
    pipeline = build_pipeline(args, root, config)
    command = {
        "download": "collect",
        "retry-failed": "retry",
    }.get(args.command, args.command)

    LOGGER.info("[VLM] prompt version: %s", PROMPT_VERSION)
    LOGGER.info("[VLM] model: %s", pipeline.model)
    LOGGER.info("[VLM] execution mode: Gemini Batch API")

    if command in {"prepare", "run"}:
        jobs = discover_video_jobs(pipeline.data_root, args.scene_id)
        if not jobs:
            parser.error(f"no matching videos found under {pipeline.data_root}")
        LOGGER.info("[PREPARE] scenes/videos discovered: %d", len(jobs))
        summary = pipeline.prepare(jobs, limit=args.limit, force=args.force)
        _print_prepare(summary)
        if command == "prepare":
            return 1 if summary["upload_failed"] else 0
        submitted = pipeline.submit()
        _print_jobs(submitted)
        return 1 if summary["upload_failed"] else 0

    if command == "submit":
        _print_jobs(pipeline.submit())
    elif command == "status":
        _print_jobs(pipeline.status())
    elif command == "collect":
        summary = pipeline.collect()
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        return 1 if summary["results_failed"] else 0
    elif command == "retry":
        _print_jobs(pipeline.retry_failed())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
