import argparse
import logging
import time
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from pipeline.retrieval.semantic_pipeline import build_corpus, process_scene


LOGGER = logging.getLogger(__name__)
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def format_duration(elapsed_s: float) -> str:
    """Format elapsed seconds as HH:MM:SS.mmm for human-readable logs."""
    total_ms = round(elapsed_s * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _path_id(path: Path) -> str:
    """Convert a relative path to a stable artifact-directory name."""
    return "__".join(path.parts)


def discover_video_jobs(
    data_root: str | Path, scene_id: str | None = None
) -> list[tuple[str, Path]]:
    """Find every video below data_root and assign a unique scene ID to it.

    A directory containing one video keeps its directory name as the scene ID
    (for example ``scene-0061/cam_front.mp4`` becomes ``scene-0061``). When a
    directory contains multiple videos, the video stem is appended so that no
    semantic artifact can overwrite or skip another video.
    """
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


def process_scene_with_timing(
    scene_id, video_path, output_root, camera, num_frames, force=False
):
    """Process one scene and log its complete wall-clock processing time."""
    started = time.perf_counter()
    LOGGER.info("scene_id=%s status=started video=%s", scene_id, video_path)
    try:
        artifact = process_scene(
            scene_id, video_path, output_root, camera, num_frames, force
        )
    except Exception as exc:
        elapsed_s = time.perf_counter() - started
        LOGGER.error(
            "scene_id=%s status=failed elapsed_s=%.3f duration=%s error=%s",
            scene_id,
            elapsed_s,
            format_duration(elapsed_s),
            exc,
        )
        raise

    elapsed_s = time.perf_counter() - started
    LOGGER.info(
        "scene_id=%s status=success elapsed_s=%.3f duration=%s artifact=%s",
        scene_id,
        elapsed_s,
        format_duration(elapsed_s),
        artifact,
    )
    return artifact


def log_total_duration(
    started_at: float,
    total: int,
    succeeded: int,
    failed: int,
    corpus_path: str | Path,
) -> float:
    """Log the total wall-clock time for a complete VLM batch."""
    elapsed_s = time.perf_counter() - started_at
    LOGGER.info(
        "pipeline_complete total=%d succeeded=%d failed=%d "
        "elapsed_s=%.3f duration=%s corpus=%s",
        total,
        succeeded,
        failed,
        elapsed_s,
        format_duration(elapsed_s),
        corpus_path,
    )
    return elapsed_s


def main():
    parser = argparse.ArgumentParser(
        description="Run Gemini VLM over every video under the configured data directory."
    )
    parser.add_argument(
        "--scene-id",
        help="Process only a matching scene directory, generated scene ID, or video stem.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all videos (this is also the default when --scene-id is omitted).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config/retrieval.yaml").read_text())
    data_root = root / cfg["semantic"]["data_root"]
    output = root / cfg["semantic"]["output_root"]
    jobs = discover_video_jobs(data_root, args.scene_id)

    if not jobs:
        target = f" matching {args.scene_id!r}" if args.scene_id else ""
        parser.error(f"no videos found{target} under {data_root}")

    LOGGER.info("discovered_videos=%d data_root=%s", len(jobs), data_root)
    pipeline_started = time.perf_counter()
    succeeded = 0
    failed = 0
    for video_scene_id, video_path in jobs:
        try:
            process_scene_with_timing(
                video_scene_id,
                video_path,
                output,
                cfg["semantic"]["camera"],
                cfg["semantic"]["num_frames"],
                args.force,
            )
            succeeded += 1
        except Exception:
            failed += 1

    build_corpus(output)
    log_total_duration(
        pipeline_started,
        len(jobs),
        succeeded,
        failed,
        output / "semantic_corpus.jsonl",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
