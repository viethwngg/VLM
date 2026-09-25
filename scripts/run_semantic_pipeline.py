import argparse, logging, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from pipeline.retrieval.semantic_pipeline import process_scene, build_corpus

LOGGER = logging.getLogger(__name__)


def format_duration(elapsed_s: float) -> str:
    """Format elapsed seconds as HH:MM:SS.mmm for human-readable logs."""
    total_ms = round(elapsed_s * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def process_scene_with_timing(scene_id, video_path, output_root, camera, num_frames, force=False):
    """Process one scene and log its complete wall-clock processing time."""
    started = time.perf_counter()
    LOGGER.info("scene_id=%s status=started", scene_id)
    try:
        artifact = process_scene(
            scene_id, video_path, output_root, camera, num_frames, force
        )
    except Exception as exc:
        elapsed_s = time.perf_counter() - started
        LOGGER.error(
            "scene_id=%s status=failed elapsed_s=%.3f duration=%s error=%s",
            scene_id, elapsed_s, format_duration(elapsed_s), exc,
        )
        raise

    elapsed_s = time.perf_counter() - started
    LOGGER.info(
        "scene_id=%s status=success elapsed_s=%.3f duration=%s artifact=%s",
        scene_id, elapsed_s, format_duration(elapsed_s), artifact,
    )
    return artifact


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scene-id"); ap.add_argument("--all", action="store_true"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--force", action="store_true"); args = ap.parse_args()
    logging.basicConfig(level=logging.INFO); root = Path(__file__).resolve().parents[1]; cfg = yaml.safe_load((root / "config/retrieval.yaml").read_text())
    data_root = root / cfg["semantic"]["data_root"]; output = root / cfg["semantic"]["output_root"]
    scenes = [args.scene_id] if args.scene_id else ([p.name for p in sorted(data_root.glob("scene-*")) if p.is_dir()] if args.all else [])
    if not scenes: ap.error("provide --scene-id or --all")
    for scene_id in scenes:
        videos = list((data_root / scene_id).glob("*.mp4"));
        if not videos:
            LOGGER.warning("scene_id=%s status=skipped reason=no_video", scene_id)
            continue
        try:
            process_scene_with_timing(
                scene_id, videos[0], output,
                cfg["semantic"]["camera"], cfg["semantic"]["num_frames"], args.force,
            )
        except Exception:
            return 1
    build_corpus(output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
