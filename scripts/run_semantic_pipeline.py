import argparse, json, logging, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from pipeline.retrieval.semantic_pipeline import process_scene, build_corpus

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scene-id"); ap.add_argument("--all", action="store_true"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--force", action="store_true"); args = ap.parse_args()
    logging.basicConfig(level=logging.INFO); root = Path(__file__).resolve().parents[1]; cfg = yaml.safe_load((root / "config/retrieval.yaml").read_text())
    data_root = root / cfg["semantic"]["data_root"]; output = root / cfg["semantic"]["output_root"]
    scenes = [args.scene_id] if args.scene_id else ([p.name for p in sorted(data_root.glob("scene-*")) if p.is_dir()] if args.all else [])
    if not scenes: ap.error("provide --scene-id or --all")
    for scene_id in scenes:
        started = time.perf_counter()
        videos = list((data_root / scene_id).glob("*.mp4"));
        if not videos:
            logging.warning("scene_id=%s status=skipped reason=no_video elapsed_s=%.3f", scene_id, time.perf_counter() - started)
            continue
        try:
            artifact = process_scene(scene_id, videos[0], output, cfg["semantic"]["camera"], cfg["semantic"]["num_frames"], args.force)
            logging.info("scene_id=%s status=success elapsed_s=%.3f artifact=%s", scene_id, time.perf_counter() - started, artifact)
        except Exception:
            logging.exception("scene_id=%s status=failed elapsed_s=%.3f", scene_id, time.perf_counter() - started)
            raise
    build_corpus(output)
if __name__ == "__main__": main()
