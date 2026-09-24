"""Scene-level semantic extraction and JSONL export."""
import json, logging
from pathlib import Path
from .frame_sampler import sample_video
from .gemini_vlm import GeminiVLM
from .searchable_text import build_searchable_text

LOGGER = logging.getLogger(__name__)

def process_scene(scene_id: str, video_path: str | Path, output_root: str | Path, camera="CAM_FRONT", num_frames=8, force=False, vlm=None) -> Path:
    out = Path(output_root) / scene_id / "semantic_metadata.json"
    if out.exists() and not force: return out
    frames = sample_video(video_path, camera, num_frames)
    evidence = [f.__dict__ for f in frames]
    scene = (vlm or GeminiVLM()).analyze(scene_id, frames, evidence)
    scene.searchable_text = build_searchable_text(scene)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scene.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out

def build_corpus(semantic_root: str | Path) -> Path:
    root = Path(semantic_root); corpus = root / "semantic_corpus.jsonl"; rows = []
    for path in sorted(root.glob("*/semantic_metadata.json")):
        data = json.loads(path.read_text(encoding="utf-8")); rows.append({k: data.get(k) for k in ("scene_id", "searchable_text", "agents", "actions", "locations", "events", "road_context", "weather", "time_of_day")})
    corpus.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return corpus
