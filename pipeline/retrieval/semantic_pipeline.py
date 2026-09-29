"""Validated semantic artifact persistence and corpus export."""
import json, logging
from pathlib import Path
from .prompts import PROMPT_VERSION
from .schemas import SemanticScene
from .searchable_text import build_searchable_text
from .taxonomy import TAXONOMY_VERSION
from .vlm_contract import PIPELINE_VERSION

LOGGER = logging.getLogger(__name__)


def is_current_scene_artifact(
    path: str | Path,
    scene_id: str,
    model: str | None = None,
) -> bool:
    """Accept resume output only when schema and semantic versions match."""
    try:
        scene = SemanticScene.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        scene.scene_id == scene_id
        and scene.provenance.prompt_version == PROMPT_VERSION
        and scene.provenance.taxonomy_version == TAXONOMY_VERSION
        and scene.provenance.pipeline_version == PIPELINE_VERSION
        and (model is None or scene.provenance.vlm_model == model)
    )


def write_scene_metadata(scene, output_root: str | Path) -> Path:
    """Persist validated metadata in the canonical artifact location."""
    out = Path(output_root) / scene.scene_id / "semantic_metadata.json"
    scene.searchable_text = build_searchable_text(scene)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(scene.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out

def build_corpus(semantic_root: str | Path) -> Path:
    root = Path(semantic_root)
    corpus = root / "semantic_corpus.jsonl"
    rows = []
    for path in sorted(root.glob("*/semantic_metadata.json")):
        try:
            scene = SemanticScene.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            LOGGER.warning("Skipping incompatible semantic artifact %s: %s", path, exc)
            continue
        data = scene.model_dump(mode="json")
        rows.append({
            key: data.get(key)
            for key in (
                "scene_id", "searchable_text", "scene", "ego", "agents",
                "events", "attention_events",
            )
        })
    corpus.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return corpus
