import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from pipeline.retrieval.schemas import SemanticScene
from pipeline.retrieval.semantic_pipeline import build_corpus, write_scene_metadata
from pipeline.retrieval.searchable_text import build_searchable_text


@pytest.fixture
def workspace_tmp_path():
    root = Path.cwd() / ".test-work" / uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def provenance():
    return {
        "vlm_model": "mock",
        "prompt_version": "road-vlm-v3",
        "taxonomy_version": "road-v2",
    }


def semantic_scene(**overrides):
    data = {
        "scene_id": "x",
        "scene": {
            "road_type": "urban_road",
            "traffic_state": "dense_traffic",
            "weather": "clear",
            "lighting": "daylight",
            "road_surface": "dry",
        },
        "ego": {"actions": ["decelerating"]},
        "agents": [],
        "events": [],
        "attention_events": [],
        "searchable_text": "",
        "provenance": provenance(),
    }
    data.update(overrides)
    return SemanticScene(**data)


def test_nested_taxonomy_normalization():
    scene = semantic_scene(agents=[{
        "type": "Pedestrian",
        "actions": ["Crossing", "invented"],
        "locations": ["Crosswalk", "invented"],
        "relations": ["Crossing Ego Path", "invented"],
    }])
    assert scene.agents[0].type == "pedestrian"
    assert scene.agents[0].actions == ["crossing"]
    assert scene.agents[0].locations == ["crosswalk"]
    assert scene.agents[0].relations == ["crossing_ego_path"]


def test_searchable_text_preserves_vlm_text():
    scene = semantic_scene(
        searchable_text="Pedestrian crossing the ego vehicle lane."
    )
    assert build_searchable_text(scene) == "Pedestrian crossing the ego vehicle lane."


def test_searchable_text_fallback_uses_nested_schema():
    scene = semantic_scene(
        agents=[{
            "type": "pedestrian",
            "actions": ["crossing"],
            "locations": ["crosswalk"],
            "relations": ["crossing_ego_path"],
        }],
        attention_events=["pedestrian_crossing"],
    )
    text = build_searchable_text(scene)
    assert "pedestrian crossing" in text.lower()
    assert "ego vehicle decelerating" in text.lower()


def test_json_serialization_matches_nested_contract():
    data = json.loads(semantic_scene().model_dump_json())
    assert data["scene_id"] == "x"
    assert data["scene"]["road_type"] == "urban_road"
    assert data["ego"]["actions"] == ["decelerating"]


def test_write_scene_metadata_rejects_empty_embedding_input(workspace_tmp_path):
    scene = semantic_scene(
        scene_id="empty",
        scene={
            "road_type": None,
            "traffic_state": None,
            "weather": None,
            "lighting": None,
            "road_surface": None,
        },
        ego={"actions": []},
        searchable_text="",
    )

    with pytest.raises(ValueError, match="no searchable_text or semantic fallback"):
        write_scene_metadata(scene, workspace_tmp_path)


def test_corpus_exports_only_current_v3_artifacts_for_model(workspace_tmp_path):
    current = semantic_scene(scene_id="current", searchable_text="Current V3 scene.")
    stale = semantic_scene(
        scene_id="stale",
        searchable_text="Stale V2 scene.",
        provenance={
            "vlm_model": "mock",
            "prompt_version": "road-vlm-v2",
            "taxonomy_version": "road-v1",
        },
    )
    other_model = semantic_scene(
        scene_id="other-model",
        searchable_text="Different model scene.",
        provenance={
            "vlm_model": "other",
            "prompt_version": "road-vlm-v3",
            "taxonomy_version": "road-v2",
        },
    )
    for scene in (current, stale, other_model):
        write_scene_metadata(scene, workspace_tmp_path)

    corpus = build_corpus(workspace_tmp_path, model="mock")
    rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]

    assert [row["scene_id"] for row in rows] == ["current"]
    assert rows[0]["searchable_text"] == "Current V3 scene."
