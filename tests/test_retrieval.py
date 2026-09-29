import json

from pipeline.retrieval.schemas import SemanticScene
from pipeline.retrieval.searchable_text import build_searchable_text


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
