import json
from pathlib import Path

import pytest

from pipeline.retrieval.gemini_vlm import GeminiVLM
from pipeline.retrieval.prompts import PROMPT_VERSION
from pipeline.retrieval.taxonomy import TAXONOMY_VERSION
from pipeline.retrieval.vlm_contract import parse_scene_payload, response_text


VALID_OUTPUT = {
    "scene": {
        "road_type": "urban_road",
        "traffic_state": "moderate_traffic",
        "weather": "clear",
        "lighting": "daylight",
        "road_surface": "dry",
    },
    "ego": {"actions": ["decelerating"]},
    "agents": [],
    "events": [],
    "attention_events": [],
    "searchable_text": "Urban road with moderate traffic.",
}


def test_v3_payload_validation_and_provenance():
    scene = parse_scene_payload(
        json.dumps(VALID_OUTPUT),
        scene_id="scene-0061",
        evidence=[],
        model="gemini-test",
    )
    assert scene.scene_id == "scene-0061"
    assert scene.provenance.prompt_version == PROMPT_VERSION
    assert scene.provenance.taxonomy_version == TAXONOMY_VERSION
    assert scene.searchable_text == "Urban road with moderate traffic."


def test_nested_taxonomy_labels_are_normalized():
    payload = dict(VALID_OUTPUT)
    payload["agents"] = [{
        "type": "Pedestrian",
        "actions": ["Crossing", "invented"],
        "locations": ["Crosswalk"],
        "relations": ["Crossing Ego Path"],
    }]
    payload["events"] = [{
        "type": "Pedestrian Crossing",
        "participants": ["Pedestrian"],
        "location": "Crosswalk",
        "temporal_transition": "Roadside To Crossing",
    }]
    scene = parse_scene_payload(
        payload, scene_id="scene-0061", evidence=[], model="gemini-test"
    )
    assert scene.agents[0].type == "pedestrian"
    assert scene.agents[0].actions == ["crossing"]
    assert scene.agents[0].relations == ["crossing_ego_path"]
    assert scene.events[0].type == "pedestrian_crossing"
    assert scene.events[0].temporal_transition == "roadside_to_crossing"


@pytest.mark.parametrize("payload", ["", "not json", "{}"])
def test_invalid_v3_output_is_rejected(payload):
    with pytest.raises(ValueError):
        parse_scene_payload(
            payload, scene_id="scene-0061", evidence=[], model="gemini-test"
        )


def test_response_text_extracts_batch_candidates_and_detects_empty():
    response = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}
    assert response_text(response) == "{}"
    assert response_text({}) == ""


def test_sync_adapter_is_retired():
    with pytest.raises(RuntimeError, match="Synchronous Gemini VLM inference was removed"):
        GeminiVLM()
    production_sources = [
        Path("scripts/run_semantic_pipeline.py"),
        Path("pipeline/retrieval/semantic_pipeline.py"),
        Path("pipeline/retrieval/gemini_batch.py"),
    ]
    assert all("generate_content" not in path.read_text(encoding="utf-8") for path in production_sources)
