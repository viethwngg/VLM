import json
import pytest
from pipeline.retrieval.schemas import SemanticScene
from pipeline.retrieval.searchable_text import build_searchable_text

def provenance(): return {"vlm_model": "mock", "prompt_version": "road-vlm-v1", "taxonomy_version": "road-v1"}
def test_taxonomy_normalization():
    s = SemanticScene(scene_id="x", agents=["Pedestrian", "invented"], road_context=["urban"], provenance=provenance())
    assert s.agents == ["pedestrian"]
def test_searchable_text():
    s = SemanticScene(scene_id="x", agents=["car"], actions=["turning_right"], provenance=provenance())
    assert "Agents: car" in build_searchable_text(s)
def test_embedding_dimensions_and_normalization():
    np = pytest.importorskip("numpy")
    from pipeline.retrieval.embedder import Embedder
    v = Embedder(dimension=8).encode(["urban car"]); assert v.shape == (1, 8); assert np.isclose(np.linalg.norm(v[0]), 1)
def test_json_serialization():
    s = SemanticScene(scene_id="x", provenance=provenance()); assert json.loads(s.model_dump_json())["scene_id"] == "x"
