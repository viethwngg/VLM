import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from pipeline.retrieval.frame_sampler import FrameSample
from pipeline.retrieval.gemini_batch import (
    GeminiBatchPipeline,
    is_scene_completed,
    make_batch_request,
    make_request_key,
    validate_batch_rows,
)
from pipeline.retrieval.prompts import PROMPT_VERSION, prompt_for_taxonomy
from pipeline.retrieval.schemas import SemanticScene
from pipeline.retrieval.semantic_pipeline import write_scene_metadata
from pipeline.retrieval.taxonomy import TAXONOMY_VERSION, taxonomy_values
from pipeline.retrieval.vlm_contract import batch_generation_config, build_scene_prompt


@pytest.fixture
def workspace_tmp_path():
    root = Path.cwd() / ".test-work" / uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


class FakeFiles:
    def __init__(self):
        self.upload = Mock(side_effect=self._upload)
        self.get = Mock(side_effect=self._get)
        self.download = Mock(side_effect=self._download)
        self.remotes = {}
        self.result_rows = []

    def _upload(self, *, file, config=None):
        name = f"files/file-{len(self.remotes) + 1}"
        mime = getattr(config, "mime_type", None) or "video/mp4"
        remote = SimpleNamespace(
            name=name,
            uri=f"https://files.example/{name}",
            mime_type=mime,
            state="ACTIVE",
            expiration_time=None,
        )
        self.remotes[name] = remote
        return remote

    def _get(self, *, name):
        return self.remotes[name]

    def _download(self, *, file, destination):
        Path(destination).write_text(
            "".join(json.dumps(row) + "\n" for row in self.result_rows),
            encoding="utf-8",
        )


class FakeBatches:
    def __init__(self):
        self.create = Mock(side_effect=self._create)
        self.get = Mock(side_effect=self._get)
        self.jobs = {}

    def _create(self, *, model, src, config):
        name = f"batches/job-{len(self.jobs) + 1}"
        job = SimpleNamespace(
            name=name,
            model=model,
            state="JOB_STATE_QUEUED",
            completion_stats=None,
            dest=None,
        )
        self.jobs[name] = job
        return job

    def _get(self, *, name):
        return self.jobs[name]


@pytest.fixture
def fake_client():
    return SimpleNamespace(files=FakeFiles(), batches=FakeBatches())


def make_pipeline(root, fake_client, batch_size=2):
    return GeminiBatchPipeline(
        data_root=root / "data",
        output_root=root / "semantic",
        batch_root=root / "vlm_batch",
        client=fake_client,
        model="gemini-test",
        batch_size=batch_size,
        retry_delay=0.01,
    )


def frames(video="video.mp4"):
    return [
        FrameSample("CAM_FRONT", 0.0, video),
        FrameSample("CAM_FRONT", 1.0, video),
    ]


def valid_vlm_output(searchable_text="Urban road with moderate traffic."):
    return {
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
        "searchable_text": searchable_text,
    }


def semantic_scene(scene_id):
    return SemanticScene(
        scene_id=scene_id,
        scene={
            "road_type": "urban_road",
            "traffic_state": "moderate_traffic",
            "weather": "clear",
            "lighting": "daylight",
            "road_surface": "dry",
        },
        ego={"actions": ["decelerating"]},
        agents=[],
        events=[],
        attention_events=[],
        searchable_text="Urban road with moderate traffic.",
        provenance={
            "vlm_model": "gemini-test",
            "prompt_version": "road-vlm-v3",
            "taxonomy_version": "road-v2",
        },
    )


def test_batch_request_reuses_prompt_and_references_file_uri():
    row = make_batch_request(
        "scene-0001", frames(), file_uri="https://files.example/video", mime_type="video/mp4"
    )
    validate_batch_rows([row])
    assert row["key"] == "scene-0001__CAM_FRONT"
    parts = row["request"]["contents"][0]["parts"]
    assert parts[0]["text"] == build_scene_prompt("scene-0001")
    assert parts[1]["file_data"] == {
        "file_uri": "https://files.example/video",
        "mime_type": "video/mp4",
    }
    assert row["request"]["generation_config"]["response_mime_type"] == "application/json"
    assert row["request"]["generation_config"] == batch_generation_config()
    assert PROMPT_VERSION == "road-vlm-v3"
    assert TAXONOMY_VERSION == "road-v2"
    assert json.dumps(taxonomy_values(), separators=(",", ":")) in prompt_for_taxonomy()


def test_request_key_deterministically_maps_scene_and_camera():
    assert make_request_key("scene-000123", "CAM_FRONT_LEFT") == (
        "scene-000123__CAM_FRONT_LEFT"
    )


def test_batch_validation_rejects_duplicate_keys():
    row = make_batch_request("scene", frames(), file_uri="uri", mime_type="video/mp4")
    with pytest.raises(ValueError, match="Duplicate"):
        validate_batch_rows([row, row])


def test_is_scene_completed_requires_valid_matching_schema(workspace_tmp_path):
    output = workspace_tmp_path / "semantic"
    scene = semantic_scene("scene-1")
    write_scene_metadata(scene, output)
    assert is_scene_completed("scene-1", output)
    assert not is_scene_completed("scene-2", output)


def test_prepare_skips_completed_splits_and_persists_manifest(
    workspace_tmp_path, fake_client, monkeypatch
):
    pipeline = make_pipeline(workspace_tmp_path, fake_client, batch_size=2)
    data = workspace_tmp_path / "data"
    jobs = []
    for index in range(4):
        path = data / f"scene-{index}" / "cam_front.mp4"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"video")
        jobs.append((f"scene-{index}", path))
    completed = semantic_scene("scene-0")
    write_scene_metadata(completed, pipeline.output_root)
    monkeypatch.setattr(
        "pipeline.retrieval.gemini_batch.sample_video",
        lambda path, camera, count: frames(str(path)),
    )

    summary = pipeline.prepare(jobs, limit=3)

    assert summary == {
        "total_videos_discovered": 4,
        "already_completed": 1,
        "need_processing": 3,
        "upload_successful": 3,
        "upload_failed": 0,
        "batch_requests_generated": 3,
        "batch_count": 2,
    }
    plan = json.loads(pipeline.plan_path.read_text(encoding="utf-8"))
    assert [batch["request_count"] for batch in plan["batches"]] == [2, 1]
    assert plan["prompt_version"] == PROMPT_VERSION
    assert plan["taxonomy_version"] == TAXONOMY_VERSION
    prepared = json.loads(pipeline.prepare_manifest_path.read_text(encoding="utf-8"))
    assert prepared["scene-1__CAM_FRONT"]["scene_id"] == "scene-1"
    assert prepared["scene-1__CAM_FRONT"]["camera"] == "CAM_FRONT"
    assert prepared["scene-1__CAM_FRONT"]["model"] == "gemini-test"
    manifest = json.loads(pipeline.upload_manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == {
        "scene-1__CAM_FRONT",
        "scene-2__CAM_FRONT",
        "scene-3__CAM_FRONT",
    }
    assert all(item["status"] == "ACTIVE" for item in manifest.values())


def test_submit_is_idempotent_for_active_jobs(workspace_tmp_path, fake_client, monkeypatch):
    pipeline = make_pipeline(workspace_tmp_path, fake_client)
    path = workspace_tmp_path / "data" / "scene" / "cam_front.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"video")
    monkeypatch.setattr(
        "pipeline.retrieval.gemini_batch.sample_video",
        lambda path, camera, count: frames(str(path)),
    )
    pipeline.prepare([("scene", path)])

    first = pipeline.submit()
    second = pipeline.submit()

    assert first[0]["job_name"] == second[0]["job_name"]
    fake_client.batches.create.assert_called_once()


def test_submit_does_not_resubmit_terminal_job_after_restart(
    workspace_tmp_path, fake_client, monkeypatch
):
    pipeline = make_pipeline(workspace_tmp_path, fake_client)
    path = workspace_tmp_path / "data" / "scene" / "cam_front.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"video")
    monkeypatch.setattr(
        "pipeline.retrieval.gemini_batch.sample_video",
        lambda path, camera, count: frames(str(path)),
    )
    pipeline.prepare([("scene", path)])
    first = pipeline.submit()
    fake_client.batches.jobs[first[0]["job_name"]].state = "JOB_STATE_FAILED"

    second = pipeline.submit()

    assert second[0]["job_name"] == first[0]["job_name"]
    fake_client.batches.create.assert_called_once()


def test_collect_uses_shared_schema_parser_and_writes_canonical_metadata(
    workspace_tmp_path, fake_client, monkeypatch
):
    pipeline = make_pipeline(workspace_tmp_path, fake_client)
    path = workspace_tmp_path / "data" / "scene" / "cam_front.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"video")
    monkeypatch.setattr(
        "pipeline.retrieval.gemini_batch.sample_video",
        lambda path, camera, count: frames(str(path)),
    )
    pipeline.prepare([("scene", path)])
    jobs = pipeline.submit()
    remote_job = fake_client.batches.jobs[jobs[0]["job_name"]]
    remote_job.state = "JOB_STATE_SUCCEEDED"
    remote_job.completion_stats = SimpleNamespace(
        successful_count=1, failed_count=0, incomplete_count=0
    )
    remote_job.dest = SimpleNamespace(file_name="files/result")
    fake_client.files.result_rows = [{
        "key": "scene__CAM_FRONT",
        "response": {
            "candidates": [{
                "content": {"parts": [{"text": json.dumps({
                    "scene": {
                        "road_type": "urban_road",
                        "traffic_state": "moderate_traffic",
                        "weather": "clear",
                        "lighting": "daylight",
                        "road_surface": "dry",
                    },
                    "ego": {"actions": ["turning_right"]},
                    "agents": [{
                        "type": "car",
                        "actions": ["turning_right"],
                        "locations": ["intersection"],
                        "relations": [],
                    }],
                    "events": [{
                        "type": "vehicle_turning",
                        "participants": ["car"],
                        "location": "intersection",
                        "temporal_transition": "straight_to_turning_right",
                    }],
                    "attention_events": [],
                    "searchable_text": "A car turns right at an urban intersection.",
                })}]}
            }]
        },
    }]

    summary = pipeline.collect()

    assert summary == {
        "jobs_downloaded": 1,
        "results_succeeded": 1,
        "results_failed": 0,
        "invalid_schema": 0,
    }
    output = pipeline.output_root / "scene" / "semantic_metadata.json"
    metadata = json.loads(output.read_text(encoding="utf-8"))
    assert metadata["scene_id"] == "scene"
    assert metadata["provenance"]["vlm_model"] == "gemini-test"
    assert metadata["scene"]["road_type"] == "urban_road"
    assert metadata["agents"][0]["type"] == "car"
    assert metadata["searchable_text"] == "A car turns right at an urban intersection."
    corpus_row = json.loads(
        (pipeline.output_root / "semantic_corpus.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert corpus_row["scene_id"] == "scene"
    assert corpus_row["searchable_text"] == metadata["searchable_text"]
    assert pipeline.failed_path.read_text(encoding="utf-8").strip() == "[]"
    assert json.loads(pipeline.successful_path.read_text(encoding="utf-8")) == [
        "scene__CAM_FRONT"
    ]


def test_retry_failed_submits_only_retryable_request(
    workspace_tmp_path, fake_client, monkeypatch
):
    pipeline = make_pipeline(workspace_tmp_path, fake_client)
    path = workspace_tmp_path / "data" / "scene" / "cam_front.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"video")
    monkeypatch.setattr(
        "pipeline.retrieval.gemini_batch.sample_video",
        lambda path, camera, count: frames(str(path)),
    )
    pipeline.prepare([("scene", path)])
    jobs = pipeline.submit()
    original = fake_client.batches.jobs[jobs[0]["job_name"]]
    original.state = "JOB_STATE_SUCCEEDED"
    original.completion_stats = SimpleNamespace(
        successful_count=0, failed_count=1, incomplete_count=0
    )
    original.dest = SimpleNamespace(file_name="files/result")
    fake_client.files.result_rows = [{
        "key": "scene__CAM_FRONT",
        "error": {"code": 503, "message": "Temporarily unavailable"},
    }]
    assert pipeline.collect()["results_failed"] == 1

    retry_jobs = pipeline.retry_failed()

    assert len(retry_jobs) == 1
    assert retry_jobs[0]["request_keys"] == ["scene__CAM_FRONT"]
    assert fake_client.batches.create.call_count == 2


def test_collect_reports_invalid_v3_schema_for_retry(
    workspace_tmp_path, fake_client, monkeypatch
):
    pipeline = make_pipeline(workspace_tmp_path, fake_client)
    path = workspace_tmp_path / "data" / "scene" / "cam_front.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"video")
    monkeypatch.setattr(
        "pipeline.retrieval.gemini_batch.sample_video",
        lambda path, camera, count: frames(str(path)),
    )
    pipeline.prepare([("scene", path)])
    jobs = pipeline.submit()
    remote_job = fake_client.batches.jobs[jobs[0]["job_name"]]
    remote_job.state = "JOB_STATE_SUCCEEDED"
    remote_job.completion_stats = SimpleNamespace(
        successful_count=1, failed_count=0, incomplete_count=0
    )
    remote_job.dest = SimpleNamespace(file_name="files/result")
    fake_client.files.result_rows = [{
        "key": "scene__CAM_FRONT",
        "response": {
            "candidates": [{"content": {"parts": [{"text": "{}"}]}}]
        },
    }]

    summary = pipeline.collect()

    assert summary["results_succeeded"] == 0
    assert summary["results_failed"] == 1
    assert summary["invalid_schema"] == 1
    failure = json.loads(pipeline.failed_path.read_text(encoding="utf-8"))[0]
    assert failure["key"] == "scene__CAM_FRONT"
    assert failure["category"] == "invalid_schema"
    assert failure["retryable"] is True
    assert json.loads(pipeline.invalid_path.read_text(encoding="utf-8")) == [failure]
    assert not (pipeline.output_root / "scene" / "semantic_metadata.json").exists()


def test_collect_preserves_success_and_retries_only_failed_request(
    workspace_tmp_path, fake_client, monkeypatch
):
    pipeline = make_pipeline(workspace_tmp_path, fake_client)
    jobs_to_prepare = []
    for scene_id in ("scene-a", "scene-b"):
        path = workspace_tmp_path / "data" / scene_id / "cam_front.mp4"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"video")
        jobs_to_prepare.append((scene_id, path))
    monkeypatch.setattr(
        "pipeline.retrieval.gemini_batch.sample_video",
        lambda path, camera, count: frames(str(path)),
    )
    pipeline.prepare(jobs_to_prepare)
    jobs = pipeline.submit()
    remote_job = fake_client.batches.jobs[jobs[0]["job_name"]]
    remote_job.state = "JOB_STATE_PARTIALLY_SUCCEEDED"
    remote_job.completion_stats = SimpleNamespace(
        successful_count=1, failed_count=1, incomplete_count=0
    )
    remote_job.dest = SimpleNamespace(file_name="files/result")
    fake_client.files.result_rows = [
        {
            "key": "scene-a__CAM_FRONT",
            "response": {
                "candidates": [{
                    "content": {"parts": [{"text": json.dumps(valid_vlm_output())}]}
                }]
            },
        },
        {
            "key": "scene-b__CAM_FRONT",
            "error": {"code": 503, "message": "Temporarily unavailable"},
        },
    ]

    assert pipeline.collect()["results_succeeded"] == 1
    success_path = pipeline.output_root / "scene-a" / "semantic_metadata.json"
    original_success = success_path.read_text(encoding="utf-8")

    retry_jobs = pipeline.retry_failed()

    assert retry_jobs[0]["request_keys"] == ["scene-b__CAM_FRONT"]
    assert success_path.read_text(encoding="utf-8") == original_success

