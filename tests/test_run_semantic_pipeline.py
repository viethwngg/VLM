import logging
import shutil
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from scripts import run_semantic_pipeline


@pytest.fixture
def workspace_tmp_path():
    root = Path.cwd() / ".test-work" / uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize(
    ("elapsed_s", "expected"),
    [(19.124, "00:00:19.124"), (61.005, "00:01:01.005"), (3661.999, "01:01:01.999")],
)
def test_format_duration(elapsed_s, expected):
    assert run_semantic_pipeline.format_duration(elapsed_s) == expected


def test_discover_video_jobs_finds_every_supported_video(workspace_tmp_path):
    (workspace_tmp_path / "scene-0001").mkdir()
    (workspace_tmp_path / "scene-0002").mkdir()
    (workspace_tmp_path / "scene-0001" / "cam_front.mp4").touch()
    (workspace_tmp_path / "scene-0002" / "camera-a.MOV").touch()
    (workspace_tmp_path / "scene-0002" / "camera-b.mkv").touch()
    (workspace_tmp_path / "scene-0002" / "notes.txt").touch()

    jobs = run_semantic_pipeline.discover_video_jobs(workspace_tmp_path)

    assert [(scene_id, path.name) for scene_id, path in jobs] == [
        ("scene-0001", "cam_front.mp4"),
        ("scene-0002__camera-a", "camera-a.MOV"),
        ("scene-0002__camera-b", "camera-b.mkv"),
    ]


def test_discover_video_jobs_can_filter_by_scene_directory(workspace_tmp_path):
    for scene in ("scene-0001", "scene-0002"):
        directory = workspace_tmp_path / scene
        directory.mkdir()
        (directory / "cam_front.mp4").touch()

    jobs = run_semantic_pipeline.discover_video_jobs(
        workspace_tmp_path, "scene-0002"
    )

    assert [(scene_id, path.name) for scene_id, path in jobs] == [
        ("scene-0002", "cam_front.mp4")
    ]


def test_discover_video_jobs_finds_nested_and_root_videos(workspace_tmp_path):
    nested = workspace_tmp_path / "train" / "scene-0001"
    nested.mkdir(parents=True)
    (nested / "front.webm").touch()
    (workspace_tmp_path / "standalone.avi").touch()

    jobs = run_semantic_pipeline.discover_video_jobs(workspace_tmp_path)

    assert [scene_id for scene_id, _ in jobs] == [
        "standalone",
        "train__scene-0001",
    ]


def test_process_scene_logs_elapsed_time(monkeypatch, caplog):
    artifact = Path("artifacts/semantic/scene-0061/semantic_metadata.json")
    process = Mock(return_value=artifact)
    monkeypatch.setattr(run_semantic_pipeline, "process_scene", process)
    monkeypatch.setattr(
        run_semantic_pipeline.time, "perf_counter", Mock(side_effect=[100.0, 119.124])
    )

    with caplog.at_level(logging.INFO):
        result = run_semantic_pipeline.process_scene_with_timing(
            "scene-0061", "video.mp4", "output", "CAM_FRONT", 8
        )

    assert result == artifact
    assert "status=started" in caplog.text
    assert "elapsed_s=19.124" in caplog.text
    assert "duration=00:00:19.124" in caplog.text


def test_process_scene_logs_failure_time(monkeypatch, caplog):
    monkeypatch.setattr(
        run_semantic_pipeline, "process_scene", Mock(side_effect=RuntimeError("failed"))
    )
    monkeypatch.setattr(
        run_semantic_pipeline.time, "perf_counter", Mock(side_effect=[10.0, 12.5])
    )

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="failed"):
        run_semantic_pipeline.process_scene_with_timing(
            "scene-0061", "video.mp4", "output", "CAM_FRONT", 8
        )

    assert "status=failed" in caplog.text
    assert "elapsed_s=2.500" in caplog.text
    assert "duration=00:00:02.500" in caplog.text


def test_log_total_duration_for_all_scenes(monkeypatch, caplog):
    monkeypatch.setattr(
        run_semantic_pipeline.time, "perf_counter", Mock(return_value=3723.456)
    )

    with caplog.at_level(logging.INFO):
        elapsed_s = run_semantic_pipeline.log_total_duration(
            started_at=100.0,
            total=10,
            succeeded=9,
            failed=1,
            corpus_path="artifacts/semantic/semantic_corpus.jsonl",
        )

    assert elapsed_s == pytest.approx(3623.456)
    assert "pipeline_complete" in caplog.text
    assert "total=10" in caplog.text
    assert "succeeded=9" in caplog.text
    assert "failed=1" in caplog.text
    assert "elapsed_s=3623.456" in caplog.text
    assert "duration=01:00:23.456" in caplog.text
