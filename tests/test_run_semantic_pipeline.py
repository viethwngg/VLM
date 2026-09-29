import shutil
from pathlib import Path
from types import SimpleNamespace
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


def test_main_run_prepares_and_submits_batch_only(monkeypatch, workspace_tmp_path):
    video = workspace_tmp_path / "scene" / "cam_front.mp4"
    video.parent.mkdir()
    video.touch()
    pipeline = SimpleNamespace(
        data_root=workspace_tmp_path,
        model="gemini-test",
        prepare=Mock(return_value={
            "total_videos_discovered": 1,
            "already_completed": 0,
            "need_processing": 1,
            "upload_successful": 1,
            "upload_failed": 0,
            "batch_requests_generated": 1,
            "batch_count": 1,
        }),
        submit=Mock(return_value=[]),
    )
    monkeypatch.setattr(run_semantic_pipeline, "build_pipeline", Mock(return_value=pipeline))

    assert run_semantic_pipeline.main(["run"]) == 0
    pipeline.prepare.assert_called_once()
    pipeline.submit.assert_called_once()


@pytest.mark.parametrize(
    ("command", "method"),
    [("submit", "submit"), ("status", "status"), ("retry", "retry_failed")],
)
def test_individual_lifecycle_commands(monkeypatch, command, method):
    pipeline = SimpleNamespace(data_root=Path("data"), model="gemini-test")
    operation = Mock(return_value=[])
    setattr(pipeline, method, operation)
    monkeypatch.setattr(run_semantic_pipeline, "build_pipeline", Mock(return_value=pipeline))

    assert run_semantic_pipeline.main([command]) == 0
    operation.assert_called_once_with()


def test_collect_exit_code_reflects_failed_results(monkeypatch):
    pipeline = SimpleNamespace(
        data_root=Path("data"),
        model="gemini-test",
        collect=Mock(return_value={
            "jobs_downloaded": 1,
            "results_succeeded": 2,
            "results_failed": 1,
            "invalid_schema": 1,
        }),
    )
    monkeypatch.setattr(run_semantic_pipeline, "build_pipeline", Mock(return_value=pipeline))

    assert run_semantic_pipeline.main(["collect"]) == 1
    pipeline.collect.assert_called_once_with()
