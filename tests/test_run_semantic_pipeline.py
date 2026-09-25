import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import run_semantic_pipeline


@pytest.mark.parametrize(
    ("elapsed_s", "expected"),
    [(19.124, "00:00:19.124"), (61.005, "00:01:01.005"), (3661.999, "01:01:01.999")],
)
def test_format_duration(elapsed_s, expected):
    assert run_semantic_pipeline.format_duration(elapsed_s) == expected


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
