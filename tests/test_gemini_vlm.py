from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from google.genai import errors

from pipeline.retrieval.frame_sampler import FrameSample
from pipeline.retrieval.gemini_vlm import GeminiVLM
from pipeline.retrieval.schemas import GeminiSceneOutput


@pytest.fixture
def setup_vlm(monkeypatch):
    monkeypatch.setattr("pipeline.retrieval.gemini_vlm.time.sleep", Mock())
    monkeypatch.setattr("pipeline.retrieval.gemini_vlm.Path.exists", lambda self: True)
    video = "video.mp4"
    remote = SimpleNamespace(name="files/test", uri="https://example.com/video",
                             mime_type="video/mp4", state="ACTIVE")
    client = SimpleNamespace(
        files=SimpleNamespace(upload=Mock(return_value=remote), get=Mock(return_value=remote)),
        models=SimpleNamespace(generate_content=Mock(return_value=SimpleNamespace(text='{}'))),
    )
    frames = [FrameSample("CAM_FRONT", timestamp, str(video)) for timestamp in (0, 1)]
    return GeminiVLM(client=client, max_retries=3, retry_delay=10, fallback_models=[]), client, frames, remote


def test_upload_disconnect_retried_and_video_deduplicated(setup_vlm):
    vlm, client, frames, remote = setup_vlm
    client.files.upload.side_effect = [httpx.RemoteProtocolError("Disconnected"), remote]
    scene = vlm.analyze("scene-0061", frames)
    assert scene.scene_id == "scene-0061"
    assert client.files.upload.call_count == 2
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert sum("file_data" in part for part in contents) == 1


def test_persistent_disconnect_is_bounded(setup_vlm):
    vlm, client, frames, _ = setup_vlm
    error = httpx.RemoteProtocolError("Disconnected")
    client.files.upload.side_effect = error
    with pytest.raises(RuntimeError, match="scene-0061.*3 attempts") as caught:
        vlm.analyze("scene-0061", frames)
    assert caught.value.__cause__ is error
    assert client.files.upload.call_count == 3
    client.models.generate_content.assert_not_called()


def test_waits_for_active_and_retries_poll_disconnect(setup_vlm):
    vlm, client, frames, remote = setup_vlm
    client.files.upload.return_value = SimpleNamespace(name=remote.name, state="PROCESSING")
    def poll(**kwargs):
        client.models.generate_content.assert_not_called()
        if client.files.get.call_count == 1:
            raise httpx.RemoteProtocolError("Disconnected")
        return remote
    client.files.get.side_effect = poll
    vlm.analyze("scene-0061", frames)
    assert client.files.get.call_count == 2
    client.files.get.assert_called_with(name=remote.name)
    client.models.generate_content.assert_called_once()


def test_processing_failure_stops_before_generation(setup_vlm):
    vlm, client, frames, remote = setup_vlm
    remote.state = "FAILED"
    with pytest.raises(RuntimeError, match="processing failed"):
        vlm.analyze("scene-0061", frames)
    client.models.generate_content.assert_not_called()


def test_processing_timeout(setup_vlm, monkeypatch):
    vlm, client, frames, remote = setup_vlm
    remote.state = "PROCESSING"
    monkeypatch.setattr("pipeline.retrieval.gemini_vlm.time.monotonic", Mock(side_effect=[0, 301]))
    with pytest.raises(TimeoutError, match="processing timed out"):
        vlm.analyze("scene-0061", frames)
    client.models.generate_content.assert_not_called()


def test_permanent_api_error_is_not_retried(setup_vlm):
    vlm, client, frames, _ = setup_vlm
    client.files.upload.side_effect = errors.ClientError(403, {"error": {"message": "Forbidden"}})
    with pytest.raises(errors.ClientError):
        vlm.analyze("scene-0061", frames)
    client.files.upload.assert_called_once()


def test_generation_retry_reuses_uploaded_video(setup_vlm):
    vlm, client, frames, _ = setup_vlm
    client.models.generate_content.side_effect = [httpx.RemoteProtocolError("Disconnected"), SimpleNamespace(text='{}')]
    vlm.analyze("scene-0061", frames)
    client.files.upload.assert_called_once()
    assert client.models.generate_content.call_count == 2


def test_generation_requests_structured_output(setup_vlm):
    vlm, client, frames, _ = setup_vlm
    vlm.analyze("scene-0061", frames)
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_schema"] is GeminiSceneOutput


def test_legacy_string_events_and_relations_are_normalized(setup_vlm):
    vlm, client, frames, _ = setup_vlm
    client.models.generate_content.return_value = SimpleNamespace(text='''{
        "events": ["vehicle_turning", "vehicle_stopping"],
        "relations": ["in_front_of", "following"]
    }''')
    scene = vlm.analyze("scene-0061", frames)
    assert [event.event for event in scene.events] == ["vehicle_turning", "vehicle_stopping"]
    assert [relation.relation for relation in scene.relations] == ["in_front_of", "following"]
    assert all(relation.subject == "unknown" and relation.object == "unknown"
               for relation in scene.relations)
    client.models.generate_content.assert_called_once()


def test_uses_sdk_parsed_structured_output(setup_vlm):
    vlm, client, frames, _ = setup_vlm
    parsed = GeminiSceneOutput(
        description="A car turns.",
        events=[{"event": "vehicle_turning", "agent": "car"}],
    )
    client.models.generate_content.return_value = SimpleNamespace(parsed=parsed, text="ignored")
    scene = vlm.analyze("scene-0061", frames)
    assert scene.description == "A car turns."
    assert scene.events[0].event == "vehicle_turning"


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_permanent_generation_error_fails_once_with_model_hint(setup_vlm, code):
    vlm, client, frames, _ = setup_vlm
    error = errors.ClientError(code, {"error": {"message": "Model unavailable"}})
    client.models.generate_content.side_effect = error
    with pytest.raises(RuntimeError, match=f"scene-0061.*{vlm.model}.*HTTP {code}.*GEMINI_MODEL") as caught:
        vlm.analyze("scene-0061", frames)
    assert caught.value.__cause__ is error
    client.models.generate_content.assert_called_once()
    client.files.upload.assert_called_once()


@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
def test_transient_generation_api_error_is_retried(setup_vlm, code):
    vlm, client, frames, _ = setup_vlm
    client.models.generate_content.side_effect = [
        errors.APIError(code, {"error": {"message": "Try again"}}),
        SimpleNamespace(text='{}'),
    ]
    assert vlm.analyze("scene-0061", frames).scene_id == "scene-0061"
    assert client.models.generate_content.call_count == 2
    client.files.upload.assert_called_once()


def test_default_model_and_overrides(monkeypatch):
    client = SimpleNamespace()
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_FALLBACK_MODELS", raising=False)
    assert GeminiVLM(client=client).model == "gemini-3.8-flash"
    assert GeminiVLM(client=client).models == [
        "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite"
    ]
    monkeypatch.setenv("GEMINI_MODEL", "env-model")
    assert GeminiVLM(client=client).model == "env-model"
    assert GeminiVLM(client=client, model="explicit-model").model == "explicit-model"


def test_persistent_overload_has_bounded_backoff_and_actionable_error(setup_vlm, monkeypatch):
    vlm, client, frames, _ = setup_vlm
    vlm.max_retries = 6
    monkeypatch.setattr("pipeline.retrieval.gemini_vlm.random.uniform", lambda low, high: high)
    sleep = Mock()
    monkeypatch.setattr("pipeline.retrieval.gemini_vlm.time.sleep", sleep)
    error = errors.ServerError(503, {"error": {"message": "High demand"}})
    client.models.generate_content.side_effect = error
    with pytest.raises(RuntimeError, match="6 attempts.*HTTP 503.*Retry later") as caught:
        vlm.analyze("scene-0061", frames)
    assert caught.value.__cause__ is error
    assert client.models.generate_content.call_count == 6
    assert [call.args[0] for call in sleep.call_args_list] == [12, 24, 48, 60, 60]
    client.files.upload.assert_called_once()


def test_overload_recovers_without_reupload(setup_vlm):
    vlm, client, frames, _ = setup_vlm
    client.models.generate_content.side_effect = [
        errors.ServerError(503, {"error": {"message": "High demand"}}),
        errors.ServerError(503, {"error": {"message": "High demand"}}),
        SimpleNamespace(text='{}'),
    ]
    assert vlm.analyze("scene-0061", frames).scene_id == "scene-0061"
    client.files.upload.assert_called_once()
    assert client.models.generate_content.call_count == 3
    assert client.models.generate_content.call_args.kwargs["config"]["automatic_function_calling"]["disable"]


def test_retry_settings_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_ATTEMPTS", "7")
    monkeypatch.setenv("GEMINI_RETRY_DELAY_SECONDS", "15")
    vlm = GeminiVLM(client=SimpleNamespace())
    assert (vlm.max_retries, vlm.retry_delay) == (7, 15)
    vlm = GeminiVLM(client=SimpleNamespace(), max_retries=2, retry_delay=1)
    assert (vlm.max_retries, vlm.retry_delay) == (2, 1)


@pytest.mark.parametrize("code", [404, 503])
def test_unavailable_model_falls_back_without_reupload(setup_vlm, code):
    vlm, client, frames, _ = setup_vlm
    vlm.models = [vlm.model, "fallback-model"]
    client.models.generate_content.side_effect = [
        errors.APIError(code, {"error": {"message": "Unavailable"}}),
        SimpleNamespace(text='{}'),
    ]
    scene = vlm.analyze("scene-0061", frames)
    assert scene.provenance.vlm_model == "fallback-model"
    assert [call.kwargs["model"] for call in client.models.generate_content.call_args_list] == [
        vlm.model, "fallback-model"
    ]
    client.files.upload.assert_called_once()


def test_fallback_models_from_environment_are_trimmed_and_deduplicated(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", " fallback ,primary,, fallback ")
    assert GeminiVLM(client=SimpleNamespace()).models == ["primary", "fallback"]


def test_empty_fallback_environment_disables_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")
    assert GeminiVLM(client=SimpleNamespace()).models == ["primary"]


@pytest.mark.parametrize("delay", [0, -1, float("inf"), float("nan")])
def test_invalid_retry_delay(delay):
    with pytest.raises(ValueError, match="retry_delay"):
        GeminiVLM(client=SimpleNamespace(), retry_delay=delay)
