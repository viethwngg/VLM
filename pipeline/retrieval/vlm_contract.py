"""Shared VLM V3 request, validation, and postprocessing contract.

This module is transport-neutral. Gemini Batch is the only execution
transport; prompt, schema, taxonomy, generation settings, and materialization
all originate here or from the versioned modules imported here.
"""

from __future__ import annotations

import json

from .prompts import PROMPT_VERSION, prompt_for_taxonomy
from .schemas import GeminiSceneOutput, SEMANTIC_PIPELINE_VERSION, SemanticScene
from .searchable_text import build_searchable_text
from .taxonomy import TAXONOMY_VERSION


DEFAULT_MODEL = "gemini-3.8-flash"
PIPELINE_VERSION = SEMANTIC_PIPELINE_VERSION


def build_scene_prompt(scene_id: str) -> str:
    """Return the single versioned prompt used by every VLM batch request."""
    return prompt_for_taxonomy() + f"\nScene ID: {scene_id}\nReturn JSON only."


def build_scene_parts(
    scene_id: str,
    frames: list,
    file_uri: str,
    mime_type: str,
) -> list[dict]:
    """Build the video and temporal context parts for one batch request."""
    parts = [{"text": build_scene_prompt(scene_id)}]
    parts.append({"file_data": {"file_uri": file_uri, "mime_type": mime_type}})
    for frame in frames:
        parts.append({
            "text": (
                f"Frame camera={frame.camera}, timestamp_s={frame.timestamp_s}, "
                f"uri={frame.frame_uri}"
            )
        })
    return parts


def batch_generation_config() -> dict:
    """Return the canonical JSON-serializable V3 Batch generation settings."""
    return {
        "response_mime_type": "application/json",
        "response_json_schema": GeminiSceneOutput.model_json_schema(),
    }


def _json_object(raw: str) -> dict:
    """Decode a model JSON response, tolerating a surrounding Markdown fence."""
    if not raw or not raw.strip():
        raise ValueError("Gemini response is empty")
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Gemini response does not contain a JSON object")
    value = json.loads(raw[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Gemini response JSON must be an object")
    return value


def parse_scene_payload(
    payload,
    *,
    scene_id: str,
    evidence: list[dict] | None,
    model: str,
) -> SemanticScene:
    """Validate and postprocess a V3 payload into canonical semantic metadata."""
    if hasattr(payload, "model_dump"):
        data = payload.model_dump()
    elif isinstance(payload, dict):
        data = dict(payload)
    elif isinstance(payload, str):
        data = _json_object(payload)
    else:
        raise TypeError(f"Unsupported Gemini payload type: {type(payload).__name__}")

    data["scene_id"] = scene_id
    data["evidence"] = evidence or data.get("evidence", [])
    data["provenance"] = {
        "vlm_provider": "gemini",
        "vlm_model": model,
        "prompt_version": PROMPT_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "pipeline_version": PIPELINE_VERSION,
    }
    scene = SemanticScene.model_validate(data)
    scene.searchable_text = build_searchable_text(scene)
    return scene


def response_text(response) -> str:
    """Extract generated text from an SDK object or Batch API response dict."""
    direct = getattr(response, "text", None) or getattr(response, "output_text", None)
    if direct:
        return direct
    if isinstance(response, dict):
        direct = response.get("text") or response.get("output_text")
        if direct:
            return direct
        texts = []
        for candidate in response.get("candidates", []):
            for part in (candidate.get("content") or {}).get("parts", []):
                if part.get("text"):
                    texts.append(part["text"])
        return "".join(texts)
    return str(response) if response is not None else ""
