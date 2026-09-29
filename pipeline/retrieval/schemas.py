"""Validated nested semantic schema for RAV-21 VLM output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .taxonomy import TAXONOMY_VERSION, allowed, normalize_label


def _labels(field: str, values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return list(dict.fromkeys(
        label for value in values if (label := allowed(field, value))
    ))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SceneContext(StrictModel):
    road_type: str | None
    traffic_state: str | None
    weather: str | None
    lighting: str | None
    road_surface: str | None

    @field_validator(
        "road_type", "traffic_state", "weather", "lighting", "road_surface",
        mode="before",
    )
    @classmethod
    def normalize_context(cls, value, info):
        return allowed(info.field_name, value) if value is not None else None


class EgoState(StrictModel):
    actions: list[str]

    @field_validator("actions", mode="before")
    @classmethod
    def normalize_actions(cls, value):
        return _labels("ego_action", value)


class RoadAgent(StrictModel):
    type: str
    actions: list[str]
    locations: list[str]
    relations: list[str]

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value):
        label = allowed("agent_type", value)
        if label is None:
            raise ValueError(f"Unsupported agent type: {value}")
        return label

    @field_validator("actions", "locations", "relations", mode="before")
    @classmethod
    def normalize_lists(cls, value, info):
        taxonomy_field = {
            "actions": "agent_action",
            "locations": "location",
            "relations": "relation",
        }[info.field_name]
        return _labels(taxonomy_field, value)


class RoadEvent(StrictModel):
    type: str
    participants: list[str]
    location: str | None
    temporal_transition: str | None

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value):
        label = allowed("road_event", value)
        if label is None:
            raise ValueError(f"Unsupported road event: {value}")
        return label

    @field_validator("participants", mode="before")
    @classmethod
    def normalize_participants(cls, values):
        if values is None:
            return []
        if isinstance(values, str):
            values = [values]
        normalized = []
        for value in values:
            label = normalize_label(value)
            if label == "ego_vehicle" or allowed("agent_type", label):
                normalized.append(label)
        return list(dict.fromkeys(normalized))

    @field_validator("location", mode="before")
    @classmethod
    def normalize_location(cls, value):
        return allowed("location", value) if value is not None else None

    @field_validator("temporal_transition", mode="before")
    @classmethod
    def normalize_transition(cls, value):
        return allowed("temporal_transition", value) if value is not None else None


class SceneContent(StrictModel):
    """Exact JSON body produced by Gemini for one video scene."""

    scene: SceneContext
    ego: EgoState
    agents: list[RoadAgent]
    events: list[RoadEvent]
    attention_events: list[str]
    searchable_text: str

    @field_validator("attention_events", mode="before")
    @classmethod
    def normalize_attention_events(cls, value):
        return _labels("attention_event", value)

    @field_validator("searchable_text", mode="before")
    @classmethod
    def normalize_searchable_text(cls, value):
        return " ".join(str(value or "").split())


class GeminiSceneOutput(SceneContent):
    """VLM-owned fields used as the structured Gemini response schema."""


class Evidence(BaseModel):
    camera: str
    timestamp_s: float | None = None
    frame_uri: str | None = None


class Provenance(BaseModel):
    vlm_provider: str = "gemini"
    vlm_model: str
    prompt_version: str
    taxonomy_version: str = TAXONOMY_VERSION
    pipeline_version: str = "semantic-pipeline-v2"


class SemanticScene(SceneContent):
    """Gemini output plus pipeline-owned identity and provenance fields."""

    scene_id: str
    evidence: list[Evidence] = Field(default_factory=list)
    provenance: Provenance
