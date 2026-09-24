"""Validated semantic schema and taxonomy normalization."""

from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .taxonomy import TAXONOMY, allowed

def _labels(field: str, values: Any) -> list[str]:
    if values is None: return []
    if isinstance(values, str): values = [values]
    return list(dict.fromkeys(x for v in values if (x := allowed(field, v))))

class Relation(BaseModel):
    subject: str
    relation: str
    object: str
    @field_validator("relation", mode="before")
    @classmethod
    def valid_relation(cls, v):
        return allowed("relations", v) or "near"

class Event(BaseModel):
    event: str
    agent: str | None = None
    action: str | None = None
    location: str | None = None
    start_s: float | None = None
    end_s: float | None = None
    @field_validator("event", mode="before")
    @classmethod
    def valid_event(cls, v):
        return allowed("events", v) or "vehicle_vehicle_interaction"
    @field_validator("agent", mode="before")
    @classmethod
    def valid_agent(cls, v): return allowed("agents", v) if v is not None else None
    @field_validator("action", mode="before")
    @classmethod
    def valid_action(cls, v): return allowed("actions", v) if v is not None else None
    @field_validator("location", mode="before")
    @classmethod
    def valid_location(cls, v): return allowed("locations", v) if v is not None else None

class TemporalRelation(BaseModel):
    event_1: str
    relation: str
    event_2: str

class Evidence(BaseModel):
    camera: str
    timestamp_s: float | None = None
    frame_uri: str | None = None

class Provenance(BaseModel):
    vlm_provider: str = "gemini"
    vlm_model: str
    prompt_version: str
    taxonomy_version: str
    pipeline_version: str = "semantic-pipeline-v1"

class SemanticScene(BaseModel):
    model_config = ConfigDict(extra="ignore")
    scene_id: str
    description: str = ""
    agents: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    road_context: list[str] = Field(default_factory=list)
    traffic_condition: str | None = None
    weather: list[str] = Field(default_factory=list)
    time_of_day: str | None = None
    relations: list[Relation] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    temporal_relations: list[TemporalRelation] = Field(default_factory=list)
    hazards: list[str] = Field(default_factory=list)
    visibility: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    searchable_text: str = ""
    provenance: Provenance

    @field_validator("agents", "actions", "locations", "road_context", "weather", "hazards", mode="before")
    @classmethod
    def normalize_lists(cls, v, info): return _labels(info.field_name, v)
    @field_validator("traffic_condition", "time_of_day", mode="before")
    @classmethod
    def normalize_optional(cls, v, info): return allowed(info.field_name, v) if v is not None else None
