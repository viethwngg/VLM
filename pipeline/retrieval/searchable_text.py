"""Deterministic fallback retrieval text for the nested semantic schema."""

from .schemas import SemanticScene


def _words(value: str) -> str:
    return value.replace("_", " ")


def build_searchable_text(scene: SemanticScene) -> str:
    """Preserve VLM retrieval text, or build a compact fallback if it is empty."""
    if scene.searchable_text.strip():
        return " ".join(scene.searchable_text.split())

    phrases: list[str] = []
    if scene.attention_events:
        phrases.append(
            "Attention events: "
            + ", ".join(_words(value) for value in scene.attention_events)
            + "."
        )
    for event in scene.events:
        parts = [
            ", ".join(_words(value) for value in event.participants),
            _words(event.type),
            _words(event.location) if event.location else "",
            _words(event.temporal_transition) if event.temporal_transition else "",
        ]
        phrase = " ".join(part for part in parts if part)
        if phrase:
            phrases.append(phrase + ".")
    for agent in scene.agents:
        details = agent.actions + agent.locations + agent.relations
        phrase = " ".join([_words(agent.type), *(_words(value) for value in details)])
        phrases.append(phrase + ".")
    if scene.ego.actions:
        phrases.append(
            "Ego vehicle "
            + ", ".join(_words(value) for value in scene.ego.actions)
            + "."
        )
    context = [
        scene.scene.road_type,
        scene.scene.traffic_state,
        scene.scene.weather,
        scene.scene.lighting,
        scene.scene.road_surface,
    ]
    context = [_words(value) for value in context if value]
    if context:
        phrases.append("Scene conditions: " + ", ".join(context) + ".")
    return " ".join(phrases).strip()
