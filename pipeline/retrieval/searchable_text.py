"""Deterministic retrieval text generation from validated metadata."""
from .schemas import SemanticScene

def build_searchable_text(scene: SemanticScene) -> str:
    sentences = []
    context = ", ".join(scene.road_context)
    conditions = ", ".join(scene.weather + ([scene.time_of_day] if scene.time_of_day else []) + ([scene.traffic_condition] if scene.traffic_condition else []))
    if context: sentences.append(f"{context.replace('_', ' ')} driving scene.")
    if scene.description: sentences.append(scene.description.strip())
    for event in scene.events:
        parts = [event.agent, event.action, event.location]
        phrase = " ".join(x.replace("_", " ") for x in parts if x)
        if phrase: sentences.append(f"{phrase}.")
    for rel in scene.relations:
        sentences.append(f"{rel.subject} {rel.relation.replace('_', ' ')} {rel.object}.")
    if conditions: sentences.append(f"Conditions: {conditions.replace('_', ' ')}.")
    fields = [f"Agents: {', '.join(scene.agents)}.", f"Actions: {', '.join(scene.actions)}.", f"Locations: {', '.join(scene.locations)}.", f"Events: {', '.join(e.event for e in scene.events)}.", f"Relations: {', '.join(r.relation for r in scene.relations)}."]
    return " ".join(sentences + fields).strip()
