"""Controlled taxonomy for RAV-21 road-scene semantic understanding.

Design goals:
- Retrieval-oriented rather than generic video captioning.
- Inspired by ROAD / ROAD++, HDD, DoTA and autonomous-driving datasets.
- Separate scene, agent, action, location, relation and event semantics.
- Avoid speculative intent or hidden causal reasoning.
"""

from __future__ import annotations

from typing import Final

TAXONOMY_VERSION = "road-v2"


# ============================================================
# GLOBAL SCENE CONTEXT
# ============================================================

ROAD_TYPES: Final[list[str]] = [
    "highway",
    "expressway",
    "urban_road",
    "residential_road",
    "rural_road",
    "intersection",
    "roundabout",
    "bridge",
    "tunnel",
    "parking_area",
    "service_road",
    "construction_area",
]

TRAFFIC_STATES: Final[list[str]] = [
    "free_flow",
    "light_traffic",
    "moderate_traffic",
    "dense_traffic",
    "traffic_congestion",
    "traffic_queue",
    "stop_and_go_traffic",
    "stopped_traffic",
]

WEATHER: Final[list[str]] = [
    "clear",
    "cloudy",
    "rain",
    "heavy_rain",
    "fog",
    "snow",
]

LIGHTING: Final[list[str]] = [
    "daylight",
    "dawn",
    "dusk",
    "night",
    "low_light",
]

ROAD_SURFACE: Final[list[str]] = [
    "dry",
    "wet",
    "flooded",
    "snow_covered",
    "partially_blocked",
]


# ============================================================
# ROAD AGENTS
# ============================================================

AGENT_TYPES: Final[list[str]] = [
    "pedestrian",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "cyclist",
    "emergency_vehicle",
    "construction_vehicle",
    "animal",
    "traffic_control",
    "road_obstacle",
    "unknown_vehicle",
]


# ============================================================
# AGENT ACTIONS
# ============================================================

AGENT_ACTIONS: Final[list[str]] = [
    # basic motion
    "moving",
    "stopped",
    "waiting",
    "approaching",
    "moving_away",

    # longitudinal
    "accelerating",
    "decelerating",
    "braking",

    # directional
    "going_straight",
    "turning_left",
    "turning_right",

    # lane behavior
    "changing_lane_left",
    "changing_lane_right",
    "merging",
    "overtaking",

    # road entry / exit
    "entering_road",
    "exiting_road",
    "entering_lane",
    "leaving_lane",

    # pedestrian / vulnerable road-user behavior
    "crossing",
    "waiting_to_cross",

    # other important behavior
    "yielding",
    "reversing",

    # visible vehicle signals
    "indicating_left",
    "indicating_right",
    "hazard_lights_on",
]


# ============================================================
# EGO VEHICLE ACTIONS
# ============================================================

EGO_ACTIONS: Final[list[str]] = [
    "moving",
    "stopped",

    "accelerating",
    "decelerating",
    "braking",

    "going_straight",
    "turning_left",
    "turning_right",

    "lane_change_left",
    "lane_change_right",

    "merging",
    "overtaking",

    "intersection_approach",
    "intersection_crossing",
    "crosswalk_approach",
    "crosswalk_passing",

    "highway_driving",
    "highway_entry",
    "highway_exit",
]


# ============================================================
# ROAD-RELATIVE LOCATIONS
# ============================================================

LOCATIONS: Final[list[str]] = [
    # relative to ego
    "ahead",
    "behind",
    "left",
    "right",

    # lane-level
    "ego_vehicle_lane",
    "adjacent_left_lane",
    "adjacent_right_lane",
    "incoming_lane",
    "opposite_lane",
    "entering_lane",

    # road-level
    "road_center",
    "roadside",
    "left_roadside",
    "right_roadside",
    "shoulder",
    "median",

    # semantic road regions
    "intersection",
    "junction",
    "crosswalk",
    "roundabout",
    "parking_area",
    "bus_stop",

    # generic relation to road
    "on_road",
    "near_road",
]


# ============================================================
# INTER-AGENT / EGO RELATIONS
# ============================================================

RELATIONS: Final[list[str]] = [
    # longitudinal
    "following_vehicle",
    "vehicle_ahead",
    "approaching_vehicle",
    "approaching_pedestrian",

    # ego-path relations
    "crossing_ego_path",
    "entering_ego_path",
    "leaving_ego_path",
    "occupying_ego_path",

    # lane interaction
    "entering_ego_lane",
    "leaving_ego_lane",
    "cutting_into_ego_lane",

    # vehicle interaction
    "merging_into_traffic",
    "overtaking_vehicle",
    "yielding_to_vehicle",
    "yielding_to_pedestrian",

    # crossing relations
    "pedestrian_crossing_vehicle_path",
    "cyclist_crossing_vehicle_path",
    "vehicle_crossing_vehicle_path",

    # proximity / trajectory
    "close_to_vehicle",
    "close_to_pedestrian",
    "converging_trajectories",
    "path_conflict",

    # road hazard relation
    "obstacle_in_travel_path",
    "blocked_travel_path",
]


# ============================================================
# SEMANTIC ROAD EVENTS
#
# These labels are especially important for retrieval.
# ============================================================

ROAD_EVENTS: Final[list[str]] = [
    # ---------------------
    # Pedestrian
    # ---------------------
    "pedestrian_crossing",
    "pedestrian_entering_road",
    "pedestrian_waiting_to_cross",
    "pedestrian_in_vehicle_path",

    # ---------------------
    # Cyclist / motorcycle
    # ---------------------
    "cyclist_crossing",
    "cyclist_on_road",
    "motorcycle_filtering",
    "motorcycle_lane_change",

    # ---------------------
    # Animal
    # ---------------------
    "animal_on_road",
    "animal_near_road",
    "animal_entering_road",

    # ---------------------
    # Traffic
    # ---------------------
    "traffic_congestion",
    "traffic_queue",
    "stop_and_go_traffic",
    "stopped_traffic",

    # ---------------------
    # Vehicle maneuver
    # ---------------------
    "vehicle_turning",
    "vehicle_lane_change",
    "vehicle_cut_in",
    "vehicle_merge",
    "vehicle_overtaking",
    "vehicle_reversing",

    # ---------------------
    # Highway
    # ---------------------
    "highway_driving",
    "highway_merge",
    "highway_entry",
    "highway_exit",

    # ---------------------
    # Intersection
    # ---------------------
    "intersection_approach",
    "intersection_crossing",
    "intersection_turn",
    "roundabout_navigation",

    # ---------------------
    # Road condition
    # ---------------------
    "blocked_lane",
    "road_obstacle",
    "road_debris",
    "road_work",
    "construction_zone",

    # ---------------------
    # Unusual road situation
    # ---------------------
    "stopped_vehicle_in_lane",
    "parked_vehicle_near_lane",
    "emergency_vehicle_present",
    "unexpected_object_on_road",
]


# ============================================================
# ATTENTION / SAFETY-RELEVANT EVENTS
#
# Keep these based on observable evidence.
# Do NOT infer driver intention or collision probability.
# ============================================================

ATTENTION_EVENTS: Final[list[str]] = [
    "pedestrian_crossing",
    "pedestrian_entering_road",
    "pedestrian_in_vehicle_path",

    "cyclist_crossing",
    "animal_on_road",
    "animal_entering_road",

    "vehicle_cut_in",
    "sudden_lane_change",
    "sudden_braking",
    "vehicle_reversing",

    "stopped_vehicle_in_lane",

    "blocked_lane",
    "road_obstacle",
    "road_debris",

    "crossing_vehicle",
    "merging_vehicle",

    "path_conflict",
    "converging_trajectories",
    "obstacle_in_travel_path",

    "road_work",
    "construction_zone",

    "emergency_vehicle_interaction",
]


# ============================================================
# TEMPORAL TRANSITIONS
#
# Useful for video understanding rather than static-frame captioning.
# ============================================================

TEMPORAL_TRANSITIONS: Final[list[str]] = [
    "moving_to_stopped",
    "stopped_to_moving",

    "moving_to_braking",
    "braking_to_stopped",

    "straight_to_turning_left",
    "straight_to_turning_right",

    "lane_keep_to_lane_change_left",
    "lane_keep_to_lane_change_right",

    "roadside_to_road",
    "roadside_to_crossing",

    "adjacent_lane_to_ego_lane",

    "outside_ego_path_to_crossing_ego_path",

    "free_flow_to_congestion",
    "moving_traffic_to_queue",

    "distant_to_close",
]


# ============================================================
# TAXONOMY EXPORT
# ============================================================

def taxonomy_values() -> dict[str, list[str]]:
    """Return complete controlled taxonomy for VLM prompt/schema."""

    return {
        "road_type": ROAD_TYPES,
        "traffic_state": TRAFFIC_STATES,
        "weather": WEATHER,
        "lighting": LIGHTING,
        "road_surface": ROAD_SURFACE,

        "agent_type": AGENT_TYPES,
        "agent_action": AGENT_ACTIONS,
        "ego_action": EGO_ACTIONS,

        "location": LOCATIONS,
        "relation": RELATIONS,

        "road_event": ROAD_EVENTS,
        "attention_event": ATTENTION_EVENTS,
        "temporal_transition": TEMPORAL_TRANSITIONS,
    }


TAXONOMY: Final[dict[str, list[str]]] = taxonomy_values()


def normalize_label(value: object) -> str:
    """Normalize a model label to the underscore-separated taxonomy form."""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def allowed(field: str, value: object) -> str | None:
    """Return a normalized label only when it belongs to the named taxonomy."""
    label = normalize_label(value)
    return label if label in TAXONOMY.get(field, ()) else None
