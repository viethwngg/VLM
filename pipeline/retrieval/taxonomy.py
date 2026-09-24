"""Versioned controlled vocabulary used by the semantic pipeline."""

TAXONOMY_VERSION = "road-v1"

AGENTS = ("pedestrian", "car", "cyclist", "motorcycle", "bus", "truck", "large_vehicle", "emergency_vehicle", "other_vehicle")
ACTIONS = ("crossing", "walking", "moving", "moving_towards", "moving_away", "turning_left", "turning_right", "stopping", "stopped", "starting", "accelerating", "decelerating", "merging", "overtaking", "lane_change_left", "lane_change_right", "waiting", "standing", "entering_road", "leaving_road")
LOCATIONS = ("road", "vehicle_lane", "ego_lane", "left_lane", "right_lane", "oncoming_lane", "sidewalk", "crosswalk", "intersection", "junction", "parking_area", "bike_lane", "shoulder")
ROAD_CONTEXT = ("urban", "suburban", "residential", "highway", "intersection", "t_intersection", "four_way_intersection", "roundabout", "straight_road", "curved_road")
TRAFFIC_CONDITION = ("free_flow", "light", "moderate", "heavy", "congested", "queue")
WEATHER = ("clear", "sunny", "cloudy", "rain", "heavy_rain", "fog", "wet_road", "dry_road")
TIME_OF_DAY = ("day", "night", "dawn", "dusk")
RELATIONS = ("near", "in_front_of", "behind", "approaching", "moving_away_from", "crossing_in_front_of", "following", "yielding_to")
EVENTS = ("pedestrian_crossing", "vehicle_turning", "vehicle_merging", "vehicle_overtaking", "lane_change", "vehicle_stopping", "pedestrian_entering_road", "vehicle_pedestrian_interaction", "vehicle_vehicle_interaction")
HAZARDS = ("pedestrian_conflict", "vehicle_conflict", "cut_in", "sudden_stop", "sudden_braking", "obstacle_on_road", "near_collision")

TAXONOMY = {"agents": AGENTS, "actions": ACTIONS, "locations": LOCATIONS, "road_context": ROAD_CONTEXT, "traffic_condition": TRAFFIC_CONDITION, "weather": WEATHER, "time_of_day": TIME_OF_DAY, "relations": RELATIONS, "events": EVENTS, "hazards": HAZARDS}

def taxonomy_values() -> dict[str, list[str]]:
    return {key: list(values) for key, values in TAXONOMY.items()}

def normalize_label(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")

def allowed(field: str, value: object) -> str | None:
    label = normalize_label(value)
    return label if label in TAXONOMY[field] else None
