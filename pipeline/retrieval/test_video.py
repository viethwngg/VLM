import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Build the path independently of the current working directory. The source
# video is stored under the project's data directory.
VIDEO_PATH = Path(__file__).resolve().parents[2] / "data" / "scene-0061" / "cam_front.mp4"


# 1. Upload video
video = client.files.upload(
    file=VIDEO_PATH
)

print("Uploaded:", video.name)


# 2. Wait Gemini process video
while not video.state or video.state.name != "ACTIVE":
    print("Processing video...")

    time.sleep(2)

    video = client.files.get(
        name=video.name
    )


print("Video ACTIVE")


# 3. Ask Gemini to understand it
response = client.interactions.create(
    model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),

    input=[
        {
            "type": "video",
            "uri": video.uri,
            "mime_type": video.mime_type
        },

        {
            "type": "text",
            "text": """
You are the semantic scene understanding component of RAV-21,
a retrieval system for autonomous-driving datasets.

Your task is NOT to write a creative video caption.

Your task is to analyze temporally ordered driving-camera frames
and produce structured, retrieval-oriented semantic metadata.

The metadata will later be used for:
1. metadata filtering,
2. semantic embedding,
3. FAISS vector search,
4. hybrid retrieval,
5. autonomous-driving event benchmark.

Focus especially on concepts similar to ROAD++:
Agent + Action + Location + Temporal Event.

IMPORTANT RULES:

1. Only describe information visually supported by the supplied frames.
2. Do not hallucinate invisible objects, actions, weather or hazards.
3. If uncertain, omit the label rather than guessing.
4. Use ONLY labels from the provided controlled vocabularies when possible.
5. Do not invent new JSON fields.
6. Track temporal relationships across frames.
7. Distinguish an object from its action.
8. Distinguish an action from its location.
9. Capture meaningful relations between road agents.
10. Description must be concise but semantically dense.
11. searchable_text should be optimized for semantic retrieval, not storytelling.
12. Use natural-language synonyms in searchable_text while keeping canonical
    taxonomy labels in structured fields.

Analyze:

AGENTS:
pedestrian, car, cyclist, motorcycle, bus, truck,
large_vehicle, emergency_vehicle, other_vehicle

ACTIONS:
crossing, walking, moving, moving_towards, moving_away,
turning_left, turning_right, stopping, stopped, starting,
accelerating, decelerating, merging, overtaking,
lane_change_left, lane_change_right, waiting, standing,
entering_road, leaving_road

LOCATIONS:
road, vehicle_lane, ego_lane, left_lane, right_lane,
oncoming_lane, sidewalk, crosswalk, intersection,
junction, parking_area, bike_lane, shoulder

ROAD_CONTEXT:
urban, suburban, residential, highway, intersection,
t_intersection, four_way_intersection, roundabout,
straight_road, curved_road

TRAFFIC_CONDITION:
free_flow, light, moderate, heavy, congested, queue

WEATHER:
clear, sunny, cloudy, rain, heavy_rain, fog,
wet_road, dry_road

TIME_OF_DAY:
day, night, dawn, dusk

RELATIONS:
near, in_front_of, behind, approaching,
moving_away_from, crossing_in_front_of,
following, yielding_to

EVENTS:
pedestrian_crossing, vehicle_turning,
vehicle_merging, vehicle_overtaking,
lane_change, vehicle_stopping,
pedestrian_entering_road,
vehicle_pedestrian_interaction,
vehicle_vehicle_interaction

HAZARDS:
pedestrian_conflict, vehicle_conflict, cut_in,
sudden_stop, sudden_braking, obstacle_on_road,
near_collision

Return valid JSON only.

The searchable_text must contain:
- road/scene context,
- important agents,
- their actions,
- locations,
- temporal events,
- meaningful relations,
- relevant environmental conditions.

Do not include IDs, model names, timestamps or confidence values
inside searchable_text unless they have semantic meaning for retrieval.
"""
        }
    ]
)

print(response.output_text)
