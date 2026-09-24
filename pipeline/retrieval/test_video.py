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
    model="gemini-3.8-flash",

    input=[
        {
            "type": "video",
            "uri": video.uri,
            "mime_type": video.mime_type
        },

        {
            "type": "text",
            "text": """
Analyze this autonomous-driving scene.

Describe:

1. Traffic condition
2. Road context
3. Important objects
4. Events occurring in the scene
5. Possible hazards

Be factual.
Do not invent objects that cannot be observed.
"""
        }
    ]
)

print(response.output_text)
