"""Gemini adapter with bounded retries and local schema validation."""
import json, logging, os, time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from .prompts import PROMPT_VERSION, prompt_for_taxonomy
from .schemas import SemanticScene, Provenance
from .searchable_text import build_searchable_text

load_dotenv()
LOGGER = logging.getLogger(__name__)

class GeminiVLM:
    def __init__(self, client=None, model: str | None = None, max_retries: int = 3):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = client or genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.max_retries = max_retries

    def analyze(self, scene_id: str, frames: list, evidence: list[dict] | None = None) -> SemanticScene:
        contents = [{"text": prompt_for_taxonomy() + f"\nScene ID: {scene_id}\nReturn JSON only."}]
        for frame in frames:
            contents.append({"text": f"Frame camera={frame.camera}, timestamp_s={frame.timestamp_s}, uri={frame.frame_uri}"})
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(model=self.model, contents=contents, config={"response_mime_type": "application/json"})
                raw = getattr(response, "text", None) or getattr(response, "output_text", None) or str(response)
                data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
                data["scene_id"] = scene_id
                data["evidence"] = evidence or data.get("evidence", [])
                data["provenance"] = {"vlm_provider": "gemini", "vlm_model": self.model, "prompt_version": PROMPT_VERSION, "taxonomy_version": "road-v1", "pipeline_version": "semantic-pipeline-v1"}
                scene = SemanticScene.model_validate(data)
                scene.searchable_text = build_searchable_text(scene)
                return scene
            except Exception as exc:
                last_error = exc; LOGGER.warning("VLM failure scene=%s attempt=%d: %s", scene_id, attempt + 1, exc)
                if attempt + 1 < self.max_retries: time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini failed for {scene_id}") from last_error
