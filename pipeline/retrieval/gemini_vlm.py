"""Gemini adapter with bounded retries and local schema validation."""
import json, logging, math, os, random, time
from pathlib import Path
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from .prompts import PROMPT_VERSION, prompt_for_taxonomy
from .schemas import GeminiSceneOutput, SemanticScene
from .searchable_text import build_searchable_text

load_dotenv()
LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACK_MODELS = ("gemini-3.7-flash", "gemini-3.5-flash-lite")

class GeminiVLM:
    def __init__(self, client=None, model: str | None = None, max_retries: int | None = None, processing_timeout: float = 300,
                 retry_delay: float | None = None, fallback_models: list[str] | tuple[str, ...] | None = None):
        max_retries = int(os.getenv("GEMINI_MAX_ATTEMPTS", "5")) if max_retries is None else max_retries
        retry_delay = float(os.getenv("GEMINI_RETRY_DELAY_SECONDS", "10")) if retry_delay is None else retry_delay
        if max_retries < 1 or processing_timeout <= 0:
            raise ValueError("max_retries and processing_timeout must be positive")
        if not math.isfinite(retry_delay) or retry_delay <= 0:
            raise ValueError("retry_delay must be finite and positive")
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        if fallback_models is None:
            configured = os.getenv("GEMINI_FALLBACK_MODELS")
            fallback_models = DEFAULT_FALLBACK_MODELS if configured is None else configured.split(",")
        self.models = list(dict.fromkeys(
            candidate.strip() for candidate in (self.model, *fallback_models) if candidate.strip()
        ))
        self.client = client or genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.max_retries = max_retries
        self.processing_timeout = processing_timeout
        self.retry_delay = retry_delay

    def _retry_wait(self, attempt, scene_id):
        # Cap the exponent and delay; jitter spreads concurrent callers apart.
        base = min(60.0, self.retry_delay * (2 ** min(attempt, 10)))
        delay = min(60.0, base + random.uniform(0, base * 0.2))
        LOGGER.info("Retrying Gemini scene=%s in %.1fs", scene_id, delay)
        time.sleep(delay)

    def _file_request(self, operation, scene_id, **kwargs):
        """Retry interrupted transfers and transient API failures only."""
        for attempt in range(self.max_retries):
            try:
                return operation(**kwargs)
            except (httpx.TransportError, errors.APIError) as exc:
                if isinstance(exc, errors.APIError) and exc.code not in (408, 429, 500, 502, 503, 504):
                    raise
                LOGGER.warning("Gemini file request failed scene=%s attempt=%d/%d: %s",
                               scene_id, attempt + 1, self.max_retries, exc)
                if attempt + 1 == self.max_retries:
                    raise RuntimeError(f"Gemini file request failed for {scene_id} after {self.max_retries} attempts") from exc
                self._retry_wait(attempt, scene_id)

    def _wait_for_file(self, remote, scene_id):
        deadline = time.monotonic() + self.processing_timeout
        while True:
            state = getattr(remote, "state", None)
            state = getattr(state, "name", state)
            if state == "ACTIVE":
                return remote
            if state == "FAILED":
                raise RuntimeError(f"Gemini video processing failed for {scene_id}: {remote.name}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Gemini video processing timed out for {scene_id}: {remote.name}")
            LOGGER.info("Waiting for Gemini video processing scene=%s file=%s", scene_id, remote.name)
            time.sleep(min(2, remaining))
            remote = self._file_request(self.client.files.get, scene_id, name=remote.name)

    def analyze(self, scene_id: str, frames: list, evidence: list[dict] | None = None) -> SemanticScene:
        contents = [{"text": prompt_for_taxonomy() + f"\nScene ID: {scene_id}\nReturn JSON only."}]
        uploaded = set()
        for frame in frames:
            uri = str(frame.frame_uri)
            # Real Gemini clients need a Files API object for local media. Mock
            # clients used by tests may not expose ``files``; retain metadata
            # text in that case so the adapter remains independently testable.
            if hasattr(self.client, "files") and Path(uri).exists() and uri not in uploaded:
                remote = self._file_request(self.client.files.upload, scene_id, file=uri)
                remote = self._wait_for_file(remote, scene_id)
                contents.append({"file_data": {"file_uri": remote.uri, "mime_type": getattr(remote, "mime_type", "video/mp4")}})
                uploaded.add(uri)
            contents.append({"text": f"Frame camera={frame.camera}, timestamp_s={frame.timestamp_s}, uri={uri}"})
        last_error = None
        model_index = 0
        for attempt in range(self.max_retries):
            active_model = self.models[model_index]
            try:
                response = self.client.models.generate_content(model=active_model, contents=contents, config={
                    "response_mime_type": "application/json",
                    "response_schema": GeminiSceneOutput,
                    "automatic_function_calling": {"disable": True},
                })
                parsed = getattr(response, "parsed", None)
                if hasattr(parsed, "model_dump"):
                    data = parsed.model_dump()
                elif isinstance(parsed, dict):
                    data = dict(parsed)
                else:
                    raw = getattr(response, "text", None) or getattr(response, "output_text", None) or str(response)
                    data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
                data["scene_id"] = scene_id
                data["evidence"] = evidence or data.get("evidence", [])
                data["provenance"] = {"vlm_provider": "gemini", "vlm_model": active_model, "prompt_version": PROMPT_VERSION, "taxonomy_version": "road-v1", "pipeline_version": "semantic-pipeline-v1"}
                scene = SemanticScene.model_validate(data)
                scene.searchable_text = build_searchable_text(scene)
                return scene
            except Exception as exc:
                can_switch_model = (
                    isinstance(exc, errors.APIError)
                    and exc.code in (404, 503)
                    and len(self.models) > 1
                )
                if (isinstance(exc, errors.APIError)
                        and exc.code not in (408, 429, 500, 502, 503, 504)
                        and not can_switch_model):
                    raise RuntimeError(
                        f"Gemini failed for {scene_id} with model {active_model} "
                        f"(HTTP {exc.code}). Check GEMINI_MODEL and API access. {exc}"
                    ) from exc
                last_error = exc
                LOGGER.warning("VLM failure scene=%s model=%s attempt=%d/%d: %s",
                               scene_id, active_model, attempt + 1, self.max_retries, exc)
                if attempt + 1 < self.max_retries:
                    if can_switch_model:
                        previous_index = model_index
                        model_index = (model_index + 1) % len(self.models)
                        LOGGER.info("Switching Gemini model scene=%s from=%s to=%s",
                                    scene_id, active_model, self.models[model_index])
                        if model_index <= previous_index:
                            self._retry_wait(attempt // len(self.models), scene_id)
                    else:
                        self._retry_wait(attempt, scene_id)
        detail = ""
        if isinstance(last_error, errors.APIError) and last_error.code == 503:
            detail = " Gemini is temporarily unavailable (HTTP 503). Retry later or configure GEMINI_FALLBACK_MODELS."
        raise RuntimeError(
            f"Gemini failed for {scene_id} after {self.max_retries} attempts "
            f"using {', '.join(self.models)}.{detail}"
        ) from last_error
