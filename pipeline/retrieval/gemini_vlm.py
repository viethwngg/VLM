"""Deprecated compatibility exports for the removed synchronous VLM adapter.

Semantic inference now runs exclusively through :mod:`gemini_batch`. Shared
VLM V3 helpers remain importable from their historical module path so callers
can migrate without prompt/schema drift.
"""

from .vlm_contract import (  # noqa: F401
    DEFAULT_MODEL,
    PIPELINE_VERSION,
    batch_generation_config,
    build_scene_parts,
    build_scene_prompt,
    parse_scene_payload,
    response_text,
)


class GeminiVLM:
    """Compatibility guard for the retired synchronous execution class."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Synchronous Gemini VLM inference was removed. "
            "Use GeminiBatchPipeline or scripts/run_semantic_pipeline.py."
        )
