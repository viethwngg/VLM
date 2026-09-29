"""Resumable file-based Gemini Batch API pipeline for RAV-21 videos."""

from __future__ import annotations

import json
import logging
import math
import mimetypes
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from .frame_sampler import sample_video
from .gemini_vlm import (
    DEFAULT_MODEL,
    build_scene_parts,
    parse_scene_payload,
    response_text,
)
from .schemas import GeminiSceneOutput
from .semantic_pipeline import build_corpus, is_current_scene_artifact, write_scene_metadata

load_dotenv()
LOGGER = logging.getLogger(__name__)

ACTIVE_JOB_STATES = {
    "JOB_STATE_QUEUED",
    "JOB_STATE_PENDING",
    "JOB_STATE_RUNNING",
    "JOB_STATE_CANCELLING",
    "JOB_STATE_UPDATING",
}
SUCCESS_JOB_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
VIDEO_MIME_TYPES = {
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_name(value) -> str:
    if value is None:
        return "JOB_STATE_UNSPECIFIED"
    return str(getattr(value, "value", None) or getattr(value, "name", None) or value)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Ignoring unreadable JSON artifact: %s", path)
        return default


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def _file_state(remote) -> str:
    return _state_name(getattr(remote, "state", None)).removeprefix("FILE_STATE_")


def _mime_type(path: Path) -> str:
    guessed = mimetypes.guess_type(path.name)[0]
    return guessed or VIDEO_MIME_TYPES.get(path.suffix.lower(), "video/mp4")


def _retryable_exception(exc: Exception) -> bool:
    return isinstance(exc, httpx.TransportError) or (
        isinstance(exc, errors.APIError) and exc.code in RETRYABLE_HTTP_CODES
    )


def _error_retryable(error) -> bool:
    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, int):
            return code in RETRYABLE_HTTP_CODES
        text = json.dumps(error, ensure_ascii=False)
    else:
        text = str(error)
    return any(str(code) in text for code in RETRYABLE_HTTP_CODES) or any(
        token in text.lower() for token in ("timeout", "temporar", "overload", "unavailable", "rate limit")
    )


def is_scene_completed(scene_id: str, output_root: str | Path) -> bool:
    """Return true only for an existing, schema-valid artifact for this scene."""
    path = Path(output_root) / scene_id / "semantic_metadata.json"
    return is_current_scene_artifact(path, scene_id)


def make_batch_request(
    scene_id: str,
    frames: list,
    *,
    file_uri: str,
    mime_type: str,
) -> dict:
    """Create one documented file-based BatchGenerateContent JSONL row."""
    return {
        "key": scene_id,
        "request": {
            "contents": [{
                "role": "user",
                "parts": build_scene_parts(scene_id, frames, file_uri, mime_type),
            }],
            "generation_config": {
                "response_mime_type": "application/json",
                "response_json_schema": GeminiSceneOutput.model_json_schema(),
            },
        },
    }


def validate_batch_rows(rows: list[dict]) -> None:
    """Fail locally before a malformed or duplicate JSONL file is submitted."""
    keys: set[str] = set()
    for position, row in enumerate(rows, 1):
        # A serialization round trip also catches values JSON cannot represent.
        parsed = json.loads(json.dumps(row, allow_nan=False))
        key = parsed.get("key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Batch row {position} has no non-empty key")
        if key in keys:
            raise ValueError(f"Duplicate batch request key: {key}")
        keys.add(key)
        request = parsed.get("request") or {}
        contents = request.get("contents") or []
        parts = [part for content in contents for part in content.get("parts", [])]
        file_parts = [part.get("file_data") for part in parts if part.get("file_data")]
        text_parts = [part.get("text") for part in parts if part.get("text")]
        if len(file_parts) != 1 or not file_parts[0].get("file_uri"):
            raise ValueError(f"Batch row {key} must reference exactly one file_uri")
        if not str(file_parts[0].get("mime_type", "")).startswith("video/"):
            raise ValueError(f"Batch row {key} has invalid video mime_type")
        if not text_parts or not text_parts[0].strip():
            raise ValueError(f"Batch row {key} has an empty prompt")


class GeminiBatchPipeline:
    """Prepare, submit, inspect, and materialize resumable Gemini batch jobs."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        output_root: str | Path,
        batch_root: str | Path,
        camera: str = "CAM_FRONT",
        num_frames: int = 8,
        model: str | None = None,
        batch_size: int | None = None,
        client=None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        processing_timeout: float = 600,
    ):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.batch_root = Path(batch_root)
        self.camera = camera
        self.num_frames = num_frames
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.batch_size = batch_size or int(os.getenv("VLM_BATCH_SIZE", "100"))
        self.max_retries = max_retries or int(os.getenv("GEMINI_MAX_ATTEMPTS", "5"))
        self.retry_delay = retry_delay or float(os.getenv("GEMINI_RETRY_DELAY_SECONDS", "10"))
        self.processing_timeout = processing_timeout
        if self.batch_size < 1 or self.max_retries < 1 or self.processing_timeout <= 0:
            raise ValueError("batch_size, max_retries, and processing_timeout must be positive")
        if not math.isfinite(self.retry_delay) or self.retry_delay <= 0:
            raise ValueError("retry_delay must be finite and positive")
        api_key = os.getenv("GEMINI_API_KEY")
        if client is None and not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini batch processing")
        self.client = client or genai.Client(api_key=api_key)
        self.batch_root.mkdir(parents=True, exist_ok=True)
        self.upload_manifest_path = self.batch_root / "upload_manifest.json"
        self.prepare_manifest_path = self.batch_root / "prepare_manifest.json"
        self.plan_path = self.batch_root / "batch_plan.json"
        self.root_requests_path = self.batch_root / "batch_requests.jsonl"
        self.root_job_path = self.batch_root / "batch_job.json"
        self.failed_path = self.batch_root / "failed_requests.json"

    def _sleep(self, attempt: int, label: str) -> None:
        base = min(60.0, self.retry_delay * (2 ** min(attempt, 10)))
        delay = min(60.0, base + random.uniform(0, base * 0.2))
        LOGGER.info("[RETRY] %s in %.1fs", label, delay)
        time.sleep(delay)

    def _call(self, operation, label: str, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return operation(**kwargs)
            except Exception as exc:
                if not _retryable_exception(exc) or attempt + 1 == self.max_retries:
                    raise
                LOGGER.warning("[RETRY] %s attempt=%d/%d error=%s", label, attempt + 1, self.max_retries, exc)
                self._sleep(attempt, label)

    def wait_until_active(self, remote, key: str):
        """Wait for Files API processing with a bounded timeout."""
        deadline = time.monotonic() + self.processing_timeout
        while True:
            state = _file_state(remote)
            if state == "ACTIVE":
                LOGGER.info("[ACTIVE] %s file=%s", key, remote.name)
                return remote
            if state == "FAILED":
                raise RuntimeError(f"Gemini file processing failed for {key}: {remote.name}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Gemini file processing timed out for {key}: {remote.name}")
            LOGGER.info("[UPLOAD] %s state=%s", key, state)
            time.sleep(min(2.0, remaining))
            remote = self._call(self.client.files.get, f"poll {key}", name=remote.name)

    def _cached_remote(self, key: str, video_path: Path, entry: dict | None):
        if not entry or entry.get("local_path") != str(video_path.resolve()):
            return None
        stat = video_path.stat()
        if entry.get("size_bytes") != stat.st_size or entry.get("mtime_ns") != stat.st_mtime_ns:
            return None
        name = entry.get("file_name")
        if not name:
            return None
        try:
            remote = self._call(self.client.files.get, f"validate {key}", name=name)
        except errors.APIError as exc:
            if exc.code == 404:
                return None
            raise
        expiration = getattr(remote, "expiration_time", None)
        if expiration is not None and expiration <= datetime.now(timezone.utc):
            return None
        if _file_state(remote) == "FAILED":
            return None
        return self.wait_until_active(remote, key)

    def upload_video(self, key: str, video_path: str | Path, manifest: dict):
        """Reuse a valid uploaded file or upload this local video once."""
        path = Path(video_path)
        remote = self._cached_remote(key, path, manifest.get(key))
        if remote is None:
            LOGGER.info("[UPLOAD] %s path=%s", key, path)
            remote = self._call(
                self.client.files.upload,
                f"upload {key}",
                file=path,
                config=types.UploadFileConfig(
                    display_name=key[:512],
                    mime_type=_mime_type(path),
                ),
            )
            remote = self.wait_until_active(remote, key)
        stat = path.stat()
        manifest[key] = {
            "local_path": str(path.resolve()),
            "file_name": remote.name,
            "file_uri": remote.uri,
            "mime_type": getattr(remote, "mime_type", None) or _mime_type(path),
            "status": "ACTIVE",
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "expiration_time": (
                remote.expiration_time.isoformat()
                if getattr(remote, "expiration_time", None) else None
            ),
            "updated_at": utc_now(),
        }
        _write_json(self.upload_manifest_path, manifest)
        return manifest[key]

    def _submitted_keys(self, *, active_only: bool = False) -> set[str]:
        keys = set()
        for job_path in self.batch_root.glob("batch_*/batch_job.json"):
            job = _read_json(job_path, {})
            state = job.get("status")
            should_include = (
                state in ACTIVE_JOB_STATES
                if active_only
                else state not in {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
            )
            if job.get("job_name") and should_include:
                keys.update(job.get("request_keys", []))
        return keys

    def _next_batch_number(self) -> int:
        numbers = []
        for path in self.batch_root.glob("batch_[0-9][0-9][0-9][0-9]"):
            try:
                numbers.append(int(path.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
        return max(numbers, default=0) + 1

    def _create_plan(self, rows: list[dict], *, kind: str) -> dict:
        validate_batch_rows(rows)
        _write_jsonl(self.root_requests_path, rows)
        batches = []
        number = self._next_batch_number()
        for offset in range(0, len(rows), self.batch_size):
            chunk = rows[offset:offset + self.batch_size]
            directory = self.batch_root / f"batch_{number:04d}"
            request_path = directory / "batch_requests.jsonl"
            _write_jsonl(request_path, chunk)
            batches.append({
                "batch_id": directory.name,
                "directory": str(directory),
                "input_jsonl": str(request_path),
                "request_count": len(chunk),
                "request_keys": [row["key"] for row in chunk],
                "status": "PREPARED",
            })
            number += 1
        plan = {
            "kind": kind,
            "model": self.model,
            "batch_size": self.batch_size,
            "request_count": len(rows),
            "created_at": utc_now(),
            "batches": batches,
        }
        _write_json(self.plan_path, plan)
        return plan

    def prepare(self, jobs: list[tuple[str, Path]], *, limit: int | None = None, force: bool = False) -> dict:
        """Upload uncompleted videos and generate validated split JSONL files."""
        upload_manifest = _read_json(self.upload_manifest_path, {})
        prepare_manifest = _read_json(self.prepare_manifest_path, {})
        submitted = self._submitted_keys()
        candidates: list[tuple[str, Path]] = []
        completed = 0
        for scene_id, video_path in jobs:
            if not force and is_scene_completed(scene_id, self.output_root):
                completed += 1
                LOGGER.info("[SKIP] %s already completed", scene_id)
                continue
            if scene_id in submitted:
                LOGGER.info("[SKIP] %s already submitted", scene_id)
                continue
            candidates.append((scene_id, Path(video_path)))
            if limit is not None and len(candidates) >= limit:
                break

        rows = []
        upload_failed = 0
        for scene_id, video_path in candidates:
            try:
                upload = self.upload_video(scene_id, video_path, upload_manifest)
                frames = sample_video(video_path, self.camera, self.num_frames)
                evidence = [frame.__dict__ for frame in frames]
                row = make_batch_request(
                    scene_id,
                    frames,
                    file_uri=upload["file_uri"],
                    mime_type=upload["mime_type"],
                )
                rows.append(row)
                prepare_manifest[scene_id] = {
                    "scene_id": scene_id,
                    "camera": self.camera,
                    "video_path": str(video_path.resolve()),
                    "file_name": upload["file_name"],
                    "file_uri": upload["file_uri"],
                    "mime_type": upload["mime_type"],
                    "evidence": evidence,
                    "request": row,
                    "prepared_at": utc_now(),
                }
                LOGGER.info("[BATCH] added %s", scene_id)
            except Exception as exc:
                upload_failed += 1
                upload_manifest[scene_id] = {
                    "local_path": str(video_path.resolve()),
                    "status": "ERROR",
                    "error": str(exc),
                    "updated_at": utc_now(),
                }
                _write_json(self.upload_manifest_path, upload_manifest)
                LOGGER.error("[FAILED] %s upload/prepare error=%s", scene_id, exc)

        _write_json(self.prepare_manifest_path, prepare_manifest)
        plan = self._create_plan(rows, kind="initial")
        summary = {
            "total_videos_discovered": len(jobs),
            "already_completed": completed,
            "need_processing": len(candidates),
            "upload_successful": len(rows),
            "upload_failed": upload_failed,
            "batch_requests_generated": len(rows),
            "batch_count": len(plan["batches"]),
        }
        plan["summary"] = summary
        _write_json(self.plan_path, plan)
        return summary

    def _batch_job_from_sdk(self, job, batch: dict, input_file_name: str) -> dict:
        stats = getattr(job, "completion_stats", None)
        return {
            **batch,
            "job_name": job.name,
            "model": self.model,
            "created_at": utc_now(),
            "input_file_name": input_file_name,
            "status": _state_name(job.state),
            "successful": getattr(stats, "successful_count", None),
            "failed": getattr(stats, "failed_count", None),
            "incomplete": getattr(stats, "incomplete_count", None),
        }

    def _refresh_job(self, job_data: dict) -> tuple[object, dict]:
        job = self._call(
            self.client.batches.get,
            f"status {job_data['job_name']}",
            name=job_data["job_name"],
        )
        stats = getattr(job, "completion_stats", None)
        job_data.update({
            "status": _state_name(job.state),
            "successful": getattr(stats, "successful_count", None),
            "failed": getattr(stats, "failed_count", None),
            "incomplete": getattr(stats, "incomplete_count", None),
            "updated_at": utc_now(),
        })
        return job, job_data

    def _submit_batch(self, batch: dict) -> dict:
        directory = Path(batch["directory"])
        job_path = directory / "batch_job.json"
        job_data = _read_json(job_path, {})
        if job_data.get("job_name"):
            _, job_data = self._refresh_job(job_data)
            _write_json(job_path, job_data)
            if job_data["status"] in ACTIVE_JOB_STATES | SUCCESS_JOB_STATES:
                LOGGER.info("[SKIP] %s existing job=%s state=%s", batch["batch_id"], job_data["job_name"], job_data["status"])
                return job_data

        input_file_name = job_data.get("input_file_name")
        if input_file_name:
            try:
                self._call(self.client.files.get, f"validate {batch['batch_id']} input", name=input_file_name)
            except Exception:
                input_file_name = None
        if not input_file_name:
            uploaded = self._call(
                self.client.files.upload,
                f"upload {batch['batch_id']} JSONL",
                file=Path(batch["input_jsonl"]),
                config=types.UploadFileConfig(
                    display_name=f"rav21-{batch['batch_id']}",
                    mime_type="jsonl",
                ),
            )
            input_file_name = uploaded.name
            job_data = {
                **batch,
                "model": self.model,
                "input_file_name": input_file_name,
                "status": "INPUT_UPLOADED",
                "created_at": utc_now(),
            }
            _write_json(job_path, job_data)

        job = self._call(
            self.client.batches.create,
            f"submit {batch['batch_id']}",
            model=self.model,
            src=input_file_name,
            config={"display_name": f"rav21-{batch['batch_id']}"},
        )
        job_data = self._batch_job_from_sdk(job, batch, input_file_name)
        _write_json(job_path, job_data)
        LOGGER.info("[SUBMIT] %s", job.name)
        return job_data

    def submit(self) -> list[dict]:
        """Submit every prepared split, without duplicating an existing job."""
        plan = _read_json(self.plan_path, {})
        if not plan.get("batches"):
            LOGGER.info("[SUBMIT] no prepared batch requests")
            return []
        jobs = [self._submit_batch(batch) for batch in plan["batches"]]
        self._write_root_job(jobs)
        return jobs

    def _write_root_job(self, jobs: list[dict]) -> None:
        """Keep the stable root job manifest in sync with per-split state."""
        states = {job.get("status") for job in jobs}
        root = {
            "job_name": jobs[0].get("job_name") if len(jobs) == 1 else None,
            "model": self.model,
            "created_at": utc_now(),
            "request_count": sum(job.get("request_count", 0) for job in jobs),
            "input_jsonl": str(self.root_requests_path),
            "status": states.pop() if len(states) == 1 else "MULTIPLE",
            "batches": jobs,
        }
        _write_json(self.root_job_path, root)

    def _job_files(self) -> list[Path]:
        return sorted(self.batch_root.glob("batch_*/batch_job.json"))

    def status(self) -> list[dict]:
        """Refresh and persist state for every submitted batch job."""
        statuses = []
        for path in self._job_files():
            data = _read_json(path, {})
            if not data.get("job_name"):
                continue
            _, data = self._refresh_job(data)
            _write_json(path, data)
            statuses.append(data)
            LOGGER.info(
                "[STATUS] batch=%s state=%s requests=%s succeeded=%s failed=%s",
                data["job_name"], data["status"], data.get("request_count"),
                data.get("successful"), data.get("failed"),
            )
        if statuses:
            self._write_root_job(statuses)
        return statuses

    def _result_file_name(self, job) -> str | None:
        destination = getattr(job, "dest", None)
        if isinstance(destination, dict):
            return destination.get("file_name") or destination.get("fileName")
        return getattr(destination, "file_name", None)

    def _parse_result_line(self, line: dict, prepare_manifest: dict) -> tuple[str | None, dict | None]:
        key = line.get("key")
        if not key or key not in prepare_manifest:
            return key, {"key": key, "error": "Unknown or missing request key", "retryable": False}
        error = line.get("error")
        response = line.get("response")
        if error or not response:
            detail = error or "Batch result has no response"
            return key, {"key": key, "error": detail, "retryable": _error_retryable(detail)}
        try:
            prepared = prepare_manifest[key]
            scene = parse_scene_payload(
                response_text(response),
                scene_id=prepared["scene_id"],
                evidence=prepared["evidence"],
                model=self.model,
            )
            artifact = write_scene_metadata(scene, self.output_root)
            LOGGER.info("[RESULT] %s success artifact=%s", key, artifact)
            return key, None
        except Exception as exc:
            LOGGER.error("[FAILED] %s result parse error=%s", key, exc)
            return key, {"key": key, "error": str(exc), "retryable": True}

    def download(self) -> dict:
        """Download successful output JSONL files and materialize scene metadata."""
        prepare_manifest = _read_json(self.prepare_manifest_path, {})
        failures_by_key = {
            item.get("key"): item for item in _read_json(self.failed_path, []) if item.get("key")
        }
        succeeded = 0
        processed_jobs = 0
        latest_jobs = []
        for path in self._job_files():
            data = _read_json(path, {})
            if not data.get("job_name"):
                continue
            job, data = self._refresh_job(data)
            if data["status"] not in SUCCESS_JOB_STATES:
                _write_json(path, data)
                latest_jobs.append(data)
                LOGGER.info("[STATUS] %s not ready state=%s", data["job_name"], data["status"])
                continue
            result_name = self._result_file_name(job)
            if not result_name:
                data["download_error"] = "Completed batch has no destination file"
                _write_json(path, data)
                latest_jobs.append(data)
                LOGGER.error("[FAILED] %s has no result file", data["job_name"])
                continue
            raw_path = Path(data["directory"]) / "batch_results_raw.jsonl"
            downloaded = self._call(
                self.client.files.download,
                f"download {data['job_name']}",
                file=result_name,
                destination=raw_path,
            )
            if downloaded is not None:
                raw_path.write_bytes(downloaded)
            processed_jobs += 1
            for line_number, raw in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), 1):
                if not raw.strip():
                    continue
                try:
                    line = json.loads(raw)
                    key, failure = self._parse_result_line(line, prepare_manifest)
                except Exception as exc:
                    key, failure = None, {
                        "key": None,
                        "error": f"Invalid result JSONL line {line_number}: {exc}",
                        "retryable": False,
                    }
                if failure:
                    failures_by_key[key or f"{data['batch_id']}:{line_number}"] = failure
                elif key:
                    failures_by_key.pop(key, None)
                    succeeded += 1
            data.update({
                "result_file_name": result_name,
                "raw_result": str(raw_path),
                "downloaded_at": utc_now(),
            })
            _write_json(path, data)
            latest_jobs.append(data)

        failures = list(failures_by_key.values())
        _write_json(self.failed_path, failures)
        if latest_jobs:
            self._write_root_job(latest_jobs)
        if succeeded:
            build_corpus(self.output_root)
        # Required stable aggregate path, while each split retains its own raw file.
        aggregate = self.batch_root / "batch_results_raw.jsonl"
        raw_files = [Path(data.get("raw_result", "")) for data in map(lambda p: _read_json(p, {}), self._job_files())]
        aggregate.write_text(
            "".join(path.read_text(encoding="utf-8") for path in raw_files if path.is_file()),
            encoding="utf-8",
        )
        return {
            "jobs_downloaded": processed_jobs,
            "results_succeeded": succeeded,
            "results_failed": len(failures),
        }

    def retry_failed(self) -> list[dict]:
        """Create and submit new batches containing retryable failed keys only."""
        failures = _read_json(self.failed_path, [])
        prepared = _read_json(self.prepare_manifest_path, {})
        submitted = self._submitted_keys(active_only=True)
        rows = []
        for failure in failures:
            key = failure.get("key")
            if not failure.get("retryable") or key not in prepared:
                continue
            if is_scene_completed(key, self.output_root) or key in submitted:
                LOGGER.info("[SKIP] %s already completed/submitted", key)
                continue
            rows.append(prepared[key]["request"])
        if not rows:
            LOGGER.info("[BATCH] no retryable failed requests")
            return []
        self._create_plan(rows, kind="retry-failed")
        return self.submit()

