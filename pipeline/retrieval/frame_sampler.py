"""Temporally distributed camera frame sampling."""
from dataclasses import dataclass
from pathlib import Path
try:
    import cv2
except ImportError:  # Frame extraction is optional for corpus-only operations.
    cv2 = None

@dataclass(frozen=True)
class FrameSample:
    camera: str
    timestamp_s: float
    frame_uri: str

def sample_video(video_path: str | Path, camera: str = "CAM_FRONT", num_frames: int = 8) -> list[FrameSample]:
    path = Path(video_path)
    if cv2 is None:
        raise RuntimeError("opencv-python-headless is required to sample video frames")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise FileNotFoundError(f"Cannot open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = float(cap.get(cv2.CAP_PROP_FPS) or 1.0)
    if count <= 0: cap.release(); raise ValueError(f"Video has no frames: {path}")
    indices = sorted(set(round(i * (count - 1) / max(1, num_frames - 1)) for i in range(num_frames)))
    result = [FrameSample(camera, index / fps, str(path)) for index in indices]
    cap.release(); return result
