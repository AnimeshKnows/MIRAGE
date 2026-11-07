"""MediaPipe hand landmark wrapper (pretrained model, not trained in this project).

Self-test:  python -m mirage.tracking.landmarks
"""
from __future__ import annotations

import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import List, Optional, Type

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .camera import Camera, CameraError, Frame

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"
NUM_LANDMARKS = 21


@dataclass(frozen=True)
class Hand:
    handedness: str            # "Left" or "Right" (the user's real hand, see note below)
    score: float               # handedness confidence, 0..1
    landmarks: np.ndarray      # (21, 3) float32: x, y in [0, 1] image coords, z relative depth
    world_landmarks: np.ndarray  # (21, 3) float32: metric coords in meters, origin near hand center


def ensure_model(path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the hand landmarker model if it is not on disk."""
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"Downloading hand landmarker model to {path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Model download failed: {e}") from e
    return path

# The Tasks API labels come out swapped for our mirrored (selfie) frames.
# Verified on this setup: raised right hand reported "Left" and vice versa.
_FLIP_HANDEDNESS = {"Left": "Right", "Right": "Left"}

def _to_array(points) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)


def _dedupe(hands: List[Hand], max_mean_dist: float = 0.08) -> List[Hand]:
    """Drop duplicate detections of the same physical hand, keeping the higher-confidence one.

    Two real hands differ by far more than 0.08 (normalized image units) on average
    across their 21 landmarks; ghost detections sit almost on top of each other.
    """
    kept: List[Hand] = []
    for h in sorted(hands, key=lambda x: x.score, reverse=True):
        is_dup = any(
            np.linalg.norm(h.landmarks[:, :2] - k.landmarks[:, :2], axis=1).mean() < max_mean_dist
            for k in kept
        )
        if not is_dup:
            kept.append(h)
    return kept


class HandTracker:
    """Wraps HandLandmarker in VIDEO mode (tracking between frames, lower latency than IMAGE mode).

        Handedness note: with Camera(mirror=True) the raw MediaPipe labels are inverted, so
        process() swaps them. Hand.handedness is the user's real hand. If you ever feed
        unmirrored frames, remove the swap.
        """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = ensure_model(Path(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker: Optional[vision.HandLandmarker] = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame: Frame) -> List[Hand]:
        if self._landmarker is None:
            raise RuntimeError("HandTracker is closed.")

        # MediaPipe requires strictly increasing timestamps in VIDEO mode.
        ts = frame.timestamp_ms
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(mp_image, ts)

        hands: List[Hand] = []
        for i, pts in enumerate(result.hand_landmarks):
            top = result.handedness[i][0]
            hands.append(
                Hand(
                    handedness=_FLIP_HANDEDNESS.get(top.category_name, top.category_name),
                    score=float(top.score),
                    landmarks=_to_array(pts),
                    world_landmarks=_to_array(result.hand_world_landmarks[i]),
                )
            )
        return _dedupe(hands)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()


def _self_test(n_frames: int = 150) -> int:
    try:
        with Camera() as cam, HandTracker() as tracker:
            print("Show your hand to the camera. Running", n_frames, "frames...")
            infer_ms: List[float] = []
            with_hand = 0
            frames = 0
            while frames < n_frames:
                frame = cam.read()
                if frame is None:
                    continue
                t = time.perf_counter()
                hands = tracker.process(frame)
                infer_ms.append((time.perf_counter() - t) * 1000)
                frames += 1
                if hands:
                    with_hand += 1
                if frames % 15 == 0:
                    desc = ", ".join(
                        f"{h.handedness}({h.score:.2f}) wrist=({h.landmarks[0, 0]:.2f},{h.landmarks[0, 1]:.2f})"
                        for h in hands
                    ) or "no hands"
                    print(f"frame {frame.index:4d}: {desc}")
            arr = np.array(infer_ms[5:] or infer_ms)  # skip warm-up frames
            print(
                f"Hands seen in {with_hand}/{frames} frames | inference "
                f"mean {arr.mean():.1f} ms, p95 {np.percentile(arr, 95):.1f} ms"
            )
        return 0
    except (CameraError, RuntimeError) as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(_self_test())