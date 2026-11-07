"""Webcam capture for MIRAGE (console only, no preview window).

Self-test:  python -m mirage.tracking.camera
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Optional, Type

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    image: np.ndarray   # BGR, uint8, shape (H, W, 3)
    timestamp_ms: int   # monotonic, milliseconds since camera opened
    index: int          # 0-based frame counter


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(
        self,
        device: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        mirror: bool = True,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.mirror = mirror
        self._cap: Optional[cv2.VideoCapture] = None
        self._t0_ns = 0
        self._index = 0

    # ---- lifecycle -------------------------------------------------------
    def open(self) -> "Camera":
        if self._cap is not None:
            return self

        if sys.platform == "win32":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        last_reason = "device could not be opened"
        for backend in backends:
            cap = cv2.VideoCapture(self.device, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ignored by some backends

            ok, frame = cap.read()  # verify the stream actually delivers
            if ok and frame is not None:
                self._cap = cap
                self._t0_ns = time.monotonic_ns()
                self._index = 0
                return self

            last_reason = "device opened but returned no frames"
            cap.release()

        raise CameraError(
            f"Cannot use camera {self.device}: {last_reason}. "
            "Close other apps using the webcam, or try device=1."
        )

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.release()

    # ---- capture ---------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return self._cap is not None

    @property
    def resolution(self) -> tuple[int, int]:
        """Actual (width, height) the driver gave us, which may differ from the request."""
        if self._cap is None:
            raise CameraError("Camera is not open.")
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def read(self) -> Optional[Frame]:
        """Grab one frame. Returns None on a dropped frame; raises if not open."""
        if self._cap is None:
            raise CameraError("Camera is not open. Call open() or use 'with'.")

        ok, image = self._cap.read()
        if not ok or image is None:
            return None

        if self.mirror:
            image = cv2.flip(image, 1)  # selfie view: your right hand appears on the right

        ts_ms = (time.monotonic_ns() - self._t0_ns) // 1_000_000
        frame = Frame(image=image, timestamp_ms=int(ts_ms), index=self._index)
        self._index += 1
        return frame


def _self_test(n_frames: int = 100) -> int:
    try:
        with Camera() as cam:
            w, h = cam.resolution
            print(f"Camera opened: {w}x{h}")
            got = dropped = 0
            start = time.perf_counter()
            while got < n_frames:
                frame = cam.read()
                if frame is None:
                    dropped += 1
                    if dropped > n_frames:
                        print("Too many dropped frames, aborting.")
                        return 1
                    continue
                got += 1
            elapsed = time.perf_counter() - start
            print(f"{got} frames in {elapsed:.2f}s -> {got / elapsed:.1f} FPS, {dropped} dropped")
            print(f"Last frame: index={frame.index}, ts={frame.timestamp_ms} ms, shape={frame.image.shape}")
        return 0
    except CameraError as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(_self_test())