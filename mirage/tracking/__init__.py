"""MIRAGE tracking package: webcam -> hand landmarks -> normalized, smoothed features.

Names are loaded lazily (PEP 562), so `import mirage.tracking` is instant and does not pull in
MediaPipe until HandTracker, Hand or ensure_model is first used.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, List

_EXPORTS = {
    "Camera": ".camera",
    "CameraError": ".camera",
    "Frame": ".camera",
    "Hand": ".hand_tracker",
    "HandTracker": ".hand_tracker",
    "ensure_model": ".hand_tracker",
    "FEATURE_DIM": ".normalize",
    "normalize_landmarks": ".normalize",
    "hand_features": ".normalize",
    "pinch_distance": ".normalize",
    "OneEuroFilter": ".one_euro",
    "HandSmoother": ".one_euro",
}

__all__ = sorted(_EXPORTS)

if TYPE_CHECKING:  # for editors and type checkers only, never executed at runtime
    from .camera import Camera, CameraError, Frame
    from .hand_tracker import Hand, HandTracker, ensure_model
    from .normalize import FEATURE_DIM, hand_features, normalize_landmarks, pinch_distance
    from .one_euro import HandSmoother, OneEuroFilter


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value  # cache so __getattr__ runs once per name
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))