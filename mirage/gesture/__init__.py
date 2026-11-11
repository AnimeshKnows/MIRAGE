"""MIRAGE gesture package: class probabilities -> stable gesture events.

Names are loaded lazily (PEP 562), so `import mirage.gesture` is instant and each
submodule is only imported when one of its names is first used.

Note: `Hand` here is the left/right/both side enum from events.py, not the tracked
hand data class in mirage.tracking.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, List

_EXPORTS = {
    "Gesture": ".events",
    "Hand": ".events",
    "EventType": ".events",
    "GestureEvent": ".events",
    "MODEL_CLASSES": ".events",
    "NUM_MODEL_CLASSES": ".events",
    "STATIC_GESTURES": ".events",
    "TWO_HAND_GESTURES": ".events",
    "DYNAMIC_GESTURES": ".events",
    "class_index": ".events",
    "gesture_from_index": ".events",
    "SwipeDetector": ".swipe",
    "GestureStateMachine": ".state_machine",
    "GestureModel": ".model",
    "RuleBasedGestureModel": ".model",
    "SampleCollector": ".recorder",
    "RECORDABLE": ".recorder",
    "save_samples": ".recorder",
    "load_samples": ".recorder",
}

__all__ = sorted(_EXPORTS)

if TYPE_CHECKING:  # for editors and type checkers only, never executed at runtime
    from .events import (
        DYNAMIC_GESTURES,
        MODEL_CLASSES,
        NUM_MODEL_CLASSES,
        STATIC_GESTURES,
        TWO_HAND_GESTURES,
        EventType,
        Gesture,
        GestureEvent,
        Hand,
        class_index,
        gesture_from_index,
    )
    from .model import GestureModel, RuleBasedGestureModel
    from .recorder import RECORDABLE, SampleCollector, load_samples, save_samples
    from .state_machine import GestureStateMachine
    from .swipe import SwipeDetector


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value  # cache so __getattr__ runs once per name
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))