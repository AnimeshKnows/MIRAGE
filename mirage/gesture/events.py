"""Shared vocabulary for the gesture layer.

Everything else in gesture/ (and later the editor) imports from here, so this
file must not import anything from the rest of the project.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Gesture(str, Enum):
    POINT = "point"
    PINCH = "pinch"
    FIST = "fist"
    PALM = "palm"
    IDLE = "idle"
    TWO_PINCH = "two_pinch"
    TWO_ROTATE = "two_rotate"
    SWIPE_LEFT = "swipe_left"


class Hand(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


class EventType(str, Enum):
    START = "start"      # gesture became active (after debounce)
    HOLD = "hold"        # gesture still active, emitted each frame
    END = "end"          # gesture stopped (after hysteresis)
    TRIGGER = "trigger"  # one-shot gesture fired (swipe)


# Classes the per-frame model outputs, in output-index order. The model never
# predicts two-hand or swipe gestures; the state machine derives those.
MODEL_CLASSES: tuple[Gesture, ...] = (
    Gesture.POINT,
    Gesture.PINCH,
    Gesture.FIST,
    Gesture.PALM,
    Gesture.IDLE,
)
NUM_MODEL_CLASSES = len(MODEL_CLASSES)
_CLASS_TO_INDEX = {g: i for i, g in enumerate(MODEL_CLASSES)}

STATIC_GESTURES = frozenset(MODEL_CLASSES)
TWO_HAND_GESTURES = frozenset({Gesture.TWO_PINCH, Gesture.TWO_ROTATE})
DYNAMIC_GESTURES = frozenset({Gesture.SWIPE_LEFT})


def class_index(gesture: Gesture) -> int:
    """Model output index for a gesture. Raises ValueError if the model does not predict it."""
    try:
        return _CLASS_TO_INDEX[gesture]
    except KeyError:
        raise ValueError(f"{gesture.value} is not a model class") from None


def gesture_from_index(index: int) -> Gesture:
    """Gesture for a model output index. Raises IndexError if out of range."""
    if not 0 <= index < NUM_MODEL_CLASSES:
        raise IndexError(f"class index {index} out of range 0..{NUM_MODEL_CLASSES - 1}")
    return MODEL_CLASSES[index]


@dataclass(frozen=True)
class GestureEvent:
    type: EventType
    gesture: Gesture
    hand: Hand
    timestamp: float          # seconds, monotonic clock (time.perf_counter)
    confidence: float = 1.0   # 0..1
    duration: float = 0.0     # seconds the gesture has been active (HOLD/END)

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.duration < 0.0 or not math.isfinite(self.duration):
            raise ValueError("duration must be finite and >= 0")

        if self.gesture in DYNAMIC_GESTURES:
            if self.type is not EventType.TRIGGER:
                raise ValueError(f"{self.gesture.value} is one-shot and must use TRIGGER")
        elif self.type is EventType.TRIGGER:
            raise ValueError(f"{self.gesture.value} is not one-shot, cannot use TRIGGER")

        if self.gesture in TWO_HAND_GESTURES and self.hand is not Hand.BOTH:
            raise ValueError(f"{self.gesture.value} requires hand=BOTH")
        if self.gesture not in TWO_HAND_GESTURES and self.hand is Hand.BOTH:
            raise ValueError(f"{self.gesture.value} cannot use hand=BOTH")

    @property
    def is_active(self) -> bool:
        """True while the gesture is in progress (START or HOLD)."""
        return self.type in (EventType.START, EventType.HOLD)