"""Swipe-left detection from wrist motion. No model involved.

One SwipeDetector per hand. Feed it the smoothed wrist position in image
coordinates (0..1, x already in display/mirrored space so left = smaller x),
NOT the wrist-relative normalized landmarks, which are always zero at the wrist.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

from .events import EventType, Gesture, GestureEvent, Hand

_Sample = Tuple[float, float, float]  # (timestamp, x, y)


class SwipeDetector:
    def __init__(
        self,
        hand: Hand,
        window_s: float = 0.4,
        min_distance: float = 0.25,       # fraction of frame width
        min_speed: float = 0.8,           # frame widths per second
        max_vertical_ratio: float = 0.6,  # vertical drift allowed relative to travel
        cooldown_s: float = 0.8,
        min_samples: int = 3,
    ) -> None:
        if hand is Hand.BOTH:
            raise ValueError("SwipeDetector needs a single hand (LEFT or RIGHT)")
        if min(window_s, min_distance, min_speed, max_vertical_ratio) <= 0:
            raise ValueError("window_s, min_distance, min_speed, max_vertical_ratio must be > 0")
        if cooldown_s < 0 or min_samples < 2:
            raise ValueError("cooldown_s must be >= 0 and min_samples >= 2")
        self.hand = hand
        self.window_s = window_s
        self.min_distance = min_distance
        self.min_speed = min_speed
        self.max_vertical_ratio = max_vertical_ratio
        self.cooldown_s = cooldown_s
        self.min_samples = min_samples
        self._samples: Deque[_Sample] = deque()
        self._cooldown_until = -math.inf

    def reset(self) -> None:
        """Call when the hand is lost. Keeps the cooldown so a flicker cannot double fire."""
        self._samples.clear()

    def update(
        self, timestamp: float, x: float, y: float, is_open: bool = True
    ) -> Optional[GestureEvent]:
        """Feed one frame. Returns a TRIGGER event when a swipe completes, else None."""
        if not (math.isfinite(timestamp) and math.isfinite(x) and math.isfinite(y)):
            return None

        # Out-of-order or repeated timestamp: history is unreliable, start over.
        if self._samples and timestamp <= self._samples[-1][0]:
            self._samples.clear()

        # Cooldown, or hand not in an open pose: no swipe possible, drop history.
        if timestamp < self._cooldown_until or not is_open:
            self._samples.clear()
            return None

        self._samples.append((timestamp, x, y))
        while self._samples and timestamp - self._samples[0][0] > self.window_s:
            self._samples.popleft()

        n = len(self._samples)
        if n < self.min_samples:
            return None

        # Rightmost point in the window (latest one on ties) is where the swipe started.
        peak_i = max(range(n), key=lambda i: (self._samples[i][1], i))
        peak_t, peak_x, _ = self._samples[peak_i]
        dt = timestamp - peak_t
        travel = peak_x - x  # positive = moved left
        if dt <= 0 or travel < self.min_distance:
            return None
        speed = travel / dt
        if speed < self.min_speed:
            return None

        ys = [s[2] for s in list(self._samples)[peak_i:]]
        if max(ys) - min(ys) > self.max_vertical_ratio * travel:
            return None

        self._samples.clear()
        self._cooldown_until = timestamp + self.cooldown_s
        confidence = min(1.0, speed / (2.0 * self.min_speed))
        return GestureEvent(
            type=EventType.TRIGGER,
            gesture=Gesture.SWIPE_LEFT,
            hand=self.hand,
            timestamp=timestamp,
            confidence=confidence,
        )