"""Per-hand gesture state machine: model probabilities in, stable events out.

Entry is strict (high confidence held for enter_time_s), exit is lenient
(confidence must stay below a lower threshold for exit_time_s). IDLE never
emits events; it is simply the state where nothing is active.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

from .events import (
    MODEL_CLASSES,
    NUM_MODEL_CLASSES,
    EventType,
    Gesture,
    GestureEvent,
    Hand,
    class_index,
)


class GestureStateMachine:
    def __init__(
        self,
        hand: Hand,
        enter_threshold: float = 0.7,
        exit_threshold: float = 0.5,
        enter_time_s: float = 0.12,
        exit_time_s: float = 0.15,
        max_gap_s: float = 0.5,
    ) -> None:
        if hand is Hand.BOTH:
            raise ValueError("GestureStateMachine is single-hand (LEFT or RIGHT)")
        if not 0.0 < exit_threshold <= enter_threshold <= 1.0:
            raise ValueError("need 0 < exit_threshold <= enter_threshold <= 1")
        if enter_time_s < 0 or exit_time_s < 0:
            raise ValueError("enter_time_s and exit_time_s must be >= 0")
        if max_gap_s <= 0:
            raise ValueError("max_gap_s must be > 0")
        self.hand = hand
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.enter_time_s = enter_time_s
        self.exit_time_s = exit_time_s
        self.max_gap_s = max_gap_s

        self._active: Optional[Gesture] = None
        self._active_since = 0.0
        self._last_conf = 0.0
        self._below_since: Optional[float] = None
        self._candidate: Optional[Gesture] = None
        self._candidate_since = 0.0
        self._last_t: Optional[float] = None

    @property
    def active(self) -> Optional[Gesture]:
        return self._active

    # ------------------------------------------------------------------ public

    def update(self, timestamp: float, probs: Sequence[float]) -> List[GestureEvent]:
        """Feed one frame of class probabilities (order = MODEL_CLASSES).

        Frames with non-finite values or a timestamp that does not advance are
        ignored. A wrong-length probability vector raises ValueError.
        """
        if len(probs) != NUM_MODEL_CLASSES:
            raise ValueError(f"expected {NUM_MODEL_CLASSES} probabilities, got {len(probs)}")
        p = [float(x) for x in probs]
        if not math.isfinite(timestamp) or not all(math.isfinite(x) for x in p):
            return []
        p = [min(1.0, max(0.0, x)) for x in p]
        if self._last_t is not None and timestamp <= self._last_t:
            return []

        events: List[GestureEvent] = []

        # A long silence means the hand was lost: close anything open, start fresh.
        if self._last_t is not None and timestamp - self._last_t > self.max_gap_s:
            events += self._end_active(timestamp, self._last_t)
            self._candidate = None
        self._last_t = timestamp

        top = max(range(NUM_MODEL_CLASSES), key=p.__getitem__)
        top_g = MODEL_CLASSES[top]
        target = top_g if (top_g is not Gesture.IDLE and p[top] >= self.enter_threshold) else None

        if self._active is not None:
            conf = p[class_index(self._active)]
            if conf >= self.exit_threshold:
                self._below_since = None
                self._last_conf = conf
                events.append(self._event(EventType.HOLD, self._active, timestamp, conf,
                                          timestamp - self._active_since))
                return events
            if self._below_since is None:
                self._below_since = timestamp
            if timestamp - self._below_since < self.exit_time_s:
                # Grace period: still active, ride out the dip.
                events.append(self._event(EventType.HOLD, self._active, timestamp, conf,
                                          timestamp - self._active_since))
                return events
            events += self._end_active(timestamp, timestamp, conf)
            # fall through: a new gesture may start building on this same frame

        if target is None:
            self._candidate = None
            return events
        if target is not self._candidate:
            self._candidate = target
            self._candidate_since = timestamp
        if timestamp - self._candidate_since >= self.enter_time_s:
            conf = p[class_index(target)]
            self._active = target
            self._active_since = timestamp
            self._last_conf = conf
            self._below_since = None
            self._candidate = None
            events.append(self._event(EventType.START, target, timestamp, conf, 0.0))
        return events

    def reset(self, timestamp: Optional[float] = None) -> List[GestureEvent]:
        """Call when the hand is lost. Ends any active gesture and clears all state."""
        events: List[GestureEvent] = []
        if self._active is not None and self._last_t is not None:
            t = self._last_t
            if timestamp is not None and math.isfinite(timestamp):
                t = max(t, timestamp)
            events = self._end_active(t, t)
        self._active = None
        self._candidate = None
        self._below_since = None
        self._last_t = None
        return events

    # ----------------------------------------------------------------- private

    def _event(self, etype: EventType, gesture: Gesture, t: float,
               conf: float, duration: float) -> GestureEvent:
        return GestureEvent(
            type=etype,
            gesture=gesture,
            hand=self.hand,
            timestamp=t,
            confidence=min(1.0, max(0.0, conf)),
            duration=max(0.0, duration),
        )

    def _end_active(self, t: float, duration_until: float,
                    conf: Optional[float] = None) -> List[GestureEvent]:
        if self._active is None:
            return []
        ev = self._event(
            EventType.END,
            self._active,
            t,
            self._last_conf if conf is None else conf,
            duration_until - self._active_since,
        )
        self._active = None
        self._below_since = None
        return [ev]