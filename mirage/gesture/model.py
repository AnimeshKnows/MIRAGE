"""Rule-based base gesture model (stand-in until the trained MLP exists).

Input is the (63,) feature vector from tracking.normalize.hand_features: 21 landmarks,
wrist at the origin, palm length (wrist to middle MCP) = 1, left hands mirrored.
Output is a (5,) probability vector in MODEL_CLASSES order, so the trained MLP can
replace this class without touching the state machine.

Self-test:  python -m mirage.gesture.model
"""
from __future__ import annotations

import math
import sys
from typing import Dict, List, Protocol

import numpy as np

from .events import MODEL_CLASSES, NUM_MODEL_CLASSES, Gesture, class_index

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3

_WRIST = 0
_THUMB_TIP = 4
_INDEX_TIP = 8
# (pip, tip) landmark indices for index, middle, ring, pinky
_FINGERS = ((6, 8), (10, 12), (14, 16), (18, 20))
_EPS = 1e-6


class GestureModel(Protocol):
    def predict(self, features: np.ndarray) -> np.ndarray:
        """(63,) normalized features -> (5,) probabilities in MODEL_CLASSES order."""
        ...


def _sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, x))
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class RuleBasedGestureModel:
    """Soft rules on finger extension and thumb-index distance.

    A finger counts as extended when its tip is farther from the wrist than its PIP joint
    (works at any hand rotation). Each rule gives a 0..1 score, idle_bias is the score
    idle always has, and the five scores are normalized. Unclear poses therefore end up
    as idle instead of a weak guess.
    """

    def __init__(
        self,
        idle_bias: float = 0.15,
        ext_center: float = 1.0,
        ext_scale: float = 0.12,
        pinch_center: float = 0.4,
        pinch_scale: float = 0.1,
    ) -> None:
        if idle_bias <= 0 or ext_scale <= 0 or pinch_scale <= 0:
            raise ValueError("idle_bias, ext_scale and pinch_scale must be > 0")
        self.idle_bias = idle_bias
        self.ext_center = ext_center
        self.ext_scale = ext_scale
        self.pinch_center = pinch_center
        self.pinch_scale = pinch_scale

    @staticmethod
    def _idle() -> np.ndarray:
        out = np.zeros(NUM_MODEL_CLASSES, dtype=np.float32)
        out[class_index(Gesture.IDLE)] = 1.0
        return out

    def finger_extension(self, pts: np.ndarray) -> List[float]:
        """0..1 extension score for index, middle, ring, pinky."""
        scores = []
        for pip, tip in _FINGERS:
            pip_d = float(np.linalg.norm(pts[pip] - pts[_WRIST]))
            tip_d = float(np.linalg.norm(pts[tip] - pts[_WRIST]))
            ratio = tip_d / max(pip_d, _EPS)
            scores.append(_sigmoid((ratio - self.ext_center) / self.ext_scale))
        return scores

    def pinch_score(self, pts: np.ndarray) -> float:
        d = float(np.linalg.norm(pts[_THUMB_TIP] - pts[_INDEX_TIP]))
        return _sigmoid((self.pinch_center - d) / self.pinch_scale)

    def predict(self, features: np.ndarray) -> np.ndarray:
        try:
            f = np.asarray(features, dtype=np.float32)
        except (TypeError, ValueError):
            return self._idle()
        if f.size != FEATURE_DIM or not np.all(np.isfinite(f)):
            return self._idle()
        pts = f.reshape(NUM_LANDMARKS, 3)

        idx, mid, ring, pinky = self.finger_extension(pts)
        pinch = self.pinch_score(pts)
        apart = 1.0 - pinch

        scores = np.zeros(NUM_MODEL_CLASSES, dtype=np.float64)
        scores[class_index(Gesture.POINT)] = idx * (1 - mid) * (1 - ring) * (1 - pinky) * apart
        scores[class_index(Gesture.PINCH)] = pinch
        scores[class_index(Gesture.FIST)] = (1 - idx) * (1 - mid) * (1 - ring) * (1 - pinky) * apart
        scores[class_index(Gesture.PALM)] = idx * mid * ring * pinky * apart
        scores[class_index(Gesture.IDLE)] = self.idle_bias
        return (scores / scores.sum()).astype(np.float32)


# ---- self-test ----------------------------------------------------------------
_MCP = {
    5: (0.4, -0.95),    # index
    9: (0.0, -1.00),    # middle
    13: (-0.35, -0.95), # ring
    17: (-0.7, -0.85),  # pinky
}


def _synthetic_hand(curl=(0.0, 0.0, 0.0, 0.0), thumb_to_index: bool = False) -> np.ndarray:
    """Right-hand shape in normalized space. curl 0 = straight, 1 = fully curled."""
    pts = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    for mcp_i, c in zip((5, 9, 13, 17), curl):
        mx, my = _MCP[mcp_i]
        mcp = np.array([mx, my, 0.0])
        pts[mcp_i] = mcp
        pts[mcp_i + 1] = mcp + [0, -0.45, 0]                       # PIP
        ext_tip = np.array([0, -0.95, 0.0])
        cur_tip = np.array([0, 0.30, 0.30])
        tip_off = (1 - c) * ext_tip + c * cur_tip
        pts[mcp_i + 2] = mcp + [0, -0.45, 0] + (tip_off - [0, -0.45, 0]) * 0.5  # DIP
        pts[mcp_i + 3] = mcp + tip_off                              # tip
    pts[1], pts[2], pts[3] = (0.4, -0.3, 0), (0.65, -0.55, 0), (0.85, -0.8, 0)
    pts[4] = pts[8] + [0.05, 0.05, 0] if thumb_to_index else (1.0, -1.0, 0)
    return pts.reshape(FEATURE_DIM)


def _synthetic_checks() -> bool:
    model = RuleBasedGestureModel()
    results: List[bool] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

    def top(p: np.ndarray) -> Gesture:
        return MODEL_CLASSES[int(np.argmax(p))]

    cases: Dict[str, tuple] = {
        "open palm": (_synthetic_hand((0, 0, 0, 0)), Gesture.PALM),
        "fist": (_synthetic_hand((1, 1, 1, 1)), Gesture.FIST),
        "point": (_synthetic_hand((0, 1, 1, 1)), Gesture.POINT),
        "pinch": (_synthetic_hand((0, 1, 1, 1), thumb_to_index=True), Gesture.PINCH),
    }
    for name, (feat, want) in cases.items():
        p = model.predict(feat)
        check(f"{name} -> {want.value}", top(p) is want and p[class_index(want)] >= 0.7,
              f"(p={p[class_index(want)]:.2f})")

    half = model.predict(_synthetic_hand((0.5, 0.5, 0.5, 0.5)))
    check("half-curled hand is not confident", float(half.max()) < 0.7, f"(max p={half.max():.2f})")

    p = model.predict(np.zeros(FEATURE_DIM))
    check("all-zero input is safe", np.isfinite(p).all() and abs(float(p.sum()) - 1) < 1e-5)
    check("NaN input -> idle", top(model.predict(np.full(FEATURE_DIM, np.nan))) is Gesture.IDLE)
    check("wrong size -> idle", top(model.predict(np.zeros(10))) is Gesture.IDLE)
    check("output is (5,) and sums to 1",
          model.predict(cases["fist"][0]).shape == (NUM_MODEL_CLASSES,)
          and abs(float(model.predict(cases["fist"][0]).sum()) - 1) < 1e-5)
    return all(results)


def _live_check(n_frames: int = 300) -> None:
    """Camera -> tracker -> smoother -> features -> rule model -> state machine (+ swipe)."""
    from ..tracking.camera import Camera
    from ..tracking.hand_tracker import HandTracker
    from ..tracking.normalize import hand_features
    from ..tracking.one_euro import HandSmoother
    from .events import EventType
    from .events import Hand as Side
    from .state_machine import GestureStateMachine
    from .swipe import SwipeDetector

    print("\nLive check (about 10 s): try palm, fist, point, pinch, then swipe your open hand left.")
    model = RuleBasedGestureModel()
    machines: Dict[Side, GestureStateMachine] = {}
    swipes: Dict[Side, SwipeDetector] = {}
    missing_since: Dict[Side, float] = {}
    hand_lost_grace_s = 0.3
    names = ", ".join(g.value for g in MODEL_CLASSES)
    t = 0.0

    with Camera() as cam, HandTracker() as tracker:
        smoother = HandSmoother()
        frames = 0
        while frames < n_frames:
            frame = cam.read()
            if frame is None:
                continue
            frames += 1
            t = frame.timestamp_ms / 1000.0
            hands = smoother.update(tracker.process(frame), frame.timestamp_ms)
            seen = set()
            for h in hands:
                if h.handedness in seen:
                    continue
                seen.add(h.handedness)
                side = Side.LEFT if h.handedness == "Left" else Side.RIGHT
                feats = hand_features(h)
                if feats is None:
                    continue
                probs = model.predict(feats)
                sm = machines.setdefault(side, GestureStateMachine(side))
                sw = swipes.setdefault(side, SwipeDetector(side))
                for e in sm.update(t, probs):
                    if e.type is not EventType.HOLD:
                        print(f"{t:6.2f}s {side.value:5s} {e.type.value:6s} {e.gesture.value} "
                              f"conf={e.confidence:.2f}")
                is_open = probs[class_index(Gesture.PALM)] >= 0.5
                ev = sw.update(t, float(h.landmarks[0, 0]), float(h.landmarks[0, 1]), is_open)
                if ev is not None:
                    print(f"{t:6.2f}s {side.value:5s} {ev.type.value:6s} {ev.gesture.value} "
                          f"conf={ev.confidence:.2f}")
                if frames % 20 == 0:
                    vals = " ".join(f"{x:.2f}" for x in probs)
                    print(f"        probs [{names}] = [{vals}]")

            # A hand only counts as lost after a grace period, so one or two missed
            # frames (common during fast motion) do not end a gesture or clear swipe history.
            for side, sm in machines.items():
                label = "Left" if side is Side.LEFT else "Right"
                if label in seen:
                    t0 = missing_since.pop(side, None)
                    if t0 is not None and t - t0 >= 0.1:
                        print(f"        (hand was missing {t - t0:.2f}s)")
                    continue
                t0 = missing_since.setdefault(side, t)
                if t - t0 >= hand_lost_grace_s:
                    for e in sm.reset(t):
                        print(f"{t:6.2f}s {side.value:5s} {e.type.value:6s} {e.gesture.value} (hand lost)")
                    swipes[side].reset()

        print(f"Measured {frames / max(t, 1e-6):.1f} FPS over {t:.1f} s")

if __name__ == "__main__":
    print("Synthetic checks:")
    if not _synthetic_checks():
        sys.exit(1)
    try:
        _live_check()
    except Exception as e:  # camera or model problems should not hide the synthetic result
        print(f"Live check skipped: {e}")