"""One Euro filter (Casiez et al. 2012) for landmark smoothing, plus a per-hand wrapper.

Cutoff rises with speed:  cutoff = min_cutoff + beta * |filtered speed|
  - min_cutoff: lower = smoother when still (more lag)
  - beta:       higher = less lag when moving (more jitter)

Self-test:  python -m mirage.tracking.one_euro
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from .hand_tracker import Hand


def _alpha(cutoff: np.ndarray, dt: float) -> np.ndarray:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Elementwise One Euro filter over a numpy array of any shape (e.g. (21, 3))."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 5.0,
        d_cutoff: float = 1.0,
        max_gap_ms: int = 500,
    ) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("min_cutoff and d_cutoff must be > 0")
        if beta < 0:
            raise ValueError("beta must be >= 0")
        if max_gap_ms <= 0:
            raise ValueError("max_gap_ms must be > 0")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.max_gap_ms = int(max_gap_ms)
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None
        self._t: Optional[int] = None

    def reset(self) -> None:
        self._x = self._dx = self._t = None

    def __call__(self, x: np.ndarray, t_ms: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)

        if not np.all(np.isfinite(x)):
            # Bad sample: hold the last good output rather than poisoning the filter state.
            return (self._x if self._x is not None else np.zeros_like(x)).astype(np.float32)

        needs_init = (
            self._x is None
            or x.shape != self._x.shape
            or t_ms < self._t                      # clock went backwards
            or t_ms - self._t > self.max_gap_ms    # hand was gone, start fresh
        )
        if needs_init:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = int(t_ms)
            return x.astype(np.float32)

        if t_ms == self._t:                        # duplicate timestamp, avoid divide by zero
            return self._x.astype(np.float32)

        dt = (t_ms - self._t) / 1000.0

        dx_raw = (x - self._x) / dt
        a_d = _alpha(np.full_like(x, self.d_cutoff), dt)
        self._dx = a_d * dx_raw + (1.0 - a_d) * self._dx

        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        self._t = int(t_ms)
        return self._x.astype(np.float32)


class HandSmoother:
    """Keeps one pair of filters per handedness (image landmarks + world landmarks).

    world_beta is much larger than beta because world landmarks are in meters (tiny values),
    while image landmarks are in 0..1 frame units. Both are starting points to tune later.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 5.0,
        world_beta: float = 50.0,
        d_cutoff: float = 1.0,
        max_gap_ms: int = 500,
    ) -> None:
        self._img_kw = dict(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff, max_gap_ms=max_gap_ms)
        self._world_kw = dict(min_cutoff=min_cutoff, beta=world_beta, d_cutoff=d_cutoff, max_gap_ms=max_gap_ms)
        self._filters: Dict[str, Tuple[OneEuroFilter, OneEuroFilter]] = {}

    def reset(self) -> None:
        self._filters.clear()

    def update(self, hands: List[Hand], t_ms: int) -> List[Hand]:
        """Smooth each hand. Output order matches input order."""
        out: List[Hand] = []
        used: set[str] = set()
        for h in hands:
            if h.handedness in used:   # two hands with the same label: cannot track them separately
                out.append(h)
                continue
            used.add(h.handedness)
            pair = self._filters.get(h.handedness)
            if pair is None:
                pair = (OneEuroFilter(**self._img_kw), OneEuroFilter(**self._world_kw))
                self._filters[h.handedness] = pair
            out.append(
                replace(
                    h,
                    landmarks=pair[0](h.landmarks, t_ms),
                    world_landmarks=pair[1](h.world_landmarks, t_ms),
                )
            )
        return out


# ---- self-test ----------------------------------------------------------------
def _synthetic_checks() -> bool:
    results: List[bool] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

    rng = np.random.default_rng(0)
    ts = [int(i * 1000 / 30) for i in range(300)]  # 30 FPS timeline

    # 1. first sample passes through unchanged
    f = OneEuroFilter()
    first = np.full((21, 3), 0.5)
    check("first sample passes through", np.allclose(f(first, 0), first, atol=1e-6))

    # 2. stationary noise is reduced
    f = OneEuroFilter()
    raw, smooth = [], []
    for t in ts:
        x = 0.5 + rng.normal(0, 0.005, size=(21, 3))
        raw.append(x)
        smooth.append(f(x, t))
    raw_std = float(np.std(np.array(raw[30:]), axis=0).mean())
    sm_std = float(np.std(np.array(smooth[30:]), axis=0).mean())
    check("still hand: jitter cut by >40%", sm_std < 0.6 * raw_std, f"(raw {raw_std:.4f} -> {sm_std:.4f})")

    # 3. a fast ramp is followed with small lag
    f = OneEuroFilter()
    speed = 1.0  # frame units per second
    last_true = last_out = 0.0
    for t in ts[:60]:
        true = speed * t / 1000.0
        last_out = float(f(np.array([true]), t)[0])
        last_true = true
    lag = last_true - last_out
    check("fast motion: lag < 0.05 units", 0.0 <= lag < 0.05, f"(lag {lag:.4f})")

    # 4. long gap resets the filter
    f = OneEuroFilter(max_gap_ms=500)
    f(np.array([0.0]), 0)
    f(np.array([0.0]), 33)
    out = f(np.array([1.0]), 2033)
    check("gap > max_gap_ms resets", abs(float(out[0]) - 1.0) < 1e-6)

    # 5. NaN holds last output
    f = OneEuroFilter()
    f(np.array([0.2]), 0)
    prev = f(np.array([0.25]), 33)
    held = f(np.array([np.nan]), 66)
    check("NaN sample holds last output", np.allclose(held, prev))

    # 6. shape change resets without crashing
    f = OneEuroFilter()
    f(np.zeros((21, 3)), 0)
    try:
        out = f(np.zeros((5,)), 33)
        check("shape change resets", out.shape == (5,))
    except Exception as e:
        check("shape change resets", False, str(e))

    # 7. duplicate timestamp does not divide by zero
    f = OneEuroFilter()
    f(np.array([0.1]), 100)
    try:
        out = f(np.array([0.3]), 100)
        check("duplicate timestamp is safe", np.all(np.isfinite(out)))
    except Exception as e:
        check("duplicate timestamp is safe", False, str(e))

    # 8. bad parameters raise
    try:
        OneEuroFilter(min_cutoff=0)
        check("bad parameters raise", False)
    except ValueError:
        check("bad parameters raise", True)

    # 9. HandSmoother keeps order, labels and shapes
    def fake_hand(label: str, offset: float) -> Hand:
        pts = rng.random((21, 3)).astype(np.float32) + offset
        return Hand(handedness=label, score=0.9, landmarks=pts, world_landmarks=pts * 0.1)

    sm = HandSmoother()
    hands = [fake_hand("Right", 0.0), fake_hand("Left", 1.0)]
    out_hands = sm.update(hands, 0)
    out_hands = sm.update([fake_hand("Right", 0.0), fake_hand("Left", 1.0)], 33)
    ok = (
        [h.handedness for h in out_hands] == ["Right", "Left"]
        and all(h.landmarks.shape == (21, 3) and h.world_landmarks.shape == (21, 3) for h in out_hands)
    )
    check("HandSmoother keeps order, labels, shapes", ok)

    return all(results)


def _live_check(n_frames: int = 150) -> None:
    from .camera import Camera
    from .hand_tracker import HandTracker

    print("\nLive check (about 6 s): HOLD ONE HAND AS STILL AS YOU CAN, index finger visible.")
    raw_pts, smooth_pts = [], []
    with Camera() as cam, HandTracker() as tracker:
        smoother = HandSmoother()
        frames = 0
        while frames < n_frames:
            frame = cam.read()
            if frame is None:
                continue
            frames += 1
            hands = tracker.process(frame)
            smoothed = smoother.update(hands, frame.timestamp_ms)
            if hands:
                raw_pts.append(hands[0].landmarks[8, :2].copy())       # index fingertip
                smooth_pts.append(smoothed[0].landmarks[8, :2].copy())

    print(f"Frames with a hand: {len(raw_pts)}/{n_frames}")
    if len(raw_pts) < 40:
        print("Need at least 40 frames with a hand. Skipping.")
        return

    raw = np.array(raw_pts[10:]) * [640, 480]       # approx pixels at 640x480
    sm = np.array(smooth_pts[10:]) * [640, 480]
    raw_step = float(np.linalg.norm(np.diff(raw, axis=0), axis=1).mean())
    sm_step = float(np.linalg.norm(np.diff(sm, axis=0), axis=1).mean())
    cut = 100.0 * (1.0 - sm_step / raw_step) if raw_step > 0 else 0.0
    print(f"Frame-to-frame jitter (px/frame): raw {raw_step:.2f} -> smoothed {sm_step:.2f} ({cut:.0f}% less)")
    print(f"Slow drift (std, px, not filtered by design): raw {raw.std(axis=0).mean():.2f}, "
          f"smoothed {sm.std(axis=0).mean():.2f}")


if __name__ == "__main__":
    print("Synthetic checks:")
    if not _synthetic_checks():
        sys.exit(1)
    try:
        _live_check()
    except Exception as e:
        print(f"Live check skipped: {e}")