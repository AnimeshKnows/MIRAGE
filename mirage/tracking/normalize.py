"""Landmark normalization: turns one detected hand into a 63-float feature vector.

Steps: translate wrist to origin, scale so wrist->middle-MCP length is 1,
mirror left hands so every hand looks like a right hand.

Self-test:  python -m mirage.tracking.normalize
"""
from __future__ import annotations

import sys
from typing import Optional

import numpy as np

from .hand_tracker import Hand

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3  # 63

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9

_EPS = 1e-6


def normalize_landmarks(landmarks: np.ndarray, handedness: str) -> Optional[np.ndarray]:
    """Normalize a (21, 3) landmark array. Returns a (21, 3) float32 array, or None if unusable.

    Pass metric landmarks (Hand.world_landmarks). Image-space landmarks are stretched by the
    frame's aspect ratio and would need correcting first.
    """
    if handedness not in ("Left", "Right"):
        raise ValueError(f"handedness must be 'Left' or 'Right', got {handedness!r}")

    pts = np.asarray(landmarks, dtype=np.float32)
    if pts.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"landmarks must have shape (21, 3), got {pts.shape}")
    if not np.all(np.isfinite(pts)):
        return None

    pts = pts - pts[WRIST]                      # 1. translate: wrist at origin

    palm = float(np.linalg.norm(pts[MIDDLE_MCP]))
    if palm < _EPS:                             # degenerate detection, cannot scale
        return None
    pts = pts / palm                            # 2. scale: palm length = 1

    if handedness == "Left":                    # 3. mirror: left hand -> right-hand shape
        pts[:, 0] = -pts[:, 0]

    return pts


def hand_features(hand: Hand) -> Optional[np.ndarray]:
    """Flat (63,) float32 feature vector for the classifier, or None if the hand is unusable."""
    pts = normalize_landmarks(hand.world_landmarks, hand.handedness)
    return None if pts is None else pts.reshape(FEATURE_DIM)


def pinch_distance(normalized: np.ndarray) -> float:
    """Thumb tip to index tip distance in palm lengths. Handy sanity metric, not a classifier."""
    return float(np.linalg.norm(normalized[THUMB_TIP] - normalized[INDEX_TIP]))


# ---- self-test ----------------------------------------------------------------
def _synthetic_checks() -> bool:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(NUM_LANDMARKS, 3)).astype(np.float32)
    ref = normalize_landmarks(base, "Right")
    results = []

    def check(name: str, ok: bool) -> None:
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    check("wrist is at origin", np.allclose(ref[WRIST], 0, atol=1e-6))
    check("palm length is 1", abs(np.linalg.norm(ref[MIDDLE_MCP]) - 1.0) < 1e-5)
    check("translation invariant", np.allclose(normalize_landmarks(base + [5.0, -3.0, 2.0], "Right"), ref, atol=1e-5))
    check("scale invariant", np.allclose(normalize_landmarks(base * 3.7, "Right"), ref, atol=1e-5))

    mirrored = base.copy()
    mirrored[:, 0] *= -1
    check("left hand mirrors onto right", np.allclose(normalize_landmarks(mirrored, "Left"), ref, atol=1e-5))

    check("all-zero input returns None", normalize_landmarks(np.zeros((21, 3)), "Right") is None)
    bad = base.copy()
    bad[3, 1] = np.nan
    check("NaN input returns None", normalize_landmarks(bad, "Right") is None)

    try:
        normalize_landmarks(base, "Middle")
        check("bad handedness raises", False)
    except ValueError:
        check("bad handedness raises", True)
    return all(results)


def _live_check(n_frames: int = 200) -> None:
    from .camera import Camera
    from .hand_tracker import HandTracker

    print("\nLive check (about 8 s): raise a hand now, then open and close your pinch.")
    with Camera() as cam, HandTracker() as tracker:
        frames = 0
        seen = 0
        while frames < n_frames:
            frame = cam.read()
            if frame is None:
                continue
            frames += 1
            hands = tracker.process(frame)      # every frame, so tracking stays stable
            if hands:
                seen += 1
            if frames % 10:
                continue
            if not hands:
                print(f"frame {frame.index:4d}: no hands")
                continue
            for hand in hands:
                feats = hand_features(hand)
                if feats is None:
                    print(f"frame {frame.index:4d}: {hand.handedness} unusable")
                else:
                    pts = feats.reshape(NUM_LANDMARKS, 3)
                    print(f"frame {frame.index:4d}: {hand.handedness:5s} features={feats.shape} "
                          f"pinch={pinch_distance(pts):.2f}")
        print(f"Hands seen in {seen}/{frames} frames")

if __name__ == "__main__":
    print("Synthetic checks:")
    ok = _synthetic_checks()
    if not ok:
        sys.exit(1)
    try:
        _live_check()
    except Exception as e:  # camera or model problems should not hide the synthetic result
        print(f"Live check skipped: {e}")