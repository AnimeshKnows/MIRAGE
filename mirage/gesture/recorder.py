"""Gesture sample recorder: collects labeled landmark features from the webcam.

Used for both the base dataset (several people, several lighting conditions) and, later,
per-user personalization. Each recording is one .npz file:

    data/raw/<person>/<lighting>_<camera>/<gesture>_<YYYYmmdd-HHMMSS>.npz
        X          (N, 63) float32   normalized features (same pipeline as runtime, smoothed)
        handedness (N,)    str       "Left" / "Right" per sample
        meta       str (JSON)        person, lighting, camera, gesture, fps, timing

The folder is the person, so train/test splits are made per person, never per frame.
data/raw/ is in .gitignore: do not publish raw recordings of other people without consent.

Self-test:  python -m mirage.gesture.recorder --self-test
Record:     python -m mirage.gesture.recorder --person animesh --lighting dim
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .events import MODEL_CLASSES

FEATURE_DIM = 63
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"
RECORDABLE = tuple(g.value for g in MODEL_CLASSES)
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

PROMPTS = {
    "point": "Index finger out, other fingers curled. Move and tilt your hand a little between samples.",
    "pinch": "Thumb and index fingertips touching. Vary the angle and the other three fingers.",
    "fist": "Closed fist. Vary the angle and how tight it is.",
    "palm": "Open hand, fingers spread or relaxed. Vary the angle and distance.",
    "idle": "Resting hand: relaxed, half-open, in transition, partly out of frame. Do NOT hold a gesture.",
}


class SampleCollector:
    """Pure collection logic (no camera): accepts valid, time-spaced feature vectors."""

    def __init__(self, target: int = 20, min_interval_s: float = 0.2) -> None:
        if target < 1:
            raise ValueError("target must be >= 1")
        if min_interval_s < 0 or not math.isfinite(min_interval_s):
            raise ValueError("min_interval_s must be finite and >= 0")
        self.target = target
        self.min_interval_s = min_interval_s
        self.rejected = 0  # invalid features, not spacing skips
        self._x: List[np.ndarray] = []
        self._hand: List[str] = []
        self._last_t = -math.inf

    @property
    def count(self) -> int:
        return len(self._x)

    @property
    def done(self) -> bool:
        return self.count >= self.target

    def offer(self, t: float, features: Optional[np.ndarray], handedness: str = "Right") -> bool:
        """Offer one frame. Returns True if it was stored as a sample."""
        if self.done:
            return False
        if features is None or not math.isfinite(t):
            self.rejected += 1
            return False
        f = np.asarray(features, dtype=np.float32)
        if f.shape != (FEATURE_DIM,) or not np.all(np.isfinite(f)):
            self.rejected += 1
            return False
        if t < self._last_t:          # clock went backwards: do not stall forever
            self._last_t = -math.inf
        if t - self._last_t < self.min_interval_s - 1e-9:   # tolerance for float timestamps
            return False
        self._x.append(f.copy())
        self._hand.append(handedness)
        self._last_t = t
        return True

    def array(self) -> np.ndarray:
        if not self._x:
            return np.zeros((0, FEATURE_DIM), dtype=np.float32)
        return np.stack(self._x).astype(np.float32)

    def handedness(self) -> List[str]:
        return list(self._hand)


def _check_name(label: str, value: str) -> str:
    if not _NAME_RE.match(value):
        raise ValueError(f"{label} must be 1-32 chars of letters, digits, '_' or '-', got {value!r}")
    return value


def save_samples(
    root: Path,
    person: str,
    lighting: str,
    camera: str,
    gesture: str,
    X: np.ndarray,
    handedness: Sequence[str],
    fps: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write one recording and return its path. Never overwrites an existing file."""
    _check_name("person", person)
    _check_name("lighting", lighting)
    _check_name("camera", camera)
    if gesture not in RECORDABLE:
        raise ValueError(f"gesture must be one of {RECORDABLE}, got {gesture!r}")
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != FEATURE_DIM or X.shape[0] < 1:
        raise ValueError(f"X must have shape (N>=1, {FEATURE_DIM}), got {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or inf")
    if len(handedness) != X.shape[0]:
        raise ValueError("handedness length must match X")

    out_dir = Path(root) / person / f"{lighting}_{camera}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{gesture}_{stamp}.npz"
    n = 1
    while path.exists():
        path = out_dir / f"{gesture}_{stamp}-{n}.npz"
        n += 1

    meta = {
        "person": person, "lighting": lighting, "camera": camera, "gesture": gesture,
        "n_samples": int(X.shape[0]), "fps": round(float(fps), 2),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "feature_dim": FEATURE_DIM, "class_order": list(RECORDABLE),
    }
    if extra:
        meta.update(extra)

    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, X=X, handedness=np.array(list(handedness), dtype="U5"),
                        meta=np.array(json.dumps(meta)))
    tmp.replace(path)
    return path


def load_samples(path: Path) -> Dict[str, Any]:
    """Load one recording: {"X": (N,63) float32, "handedness": [str], "meta": dict}."""
    with np.load(Path(path), allow_pickle=False) as z:
        return {
            "X": z["X"].astype(np.float32),
            "handedness": [str(h) for h in z["handedness"]],
            "meta": json.loads(str(z["meta"])),
        }


# ---- live recording -----------------------------------------------------------
def _record_live(
    person: str, lighting: str, camera_pos: str, gestures: Sequence[str], samples: int,
    min_interval: float, countdown_s: float, timeout_s: float, min_score: float, root: Path,
) -> List[Path]:
    from ..tracking.camera import Camera
    from ..tracking.hand_tracker import HandTracker
    from ..tracking.normalize import hand_features
    from ..tracking.one_euro import HandSmoother

    saved: List[Path] = []
    with Camera() as cam, HandTracker() as tracker:
        smoother = HandSmoother()

        def step():
            frame = cam.read()
            if frame is None:
                return None
            return frame, smoother.update(tracker.process(frame), frame.timestamp_ms)

        for g in gestures:
            print(f"\n=== {g.upper()} ===\n{PROMPTS[g]}")
            input("Press Enter when ready... ")

            end = time.monotonic() + countdown_s      # keep tracking warm, discard frames
            shown = None
            while time.monotonic() < end:
                step()
                left = math.ceil(end - time.monotonic())
                if left != shown and left > 0:
                    print(f"  starting in {left}...")
                    shown = left

            collector = SampleCollector(samples, min_interval)
            t0 = time.monotonic()
            frames = no_hand = 0
            last_printed = 0
            while not collector.done and time.monotonic() - t0 < timeout_s:
                got = step()
                if got is None:
                    continue
                frame, hands = got
                frames += 1
                good = [h for h in hands if h.score >= min_score]
                if not good:
                    no_hand += 1
                    continue
                h = max(good, key=lambda x: x.score)
                collector.offer(frame.timestamp_ms / 1000.0, hand_features(h), h.handedness)
                if collector.count - last_printed >= 5 or collector.done:
                    last_printed = collector.count
                    print(f"  {collector.count}/{samples}")

            elapsed = max(time.monotonic() - t0, 1e-6)
            if not collector.done:
                print(f"  Timed out with {collector.count}/{samples} samples "
                      f"(no hand in {no_hand}/{frames} frames). Not saved; rerun with --gestures {g}.")
                continue
            path = save_samples(
                root, person, lighting, camera_pos, g, collector.array(), collector.handedness(),
                fps=frames / elapsed, extra={"rejected": collector.rejected, "frames": frames},
            )
            saved.append(path)
            print(f"  Saved {path}  ({frames / elapsed:.1f} FPS)")
    return saved


# ---- self-test ----------------------------------------------------------------
def _synthetic_checks() -> bool:
    import tempfile

    results: List[bool] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

    rng = np.random.default_rng(0)

    def feat() -> np.ndarray:
        return rng.normal(size=FEATURE_DIM).astype(np.float32)

    c = SampleCollector(target=5, min_interval_s=0.0)
    for i in range(20):
        c.offer(i / 30, feat())
    check("stops at target", c.count == 5 and c.done and c.array().shape == (5, FEATURE_DIM))

    c = SampleCollector(target=100, min_interval_s=0.2)
    acc = sum(c.offer(i / 10, feat()) for i in range(20))       # 10 FPS for 2 s
    check("spacing: 10 FPS with 0.2 s interval keeps every 2nd frame", acc == 10, f"(kept {acc})")

    c = SampleCollector(target=10, min_interval_s=0.0)
    c.offer(0.0, None)
    c.offer(0.1, np.full(FEATURE_DIM, np.nan))
    c.offer(0.2, np.zeros(10))
    c.offer(float("nan"), feat())
    ok_one = c.offer(0.3, feat())
    check("rejects None / NaN / wrong shape / bad time", c.rejected == 4 and ok_one and c.count == 1)

    c = SampleCollector(target=10, min_interval_s=1.0)
    c.offer(100.0, feat())
    check("backwards clock does not stall", c.offer(5.0, feat()) and c.count == 2)

    for bad in ({"target": 0}, {"min_interval_s": -1.0}):
        try:
            SampleCollector(**bad)
            check(f"bad args {bad} raise", False)
        except ValueError:
            check(f"bad args {bad} raise", True)

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        X = np.stack([feat() for _ in range(7)])
        hands = ["Right"] * 4 + ["Left"] * 3
        p1 = save_samples(root, "anna", "dim", "front", "pinch", X, hands, fps=9.8)
        loaded = load_samples(p1)
        check("save/load round trip", np.allclose(loaded["X"], X) and loaded["handedness"] == hands
              and loaded["meta"]["gesture"] == "pinch" and loaded["meta"]["n_samples"] == 7)
        check("path layout", p1.parent.name == "dim_front" and p1.parent.parent.name == "anna"
              and p1.name.startswith("pinch_"))
        p2 = save_samples(root, "anna", "dim", "front", "pinch", X, hands)
        check("never overwrites", p2 != p1 and p1.exists() and p2.exists())
        check("no temp files left", not list(root.rglob("*.tmp.npz")))

        for label, kwargs in {
            "path traversal person": dict(person="../x", lighting="dim", camera="front", gesture="pinch"),
            "bad gesture": dict(person="a", lighting="dim", camera="front", gesture="wave"),
            "two-hand gesture": dict(person="a", lighting="dim", camera="front", gesture="two_pinch"),
        }.items():
            try:
                save_samples(root, X=X, handedness=hands, **kwargs)
                check(f"rejects {label}", False)
            except ValueError:
                check(f"rejects {label}", True)
        try:
            save_samples(root, "a", "dim", "front", "fist", np.zeros((0, FEATURE_DIM)), [])
            check("rejects empty X", False)
        except ValueError:
            check("rejects empty X", True)
        try:
            save_samples(root, "a", "dim", "front", "fist", np.full((2, FEATURE_DIM), np.nan), ["Right"] * 2)
            check("rejects NaN X", False)
        except ValueError:
            check("rejects NaN X", True)
    return all(results)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Record labeled gesture samples from the webcam.")
    ap.add_argument("--person", help="who is being recorded (letters, digits, _ or -)")
    ap.add_argument("--lighting", help="e.g. bright, normal, dim")
    ap.add_argument("--camera", default="front", help="camera position label, e.g. front, above")
    ap.add_argument("--gestures", nargs="+", choices=RECORDABLE, default=list(RECORDABLE))
    ap.add_argument("--samples", type=int, default=20, help="samples per gesture (base dataset: 100+)")
    ap.add_argument("--min-interval", type=float, default=0.2, help="min seconds between samples")
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument("--timeout", type=float, default=30.0, help="seconds allowed per gesture")
    ap.add_argument("--min-score", type=float, default=0.8, help="min handedness confidence to accept")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        print("Synthetic checks:")
        return 0 if _synthetic_checks() else 1

    if not args.person or not args.lighting:
        ap.error("--person and --lighting are required (or use --self-test)")
    try:
        _check_name("person", args.person)
        _check_name("lighting", args.lighting)
        _check_name("camera", args.camera)
        SampleCollector(args.samples, args.min_interval)
    except ValueError as e:
        ap.error(str(e))

    try:
        saved = _record_live(args.person, args.lighting, args.camera, args.gestures, args.samples,
                             args.min_interval, args.countdown, args.timeout, args.min_score, args.root)
    except KeyboardInterrupt:
        print("\nStopped by user. Finished recordings are kept, the current gesture is discarded.")
        return 130
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    print(f"\nDone: {len(saved)}/{len(args.gestures)} gestures saved under {args.root / args.person}")
    return 0 if len(saved) == len(args.gestures) else 1


if __name__ == "__main__":
    sys.exit(main())