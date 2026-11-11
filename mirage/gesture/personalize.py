"""Per-user gesture personalization: class prototypes (nearest centroid) on an embedding.

The base model is never changed. A user records a few samples per gesture, we average
their embeddings into one prototype per class, and at runtime blend the base model's
probabilities with a softmax over cosine similarity to those prototypes.

Overfitting guard (plan section 5.3): part of the user's samples is held out. Personalization
is switched on only if it STRICTLY beats the base model on that held-out part; otherwise the
base model is used unchanged. After the check, prototypes are refit on all samples.

The embedder maps (N, 63) features to (N, D). The default is the identity (raw normalized
landmarks), so this works before the MLP exists; pass the MLP's penultimate layer later.

Self-test:  python -m mirage.gesture.personalize
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .events import MODEL_CLASSES, NUM_MODEL_CLASSES, Gesture, class_index
from .model import FEATURE_DIM, GestureModel

Embedder = Callable[[np.ndarray], np.ndarray]
_EPS = 1e-8


def identity_embedder(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float32)


def _l2(E: np.ndarray) -> np.ndarray:
    return E / np.maximum(np.linalg.norm(E, axis=-1, keepdims=True), _EPS)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def _rotations(angles: np.ndarray) -> np.ndarray:
    """(N, 3) radians about x, y, z -> (N, 3, 3) rotation matrices."""
    n = len(angles)
    cx, cy, cz = np.cos(angles).T
    sx, sy, sz = np.sin(angles).T
    rx, ry, rz = (np.zeros((n, 3, 3)) for _ in range(3))
    rx[:, 0, 0] = 1; rx[:, 1, 1] = cx; rx[:, 1, 2] = -sx; rx[:, 2, 1] = sx; rx[:, 2, 2] = cx
    ry[:, 0, 0] = cy; ry[:, 0, 2] = sy; ry[:, 1, 1] = 1; ry[:, 2, 0] = -sy; ry[:, 2, 2] = cy
    rz[:, 0, 0] = cz; rz[:, 0, 1] = -sz; rz[:, 1, 0] = sz; rz[:, 1, 1] = cz; rz[:, 2, 2] = 1
    return rz @ ry @ rx


def augment(
    X: np.ndarray, n_aug: int = 5, rng: Optional[np.random.Generator] = None,
    max_deg: float = 8.0, noise: float = 0.01,
) -> np.ndarray:
    """Original samples plus n_aug jittered copies (small rotation + noise, re-normalized)."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != FEATURE_DIM:
        raise ValueError(f"X must have shape (N, {FEATURE_DIM}), got {X.shape}")
    rng = rng if rng is not None else np.random.default_rng(0)
    out = [X]
    for _ in range(max(0, n_aug)):
        pts = X.reshape(-1, 21, 3).astype(np.float64)
        ang = np.deg2rad(rng.uniform(-max_deg, max_deg, size=(len(X), 3)))
        pts = np.einsum("nij,nkj->nki", _rotations(ang), pts)
        pts = pts + rng.normal(0, noise, size=pts.shape)
        pts = pts - pts[:, 0:1, :]                                   # wrist back to origin
        palm = np.maximum(np.linalg.norm(pts[:, 9, :], axis=1), _EPS)
        out.append((pts / palm[:, None, None]).reshape(-1, FEATURE_DIM).astype(np.float32))
    return np.concatenate(out)


@dataclass
class FitReport:
    enabled: bool
    reason: str
    base_acc: float = float("nan")
    personal_acc: float = float("nan")
    alpha: float = 0.0
    n_train: int = 0
    n_heldout: int = 0
    classes_with_prototype: List[str] = field(default_factory=list)
    acc_by_alpha: Dict[float, float] = field(default_factory=dict)

    def summary(self) -> str:
        head = "ON" if self.enabled else "OFF (base model)"
        s = f"personalization {head}: {self.reason}"
        if self.n_heldout:
            s += (f" | held-out {self.n_heldout}: base {self.base_acc:.2f}, "
                  f"personalized {self.personal_acc:.2f}, alpha {self.alpha:.2f}")
        return s


class PrototypePersonalizer:
    def __init__(
        self,
        base: GestureModel,
        embedder: Embedder = identity_embedder,
        temperature: float = 20.0,
        alphas: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
        min_per_class: int = 5,
        holdout_every: int = 4,
        n_aug: int = 5,
        seed: int = 0,
    ) -> None:
        if temperature <= 0 or min_per_class < 1 or holdout_every < 2:
            raise ValueError("temperature > 0, min_per_class >= 1, holdout_every >= 2 required")
        if not alphas or any(not 0.0 < a <= 1.0 for a in alphas):
            raise ValueError("alphas must be non-empty values in (0, 1]")
        self.base = base
        self.embedder = embedder
        self.temperature = temperature
        self.alphas = tuple(sorted(alphas))
        self.min_per_class = min_per_class
        self.holdout_every = holdout_every
        self.n_aug = n_aug
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        """Back to the plain base model."""
        self.enabled = False
        self.alpha = 0.0
        self.prototypes: Optional[np.ndarray] = None       # (C, D), L2-normalized rows
        self.has_prototype = np.zeros(NUM_MODEL_CLASSES, dtype=bool)

    # ----------------------------------------------------------------- inference
    def _proto_probs(self, E: np.ndarray, prototypes: np.ndarray, has: np.ndarray) -> np.ndarray:
        """(N, D) embeddings -> (N, C) probabilities from similarity to prototypes."""
        sims = _l2(E) @ prototypes.T                      # (N, C) cosine similarity
        sims = np.where(has[None, :], sims, -np.inf)
        out = np.zeros_like(sims)
        for i in range(len(sims)):
            z = self.temperature * sims[i]
            out[i] = _softmax(z)
        return out

    def predict(self, features: np.ndarray) -> np.ndarray:
        base_p = np.asarray(self.base.predict(features), dtype=np.float32)
        if not self.enabled or self.prototypes is None:
            return base_p
        try:
            f = np.asarray(features, dtype=np.float32)
        except (TypeError, ValueError):
            return base_p
        if f.size != FEATURE_DIM or not np.all(np.isfinite(f)):
            return base_p
        e = self.embedder(f.reshape(1, FEATURE_DIM))
        proto_p = self._proto_probs(e, self.prototypes, self.has_prototype)[0]
        p = (1.0 - self.alpha) * base_p + self.alpha * proto_p
        return (p / p.sum()).astype(np.float32)

    # ------------------------------------------------------------------- fitting
    def _build_prototypes(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        protos = np.zeros((NUM_MODEL_CLASSES, 1), dtype=np.float32)
        has = np.zeros(NUM_MODEL_CLASSES, dtype=bool)
        cols = None
        for c in range(NUM_MODEL_CLASSES):
            Xc = X[y == c]
            if len(Xc) < self.min_per_class:
                continue
            E = _l2(self.embedder(augment(Xc, self.n_aug, rng)))
            centroid = _l2(E.mean(axis=0, keepdims=True))[0]
            if cols is None:
                cols = centroid.shape[0]
                protos = np.zeros((NUM_MODEL_CLASSES, cols), dtype=np.float32)
            protos[c] = centroid
            has[c] = True
        return protos, has

    def fit(self, X: np.ndarray, y: Sequence[int]) -> FitReport:
        """Fit from the user's recorded samples. X (N, 63), y class indices (MODEL_CLASSES order)."""
        self.reset()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2 or X.shape[1] != FEATURE_DIM or len(X) != len(y):
            raise ValueError(f"X must be (N, {FEATURE_DIM}) with N labels, got {X.shape} and {y.shape}")
        if len(y) and (y.min() < 0 or y.max() >= NUM_MODEL_CLASSES):
            raise ValueError("labels must be class indices in MODEL_CLASSES order")
        ok = np.all(np.isfinite(X), axis=1)
        X, y = X[ok], y[ok]

        hold = np.zeros(len(y), dtype=bool)
        for c in range(NUM_MODEL_CLASSES):
            idx = np.flatnonzero(y == c)
            hold[idx[(np.arange(len(idx)) % self.holdout_every) == self.holdout_every - 1]] = True
        Xt, yt, Xh, yh = X[~hold], y[~hold], X[hold], y[hold]

        rng = np.random.default_rng(self.seed)
        protos, has = self._build_prototypes(Xt, yt, rng)
        rep = FitReport(False, "", n_train=len(yt), n_heldout=len(yh),
                        classes_with_prototype=[MODEL_CLASSES[c].value for c in np.flatnonzero(has)])
        if not has.any():
            rep.reason = f"no class has {self.min_per_class}+ training samples"
            return rep
        if len(yh) < 4:
            rep.reason = "too few samples to validate (record more per gesture)"
            return rep

        base_p = np.stack([np.asarray(self.base.predict(x), dtype=np.float64) for x in Xh])
        rep.base_acc = float(np.mean(base_p.argmax(1) == yh))
        proto_p = self._proto_probs(self.embedder(Xh), protos, has)
        best_acc, best_alpha = rep.base_acc, 0.0
        for a in self.alphas:                              # ascending: ties keep the smaller alpha
            acc = float(np.mean(((1 - a) * base_p + a * proto_p).argmax(1) == yh))
            rep.acc_by_alpha[a] = acc
            if acc > best_acc + 1e-9:
                best_acc, best_alpha = acc, a
        rep.personal_acc, rep.alpha = best_acc, best_alpha

        if best_alpha == 0.0:
            rep.reason = "did not beat the base model on held-out samples"
            return rep

        # Refit on everything (held-out included) now that the check passed.
        protos, has = self._build_prototypes(X, y, np.random.default_rng(self.seed))
        self.prototypes, self.has_prototype = protos, has
        self.alpha, self.enabled = best_alpha, True
        rep.enabled, rep.reason = True, "beat the base model on held-out samples"
        rep.classes_with_prototype = [MODEL_CLASSES[c].value for c in np.flatnonzero(has)]
        return rep

    # --------------------------------------------------------------- persistence
    def state(self) -> Dict[str, Any]:
        """Plain arrays and numbers, ready to be stored per user (see storage/profiles.py)."""
        return {
            "enabled": bool(self.enabled), "alpha": float(self.alpha),
            "temperature": float(self.temperature),
            "class_order": [g.value for g in MODEL_CLASSES],
            "has_prototype": self.has_prototype.copy(),
            "prototypes": None if self.prototypes is None else self.prototypes.copy(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        if list(state["class_order"]) != [g.value for g in MODEL_CLASSES]:
            raise ValueError("class order in saved state does not match this version")
        protos = state["prototypes"]
        has = np.asarray(state["has_prototype"], dtype=bool)
        if has.shape != (NUM_MODEL_CLASSES,):
            raise ValueError("bad has_prototype shape")
        if state["enabled"] and (protos is None or np.asarray(protos).shape[0] != NUM_MODEL_CLASSES):
            raise ValueError("enabled state needs one prototype row per class")
        self.enabled = bool(state["enabled"])
        self.alpha = float(state["alpha"])
        self.temperature = float(state["temperature"])
        self.has_prototype = has
        self.prototypes = None if protos is None else np.asarray(protos, dtype=np.float32)


# ---- self-test ----------------------------------------------------------------
def _synthetic_user(rng: np.random.Generator, quirks: Dict[str, tuple], n: int = 20):
    """Fake recordings of one user. Only the gestures named in `quirks` are generated."""
    from .model import _synthetic_hand

    X, y = [], []
    for g in MODEL_CLASSES:
        if g.value not in quirks:
            continue
        curl = quirks[g.value]
        for _ in range(n):
            c = tuple(float(np.clip(v + rng.normal(0, 0.05), 0, 1)) for v in curl)
            hand = _synthetic_hand(c, thumb_to_index=(g is Gesture.PINCH))
            X.append(hand + rng.normal(0, 0.01, size=FEATURE_DIM).astype(np.float32))
            y.append(class_index(g))
    return np.stack(X).astype(np.float32), np.array(y)


def _synthetic_checks() -> bool:
    from .model import RuleBasedGestureModel

    results: List[bool] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

    rng = np.random.default_rng(3)
    base = RuleBasedGestureModel()

    # A user whose relaxed palm reads as idle to the rule model and whose idle hand reads as a fist.
    quirky = {"point": (0.0, 0.8, 0.8, 0.8), "pinch": (0.0, 0.8, 0.8, 0.8),
              "fist": (0.95, 0.95, 0.95, 0.95), "palm": (0.38, 0.38, 0.38, 0.38), "idle": (0.62, 0.62, 0.62, 0.62)}
    X, y = _synthetic_user(rng, quirky)
    perm = rng.permutation(len(y))
    p = PrototypePersonalizer(base)
    rep = p.fit(X[perm], y[perm])
    print("   ", rep.summary())
    check("quirky user: personalization turns on and beats base",
          rep.enabled and rep.personal_acc > rep.base_acc, f"(base {rep.base_acc:.2f} -> {rep.personal_acc:.2f})")

    Xt, yt = _synthetic_user(np.random.default_rng(99), quirky, n=15)
    base_acc = np.mean([base.predict(x).argmax() == t for x, t in zip(Xt, yt)])
    pers_acc = np.mean([p.predict(x).argmax() == t for x, t in zip(Xt, yt)])
    check("unseen samples of the same user: personalized >= base", pers_acc >= base_acc,
          f"(base {base_acc:.2f}, personalized {pers_acc:.2f})")

    probs = p.predict(Xt[0])
    check("output is (5,), finite, sums to 1", probs.shape == (NUM_MODEL_CLASSES,)
          and np.isfinite(probs).all() and abs(float(probs.sum()) - 1) < 1e-5)
    check("NaN / wrong-size input falls back to base",
          p.predict(np.full(FEATURE_DIM, np.nan)).argmax() == class_index(Gesture.IDLE)
          and p.predict(np.zeros(10)).argmax() == class_index(Gesture.IDLE))

    clean = {"point": (0.0, 1.0, 1.0, 1.0), "pinch": (0.0, 1.0, 1.0, 1.0),
             "fist": (1.0, 1.0, 1.0, 1.0), "palm": (0.0, 0.0, 0.0, 0.0)}   # base is already ~perfect
    Xc, yc = _synthetic_user(np.random.default_rng(5), clean)
    p2 = PrototypePersonalizer(base)
    rep2 = p2.fit(Xc, yc)
    print("   ", rep2.summary())
    same = all(np.allclose(p2.predict(x), base.predict(x)) for x in Xc[::7])
    check("user the base already handles: stays OFF, predictions identical to base",
          not rep2.enabled and same and rep2.base_acc >= 0.99, f"(base {rep2.base_acc:.2f})")

    rep3 = PrototypePersonalizer(base).fit(X[:10], y[:10])
    check("too few samples: stays on base model", not rep3.enabled and rep3.reason != "", f"({rep3.reason})")

    st = p.state()
    p4 = PrototypePersonalizer(base)
    p4.load_state(st)
    check("state round trip gives identical predictions",
          all(np.allclose(p.predict(x), p4.predict(x)) for x in Xt[::9]))
    bad = dict(st, class_order=["a", "b", "c", "d", "e"])
    try:
        p4.load_state(bad)
        check("rejects state with different class order", False)
    except ValueError:
        check("rejects state with different class order", True)

    aug1 = augment(Xt[:6], 3, np.random.default_rng(1))
    aug2 = augment(Xt[:6], 3, np.random.default_rng(1))
    palm_len = np.linalg.norm(aug1.reshape(-1, 21, 3)[:, 9, :], axis=1)
    check("augment: shape, finite, seeded, palm length 1 on jittered copies",
          aug1.shape == (24, FEATURE_DIM) and np.isfinite(aug1).all() and np.allclose(aug1, aug2)
          and np.allclose(palm_len[6:], 1.0, atol=1e-4) and np.allclose(aug1[:6], Xt[:6]))

    for label, kw in {"temperature 0": dict(temperature=0), "empty alphas": dict(alphas=())}.items():
        try:
            PrototypePersonalizer(base, **kw)
            check(f"rejects {label}", False)
        except ValueError:
            check(f"rejects {label}", True)
    return all(results)


if __name__ == "__main__":
    print("Synthetic checks:")
    sys.exit(0 if _synthetic_checks() else 1)