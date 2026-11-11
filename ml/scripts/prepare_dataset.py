"""Build the training dataset from raw recordings, split by PERSON (never by frame).

Reads   data/raw/<person>/<lighting>_<camera>/<gesture>_<timestamp>.npz   (see mirage.gesture.recorder)
Writes  data/processed/dataset.npz  with X, y, person, lighting, camera, handedness, fps, split,
        class_names, meta. Training scripts can also do leave-one-person-out from the `person` array.

Run from the project root:
    python -m ml.scripts.prepare_dataset
    python -m ml.scripts.prepare_dataset --test-people anna --val-people ben
    python -m ml.scripts.prepare_dataset --self-test
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mirage.gesture.events import MODEL_CLASSES, Gesture, class_index
from mirage.gesture.recorder import FEATURE_DIM, RECORDABLE, load_samples, save_samples

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "data" / "raw"
DEFAULT_OUT = ROOT / "data" / "processed" / "dataset.npz"
CLASS_NAMES = [g.value for g in MODEL_CLASSES]
SPLITS = ("train", "val", "test")


# ---- collect ------------------------------------------------------------------
def collect(raw: Path) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Load every valid recording under raw. Bad files are skipped with a warning."""
    warns: List[str] = []
    xs, ys, people, lights, cams, hands, fps = [], [], [], [], [], [], []
    files = sorted(p for p in Path(raw).rglob("*.npz") if not p.name.endswith(".tmp.npz"))
    for p in files:
        try:
            rec = load_samples(p)
            meta, X = rec["meta"], rec["X"]
            gesture = meta["gesture"]
            if gesture not in RECORDABLE:
                raise ValueError(f"unknown gesture {gesture!r}")
            if X.ndim != 2 or X.shape[1] != FEATURE_DIM or len(X) == 0:
                raise ValueError(f"bad X shape {X.shape}")
            if not np.all(np.isfinite(X)):
                raise ValueError("X has NaN or inf")
            if len(rec["handedness"]) != len(X):
                raise ValueError("handedness length does not match X")
            person, lighting, camera = str(meta["person"]), str(meta["lighting"]), str(meta["camera"])
        except Exception as e:  # one bad file must not kill the whole build
            warns.append(f"skipped {p}: {e}")
            continue
        n = len(X)
        xs.append(X)
        ys.append(np.full(n, class_index(Gesture(gesture)), dtype=np.int64))
        people += [person] * n
        lights += [lighting] * n
        cams += [camera] * n
        hands += rec["handedness"]
        fps += [float(meta.get("fps", 0.0))] * n

    if not xs:
        return {}, warns
    return {
        "X": np.concatenate(xs).astype(np.float32),
        "y": np.concatenate(ys),
        "person": np.array(people, dtype="U32"),
        "lighting": np.array(lights, dtype="U32"),
        "camera": np.array(cams, dtype="U32"),
        "handedness": np.array(hands, dtype="U5"),
        "fps": np.array(fps, dtype=np.float32),
    }, warns


# ---- split --------------------------------------------------------------------
def assign_splits(
    people: Sequence[str],
    test_people: Optional[Sequence[str]] = None,
    val_people: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> Tuple[Dict[str, str], List[str]]:
    """Assign whole people to train/val/test. Returns ({person: split}, warnings)."""
    people = sorted({str(p) for p in people})
    warns: List[str] = []
    test, val = list(test_people or []), list(val_people or [])
    for name in test + val:
        if name not in people:
            raise ValueError(f"unknown person {name!r}; recordings exist for {people}")
    if set(test) & set(val):
        raise ValueError("a person cannot be in both test and val")

    if not test and not val:
        n = len(people)
        order = [str(p) for p in np.random.default_rng(seed).permutation(people)]
        if n >= 4:
            n_test, n_val = max(1, round(0.2 * n)), max(1, round(0.15 * n))
        elif n >= 2:
            n_test, n_val = 1, 0
        else:
            n_test = n_val = 0
        test, val = order[:n_test], order[n_test:n_test + n_val]
        if n == 1:
            warns.append("only one person recorded: everything is train, so there is NO unseen-person "
                         "test set. Do not report accuracy from this split.")
        elif n_val == 0:
            warns.append("fewer than 4 people: no validation split, tune with leave-one-person-out instead.")

    split = {p: "train" for p in people}
    split.update({p: "test" for p in test})
    split.update({p: "val" for p in val})
    if len(test) == 1 and len(people) > 1:
        warns.append("test set is a single person: accuracy will be noisy. Plan: you plus at least 5 others.")
    return split, warns


# ---- report -------------------------------------------------------------------
def report(data: Dict[str, np.ndarray], split_of: Dict[str, str]) -> List[str]:
    """Print counts, return warnings about missing or imbalanced classes."""
    warns: List[str] = []
    y, person = data["y"], data["person"]
    print(f"\nTotal: {len(y)} samples, {len(set(person))} people, classes {CLASS_NAMES}")

    print("\nPer person (person, split, lighting | counts per class):")
    for p in sorted(set(person)):
        m = person == p
        counts = np.bincount(y[m], minlength=len(CLASS_NAMES))
        light = ",".join(sorted(set(data["lighting"][m])))
        print(f"  {p:14s} {split_of[str(p)]:5s} {light:22s} " + " ".join(
            f"{n}:{c:4d}" for n, c in zip(CLASS_NAMES, counts)))

    print("\nPer split:")
    row_split = np.array([split_of[str(p)] for p in person])
    for s in SPLITS:
        m = row_split == s
        if not m.any():
            print(f"  {s:5s} empty")
            continue
        counts = np.bincount(y[m], minlength=len(CLASS_NAMES))
        who = ",".join(sorted(set(person[m])))
        print(f"  {s:5s} {int(m.sum()):5d} samples, people [{who}] | " + " ".join(
            f"{n}:{c}" for n, c in zip(CLASS_NAMES, counts)))
        for n, c in zip(CLASS_NAMES, counts):
            if c == 0:
                warns.append(f"{s} split has no '{n}' samples")
        if s == "train" and counts.min() > 0 and counts.max() / counts.min() > 3:
            warns.append(f"train classes are imbalanced (max/min = {counts.max() / counts.min():.1f}); "
                         "use class weights or record more of the small class")
    return warns


# ---- build / io ---------------------------------------------------------------
def build(
    raw: Path, out: Path,
    test_people: Optional[Sequence[str]] = None,
    val_people: Optional[Sequence[str]] = None,
    seed: int = 0,
    quiet: bool = False,
) -> Dict[str, Any]:
    data, warns = collect(raw)
    if not data:
        raise RuntimeError(f"no valid recordings found under {raw}" + (f" ({len(warns)} skipped)" if warns else ""))
    split_of, w = assign_splits(sorted({str(p) for p in data["person"]}), test_people, val_people, seed)
    warns += w
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            warns += report(data, split_of)
    else:
        warns += report(data, split_of)

    row_split = np.array([split_of[str(p)] for p in data["person"]], dtype="U5")
    meta = {
        "class_names": CLASS_NAMES, "seed": seed, "n_samples": int(len(data["y"])),
        "splits": {s: sorted(p for p, v in split_of.items() if v == s) for s in SPLITS},
        "feature_dim": FEATURE_DIM,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, split=row_split, class_names=np.array(CLASS_NAMES),
                        meta=np.array(json.dumps(meta)), **data)
    tmp.replace(out)
    return {"path": out, "meta": meta, "warnings": warns, "split_of": split_of}


def load_dataset(path: Path) -> Dict[str, Any]:
    """Load dataset.npz: arrays plus `meta` (dict) and `class_names` (list)."""
    with np.load(Path(path), allow_pickle=False) as z:
        out: Dict[str, Any] = {k: z[k] for k in z.files if k not in ("meta", "class_names")}
        out["meta"] = json.loads(str(z["meta"]))
        out["class_names"] = [str(c) for c in z["class_names"]]
    return out


# ---- self-test ----------------------------------------------------------------
def _synthetic_checks() -> bool:
    results: List[bool] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

    rng = np.random.default_rng(1)

    def fake_raw(root: Path, people: Sequence[str], lights=("bright", "dim"), n=8) -> None:
        for person in people:
            for light in lights:
                for g in RECORDABLE:
                    center = np.full(FEATURE_DIM, float(RECORDABLE.index(g)), dtype=np.float32)
                    X = center + rng.normal(0, 0.1, size=(n, FEATURE_DIM)).astype(np.float32)
                    save_samples(root, person, light, "front", g, X, ["Right"] * n, fps=10.0)

    people = ["anna", "ben", "cara", "dev", "eli"]
    with tempfile.TemporaryDirectory() as d:
        raw, out = Path(d) / "raw", Path(d) / "out" / "dataset.npz"
        fake_raw(raw, people)
        res = build(raw, out, seed=0, quiet=True)
        ds = load_dataset(out)
        n_expected = len(people) * 2 * len(RECORDABLE) * 8
        check("shapes and count", ds["X"].shape == (n_expected, FEATURE_DIM) and len(ds["y"]) == n_expected)
        check("labels match class order",
              all(np.allclose(ds["X"][ds["y"] == i].mean(), float(RECORDABLE.index(CLASS_NAMES[i])), atol=0.05)
                  for i in range(len(CLASS_NAMES))))
        per_person = {p: set(ds["split"][ds["person"] == p]) for p in people}
        check("every person is in exactly one split", all(len(s) == 1 for s in per_person.values()))
        sp = res["meta"]["splits"]
        check("test and val are unseen people", not (set(sp["test"]) & set(sp["train"]))
              and not (set(sp["val"]) & set(sp["train"])) and bool(sp["test"]) and bool(sp["val"]), f"({sp})")
        check("all classes present in every split", all(
            set(ds["y"][ds["split"] == s]) == set(range(len(CLASS_NAMES))) for s in SPLITS))
        check("deterministic for same seed", build(raw, Path(d) / "o2.npz", seed=0, quiet=True)["meta"]["splits"] == sp)

        r2 = build(raw, Path(d) / "o3.npz", test_people=["cara"], val_people=["anna"], quiet=True)
        check("explicit test/val people honored", r2["meta"]["splits"]["test"] == ["cara"]
              and r2["meta"]["splits"]["val"] == ["anna"])
        for label, kw in {"unknown person": dict(test_people=["zed"]),
                          "person in both": dict(test_people=["ben"], val_people=["ben"])}.items():
            try:
                build(raw, Path(d) / "o4.npz", quiet=True, **kw)
                check(f"rejects {label}", False)
            except ValueError:
                check(f"rejects {label}", True)

        (raw / "anna" / "bright_front" / "junk.npz").write_bytes(b"not a real npz")
        r3 = build(raw, Path(d) / "o5.npz", quiet=True)
        check("corrupt file skipped with warning", any("junk.npz" in w for w in r3["warnings"])
              and r3["meta"]["n_samples"] == n_expected)
        check("no temp files left", not list(Path(d).rglob("*.tmp.npz")))

    with tempfile.TemporaryDirectory() as d:
        raw = Path(d) / "raw"
        fake_raw(raw, ["solo"], lights=("dim",))
        r = build(raw, Path(d) / "o.npz", quiet=True)
        check("single person warns about no unseen test",
              any("NO unseen-person" in w for w in r["warnings"]) and r["meta"]["splits"]["test"] == [])

    with tempfile.TemporaryDirectory() as d:
        try:
            build(Path(d) / "empty", Path(d) / "o.npz", quiet=True)
            check("empty raw dir raises", False)
        except RuntimeError:
            check("empty raw dir raises", True)
    return all(results)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build dataset.npz from raw recordings, split by person.")
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--test-people", nargs="*", default=None, help="people held out for the final test")
    ap.add_argument("--val-people", nargs="*", default=None, help="people used for validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        print("Synthetic checks:")
        return 0 if _synthetic_checks() else 1
    try:
        res = build(args.raw, args.out, args.test_people, args.val_people, args.seed)
    except (RuntimeError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1
    for w in res["warnings"]:
        print(f"WARNING: {w}")
    print(f"\nSaved {res['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())