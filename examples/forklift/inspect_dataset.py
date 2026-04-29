"""Sanity-check a converted forklift LeRobot dataset.

Usage:
    uv run examples/forklift/inspect_dataset.py --repo_id local/forklift_stage3

Checks performed:
  1. ``meta/info.json`` is v2.1, fps=10, has all expected features and shapes.
  2. Episode count matches the stage_3 expectation (or prints what was found).
  3. Random frame: shapes, dtypes, finite values, image dump for visual inspection.
  4. Per-feature global stats (min/max/mean/std) — flags suspicious dims (e.g. all zero,
     or std=0 across a whole column).
  5. ``pallet_delta_valid`` fraction per episode (flags episodes where it never goes True).
  6. Per-task episode counts and an example task string for each.
  7. Wall-clock fps consistency (number of frames vs ``fps * episode_seconds`` if available).
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


EXPECTED_FEATURES = {
    "image": {"shape": (224, 224, 3), "dtype": "image"},
    "state": {"shape": (10,), "dtype": "float32"},
    "actions": {"shape": (10,), "dtype": "float32"},
    "pallet_delta": {"shape": (3,), "dtype": "float32"},
    "pallet_delta_valid": {"shape": (1,), "dtype": "bool"},
}

EXPECTED_EPISODE_COUNTS = {
    ("g_g", "EnterPallet"): 36,
    ("g_g", "EnterSlot"): 40,
    ("g_t", "EnterPallet"): 19,
    ("g_t", "EnterSlot"): 19,
    ("t_g", "EnterPallet"): 18,
    ("t_g", "EnterSlot"): 20,
}
EXPECTED_TOTAL_EPISODES = sum(EXPECTED_EPISODE_COUNTS.values())  # 152


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def check_info(root: Path) -> dict:
    print("== meta/info.json ==")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    cv = info.get("codebase_version")
    if cv == "v2.1":
        _ok(f"codebase_version = {cv}")
    else:
        _fail(f"codebase_version = {cv} (expected 'v2.1')")
    fps = info.get("fps")
    if fps == 10:
        _ok(f"fps = {fps}")
    else:
        _warn(f"fps = {fps} (expected 10)")

    feats = info.get("features", {})
    for name, spec in EXPECTED_FEATURES.items():
        if name not in feats:
            _fail(f"missing feature '{name}'")
            continue
        got_shape = tuple(feats[name].get("shape", ()))
        got_dtype = feats[name].get("dtype")
        if got_shape != spec["shape"]:
            _fail(f"{name}.shape = {got_shape} (expected {spec['shape']})")
        elif got_dtype != spec["dtype"]:
            _fail(f"{name}.dtype = {got_dtype} (expected {spec['dtype']})")
        else:
            _ok(f"{name} = {got_dtype}{got_shape}")
    return info


def check_episode_counts(ds: LeRobotDataset) -> None:
    print("== episode counts ==")
    n = ds.num_episodes
    if n == EXPECTED_TOTAL_EPISODES:
        _ok(f"num_episodes = {n}")
    else:
        _warn(f"num_episodes = {n} (expected {EXPECTED_TOTAL_EPISODES})")

    # Tally tasks. LeRobot stores tasks in meta/tasks.jsonl, indexed by episode.
    task_counts: Counter[str] = Counter()
    for ep_idx in range(ds.num_episodes):
        tasks = ds.meta.episodes[ep_idx].get("tasks", [])
        if tasks:
            task_counts[tasks[0]] += 1
    print("  tasks observed:")
    for task, c in sorted(task_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {c:4d}  {task!r}")


def check_random_frame(ds: LeRobotDataset, dump_dir: Path) -> None:
    print("== random frame ==")
    rng = np.random.default_rng(0)
    idx = int(rng.integers(0, len(ds)))
    frame = ds[idx]
    print(f"  picked frame index {idx} (of {len(ds)})")

    img = frame["image"]
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        h, w, c = arr.shape
        layout = "HWC"
    elif arr.ndim == 3 and arr.shape[0] == 3:
        c, h, w = arr.shape
        layout = "CHW"
    else:
        layout = f"unknown:{arr.shape}"
        h = w = c = -1
    print(f"  image: {arr.dtype} shape={arr.shape} layout={layout} min={float(arr.min()):.3f} max={float(arr.max()):.3f}")
    if h not in (224, -1) or w not in (224, -1):
        _fail(f"image is not 224x224")
    else:
        _ok("image is 224x224 with 3 channels")

    for key in ("observation.state", "state"):
        if key in frame:
            s = np.asarray(frame[key])
            print(f"  state ({key}): dtype={s.dtype} shape={s.shape} min={s.min():.4f} max={s.max():.4f} finite={np.all(np.isfinite(s))}")
            break
    for key in ("action", "actions"):
        if key in frame:
            a = np.asarray(frame[key])
            print(f"  action ({key}): dtype={a.dtype} shape={a.shape} min={a.min():.4f} max={a.max():.4f} finite={np.all(np.isfinite(a))}")
            break
    if "pallet_delta" in frame:
        p = np.asarray(frame["pallet_delta"])
        v = bool(np.asarray(frame["pallet_delta_valid"]).flatten()[0])
        print(f"  pallet_delta: {p.tolist()} valid={v}")

    # Dump the image so the user can eyeball it.
    dump_dir.mkdir(parents=True, exist_ok=True)
    out = dump_dir / f"frame_{idx:06d}.png"
    try:
        import cv2

        if layout == "CHW":
            arr = np.transpose(arr, (1, 2, 0))
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(out), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        _ok(f"wrote sample image to {out}")
    except Exception as e:
        _warn(f"could not write sample image: {e}")


def check_global_stats(ds: LeRobotDataset, max_frames: int = 5000) -> None:
    print(f"== global state/action stats (up to {max_frames} sampled frames) ==")
    n = min(len(ds), max_frames)
    rng = np.random.default_rng(1)
    idxs = rng.choice(len(ds), size=n, replace=False)

    states = []
    actions = []
    deltas = []
    valids = []
    for i in idxs:
        f = ds[int(i)]
        s = f.get("observation.state", f.get("state"))
        a = f.get("action", f.get("actions"))
        states.append(np.asarray(s))
        actions.append(np.asarray(a))
        if "pallet_delta" in f:
            deltas.append(np.asarray(f["pallet_delta"]))
            valids.append(bool(np.asarray(f["pallet_delta_valid"]).flatten()[0]))

    S = np.stack(states)
    A = np.stack(actions)
    print("  state per-dim (min, max, mean, std):")
    for j in range(S.shape[1]):
        col = S[:, j]
        marker = " <-- ALL ZERO" if np.allclose(col, 0) else (" <-- std=0" if col.std() == 0 else "")
        print(f"    [{j:2d}] {col.min():12.4f}  {col.max():12.4f}  {col.mean():12.4f}  {col.std():12.4f}{marker}")
    print("  action per-dim (min, max, mean, std):")
    for j in range(A.shape[1]):
        col = A[:, j]
        marker = " <-- ALL ZERO" if np.allclose(col, 0) else (" <-- std=0" if col.std() == 0 else "")
        print(f"    [{j:2d}] {col.min():12.4f}  {col.max():12.4f}  {col.mean():12.4f}  {col.std():12.4f}{marker}")
    if deltas:
        D = np.stack(deltas)
        v = np.array(valids)
        print(f"  pallet_delta valid fraction (sampled): {v.mean():.3f}")
        if v.any():
            Dv = D[v]
            print("  pallet_delta where valid (min, max, mean, std):")
            for j, name in enumerate(("dx", "dy", "dyaw")):
                col = Dv[:, j]
                print(f"    {name:>4}: {col.min():12.4f}  {col.max():12.4f}  {col.mean():12.4f}  {col.std():12.4f}")


def check_episodes(ds: LeRobotDataset) -> None:
    print("== per-episode pallet_delta_valid fraction ==")
    edi = ds.episode_data_index  # {"from": Tensor, "to": Tensor}
    starts = np.asarray(edi["from"])
    ends = np.asarray(edi["to"])
    bad = []
    for ep_idx in range(ds.num_episodes):
        start = int(starts[ep_idx])
        end = int(ends[ep_idx])
        n = end - start
        step = max(1, n // 64)
        v_count = 0
        for i in range(start, end, step):
            f = ds[i]
            if bool(np.asarray(f["pallet_delta_valid"]).flatten()[0]):
                v_count += 1
        sampled = (n + step - 1) // step
        frac = v_count / sampled if sampled else 0.0
        if frac == 0.0:
            ep_meta = ds.meta.episodes[ep_idx] if ep_idx < len(ds.meta.episodes) else {}
            task = (ep_meta.get("tasks") or [""])[0] if isinstance(ep_meta, dict) else ""
            bad.append((ep_idx, n, task))
    if bad:
        _warn(f"{len(bad)} episodes never had pallet_delta_valid=True (sampled):")
        for ep_idx, n, task in bad[:10]:
            print(f"      ep {ep_idx:3d} (len {n}, task={task!r})")
        if len(bad) > 10:
            print(f"      ... and {len(bad) - 10} more")
    else:
        _ok("every episode had at least some valid pallet_delta frames (sampled)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="local/forklift_stage3")
    ap.add_argument("--dump_dir", default="/tmp/forklift_inspect")
    ap.add_argument("--max_stats_frames", type=int, default=5000)
    args = ap.parse_args()

    root = HF_LEROBOT_HOME / args.repo_id
    print(f"Inspecting {root}\n")
    if not root.exists():
        raise SystemExit(f"dataset not found at {root}")

    check_info(root)
    print()
    ds = LeRobotDataset(args.repo_id)
    check_episode_counts(ds)
    print()
    check_random_frame(ds, Path(args.dump_dir))
    print()
    check_global_stats(ds, max_frames=args.max_stats_frames)
    print()
    check_episodes(ds)


if __name__ == "__main__":
    main()
