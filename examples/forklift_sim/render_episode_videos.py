"""Render each episode of the sim LeRobot dataset to a browsable MP4 for review.

For every episode this produces one MP4 tiling all image features (2x2 for four
cameras) with the per-frame task string, episode/frame index, and timestamp burned
into a caption bar — plus an ``index.html`` gallery so the whole dataset can be
skimmed in a browser. Meant to pair with ``edit_tasks.py``: watch the videos, fill
in the ``new_task`` column of the exported CSV, then apply.

Reads the parquet files directly (images are embedded PNG bytes in the v2.1 image
dtype), so it does not need torch or the LeRobot loader.

Usage:
    uv run examples/forklift_sim/render_episode_videos.py \
        --repo_id local/forklift_sim
    open ~/.cache/huggingface/lerobot/local/forklift_sim_review/index.html
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import html
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tyro

HF_LEROBOT_HOME = Path("~/.cache/huggingface/lerobot").expanduser()

CAPTION_H = 46
TILE_COLS = 2


def _load_meta(ds_root: Path) -> tuple[dict, list[dict], dict[int, str]]:
    info = json.loads((ds_root / "meta" / "info.json").read_text())
    episodes = [json.loads(l) for l in (ds_root / "meta" / "episodes.jsonl").read_text().splitlines() if l]
    tasks = {}
    for l in (ds_root / "meta" / "tasks.jsonl").read_text().splitlines():
        if l:
            d = json.loads(l)
            tasks[d["task_index"]] = d["task"]
    return info, episodes, tasks


def _image_keys(info: dict) -> list[str]:
    keys = [k for k, f in info["features"].items() if f["dtype"] in ("image", "video")]
    # Stable, sensible tiling order.
    order = ["image", "third_person_image", "left_image", "right_image"]
    return sorted(keys, key=lambda k: order.index(k) if k in order else len(order))


def _decode(cell) -> np.ndarray:
    """Decode one embedded HF Image cell ({bytes,path}) to a BGR uint8 array."""
    data = cell["bytes"] if isinstance(cell, dict) else cell
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("failed to decode embedded image")
    return img


def _render_episode(
    parquet_path: str, out_path: str, image_keys: list[str], tasks: dict[int, str], fps: int
) -> tuple[int, int]:
    df = pd.read_parquet(parquet_path)
    ep_index = int(df["episode_index"].iloc[0])

    first = _decode(df[image_keys[0]].iloc[0])
    th, tw = first.shape[:2]
    rows = (len(image_keys) + TILE_COLS - 1) // TILE_COLS
    W, H = tw * TILE_COLS, th * rows + CAPTION_H

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {out_path}")

    for _, row in df.iterrows():
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for i, key in enumerate(image_keys):
            r, c = divmod(i, TILE_COLS)
            canvas[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = _decode(row[key])
        task = tasks.get(int(row["task_index"]), "?")
        t = float(row["timestamp"])
        cv2.putText(
            canvas,
            f"ep {ep_index:04d}  f {int(row['frame_index']):4d}  {t:6.1f}s",
            (8, H - CAPTION_H + 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, task, (8, H - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 220, 90), 1, cv2.LINE_AA,
        )
        writer.write(canvas)
    writer.release()
    return ep_index, len(df)


def _episode_segments(df: pd.DataFrame, tasks: dict[int, str]) -> list[tuple[str, float, float]]:
    segs = []
    ti = df["task_index"].to_numpy()
    ts = df["timestamp"].to_numpy()
    start = 0
    for i in range(1, len(ti) + 1):
        if i == len(ti) or ti[i] != ti[start]:
            segs.append((tasks.get(int(ti[start]), "?"), float(ts[start]), float(ts[i - 1])))
            start = i
    return segs


def _write_index(out_dir: Path, entries: list[dict]) -> None:
    rows = []
    for e in entries:
        segs = " → ".join(
            f"{html.escape(t)} <span class=t>({a:.0f}–{b:.0f}s)</span>" for t, a, b in e["segments"]
        )
        rows.append(
            f'<tr><td><a href="{e["video"]}">ep {e["episode_index"]:04d}</a></td>'
            f'<td>{e["frames"]}</td><td>{segs}</td></tr>'
        )
    page = (
        "<!doctype html><meta charset=utf-8><title>forklift_sim review</title>"
        "<style>body{font:14px system-ui;margin:2em}table{border-collapse:collapse}"
        "td,th{padding:4px 12px;border-bottom:1px solid #ddd;text-align:left}"
        ".t{color:#888}</style>"
        f"<h1>forklift_sim episodes ({len(entries)})</h1>"
        "<p>Click an episode to watch. Cameras: top | third_person / left | right.</p>"
        "<table><tr><th>episode</th><th>frames</th><th>task segments</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    (out_dir / "index.html").write_text(page)


def main(
    repo_id: str = "local/forklift_sim",
    out_dir: str | None = None,
    *,
    workers: int = 8,
    limit_episodes: int | None = None,
) -> None:
    ds_root = HF_LEROBOT_HOME / repo_id
    out = Path(out_dir) if out_dir else ds_root.parent / (ds_root.name + "_review")
    (out / "videos").mkdir(parents=True, exist_ok=True)

    info, episodes, tasks = _load_meta(ds_root)
    image_keys = _image_keys(info)
    fps = int(info["fps"])
    print(f"{len(episodes)} episodes, cameras: {image_keys}, fps={fps}")
    if limit_episodes is not None:
        episodes = episodes[:limit_episodes]

    jobs = []
    for ep in episodes:
        i = ep["episode_index"]
        chunk = i // info["chunks_size"]
        pq = ds_root / info["data_path"].format(episode_chunk=chunk, episode_index=i)
        jobs.append((str(pq), str(out / "videos" / f"episode_{i:06d}.mp4")))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_render_episode, pq, mp4, image_keys, tasks, fps): (pq, mp4) for pq, mp4 in jobs}
        done = 0
        for f in as_completed(futs):
            ep_index, n = f.result()
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"  rendered {done}/{len(jobs)}")

    entries = []
    for pq, mp4 in jobs:
        df = pd.read_parquet(pq, columns=["episode_index", "task_index", "timestamp"])
        entries.append(
            {
                "episode_index": int(df["episode_index"].iloc[0]),
                "frames": len(df),
                "video": f"videos/{Path(mp4).name}",
                "segments": _episode_segments(df, tasks),
            }
        )
    entries.sort(key=lambda e: e["episode_index"])
    _write_index(out, entries)
    print(f"Done. Open {out / 'index.html'}")


if __name__ == "__main__":
    tyro.cli(main)
