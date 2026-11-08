"""Export / apply per-segment language edits for the sim LeRobot dataset (v2.1).

Workflow (pairs with ``render_episode_videos.py``):

1. ``export`` writes ``task_edits.csv`` with one row per episode x task segment:
       episode_index, segment, start_frame, end_frame, start_s, end_s, task, new_task
2. Watch the episode videos and fill in ``new_task`` for any row you want to change
   (leave blank to keep). Different episodes may get different text for the same
   original task — e.g. "Engage the forks under the left pallet".
3. ``apply`` rewrites meta/tasks.jsonl, meta/episodes.jsonl, meta/info.json and the
   per-frame ``task_index`` column of each touched episode's parquet file.

``apply`` backs up the three meta files and each touched episode's original
task_index column into ``<dataset>/task_edit_backup/`` before writing, and is
idempotent: re-running with the same CSV is a no-op.

Usage:
    uv run examples/forklift_sim/edit_tasks.py export --repo_id local/forklift_sim
    uv run examples/forklift_sim/edit_tasks.py apply  --repo_id local/forklift_sim
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tyro

HF_LEROBOT_HOME = Path("~/.cache/huggingface/lerobot").expanduser()
CSV_NAME = "task_edits.csv"


def _load_meta(ds_root: Path) -> tuple[dict, list[dict], dict[int, str]]:
    info = json.loads((ds_root / "meta" / "info.json").read_text())
    episodes = [json.loads(l) for l in (ds_root / "meta" / "episodes.jsonl").read_text().splitlines() if l]
    tasks = {}
    for l in (ds_root / "meta" / "tasks.jsonl").read_text().splitlines():
        if l:
            d = json.loads(l)
            tasks[d["task_index"]] = d["task"]
    return info, episodes, tasks


def _parquet_path(ds_root: Path, info: dict, episode_index: int) -> Path:
    chunk = episode_index // info["chunks_size"]
    return ds_root / info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)


def _segments(df: pd.DataFrame) -> list[tuple[int, int, int]]:
    """Consecutive same-task_index runs -> (task_index, start_frame, end_frame)."""
    ti = df["task_index"].to_numpy()
    out = []
    start = 0
    for i in range(1, len(ti) + 1):
        if i == len(ti) or ti[i] != ti[start]:
            out.append((int(ti[start]), start, i - 1))
            start = i
    return out


def export(repo_id: str = "local/forklift_sim", out_csv: str | None = None) -> None:
    """Write one CSV row per episode x task segment, with an empty new_task column."""
    ds_root = HF_LEROBOT_HOME / repo_id
    info, episodes, tasks = _load_meta(ds_root)
    fps = float(info["fps"])

    rows = []
    for ep in episodes:
        i = ep["episode_index"]
        df = pd.read_parquet(_parquet_path(ds_root, info, i), columns=["task_index"])
        for seg_no, (ti, a, b) in enumerate(_segments(df)):
            rows.append(
                {
                    "episode_index": i,
                    "segment": seg_no,
                    "start_frame": a,
                    "end_frame": b,
                    "start_s": round(a / fps, 1),
                    "end_s": round(b / fps, 1),
                    "task": tasks[ti],
                    "new_task": "",
                }
            )
    out = Path(out_csv) if out_csv else ds_root.parent / (ds_root.name + "_review") / CSV_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} segment rows for {len(episodes)} episodes to {out}")
    print("Fill in new_task (leave blank to keep), then run: edit_tasks.py apply")


def apply(repo_id: str = "local/forklift_sim", csv: str | None = None) -> None:
    """Apply the filled-in CSV: rewrite task metadata and per-frame task_index."""
    ds_root = HF_LEROBOT_HOME / repo_id
    info, episodes, tasks = _load_meta(ds_root)
    csv_path = Path(csv) if csv else ds_root.parent / (ds_root.name + "_review") / CSV_NAME
    edits = pd.read_csv(csv_path, dtype={"new_task": str}, keep_default_na=False)
    edits = edits[edits["new_task"].str.strip() != ""]
    if edits.empty:
        print("No non-empty new_task rows; nothing to do.")
        return

    backup = ds_root / "task_edit_backup"
    backup.mkdir(exist_ok=True)
    for name in ("tasks.jsonl", "episodes.jsonl", "info.json"):
        src = ds_root / "meta" / name
        dst = backup / name
        if not dst.exists():  # keep the *original* on repeated applies
            shutil.copy2(src, dst)

    # Global task table: existing entries keep their indices; new strings appended.
    task_to_index = {t: i for i, t in tasks.items()}

    def index_for(text: str) -> int:
        if text not in task_to_index:
            task_to_index[text] = max(task_to_index.values(), default=-1) + 1
        return task_to_index[text]

    ep_meta = {e["episode_index"]: e for e in episodes}
    n_frames_changed = 0
    for ep_index, group in edits.groupby("episode_index"):
        p = _parquet_path(ds_root, info, int(ep_index))
        table = pq.read_table(p)
        ti = table["task_index"].to_numpy().copy()

        orig_backup = backup / f"task_index_ep{ep_index:06d}.npy"
        if not orig_backup.exists():
            np.save(orig_backup, ti)

        for _, row in group.iterrows():
            a, b = int(row["start_frame"]), int(row["end_frame"])
            new_idx = index_for(str(row["new_task"]).strip())
            changed = np.sum(ti[a : b + 1] != new_idx)
            ti[a : b + 1] = new_idx
            n_frames_changed += int(changed)

        idx = table.schema.get_field_index("task_index")
        table = table.set_column(idx, "task_index", pa.array(ti, type=table.schema.field("task_index").type))
        tmp = p.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        tmp.replace(p)

        used = sorted(set(int(x) for x in ti))
        index_to_task = {i: t for t, i in task_to_index.items()}
        ep_meta[int(ep_index)]["tasks"] = [index_to_task[i] for i in used]

    with open(ds_root / "meta" / "tasks.jsonl", "w") as f:
        for text, i in sorted(task_to_index.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": i, "task": text}) + "\n")
    with open(ds_root / "meta" / "episodes.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    info["total_tasks"] = len(task_to_index)
    (ds_root / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    print(
        f"Applied {len(edits)} segment edits across {edits['episode_index'].nunique()} episodes "
        f"({n_frames_changed} frames changed). Task table now has {len(task_to_index)} entries."
    )
    print(f"Originals backed up in {backup}. Re-render videos to see the new labels.")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"export": export, "apply": apply})
