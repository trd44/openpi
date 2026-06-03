#!/usr/bin/env python3
"""Offline evaluation of the fine-tuned forklift policy against its training data.

This is the *same client/server flow* the docker setup uses for inference (see
``examples/forklift/compose.yml``): a running ``openpi`` policy server is queried
over a websocket, exactly like ``examples/libero/main.py`` does. The only
difference is that the "environment" here is the recorded LeRobot dataset rather
than a live robot — we feed the model the recorded observations (open-loop /
teacher-forced) and compare its predicted action chunk against the actions that
were actually recorded.

What it answers:

  * Is the model *systematically* off on any action axis (a consistent +/- bias,
    or a scale error where it under/over-shoots), as opposed to just noisy?
  * Does prediction quality degrade across the 10-step action chunk horizon?
  * Are some tasks fit much worse than others?
  * Which specific episodes does the model fit worst (so you can go look at the
    video / rosbag for those)?

It prints a human-readable report, writes a JSON report with every number, and
(optionally) a per-episode CSV and diagnostic plots.

Run order (see INFERENCE.md / compose.yml):

  1. Serve the fine-tuned checkpoint by short name (downloads from HF if needed):
       uv run examples/forklift/serve.py --model full-15000
  2. Run this script against the dataset the model was trained on:
       uv run examples/forklift/eval_on_training_data.py --repo-id tduggan93/forklift

The run is labelled by the served model automatically (from server metadata), so
results land in data/forklift/eval/<model>/. Serve each model in turn and re-run to
compare them. Use --samples-per-obs N for the best-of-N multimodality diagnostic.

Or do both with one command via docker compose (see compose.yml):
       MODEL=full-15000 EVAL_ARGS="--repo-id tduggan93/forklift" \\
       docker compose -f examples/forklift/compose.yml up --build
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import tempfile

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

# Must match the dataset/policy definitions in examples/forklift/*.py.
ACTION_DIM = 5
ACTION_DIM_NAMES = ["drive", "steer_rate", "lift", "shift", "tilt"]
ACTION_HORIZON = 10  # Pi0Config(action_horizon=10)

# An axis whose mean signed error exceeds this fraction of its ground-truth std
# is flagged as a *systematic* bias (not just noise).
BIAS_FRAC_OF_STD = 0.25
# Linear-fit slope this far from 1.0 is flagged as a scale (gain) error.
SLOPE_TOL = 0.15


@dataclasses.dataclass
class Args:
    # --- which dataset / which server -----------------------------------------
    repo_id: str = "tduggan93/forklift"
    """LeRobot repo id of the training dataset (local or on the hub)."""
    host: str = "0.0.0.0"
    """Host of the running openpi policy server."""
    port: int = 8000
    """Port of the running openpi policy server."""

    # --- what to evaluate -----------------------------------------------------
    replan_steps: int = ACTION_HORIZON
    """Stride between inference calls within an episode. Defaults to the full
    action horizon, so every recorded step is predicted exactly once (cheapest
    full coverage). Set to 1 for the densest possible evaluation (10x slower)."""
    max_episodes: int | None = None
    """Evaluate only the first N episodes (debug / quick look). None = all."""
    prompt_override: str | None = None
    """Force this prompt for every episode instead of the dataset task string.
    Useful for measuring the model's prompt sensitivity."""
    samples_per_obs: int = 1
    """How many action chunks to sample per observation. >1 enables the best-of-N
    diagnostic: because pi0.5 is stochastic and the expert is multimodal, a large
    gap between single-sample and best-of-N error means the model's prediction is
    fine but doesn't match the single logged trajectory (multimodality), whereas
    no gap means the model genuinely can't fit that axis. N>1 costs N x inference,
    so pair it with --max-episodes. The primary stats always use the first sample
    (deployment-representative); best-of-N is reported alongside."""
    label: str | None = None
    """Tag for this run; names the output subdirectory and is recorded in the
    report so multiple models are easy to compare. Defaults to the served model's
    name (from server metadata) when available."""

    # --- flagging thresholds --------------------------------------------------
    outlier_sigma: float = 2.0
    """Episodes with normalized error above mean + this*std are flagged."""
    top_k_worst: int = 15
    """How many worst episodes to always list, regardless of the sigma cutoff."""

    # --- outputs --------------------------------------------------------------
    output_dir: str = "data/forklift/eval"
    """Directory for the JSON report, per-episode CSV, and plots."""
    make_plots: bool = True
    """Write residual/scatter diagnostic plots (needs matplotlib)."""
    seed: int = 0


def _to_uint8_hwc_rgb(image) -> np.ndarray:
    """Convert a LeRobot image (torch/np, CHW or HWC, float[0,1] or uint8) to the
    uint8 HWC RGB array the policy server expects — matching what the live ROS
    node sends at runtime."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8) if np.issubdtype(arr.dtype, np.floating) else arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def _frame_value(frame: dict, *keys) -> np.ndarray:
    for k in keys:
        if k in frame:
            return np.asarray(frame[k], dtype=np.float32)
    raise KeyError(f"none of {keys} present in frame (have: {list(frame)})")


def _episode_task(ds: LeRobotDataset, ep_idx: int) -> str:
    if ep_idx < len(ds.meta.episodes):
        tasks = ds.meta.episodes[ep_idx].get("tasks") or []
        if tasks:
            return tasks[0]
    return ""


def collect_predictions(ds: LeRobotDataset, client, args: Args):
    """Run the policy over the dataset and return matched (pred, gt) records.

    Returns a dict of stacked arrays, one row per matched (query, horizon-offset):
        pred    (M, 5)     model prediction (first sample — deployment-representative)
        gt      (M, 5)     recorded action
        offset  (M,)       horizon index 0..H-1 of this prediction within its chunk
        ep      (M,)       episode index
        samples (M, N, 5)  all N sampled predictions (only when samples_per_obs>1, else None)
    plus ep_tasks: {ep_idx: task string} and per-episode frame counts.
    """
    edi = ds.episode_data_index
    starts = np.asarray(edi["from"])
    ends = np.asarray(edi["to"])
    n_episodes = ds.num_episodes
    if args.max_episodes is not None:
        n_episodes = min(n_episodes, args.max_episodes)
    n_samples = max(1, args.samples_per_obs)

    preds, gts, offsets, eps = [], [], [], []
    samples_all: list[np.ndarray] = []
    ep_tasks: dict[int, str] = {}
    ep_lengths: dict[int, int] = {}

    for ep_idx in tqdm.tqdm(range(n_episodes), desc="episodes"):
        start, end = int(starts[ep_idx]), int(ends[ep_idx])
        ep_len = end - start
        ep_lengths[ep_idx] = ep_len
        task = args.prompt_override or _episode_task(ds, ep_idx)
        ep_tasks[ep_idx] = task

        # Ground-truth actions for the whole episode (so we can slice chunks).
        ep_actions = np.stack(
            [_frame_value(ds[i], "action", "actions") for i in range(start, end)]
        )

        for local_t in range(0, ep_len, args.replan_steps):
            frame = ds[start + local_t]
            obs = {
                "observation/image": _to_uint8_hwc_rgb(frame["image"]),
                "observation/state": _frame_value(frame, "observation.state", "state"),
                "prompt": task,
            }
            # Sample the policy n_samples times for the same observation.
            chunks = [
                np.asarray(client.infer(obs)["actions"], dtype=np.float32) for _ in range(n_samples)
            ]
            chunk = chunks[0]  # primary (deployment-representative) sample
            # chunk is (H, 5); compare each step against the recorded future action,
            # clipped at the episode boundary.
            horizon = min(chunk.shape[0], ep_len - local_t)
            for k in range(horizon):
                preds.append(chunk[k])
                gts.append(ep_actions[local_t + k])
                offsets.append(k)
                eps.append(ep_idx)
                if n_samples > 1:
                    samples_all.append(np.stack([c[k] for c in chunks]))  # (N, 5)

    return {
        "pred": np.asarray(preds, dtype=np.float32),
        "gt": np.asarray(gts, dtype=np.float32),
        "offset": np.asarray(offsets, dtype=np.int32),
        "ep": np.asarray(eps, dtype=np.int32),
        "samples": np.asarray(samples_all, dtype=np.float32) if n_samples > 1 else None,
        "ep_tasks": ep_tasks,
        "ep_lengths": ep_lengths,
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _linfit(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Least-squares fit pred ~= slope*gt + intercept. slope!=1 => scale error."""
    if gt.std() < 1e-9:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(gt, pred, 1)
    return float(slope), float(intercept)


def per_dim_stats(pred: np.ndarray, gt: np.ndarray) -> list[dict]:
    out = []
    for j in range(ACTION_DIM):
        p, g = pred[:, j], gt[:, j]
        err = p - g
        gt_std = float(g.std())
        gt_range = float(g.max() - g.min())
        rmse = float(np.sqrt(np.mean(err**2)))
        me = float(err.mean())  # signed bias
        slope, intercept = _linfit(g, p)
        r = _safe_corr(g, p)
        # R^2 of the prediction as an estimator of gt (1 - SSE/SST).
        sst = float(np.sum((g - g.mean()) ** 2))
        r2 = float(1.0 - np.sum(err**2) / sst) if sst > 1e-12 else float("nan")

        flags = []
        if gt_std > 1e-6 and abs(me) > BIAS_FRAC_OF_STD * gt_std:
            flags.append(f"BIAS({me:+.3f}, {me / gt_std:+.2f}sd)")
        if not np.isnan(slope) and abs(slope - 1.0) > SLOPE_TOL:
            kind = "under-shoots" if slope < 1 else "over-shoots"
            flags.append(f"SCALE(slope={slope:.2f}, {kind})")

        out.append(
            {
                "dim": j,
                "name": ACTION_DIM_NAMES[j],
                "mae": float(np.mean(np.abs(err))),
                "rmse": rmse,
                "bias_me": me,
                "err_std": float(err.std()),
                "nrmse_std": float(rmse / gt_std) if gt_std > 1e-9 else float("nan"),
                "nrmse_range": float(rmse / gt_range) if gt_range > 1e-9 else float("nan"),
                "pearson_r": r,
                "r2": r2,
                "fit_slope": slope,
                "fit_intercept": intercept,
                "gt_mean": float(g.mean()),
                "gt_std": gt_std,
                "gt_min": float(g.min()),
                "gt_max": float(g.max()),
                "flags": flags,
            }
        )
    return out


def best_of_n_stats(samples: np.ndarray, gt: np.ndarray) -> dict:
    """Best-of-N diagnostic for multimodality.

    For each matched step, pick the sample whose (per-dim-normalized) L2 distance to
    the recorded action is smallest, then report per-axis nRMSE of those best samples.

    A big improvement over single-sample nRMSE means the model *can* produce the
    expert action — it just doesn't on every draw (multimodal target, stochastic
    sampler). Little improvement means the model genuinely can't fit that axis.

    samples: (M, N, 5), gt: (M, 5).
    """
    dim_std = gt.std(axis=0) + 1e-9
    # Normalized per-step, per-sample L2 across the 5 axes.
    err = (samples - gt[:, None, :]) / dim_std[None, None, :]  # (M, N, 5)
    l2 = np.sqrt((err**2).sum(axis=2))  # (M, N)
    best_idx = l2.argmin(axis=1)  # (M,)
    best_pred = samples[np.arange(samples.shape[0]), best_idx]  # (M, 5)
    per_dim = []
    for j in range(ACTION_DIM):
        e = best_pred[:, j] - gt[:, j]
        rmse = float(np.sqrt(np.mean(e**2)))
        per_dim.append(
            {
                "name": ACTION_DIM_NAMES[j],
                "rmse": rmse,
                "nrmse_std": float(rmse / dim_std[j]),
            }
        )
    overall = (best_pred - gt) / dim_std
    return {
        "n_samples": int(samples.shape[1]),
        "per_dim": per_dim,
        "overall_norm_rmse": float(np.sqrt(np.mean(overall**2))),
    }


def per_offset_rmse(pred, gt, offset) -> dict:
    """RMSE (averaged over dims, normalized per-dim by gt std) at each horizon step."""
    dim_std = gt.std(axis=0) + 1e-9
    out = {}
    for k in sorted(set(offset.tolist())):
        m = offset == k
        if not m.any():
            continue
        err = (pred[m] - gt[m]) / dim_std
        out[int(k)] = {
            "count": int(m.sum()),
            "norm_rmse": float(np.sqrt(np.mean(err**2))),
            "per_dim_rmse": [
                float(np.sqrt(np.mean((pred[m, j] - gt[m, j]) ** 2))) for j in range(ACTION_DIM)
            ],
        }
    return out


def per_task_stats(pred, gt, ep, ep_tasks) -> dict:
    dim_std = gt.std(axis=0) + 1e-9
    task_of_row = np.array([ep_tasks.get(int(e), "") for e in ep])
    out = {}
    for task in sorted(set(task_of_row.tolist())):
        m = task_of_row == task
        err = (pred[m] - gt[m]) / dim_std
        out[task] = {
            "count": int(m.sum()),
            "n_episodes": len({int(e) for e in ep[m]}),
            "norm_rmse": float(np.sqrt(np.mean(err**2))),
            "per_dim_rmse": [
                float(np.sqrt(np.mean((pred[m, j] - gt[m, j]) ** 2))) for j in range(ACTION_DIM)
            ],
        }
    return out


def per_episode_stats(pred, gt, ep, ep_tasks, ep_lengths) -> list[dict]:
    """Per-episode error, normalized per-dim by the *global* gt std so dims are
    comparable, then averaged over dims and frames. Sortable for worst-episode
    flagging."""
    dim_std = gt.std(axis=0) + 1e-9
    rows = []
    for e in sorted(set(ep.tolist())):
        m = ep == e
        norm_err = (pred[m] - gt[m]) / dim_std
        rows.append(
            {
                "episode": int(e),
                "task": ep_tasks.get(int(e), ""),
                "ep_len": int(ep_lengths.get(int(e), m.sum())),
                "n_pred": int(m.sum()),
                "norm_rmse": float(np.sqrt(np.mean(norm_err**2))),
                "norm_mae": float(np.mean(np.abs(norm_err))),
                "per_dim_rmse": [
                    float(np.sqrt(np.mean((pred[m, j] - gt[m, j]) ** 2))) for j in range(ACTION_DIM)
                ],
            }
        )
    return rows


def flag_outliers(ep_rows: list[dict], sigma: float, top_k: int) -> dict:
    vals = np.array([r["norm_rmse"] for r in ep_rows])
    mean, std = float(vals.mean()), float(vals.std())
    cutoff = mean + sigma * std
    ranked = sorted(ep_rows, key=lambda r: r["norm_rmse"], reverse=True)
    flagged = [r for r in ranked if r["norm_rmse"] > cutoff]
    return {
        "mean": mean,
        "std": std,
        "cutoff": cutoff,
        "n_flagged": len(flagged),
        "flagged_episodes": [r["episode"] for r in flagged],
        "worst": ranked[:top_k],
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(report: dict) -> None:
    meta = report["meta"]
    print("\n" + "=" * 78)
    print("FORKLIFT POLICY vs TRAINING DATA — open-loop evaluation")
    print("=" * 78)
    print(f"label / model : {meta.get('label')}   (server: {meta.get('server_metadata') or '{}'})")
    print(f"dataset       : {meta['repo_id']}")
    print(f"episodes      : {meta['n_episodes_evaluated']}")
    print(f"inferences    : {meta['n_inferences']}  (replan stride = {meta['replan_steps']}, "
          f"samples/obs = {meta.get('samples_per_obs', 1)})")
    print(f"matched preds : {meta['n_matched']}  (pred/gt pairs across all horizon offsets)")

    print("\n-- per-axis fit (raw action units) " + "-" * 42)
    hdr = f"{'axis':>11} {'bias':>9} {'mae':>9} {'rmse':>9} {'nRMSE/sd':>8} {'r':>6} {'R²':>7} {'slope':>7}"
    print(hdr)
    print("-" * len(hdr))
    for d in report["per_dim"]:
        print(
            f"{d['name']:>11} {d['bias_me']:>+9.4f} {d['mae']:>9.4f} {d['rmse']:>9.4f} "
            f"{d['nrmse_std']:>8.3f} {d['pearson_r']:>6.3f} {d['r2']:>7.3f} {d['fit_slope']:>7.3f}"
        )
    print("\n  nRMSE/sd < ~0.5 = clearly better than predicting the mean; ~1.0 = no better than the mean.")
    flagged_dims = [d for d in report["per_dim"] if d["flags"]]
    if flagged_dims:
        print("\n  systematic issues:")
        for d in flagged_dims:
            print(f"    [{d['name']}] " + "; ".join(d["flags"]))
    else:
        print("\n  no systematic per-axis bias or scale error detected.")

    if "best_of_n" in report:
        bon = report["best_of_n"]
        print(f"\n-- best-of-{bon['n_samples']} (multimodality diagnostic) " + "-" * 35)
        print(f"  {'axis':>11} {'1-sample nRMSE/sd':>17} {'best-of-N nRMSE/sd':>18} {'gap':>8}")
        for d_single, d_best in zip(report["per_dim"], bon["per_dim"], strict=True):
            gap = d_single["nrmse_std"] - d_best["nrmse_std"]
            print(
                f"  {d_single['name']:>11} {d_single['nrmse_std']:>17.3f} "
                f"{d_best['nrmse_std']:>18.3f} {gap:>+8.3f}"
            )
        print("\n  Large gap  => target is multimodal / sampler is stochastic; the model")
        print("               CAN hit the expert action, just not on every single draw.")
        print("  Small gap  => the model genuinely can't fit that axis (real deficiency).")

    print("\n-- error vs horizon step (normalized RMSE over dims) " + "-" * 24)
    off = report["per_offset"]
    off_max = max((o["norm_rmse"] for o in off.values()), default=0.0) or 1.0
    for k in sorted(off, key=int):
        bar = "#" * round(off[k]["norm_rmse"] / off_max * 40)
        print(f"  step {int(k):2d}  nRMSE={off[k]['norm_rmse']:.3f}  {bar}")

    print("\n-- per-task fit " + "-" * 62)
    for task, t in sorted(report["per_task"].items(), key=lambda kv: -kv[1]["norm_rmse"]):
        print(f"  nRMSE={t['norm_rmse']:.3f}  ({t['n_episodes']:>3d} eps, {t['count']:>6d} preds)  {task!r}")

    print("\n-- worst episodes " + "-" * 60)
    fl = report["outliers"]
    print(
        f"  episode normalized RMSE: mean={fl['mean']:.3f} std={fl['std']:.3f} "
        f"-> outlier cutoff (mean+{report['meta']['outlier_sigma']}sd)={fl['cutoff']:.3f}"
    )
    print(f"  {fl['n_flagged']} episode(s) over the cutoff: {fl['flagged_episodes']}")
    print(f"\n  {'ep':>4} {'nRMSE':>7} {'len':>5}  per-dim rmse [drv ster lift shft tilt]   task")
    for r in fl["worst"]:
        flag = " *" if r["norm_rmse"] > fl["cutoff"] else "  "
        pdr = " ".join(f"{v:5.2f}" for v in r["per_dim_rmse"])
        print(f"{flag}{r['episode']:>4} {r['norm_rmse']:>7.3f} {r['ep_len']:>5}  [{pdr}]  {r['task']!r}")
    print("\n  (* = flagged outlier. Pull the rosbag/video for these to see what's hard.)")
    print("=" * 78 + "\n")


def write_plots(records: dict, report: dict, out_dir: pathlib.Path) -> None:
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning("matplotlib unavailable, skipping plots: %s", e)
        return

    pred, gt = records["pred"], records["gt"]

    # 1) pred-vs-gt scatter per axis (systematic bias/scale shows as off-diagonal).
    fig, axes = plt.subplots(1, ACTION_DIM, figsize=(4 * ACTION_DIM, 4))
    for j, ax in enumerate(np.atleast_1d(axes)):
        g, p = gt[:, j], pred[:, j]
        ax.scatter(g, p, s=2, alpha=0.2)
        lo, hi = float(min(g.min(), p.min())), float(max(g.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
        d = report["per_dim"][j]
        if not np.isnan(d["fit_slope"]):
            xs = np.array([lo, hi])
            ax.plot(xs, d["fit_slope"] * xs + d["fit_intercept"], "r-", lw=1, label="fit")
        ax.set_title(f"{ACTION_DIM_NAMES[j]}\nr={d['pearson_r']:.2f} slope={d['fit_slope']:.2f}")
        ax.set_xlabel("ground truth")
        ax.set_ylabel("predicted")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "pred_vs_gt_scatter.png", dpi=110)
    plt.close(fig)

    # 2) residual histograms per axis (bias = histogram not centered on 0).
    fig, axes = plt.subplots(1, ACTION_DIM, figsize=(4 * ACTION_DIM, 3.5))
    for j, ax in enumerate(np.atleast_1d(axes)):
        err = pred[:, j] - gt[:, j]
        ax.hist(err, bins=60)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.axvline(err.mean(), color="r", lw=1, label=f"bias={err.mean():+.3f}")
        ax.set_title(ACTION_DIM_NAMES[j])
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "residual_hist.png", dpi=110)
    plt.close(fig)

    # 3) per-episode normalized RMSE bar (worst on the right).
    rows = sorted(report["per_episode"], key=lambda r: r["norm_rmse"])
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.12), 4))
    ax.bar(range(len(rows)), [r["norm_rmse"] for r in rows])
    ax.axhline(report["outliers"]["cutoff"], color="r", ls="--", lw=1, label="outlier cutoff")
    ax.set_xlabel("episode (sorted)")
    ax.set_ylabel("normalized RMSE")
    ax.set_title("per-episode fit quality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_episode_rmse.png", dpi=110)
    plt.close(fig)

    logging.info("wrote plots to %s", out_dir)


def write_csv(ep_rows: list[dict], path: pathlib.Path) -> None:
    import csv

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "task", "ep_len", "n_pred", "norm_rmse", "norm_mae", *[f"rmse_{n}" for n in ACTION_DIM_NAMES]])
        for r in sorted(ep_rows, key=lambda r: r["norm_rmse"], reverse=True):
            w.writerow(
                [r["episode"], r["task"], r["ep_len"], r["n_pred"], f"{r['norm_rmse']:.6f}", f"{r['norm_mae']:.6f}", *[f"{v:.6f}" for v in r["per_dim_rmse"]]]
            )


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    np.random.seed(args.seed)

    root = HF_LEROBOT_HOME / args.repo_id
    logging.info("loading dataset %s (cache: %s)", args.repo_id, root)
    ds = LeRobotDataset(args.repo_id)
    logging.info("dataset has %d episodes, %d frames, fps=%s", ds.num_episodes, len(ds), ds.fps)

    logging.info("connecting to policy server at ws://%s:%d", args.host, args.port)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_meta = client.get_server_metadata()
    logging.info("server metadata: %s", server_meta)

    # Label this run (for the output folder + report) from the served model when
    # the user didn't pass one explicitly.
    label = args.label or (server_meta or {}).get("model") or "eval"

    records = collect_predictions(ds, client, args)
    pred, gt, offset, ep = records["pred"], records["gt"], records["offset"], records["ep"]
    if pred.size == 0:
        raise SystemExit("no predictions collected — is the dataset empty?")

    n_inferences = int(
        sum((length + args.replan_steps - 1) // args.replan_steps for length in records["ep_lengths"].values())
    )

    ep_rows = per_episode_stats(pred, gt, ep, records["ep_tasks"], records["ep_lengths"])
    report = {
        "meta": {
            "label": label,
            "server_metadata": server_meta,
            "repo_id": args.repo_id,
            "host": args.host,
            "port": args.port,
            "replan_steps": args.replan_steps,
            "samples_per_obs": max(1, args.samples_per_obs),
            "outlier_sigma": args.outlier_sigma,
            "n_episodes_evaluated": len(records["ep_lengths"]),
            "n_inferences": n_inferences,
            "n_matched": int(pred.shape[0]),
            "action_dim_names": ACTION_DIM_NAMES,
        },
        "per_dim": per_dim_stats(pred, gt),
        "per_offset": per_offset_rmse(pred, gt, offset),
        "per_task": per_task_stats(pred, gt, ep, records["ep_tasks"]),
        "per_episode": ep_rows,
        "outliers": flag_outliers(ep_rows, args.outlier_sigma, args.top_k_worst),
    }
    if records["samples"] is not None:
        report["best_of_n"] = best_of_n_stats(records["samples"], gt)

    print_report(report)

    # Each run lands in its own subfolder so models are easy to compare.
    out_dir = pathlib.Path(args.output_dir) / label
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Fail fast if the directory exists but isn't writable (e.g. created as
        # root by a previous `docker compose` run via the ./data volume mount).
        (out_dir / ".write_test").touch()
        (out_dir / ".write_test").unlink()
    except (PermissionError, OSError) as e:
        fallback = pathlib.Path(tempfile.gettempdir()) / "forklift_eval"
        logging.warning(
            "cannot write to %s (%s); falling back to %s. "
            "Tip: `sudo chown -R $USER data` or pass --output-dir to a writable path.",
            out_dir,
            e,
            fallback,
        )
        out_dir = fallback
        out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "eval_report.json"
    json_path.write_text(json.dumps(report, indent=2))
    logging.info("wrote JSON report to %s", json_path)

    csv_path = out_dir / "per_episode.csv"
    write_csv(ep_rows, csv_path)
    logging.info("wrote per-episode CSV to %s", csv_path)

    if args.make_plots:
        write_plots(records, report, out_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
