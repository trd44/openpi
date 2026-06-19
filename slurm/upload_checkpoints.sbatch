#!/bin/bash
#SBATCH --job-name=hf-upload-ckpts
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/tufts/hrilab/tdugga02/openpi/slurm/logs/%x-%j.out
#SBATCH --error=/cluster/tufts/hrilab/tdugga02/openpi/slurm/logs/%x-%j.err

set -euo pipefail

module load uv/0.9.18
export UV_CACHE_DIR=/cluster/tufts/hrilab/tdugga02/.uv-cache
export HF_HOME=/cluster/tufts/hrilab/tdugga02/.cache/huggingface

cd /cluster/tufts/hrilab/tdugga02/openpi

uv run python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
base = "/cluster/tufts/hrilab/tdugga02/openpi/checkpoints/pi05_forklift/forklift-full"
jobs = [
    # (step, repo_id, full_overwrite)
    ("15000", "tduggan93/pi05-forklift-full-15000", True),   # exists -> clean overwrite
    ("29999", "tduggan93/pi05-forklift-full-29999", False),  # new repo
]
for step, repo_id, overwrite in jobs:
    print(f"\n=== {repo_id} (step {step}) ===", flush=True)
    url = api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    print(f"repo ready (public): {url}", flush=True)
    kwargs = dict(
        folder_path=f"{base}/{step}",
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["train_state/*", "train_state"],  # inference-only, skip optimizer state
        commit_message=f"pi05 forklift full-finetune checkpoint (step {step}, inference-only)",
    )
    if overwrite:
        kwargs["delete_patterns"] = ["*"]  # remove pre-existing remote files -> full overwrite
    api.upload_folder(**kwargs)
    print(f"DONE: {repo_id}", flush=True)
print("\nALL UPLOADS COMPLETE", flush=True)
PY
