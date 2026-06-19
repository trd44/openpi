#!/bin/bash
#SBATCH --job-name=pi05-forklift
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/cluster/tufts/hrilab/tdugga02/openpi/slurm/logs/%x-%j.out
#SBATCH --error=/cluster/tufts/hrilab/tdugga02/openpi/slurm/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tdugga02@tufts.edu

set -euo pipefail

module load uv/0.9.18
export UV_CACHE_DIR=/cluster/tufts/hrilab/tdugga02/.uv-cache
export HF_HOME=/cluster/tufts/hrilab/tdugga02/.cache/huggingface
export WANDB_DIR=/cluster/tufts/hrilab/tdugga02/.cache/wandb
# OPENPI_DATA_HOME is where the pi05_base checkpoint + gs:// assets download to.
# Without this it defaults to ~/.cache/openpi and fills the small home quota.
export OPENPI_DATA_HOME=/cluster/tufts/hrilab/tdugga02/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /cluster/tufts/hrilab/tdugga02/openpi

nvidia-smi
uv run python -c "import jax; print('JAX devices:', jax.devices())"

# First submission uses --overwrite. On a requeue (e.g. if 48h isn't enough),
# delete the --overwrite flag so train.py resumes from the latest checkpoint
# in checkpoints/pi05_forklift/forklift-full/.
uv run scripts/train.py pi05_forklift --exp-name=forklift-full --overwrite
