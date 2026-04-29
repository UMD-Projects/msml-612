#!/usr/bin/env bash
# Wrapper to avoid shell-quote nesting issues through gcloud ssh + tmux.
# Usage: run_fid_sample.sh <wandb_id> <n_samples> [<diffusion_steps>]
set -e
WANDB_ID="${1:?wandb_id required}"
N_SAMPLES="${2:-2048}"
DIFFUSION_STEPS="${3:-50}"
OUT_DIR="/tmp/fid_samples/${WANDB_ID}"
LOG="/tmp/fid_${WANDB_ID}.log"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flaxdiff
cd /home/mrwhite0racle
mkdir -p "${OUT_DIR}"

echo "[wrapper] starting sample_for_fid.py wandb_id=${WANDB_ID} n=${N_SAMPLES} steps=${DIFFUSION_STEPS}" | tee -a "${LOG}"
python sample_for_fid.py \
    --wandb_id "${WANDB_ID}" \
    --n_samples "${N_SAMPLES}" \
    --diffusion_steps "${DIFFUSION_STEPS}" \
    --out_dir "${OUT_DIR}" 2>&1 | tee -a "${LOG}"
echo "[wrapper] done. exit=$?" | tee -a "${LOG}"
