#!/usr/bin/env bash
# Full per-run FID pipeline: sample N images -> torch-fidelity -> wandb push.
# Usage: run_full_fid.sh <wandb_id> [<n_samples>] [<diffusion_steps>]
set -e
WANDB_ID="${1:?wandb_id required}"
N_SAMPLES="${2:-2048}"
DIFFUSION_STEPS="${3:-50}"
OUT_DIR="/tmp/fid_samples/${WANDB_ID}"
LOG="/tmp/fid_${WANDB_ID}.log"
METRICS_JSON="/tmp/fid_metrics_${WANDB_ID}.json"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flaxdiff
cd /home/mrwhite0racle

# Step 1: sample
echo "[$(date +%H:%M:%S)] [pipeline] sampling ${WANDB_ID} n=${N_SAMPLES}" | tee -a "${LOG}"
mkdir -p "${OUT_DIR}"
existing=$(ls "${OUT_DIR}" 2>/dev/null | wc -l)
if [ "${existing}" -ge "${N_SAMPLES}" ]; then
    echo "[pipeline] already have ${existing} samples, skipping sampling" | tee -a "${LOG}"
else
    python sample_for_fid.py \
        --wandb_id "${WANDB_ID}" \
        --n_samples "${N_SAMPLES}" \
        --diffusion_steps "${DIFFUSION_STEPS}" \
        --out_dir "${OUT_DIR}" 2>&1 | tee -a "${LOG}"
    samp_exit=${PIPESTATUS[0]}
    if [ "${samp_exit}" -ne 0 ]; then
        echo "[pipeline] sampling FAILED exit=${samp_exit}" | tee -a "${LOG}"
        exit ${samp_exit}
    fi
fi

# Step 2: torch-fidelity + wandb push
echo "[$(date +%H:%M:%S)] [pipeline] computing FID for ${WANDB_ID}" | tee -a "${LOG}"
python compute_fid.py \
    --wandb_id "${WANDB_ID}" \
    --samples_dir "${OUT_DIR}" \
    --ref_dir /tmp/oxford_ref \
    --out_json "${METRICS_JSON}" \
    --push_wandb 2>&1 | tee -a "${LOG}"
echo "[$(date +%H:%M:%S)] [pipeline] DONE ${WANDB_ID}" | tee -a "${LOG}"
