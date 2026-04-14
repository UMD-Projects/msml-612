#!/bin/bash
# launch_experiment.sh — create a queued spot TPU that automatically runs an experiment.
#
# Usage:
#   bash project/spot_pipeline/launch_experiment.sh \
#     --experiment hybrid_dit+hilbert_3to1 \
#     --tpu-name msml612-spot-hybrid-3to1 \
#     --zone europe-west4-a \
#     --accelerator v6e-4
#
# The TPU comes up with bootstrap.sh as its startup script. Bootstrap reads the
# experiment name from TPU metadata and runs the corresponding training. On
# preemption, GCP recreates the TPU and the same startup script runs again,
# auto-resuming from the last wandb checkpoint.

set -euo pipefail

# -- Defaults -----------------------------------------------------------------
EXPERIMENT_NAME=""
TPU_NAME=""
ZONE="europe-west4-a"
ACCELERATOR="v6e-4"
GCS_BUCKET="msml612-diffusion-data"
RUNTIME=""  # auto-detect from accelerator
SPOT="--spot"
QUEUED=true
WANDB_KEY=""
HF_TOKEN=""

# -- Parse args ---------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment)    EXPERIMENT_NAME="$2"; shift 2 ;;
        --tpu-name)      TPU_NAME="$2"; shift 2 ;;
        --zone)          ZONE="$2"; shift 2 ;;
        --accelerator)   ACCELERATOR="$2"; shift 2 ;;
        --gcs-bucket)    GCS_BUCKET="$2"; shift 2 ;;
        --runtime)       RUNTIME="$2"; shift 2 ;;
        --no-spot)       SPOT=""; shift ;;
        --no-queued)     QUEUED=false; shift ;;
        --wandb-key)     WANDB_KEY="$2"; shift 2 ;;
        --hf-token)      HF_TOKEN="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXPERIMENT_NAME" ]] || [[ -z "$TPU_NAME" ]]; then
    echo "Usage: $0 --experiment NAME --tpu-name NAME [--zone ZONE] [--accelerator TYPE]"
    exit 1
fi

# Auto-detect runtime version from accelerator type
if [[ -z "$RUNTIME" ]]; then
    case "$ACCELERATOR" in
        v6e*) RUNTIME="v2-alpha-tpuv6e" ;;
        v5e*|v5litepod*) RUNTIME="v2-alpha-tpuv5-lite" ;;
        v5p*) RUNTIME="v2-alpha-tpuv5" ;;
        v4*) RUNTIME="tpu-ubuntu2204-base" ;;
        *) echo "Unknown accelerator: $ACCELERATOR"; exit 1 ;;
    esac
fi

# Read wandb / HF tokens from local files if not provided
if [[ -z "$WANDB_KEY" ]] && [[ -f "$HOME/.netrc" ]]; then
    WANDB_KEY=$(awk '/machine api.wandb.ai/,/password/' "$HOME/.netrc" | grep -oP 'password \K\S+' || true)
fi
if [[ -z "$HF_TOKEN" ]]; then
    HF_TOKEN=$(grep -oP 'HF_TOKEN=\K\S+' "$HOME/.bashrc" 2>/dev/null || true)
fi

# Find bootstrap.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_SH="$SCRIPT_DIR/bootstrap.sh"
if [[ ! -f "$BOOTSTRAP_SH" ]]; then
    echo "ERROR: bootstrap.sh not found at $BOOTSTRAP_SH"
    exit 1
fi

echo "==================================================================="
echo "Launching spot experiment"
echo "==================================================================="
echo "  Experiment:    $EXPERIMENT_NAME"
echo "  TPU name:      $TPU_NAME"
echo "  Zone:          $ZONE"
echo "  Accelerator:   $ACCELERATOR"
echo "  Runtime:       $RUNTIME"
echo "  GCS bucket:    $GCS_BUCKET"
echo "  Spot:          $([[ -n "$SPOT" ]] && echo yes || echo no)"
echo "  Queued:        $QUEUED"
echo "==================================================================="

# Build the metadata flag with the bootstrap script and experiment config
# We use 'startup-script' (file contents) and individual metadata key/values for
# experiment_name, gcs_bucket, wandb_api_key, hf_token.
METADATA_ARGS=(
    "--metadata-from-file=startup-script=$BOOTSTRAP_SH"
    "--metadata=experiment_name=$EXPERIMENT_NAME,gcs_bucket=$GCS_BUCKET,wandb_api_key=$WANDB_KEY,hf_token=$HF_TOKEN"
)

if [[ "$QUEUED" == "true" ]]; then
    QR_NAME="qr-${TPU_NAME}"
    echo "Submitting queued resource request: $QR_NAME"
    gcloud compute tpus queued-resources create "$QR_NAME" \
        --node-id="$TPU_NAME" \
        --zone="$ZONE" \
        --accelerator-type="$ACCELERATOR" \
        --runtime-version="$RUNTIME" \
        $SPOT \
        "${METADATA_ARGS[@]}"
    echo "Queued. Check status with:"
    echo "  gcloud compute tpus queued-resources list --zone=$ZONE"
else
    echo "Creating TPU directly (not queued)..."
    gcloud compute tpus tpu-vm create "$TPU_NAME" \
        --zone="$ZONE" \
        --accelerator-type="$ACCELERATOR" \
        --version="$RUNTIME" \
        $SPOT \
        "${METADATA_ARGS[@]}"
fi

echo "Done. The TPU will boot, run bootstrap.sh, and start training automatically."
