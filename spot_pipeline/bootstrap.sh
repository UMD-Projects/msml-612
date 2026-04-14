#!/bin/bash
# bootstrap.sh — runs on every spot TPU boot (including after preemption recovery).
#
# Reads experiment configuration from TPU metadata, sets up the conda environment,
# pulls the latest FlaxDiff code, then launches the configured training experiment
# with auto-resume from the latest wandb checkpoint.
#
# Set as TPU startup-script via tpu_tool.sh / launch_experiment.sh.
#
# Required TPU metadata fields:
#   experiment_name   - one of: simple_dit, simple_dit+hilbert, hybrid_dit_3to1,
#                       hybrid_dit+hilbert_3to1, hybrid_dit+hilbert_1to1,
#                       hybrid_dit+hilbert_all_ssm
#   gcs_bucket        - GCS bucket for code/configs/checkpoints (e.g. msml612-diffusion-data)
#   wandb_api_key     - wandb API key (set in metadata at TPU creation time)
#   hf_token          - HuggingFace token (optional, only if pulling from HF)
#
# Logs:
#   /tmp/bootstrap.log on the TPU
#   gs://${GCS_BUCKET}/spot_logs/${HOSTNAME}_${TIMESTAMP}.log (uploaded periodically)

set -uo pipefail

# ----- Logging setup ----------------------------------------------------------
LOG_FILE=/tmp/bootstrap.log
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] === bootstrap.sh started ==="

# Best-effort: upload the log to GCS every 60s so we can debug failures off-TPU.
upload_log_loop() {
    while true; do
        sleep 60
        gsutil -q cp "$LOG_FILE" "gs://${GCS_BUCKET}/spot_logs/$(hostname)_bootstrap.log" 2>/dev/null || true
    done
}

# ----- Read TPU metadata ------------------------------------------------------
META_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
META_HDR="Metadata-Flavor: Google"

read_meta() {
    curl -sf -H "$META_HDR" "$META_URL/$1" 2>/dev/null || echo ""
}

EXPERIMENT_NAME=$(read_meta experiment_name)
GCS_BUCKET=$(read_meta gcs_bucket)
WANDB_API_KEY=$(read_meta wandb_api_key)
HF_TOKEN=$(read_meta hf_token)

if [[ -z "$EXPERIMENT_NAME" ]]; then
    echo "ERROR: experiment_name metadata is required"
    exit 1
fi
if [[ -z "$GCS_BUCKET" ]]; then
    GCS_BUCKET="msml612-diffusion-data"
    echo "WARN: gcs_bucket metadata not set, using default: $GCS_BUCKET"
fi

export GCS_BUCKET WANDB_API_KEY HF_TOKEN
upload_log_loop &  # background log uploader

echo "[$(date)] experiment_name=$EXPERIMENT_NAME"
echo "[$(date)] gcs_bucket=$GCS_BUCKET"
echo "[$(date)] hostname=$(hostname)"

# ----- Switch to the mrwhite0racle user (if running as root) ------------------
if [[ "$(id -un)" != "mrwhite0racle" ]]; then
    if id mrwhite0racle &>/dev/null; then
        echo "[$(date)] Re-execing as mrwhite0racle..."
        exec sudo -u mrwhite0racle -E -H bash "$0" "$@"
    fi
fi

cd "$HOME"

# ----- Run setup_tpu.sh if conda environment is missing -----------------------
if [[ ! -x "$HOME/miniconda3/envs/flaxdiff/bin/python" ]]; then
    echo "[$(date)] conda env not found, running setup_tpu.sh..."
    if [[ ! -f "$HOME/setup_tpu.sh" ]]; then
        gsutil cp "gs://${GCS_BUCKET}/bootstrap/setup_tpu.sh" "$HOME/setup_tpu.sh" \
            || curl -sLo "$HOME/setup_tpu.sh" \
                https://raw.githubusercontent.com/AshishKumar4/tpu-tools/main/setup_tpu.sh
    fi
    chmod +x "$HOME/setup_tpu.sh"
    bash "$HOME/setup_tpu.sh" --dev --mount-gcs="$GCS_BUCKET"
else
    echo "[$(date)] conda env already present, skipping setup_tpu.sh"
fi

export PATH="$HOME/miniconda3/envs/flaxdiff/bin:$HOME/miniconda3/bin:$PATH"

# Make sure libgl1 is installed (opencv-python-headless dep on some TPU images)
sudo apt-get install -y libgl1-mesa-glx 2>&1 | tail -2 || true

# ----- Pull latest FlaxDiff code ---------------------------------------------
mkdir -p "$HOME/research"
cd "$HOME/research"

if [[ ! -d ".git" ]]; then
    echo "[$(date)] Cloning FlaxDiff..."
    git clone https://github.com/AshishKumar4/FlaxDiff.git . || {
        echo "ERROR: git clone failed"
        exit 1
    }
else
    echo "[$(date)] Pulling latest FlaxDiff..."
    git fetch origin main && git reset --hard origin/main || true
fi

# Force-reinstall flaxdiff package from the latest source (so bug fixes propagate)
"$HOME/miniconda3/envs/flaxdiff/bin/pip" install -e . --quiet 2>&1 | tail -2 || \
    "$HOME/miniconda3/envs/flaxdiff/bin/pip" install --force-reinstall --no-deps git+https://github.com/AshishKumar4/FlaxDiff.git --quiet 2>&1 | tail -2

# ----- Pull latest project configs (ablation_baseline.sh etc) ----------------
PROJECT_DIR="$HOME/msml612_project"
if [[ ! -d "$PROJECT_DIR/.git" ]]; then
    git clone https://github.com/UMD-Projects/msml-612.git "$PROJECT_DIR" 2>/dev/null || \
        mkdir -p "$PROJECT_DIR/configs"
fi

# Copy the ablation script to the canonical location expected by run scripts
ABLATION_SCRIPT="$HOME/research/ablation_baseline.sh"
if [[ -f "$PROJECT_DIR/project/configs/ablation_baseline.sh" ]]; then
    cp "$PROJECT_DIR/project/configs/ablation_baseline.sh" "$ABLATION_SCRIPT"
elif [[ -f "$PROJECT_DIR/configs/ablation_baseline.sh" ]]; then
    cp "$PROJECT_DIR/configs/ablation_baseline.sh" "$ABLATION_SCRIPT"
else
    # Fall back: pull from GCS
    gsutil cp "gs://${GCS_BUCKET}/bootstrap/ablation_baseline.sh" "$ABLATION_SCRIPT" || true
fi

# ----- Look up wandb run ID for resume ---------------------------------------
WANDB_ID_FILE="gs://${GCS_BUCKET}/experiments/${EXPERIMENT_NAME}/wandb_id.txt"
RESUME_ARG=""
EXISTING_WANDB_ID=$(gsutil cat "$WANDB_ID_FILE" 2>/dev/null || echo "")
if [[ -n "$EXISTING_WANDB_ID" ]]; then
    echo "[$(date)] Resuming wandb run: $EXISTING_WANDB_ID"
    RESUME_ARG="--resume_last_run $EXISTING_WANDB_ID"
else
    echo "[$(date)] No previous wandb run found for $EXPERIMENT_NAME, starting fresh"
fi

# ----- wandb login -----------------------------------------------------------
if [[ -n "$WANDB_API_KEY" ]]; then
    echo "machine api.wandb.ai login user password $WANDB_API_KEY" > "$HOME/.netrc"
    chmod 600 "$HOME/.netrc"
fi

# ----- Launch the experiment -------------------------------------------------
TRAIN_LOG="$HOME/training_${EXPERIMENT_NAME}.log"
echo "[$(date)] Launching experiment: $EXPERIMENT_NAME"
echo "[$(date)] Resume arg: $RESUME_ARG"
echo "[$(date)] Training log: $TRAIN_LOG"

# Run the ablation script with the resume flag if applicable.
# The script itself contains the wandb run details and disk watchdog.
bash "$ABLATION_SCRIPT" "$EXPERIMENT_NAME" $RESUME_ARG 2>&1 | tee -a "$TRAIN_LOG"

EXIT_CODE=${PIPESTATUS[0]}
echo "[$(date)] Training exited with code $EXIT_CODE"

# Upload the final training log to GCS (best-effort)
gsutil cp "$TRAIN_LOG" "gs://${GCS_BUCKET}/spot_logs/$(hostname)_${EXPERIMENT_NAME}.log" 2>/dev/null || true

exit $EXIT_CODE
