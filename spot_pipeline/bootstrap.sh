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

set -o pipefail  # NOT set -u — startup-script runs as root with no HOME set.

# ----- Logging setup ----------------------------------------------------------
LOG_FILE=/tmp/bootstrap.log
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] === bootstrap.sh started ==="

# When invoked as a GCP startup-script (under systemd with a minimal env), HOME
# is not set at all. Pick a sane default so subsequent cd "$HOME" etc. work.
export HOME="${HOME:-/home/mrwhite0racle}"
export USER="${USER:-$(id -un)}"

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
# Extra CLI args forwarded verbatim to ablation_baseline.sh. Lets a QR override
# dataset/batch-size/etc. without needing a new experiment case in the script.
# Example for LAION scale: "--dataset laion12m_coco --batch_size 128 --learning_rate 0.0001 --num_heads 12 --emb_features 768 --epochs 2000 --steps_per_epoch 50000 --val_steps_per_epoch 4"
EXTRA_CLI_ARGS=$(read_meta extra_cli_args)

if [[ -z "$EXPERIMENT_NAME" ]]; then
    echo "ERROR: experiment_name metadata is required"
    exit 1
fi
if [[ -z "$GCS_BUCKET" ]]; then
    GCS_BUCKET="msml612-diffusion-data"
    echo "WARN: gcs_bucket metadata not set, using default: $GCS_BUCKET"
fi

export GCS_BUCKET WANDB_API_KEY HF_TOKEN EXTRA_CLI_ARGS
upload_log_loop &  # background log uploader

echo "[$(date)] experiment_name=$EXPERIMENT_NAME"
echo "[$(date)] gcs_bucket=$GCS_BUCKET"
echo "[$(date)] hostname=$(hostname)"

# ----- Switch to the mrwhite0racle user (if running as root) ------------------
# On a supervisor-spawned spot TPU nobody has SSH'd in, so the mrwhite0racle
# user may not exist yet. Create it if necessary, then re-exec as that user.
# Do this only when we're running as root (GCP startup-script default).
if [[ "$(id -un)" == "root" ]]; then
    if ! id mrwhite0racle &>/dev/null; then
        echo "[$(date)] Creating mrwhite0racle user (startup-script context)..."
        useradd -m -s /bin/bash -u 1000 -G sudo mrwhite0racle 2>/dev/null \
            || useradd -m -s /bin/bash -G sudo mrwhite0racle
        # Passwordless sudo for the user so apt-get in bootstrap still works
        echo "mrwhite0racle ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/mrwhite0racle
        chmod 0440 /etc/sudoers.d/mrwhite0racle
    fi
    echo "[$(date)] Re-execing as mrwhite0racle..."
    # Copy the script to the user's home because GCP stages startup-scripts in
    # /tmp/metadata-scripts*/ which is root-only and mrwhite0racle cannot even
    # traverse the directory. Read once as root, write to a user-owned path.
    SELF="$0"
    CP_SCRIPT=/home/mrwhite0racle/bootstrap.sh
    cp "$SELF" "$CP_SCRIPT"
    chown mrwhite0racle:mrwhite0racle "$CP_SCRIPT"
    chmod 755 "$CP_SCRIPT"
    exec sudo -u mrwhite0racle -E -H bash -c "export HOME=/home/mrwhite0racle; cd ~ && bash '$CP_SCRIPT' $*"
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

# ----- Patch training.py: setrlimit hard-limit raise crashes on newer TPU
# images (CAP_SYS_RESOURCE revoked). Wrap the two setrlimit calls in
# try/except so training.py can boot. Idempotent — runs every bootstrap.
echo "[$(date)] Patching training.py setrlimit..."
python3 - <<'PYEOF'
PATH = "training.py"
try:
    with open(PATH) as f: s = f.read()
    old = """    resource.setrlimit(
        resource.RLIMIT_CORE,
        (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    resource.setrlimit(
        resource.RLIMIT_OFILE,
        (65535, 65535))"""
    new = """    try:
        resource.setrlimit(
            resource.RLIMIT_CORE,
            (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (ValueError, OSError):
        pass
    try:
        cur_soft, cur_hard = resource.getrlimit(resource.RLIMIT_OFILE)
        target = 65535
        new_hard = cur_hard
        new_soft = min(target, cur_hard) if cur_hard != resource.RLIM_INFINITY else target
        resource.setrlimit(resource.RLIMIT_OFILE, (new_soft, new_hard))
    except (ValueError, OSError):
        pass"""
    if old in s:
        with open(PATH, "w") as f: f.write(s.replace(old, new))
        print("training.py setrlimit patched")
    elif new in s:
        print("training.py already patched")
    else:
        print("training.py: setrlimit block not found (already patched upstream?)")
except FileNotFoundError:
    print("training.py not found — skipping setrlimit patch")
PYEOF

# Force-reinstall flaxdiff package from the latest source (so bug fixes propagate)
"$HOME/miniconda3/envs/flaxdiff/bin/pip" install -e . --quiet 2>&1 | tail -2 || \
    "$HOME/miniconda3/envs/flaxdiff/bin/pip" install --force-reinstall --no-deps git+https://github.com/AshishKumar4/FlaxDiff.git --quiet 2>&1 | tail -2

# ----- Pull latest project configs (ablation_baseline.sh etc) ----------------
PROJECT_DIR="$HOME/msml612_project"
if [[ ! -d "$PROJECT_DIR/.git" ]]; then
    git clone https://github.com/UMD-Projects/msml-612.git "$PROJECT_DIR" 2>/dev/null || \
        mkdir -p "$PROJECT_DIR/configs"
fi

# Copy the ablation script to the canonical location expected by run scripts.
# Prefer GCS because that's what we push the fixed version to — the UMD-Projects
# msml-612 GitHub repo clone can lag behind and was causing stale scripts to
# win (missing +2d/1:1 cases). Fall back to the project clone only if GCS is
# unreachable.
ABLATION_SCRIPT="$HOME/research/ablation_baseline.sh"
if gsutil cp "gs://${GCS_BUCKET}/bootstrap/ablation_baseline.sh" "$ABLATION_SCRIPT"; then
    echo "[$(date)] ablation_baseline.sh fetched from GCS"
elif [[ -f "$PROJECT_DIR/project/configs/ablation_baseline.sh" ]]; then
    cp "$PROJECT_DIR/project/configs/ablation_baseline.sh" "$ABLATION_SCRIPT"
elif [[ -f "$PROJECT_DIR/configs/ablation_baseline.sh" ]]; then
    cp "$PROJECT_DIR/configs/ablation_baseline.sh" "$ABLATION_SCRIPT"
fi
chmod +x "$ABLATION_SCRIPT" 2>/dev/null || true

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

# Run the ablation script with the resume flag and any extra CLI args forwarded
# from QR metadata (e.g. LAION overrides). Unquoted intentionally so that
# EXTRA_CLI_ARGS can contain multiple --flag value pairs.
bash "$ABLATION_SCRIPT" "$EXPERIMENT_NAME" $RESUME_ARG $EXTRA_CLI_ARGS 2>&1 | tee -a "$TRAIN_LOG"

EXIT_CODE=${PIPESTATUS[0]}
echo "[$(date)] Training exited with code $EXIT_CODE"

# Upload the final training log to GCS (best-effort)
gsutil cp "$TRAIN_LOG" "gs://${GCS_BUCKET}/spot_logs/$(hostname)_${EXPERIMENT_NAME}.log" 2>/dev/null || true

exit $EXIT_CODE
