#!/bin/bash
# upload_bootstrap.sh — push bootstrap.sh and supporting files to GCS so new TPUs can fetch them.

set -euo pipefail

GCS_BUCKET="${1:-msml612-diffusion-data}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SCRIPT_DIR = .../MSML612/project/spot_pipeline
# PROJECT_ROOT should be .../MSML612/project so that project/configs and
# project/../tpu-tools both resolve correctly.
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Uploading bootstrap files to gs://${GCS_BUCKET}/bootstrap/..."

# Bootstrap script (run on every TPU boot)
gsutil cp "$SCRIPT_DIR/bootstrap.sh" "gs://${GCS_BUCKET}/bootstrap/bootstrap.sh"

# setup_tpu.sh (installs conda + deps; called by bootstrap.sh on first boot).
# Look in tpu-tools next to the repo root, and also in the local ml-poc-notebooks
# canonical location as a fallback.
SETUP_TPU_CANDIDATES=(
    "$REPO_ROOT/tpu-tools/setup_tpu.sh"
    "$HOME/Desktop/ml-poc-notebooks/diffusion experiments/tpu-tools/setup_tpu.sh"
)
for SETUP_TPU in "${SETUP_TPU_CANDIDATES[@]}"; do
    if [[ -f "$SETUP_TPU" ]]; then
        gsutil cp "$SETUP_TPU" "gs://${GCS_BUCKET}/bootstrap/setup_tpu.sh"
        break
    fi
done
if [[ ! -f "$SETUP_TPU" ]]; then
    echo "WARN: setup_tpu.sh not found; skipping"
fi

# ablation_baseline.sh (the experiment runner; bootstrap.sh execs this).
ABLATION="$PROJECT_ROOT/configs/ablation_baseline.sh"
if [[ -f "$ABLATION" ]]; then
    gsutil cp "$ABLATION" "gs://${GCS_BUCKET}/bootstrap/ablation_baseline.sh"
else
    echo "ERROR: ablation_baseline.sh not found at $ABLATION"
    exit 1
fi

echo "Done. Files uploaded:"
gsutil ls "gs://${GCS_BUCKET}/bootstrap/"
