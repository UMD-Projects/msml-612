#!/bin/bash
# upload_bootstrap.sh — push bootstrap.sh and supporting files to GCS so new TPUs can fetch them.

set -euo pipefail

GCS_BUCKET="${1:-msml612-diffusion-data}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Uploading bootstrap files to gs://${GCS_BUCKET}/bootstrap/..."

# Bootstrap script (run on every TPU boot)
gsutil cp "$SCRIPT_DIR/bootstrap.sh" "gs://${GCS_BUCKET}/bootstrap/bootstrap.sh"

# setup_tpu.sh (installs conda + deps; called by bootstrap.sh on first boot)
SETUP_TPU="$PROJECT_ROOT/tpu-tools/setup_tpu.sh"
if [[ -f "$SETUP_TPU" ]]; then
    gsutil cp "$SETUP_TPU" "gs://${GCS_BUCKET}/bootstrap/setup_tpu.sh"
else
    echo "WARN: setup_tpu.sh not found at $SETUP_TPU; skipping"
fi

# ablation_baseline.sh (the experiment runner; bootstrap.sh execs this)
ABLATION="$PROJECT_ROOT/configs/ablation_baseline.sh"
if [[ -f "$ABLATION" ]]; then
    gsutil cp "$ABLATION" "gs://${GCS_BUCKET}/bootstrap/ablation_baseline.sh"
fi

echo "Done. Files uploaded:"
gsutil ls "gs://${GCS_BUCKET}/bootstrap/"
