#!/bin/bash
# Run a sequence of ablation experiments on a single TPU.
# Usage: ./run_all_ablations.sh <space-separated experiment list>
# Example: ./run_all_ablations.sh hybrid_dit+hilbert_3to1 hybrid_dit+hilbert_1to1

set -u  # don't use set -e — we want to continue past failures

if [ $# -eq 0 ]; then
    echo "Usage: $0 <experiment1> [experiment2] ..."
    echo "Available: simple_dit simple_dit+hilbert hybrid_dit_3to1 hybrid_dit+hilbert_3to1 hybrid_dit+hilbert_1to1 hybrid_dit+hilbert_all_ssm"
    exit 1
fi

LOG_DIR=/home/mrwhite0racle/ablation_logs_fixed
mkdir -p "$LOG_DIR"

for exp in "$@"; do
    log_file="${LOG_DIR}/${exp}.log"
    echo "=================================================================" | tee -a "$log_file"
    echo "[$(date)] Starting experiment: $exp" | tee -a "$log_file"
    echo "=================================================================" | tee -a "$log_file"

    # Clean orphaned orbax tmp files (avoid the 'invalid literal for int()' crash)
    find /home/mrwhite0racle/research/checkpoints -name '*.orbax-checkpoint-tmp*' -exec rm -rf {} + 2>/dev/null

    # Run the experiment (blocking; ablation_baseline.sh has its own watchdog)
    bash /home/mrwhite0racle/research/ablation_baseline.sh "$exp" 2>&1 | tee -a "$log_file"

    rc=$?
    echo "[$(date)] Experiment $exp finished with exit code $rc" | tee -a "$log_file"
done

echo "[$(date)] All experiments done."
