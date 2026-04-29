#!/usr/bin/env bash
# Orchestrate FID computation for top runs across two EU v6e TPUs.
# Distributes work: each TPU runs its assigned runs serially via tmux,
# the local script just kicks off both and waits.
#
# Usage: orchestrate_fid.sh [n_samples] [diffusion_steps]
set -e
N_SAMPLES="${1:-2048}"
STEPS="${2:-50}"

ZONE=europe-west4-a
TPU_A=msml612-d-rescue-eu
TPU_B=msml612-d-zz1to1-eu

# Allocation — 4 runs per TPU
RUNS_A=("6lkmptiw" "xf18pnxa" "mq05643r" "srm8avoo")
RUNS_B=("tqlyjd5z" "qr2702yx" "3u0fbiwp" "azioyhka")
# leftover (run if time): m7otn3bn

# Build the remote script that runs the assigned IDs serially in tmux
make_remote_script() {
    local label="$1"; shift
    local runs="$@"
    cat <<REMOTE
#!/bin/bash
LOG=/tmp/orchestrate_${label}.log
echo "[orch ${label}] start $(date)" | tee -a \$LOG
for rid in ${runs}; do
    echo "[orch ${label}] === \$rid ===" | tee -a \$LOG
    bash /home/mrwhite0racle/run_full_fid.sh \$rid ${N_SAMPLES} ${STEPS} 2>&1 | tee -a \$LOG
    echo "[orch ${label}] FINISHED \$rid exit=\${PIPESTATUS[0]} $(date)" | tee -a \$LOG
done
echo "[orch ${label}] all done $(date)" | tee -a \$LOG
REMOTE
}

# Push and run each
for tpu_label in A B; do
    if [ "$tpu_label" = "A" ]; then tpu=$TPU_A; runs="${RUNS_A[@]}"; else tpu=$TPU_B; runs="${RUNS_B[@]}"; fi
    echo "[main] launching $tpu with: $runs"
    make_remote_script "$tpu_label" $runs > "/tmp/orch_${tpu_label}.sh"
    gcloud compute tpus tpu-vm scp "/tmp/orch_${tpu_label}.sh" "${tpu}:/home/mrwhite0racle/orch.sh" --zone=$ZONE 2>&1 | tail -1
    gcloud compute tpus tpu-vm ssh $tpu --zone=$ZONE --worker=0 --command="chmod +x /home/mrwhite0racle/orch.sh; tmux kill-session -t orch 2>/dev/null; tmux new-session -d -s orch 'bash /home/mrwhite0racle/orch.sh'; sleep 1; tmux ls" 2>&1 | tail -3
done
echo "[main] both TPUs launched. Tail logs with:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_A --zone=$ZONE --worker=0 --command='tail -f /tmp/orchestrate_A.log'"
echo "  gcloud compute tpus tpu-vm ssh $TPU_B --zone=$ZONE --worker=0 --command='tail -f /tmp/orchestrate_B.log'"
