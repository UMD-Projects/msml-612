#!/usr/bin/env bash
# Pull all /tmp/fid_metrics_*.json from both EU TPUs into project/data/fid/.
set -e
ZONE=europe-west4-a
LOCAL=/home/mrwhite0racle/Desktop/UMDCourseWork/MSML612/project/data/fid
mkdir -p "$LOCAL"

for tpu in msml612-d-rescue-eu msml612-d-zz1to1-eu; do
    echo "[pull] $tpu"
    # List files first
    files=$(gcloud compute tpus tpu-vm ssh "$tpu" --zone=$ZONE --worker=0 \
        --command='ls /tmp/fid_metrics_*.json 2>/dev/null' 2>/dev/null)
    if [ -z "$files" ]; then
        echo "  no metrics files"; continue
    fi
    for f in $files; do
        # SCP one at a time
        gcloud compute tpus tpu-vm scp "${tpu}:${f}" "$LOCAL/" --zone=$ZONE 2>&1 | tail -1
    done
done

echo "[pull] aggregate"
python3 /home/mrwhite0racle/Desktop/UMDCourseWork/MSML612/project/aggregate_fid.py \
    "$LOCAL/fid_metrics_*.json" \
    -o /home/mrwhite0racle/Desktop/UMDCourseWork/MSML612/project/docs/fid_results.md
