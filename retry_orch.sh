#!/bin/bash
# Retry the broken runs with the fixed sample_for_fid.py.
# Wipe their sample dirs first so we get fresh samples.
LOG=/tmp/retry_B.log
echo "[retry] start $(date)" | tee -a $LOG
for rid in azioyhka tqlyjd5z qr2702yx mq05643r; do
    echo "[retry] === $rid ===" | tee -a $LOG
    rm -rf "/tmp/fid_samples/$rid"
    rm -f "/tmp/fid_metrics_$rid.json"
    bash /home/mrwhite0racle/run_full_fid.sh "$rid" 1024 50 2>&1 | tee -a $LOG
    echo "[retry] FINISHED $rid exit=${PIPESTATUS[0]} $(date)" | tee -a $LOG
done
echo "[retry] all done $(date)" | tee -a $LOG
