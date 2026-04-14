# Spot TPU Training Pipeline

A reproducible, fire-and-forget pipeline for running ablation experiments on Google Cloud spot/preemptible TPUs. Designed around the recommended Google Cloud patterns for spot TPU usage:

1. **Queued Resources API** — let GCP queue and provision TPUs when capacity is available, instead of polling/retrying manually.
2. **Startup scripts in TPU metadata** — every spot TPU bootstraps itself on every boot (including after preemption recovery). No manual SSH dance.
3. **State on GCS, not on the TPU** — checkpoints, code, and the experiment manifest all live on GCS. The TPU disk is treated as ephemeral.
4. **Auto-resume from checkpoint** — when a preempted TPU comes back, the startup script picks up the same experiment from where it left off using `--resume_last_run <wandb_id>`.

References:
- https://cloud.google.com/tpu/docs/queued-resources
- https://cloud.google.com/tpu/docs/preemption
- https://cloud.google.com/tpu/docs/setup-gcp-account#create-startup-script

## Cost model

The bottleneck for GCS cost is **cross-region egress**, NOT storage or operations. We co-locate everything per-region:

| TPU zone | Checkpoint bucket | Dataset bucket |
|----------|-------------------|---------------|
| `us-central2-b` (v4) | `gs://msml612-diffusion-data/` | same — already there |
| `europe-west4-a` (v6e spot) | TBD: `gs://msml612-checkpoints-eu/` | for Oxford Flowers ablations, the dataset is loaded via TFDS locally; not an issue |

Estimated extra GCS cost per month with the supervisor pipeline running:
- Checkpoint storage with 7-day lifecycle: ~$2-3/month
- API operations: <$0.50/month
- Cross-region egress: $0 (avoided by region-co-location)

## Files

| File | Purpose |
|------|---------|
| `bootstrap.sh` | Startup script that runs on every spot TPU boot. Installs deps, mounts GCS, pulls latest code from FlaxDiff, then launches the configured experiment with auto-resume. |
| `manifest.json` | Source-of-truth list of experiments to run, their assigned TPU, and the wandb run ID for resume. |
| `launch_experiment.sh` | Convenience wrapper that creates a queued spot TPU with the right metadata to run a specific experiment. |
| `supervisor.py` | Long-running Python supervisor that watches the manifest, queues TPUs, and re-queues preempted ones. |
| `upload_bootstrap.sh` | Uploads the latest `bootstrap.sh` to the GCS bucket so new TPUs can fetch it. |

## How to use

### One-time setup

```bash
# Upload the bootstrap script to GCS
bash project/spot_pipeline/upload_bootstrap.sh
```

### Launch a single experiment on a queued spot TPU

```bash
bash project/spot_pipeline/launch_experiment.sh \
  --experiment hybrid_dit+hilbert_3to1 \
  --tpu-name msml612-spot-hybrid-3to1 \
  --zone europe-west4-a \
  --accelerator v6e-4
```

This will:
1. Submit a queued resource request for the TPU (no manual retry needed)
2. Set the startup script to fetch `bootstrap.sh` from GCS and run it
3. Pass the experiment name as TPU metadata so `bootstrap.sh` knows what to launch
4. Return immediately — GCP handles the rest

### Run the full ablation grid in fire-and-forget mode

```bash
python3 project/spot_pipeline/supervisor.py \
  --manifest project/spot_pipeline/manifest.json \
  --bucket msml612-diffusion-data
```

The supervisor:
- Reads `manifest.json` for the list of experiments
- For each unstarted experiment: queues a TPU
- For each running experiment: monitors wandb; if the TPU dies, re-queues
- Writes back to `manifest.json` with current status (TPU name, wandb id, etc.)
- Survives SSH disconnects when run inside a tmux/screen session

## Failure modes and recovery

| Failure | Recovery |
|---------|---------|
| Spot TPU preempted mid-training | Queued resources auto-recreate. New TPU boots → `bootstrap.sh` → resumes from latest wandb checkpoint. ~10-15 min lost. |
| TPU "internal error" on creation | Queued resources retry automatically until capacity is available. |
| `bootstrap.sh` itself fails | Logs go to `/tmp/bootstrap.log` on the TPU AND copied to `gs://bucket/spot_logs/<tpu_name>.log` for off-TPU debugging. |
| Wandb checkpoint missing (run never reached top-5) | Training restarts from epoch 0 for that experiment. Acceptable for current Oxford Flowers ablation runs. |

## Design choices

### Why startup script, not custom image?

A custom TPU image would skip the ~10-min conda install on every boot, but building/maintaining a custom image is 10x more work. For our 3h training jobs, 10 min of setup overhead is acceptable.

### Why queued resources, not manual create + retry?

Manual create returns immediately on capacity-out errors. We were burning hours wrapping it in retry loops. Queued resources is the same API call, but GCP handles the queueing for free.

### Why is the manifest a JSON file, not a database?

For 6-12 ablation experiments, a JSON file is plenty. Database is overkill and adds infra complexity.
