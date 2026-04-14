#!/bin/bash
# =============================================================================
# Ablation Study: Architecture Comparison
# Base config from wandb run j8denkrd (simple_dit+hilbert LDM on Oxford Flowers)
# All experiments share these params — only --architecture varies.
# =============================================================================

set -e

export PATH=/home/mrwhite0racle/miniconda3/envs/flaxdiff/bin:$PATH
cd /home/mrwhite0racle/research

# -- Disk management: prevent wandb cache from filling root disk --
#
# IMPORTANT: This watchdog must NEVER touch ~/.cache/wandb/ — wandb writes
# active run state and artifact downloads there, and racing the trainer caused
# FileNotFoundError crashes mid-training (run wg46wnui, ep ~41). Instead, we
# point WANDB_CACHE_DIR at a tmp location and use the official `wandb artifact
# cache cleanup` CLI which respects in-flight downloads.
export WANDB_CACHE_DIR=/tmp/wandb-cache
mkdir -p "$WANDB_CACHE_DIR"

cleanup_wandb_cache() {
    while true; do
        # Sleep first so we don't race the wandb init at training startup
        sleep 600  # 10 minutes between checks
        usage=$(df / --output=pcent | tail -1 | tr -d ' %')
        if [ "$usage" -le 75 ]; then
            continue  # plenty of headroom, do nothing
        fi
        before=$(du -sm "${WANDB_CACHE_DIR}" 2>/dev/null | cut -f1)
        # Use wandb's official cache cleanup which respects in-flight artifacts.
        # Cap the cache to 5GB; wandb deletes oldest artifacts beyond that.
        if /home/mrwhite0racle/miniconda3/envs/flaxdiff/bin/wandb artifact cache cleanup 5GB \
                >> /tmp/disk_watchdog.log 2>&1; then
            after=$(du -sm "${WANDB_CACHE_DIR}" 2>/dev/null | cut -f1)
            echo "[$(date)] wandb artifact cache cleanup: ${before:-?}MB -> ${after:-?}MB (disk was ${usage}%)" \
                >> /tmp/disk_watchdog.log
        else
            # Fallback if `wandb artifact cache cleanup` is unavailable: only
            # touch our tmp WANDB_CACHE_DIR, NEVER touch ~/.cache/wandb/.
            # Delete files older than 30 minutes to avoid touching anything
            # the trainer is actively using.
            find "${WANDB_CACHE_DIR}" -type f -mmin +30 -delete 2>/dev/null || true
            after=$(du -sm "${WANDB_CACHE_DIR}" 2>/dev/null | cut -f1)
            echo "[$(date)] fallback cleanup: ${before:-?}MB -> ${after:-?}MB (disk was ${usage}%)" \
                >> /tmp/disk_watchdog.log
        fi
    done
}
cleanup_wandb_cache &
WATCHDOG_PID=$!
trap "kill $WATCHDOG_PID 2>/dev/null" EXIT

run_experiment() {
    local arch="$1"
    shift
    python training.py \
      --autoencoder stable_diffusion \
      --autoencoder_opts '{"modelname":"pcuenq/sd-vae-ft-mse-flax"}' \
      --dataset oxford_flowers102 \
      --dataset_path /home/mrwhite0racle/gcs_mount \
      --batch_size 64 \
      --image_size 256 \
      --epochs 150 \
      --noise_schedule edm \
      --learning_rate 0.0003 \
      --optimizer adamw \
      --emb_features 512 \
      --num_layers 16 \
      --num_heads 8 \
      --patch_size 2 \
      --mlp_ratio 4 \
      --norm_groups 0 \
      --dtype float32 \
      --precision default \
      --only_pure_attention True \
      --distributed_training True \
      --val_metrics clip clip_score \
      --best_tracker_metric val/clip_score \
      --wandb_project msml612-training \
      --wandb_entity umd-projects \
      --GRAIN_WORKER_BUFFER_SIZE 100 \
      --architecture "$arch" \
      "$@"
}

case "$1" in
  # 1. DiT baseline (all-attention, raster scan)
  simple_dit)
    run_experiment simple_dit
    ;;

  # 2. DiT + Hilbert (all-attention, hilbert scan)
  simple_dit+hilbert)
    run_experiment simple_dit+hilbert
    ;;

  # 3. Hybrid SSM+Attention 3:1 (raster scan)
  hybrid_dit_3to1)
    run_experiment hybrid_dit --ssm_attention_ratio 3:1 --ssm_state_dim 64
    ;;

  # 4. Hybrid SSM+Attention 3:1 + Hilbert (THE NOVEL METHOD)
  hybrid_dit+hilbert_3to1)
    run_experiment hybrid_dit+hilbert --ssm_attention_ratio 3:1 --ssm_state_dim 64
    ;;

  # 5. Hybrid SSM+Attention 1:1 + Hilbert
  hybrid_dit+hilbert_1to1)
    run_experiment hybrid_dit+hilbert --ssm_attention_ratio 1:1 --ssm_state_dim 64
    ;;

  # 6. All-SSM + Hilbert
  hybrid_dit+hilbert_all_ssm)
    run_experiment hybrid_dit+hilbert --ssm_attention_ratio all-ssm --ssm_state_dim 64
    ;;

  *)
    echo "Usage: $0 {simple_dit|simple_dit+hilbert|hybrid_dit_3to1|hybrid_dit+hilbert_3to1|hybrid_dit+hilbert_1to1|hybrid_dit+hilbert_all_ssm}"
    exit 1
    ;;
esac
