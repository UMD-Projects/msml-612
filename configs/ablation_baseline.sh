#!/bin/bash
# =============================================================================
# Ablation Study: Architecture Comparison
# Base config from wandb run j8denkrd (simple_dit+hilbert LDM on Oxford Flowers)
# All experiments share these params — only --architecture varies.
# =============================================================================

set -e

export PATH=/home/mrwhite0racle/miniconda3/envs/flaxdiff/bin:$PATH
cd /home/mrwhite0racle/research

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
      --val_metrics clip \
      --best_tracker_metric val/clip_similarity \
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
