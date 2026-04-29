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
    # Training recipe calibrated against the HPO sweep in the prior
    # umd-projects/mlops-msml605-project (734 runs, 92 with CLIPScore metric).
    #
    # HPO findings for oxford_flowers102 (64 runs):
    #   - ZERO top runs used an LR schedule — flat LR dominated across 92
    #     configs. Our recent cosine decay (1e-4 peak → 5e-5 end) was
    #     non-representative of what actually works.
    #   - Best cs_sim 0.7163: simple_dit+hilbert at flat LR 3.28e-4, bs=64,
    #     150 epochs (run lt7rjtx3).
    #   - LR sweet spot: [5e-5, 5e-4]; below 2e-5 the model fails to converge
    #     in 19k steps; above 5e-4 there is no visible benefit.
    #   - Batch 64, 150 epochs, emb 512, L=16, H=8, patch 2 is the HPO consensus
    #     config for DiT variants at 256x256 on this tiny dataset.
    #   - All HPO-best runs used the codebase defaults for dropout (0.1),
    #     EMA decay (0.999), and augmentation (flip + ColorJitter).
    #
    # We therefore REVERT the canon-inspired knobs from the last rerun (peak
    # 1e-4, end 5e-5, dropout 0, EMA 0.9999, flip_only) and go back to the
    # HPO-validated flat 3e-4 recipe. Our recent architecture fixes (2D sincos
    # applied in raster too, identity RoPE in Hilbert mode, reverted C/D init
    # to lecun_normal/N(0,1)) stay in place — they are orthogonal to the
    # training recipe.
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
      --dropout_rate 0.1 \
      --ema_decay 0.999 \
      --augmentation_mode flip_jitter \
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

# The case statement below forwards `"${@:2}"` (everything after the first
# positional arg which is the ablation name) so callers can append arbitrary
# training.py overrides. Examples:
#   bash ablation_baseline.sh simple_dit --dataset laion12m_coco --epochs 1
#   bash ablation_baseline.sh hybrid_dit_3to1 --batch_size 128 --steps_per_epoch 500
# Because argparse picks the LAST occurrence of each flag, trailing overrides
# always win over the defaults set inside run_experiment.
case "$1" in
  # 1. DiT baseline (all-attention, raster scan)
  simple_dit)
    run_experiment simple_dit "${@:2}"
    ;;

  # 2. DiT + Hilbert (all-attention, hilbert scan)
  simple_dit+hilbert)
    run_experiment simple_dit+hilbert "${@:2}"
    ;;

  # 3. Hybrid SSM+Attention 3:1 (raster scan)
  hybrid_dit_3to1)
    run_experiment hybrid_dit --ssm_attention_ratio 3:1 --ssm_state_dim 64 "${@:2}"
    ;;

  hybrid_dit_1to1)
    run_experiment hybrid_dit --ssm_attention_ratio 1:1 --ssm_state_dim 64 "${@:2}"
    ;;

  hybrid_dit_all_ssm)
    run_experiment hybrid_dit --ssm_attention_ratio all-ssm --ssm_state_dim 64 "${@:2}"
    ;;

  # 4. Hybrid SSM+Attention 3:1 + Hilbert (THE NOVEL METHOD)
  hybrid_dit+hilbert_3to1)
    run_experiment hybrid_dit+hilbert --ssm_attention_ratio 3:1 --ssm_state_dim 64 "${@:2}"
    ;;

  # 5. Hybrid SSM+Attention 1:1 + Hilbert
  hybrid_dit+hilbert_1to1)
    run_experiment hybrid_dit+hilbert --ssm_attention_ratio 1:1 --ssm_state_dim 64 "${@:2}"
    ;;

  # 6. All-SSM + Hilbert
  hybrid_dit+hilbert_all_ssm)
    run_experiment hybrid_dit+hilbert --ssm_attention_ratio all-ssm --ssm_state_dim 64 "${@:2}"
    ;;

  # 7. DiT + Zigzag (all-attention, ZigMa-style serpentine scan)
  simple_dit+zigzag)
    run_experiment simple_dit+zigzag "${@:2}"
    ;;

  # 8. Hybrid SSM+Attention 3:1 + Zigzag (ZigMa's best scan order for SSM)
  hybrid_dit+zigzag_3to1)
    run_experiment hybrid_dit+zigzag --ssm_attention_ratio 3:1 --ssm_state_dim 64 "${@:2}"
    ;;

  # 9. Hybrid SSM+Attention 1:1 + Zigzag
  hybrid_dit+zigzag_1to1)
    run_experiment hybrid_dit+zigzag --ssm_attention_ratio 1:1 --ssm_state_dim 64 "${@:2}"
    ;;

  hybrid_dit+zigzag_all_ssm)
    run_experiment hybrid_dit+zigzag --ssm_attention_ratio all-ssm --ssm_state_dim 64 "${@:2}"
    ;;

  # ---------------------------------------------------------------------
  # Direction α — Spatial-Mamba-style 2D state fusion inside SSM blocks.
  # The '+2d' suffix enables multi-dilation depthwise conv after the SSM
  # scan, recovering 2D local structure that the 1D scan scrambles.
  # See docs/research/04_direction_alpha_plan.md for the full design.
  # ---------------------------------------------------------------------

  # A1. Hybrid + 2D fusion, raster scan
  hybrid_dit+2d_3to1)
    run_experiment hybrid_dit+2d --ssm_attention_ratio 3:1 --ssm_state_dim 64 "${@:2}"
    ;;

  # A2. Hybrid + 2D fusion + Hilbert scan (mechanism proof: does 2D fusion
  # rescue Hilbert?)
  hybrid_dit+2d+hilbert_3to1)
    run_experiment hybrid_dit+2d+hilbert --ssm_attention_ratio 3:1 --ssm_state_dim 64 "${@:2}"
    ;;

  # A3. Hybrid + 2D fusion + Zigzag scan
  hybrid_dit+2d+zigzag_3to1)
    run_experiment hybrid_dit+2d+zigzag --ssm_attention_ratio 3:1 --ssm_state_dim 64 "${@:2}"
    ;;

  # A4. Hybrid + 2D fusion, 1:1 ratio
  hybrid_dit+2d_1to1)
    run_experiment hybrid_dit+2d --ssm_attention_ratio 1:1 --ssm_state_dim 64 "${@:2}"
    ;;

  hybrid_dit+2d+hilbert_1to1)
    run_experiment hybrid_dit+2d+hilbert --ssm_attention_ratio 1:1 --ssm_state_dim 64 "${@:2}"
    ;;

  hybrid_dit+2d+zigzag_1to1)
    run_experiment hybrid_dit+2d+zigzag --ssm_attention_ratio 1:1 --ssm_state_dim 64 "${@:2}"
    ;;

  # A5. Hybrid + 2D fusion, all-SSM (the stress test — biggest expected win)
  hybrid_dit+2d_all_ssm)
    run_experiment hybrid_dit+2d --ssm_attention_ratio all-ssm --ssm_state_dim 64 "${@:2}"
    ;;

  # A6. Hybrid + 2D fusion + Hilbert, all-SSM (hilbert's worst case + fix)
  hybrid_dit+2d+hilbert_all_ssm)
    run_experiment hybrid_dit+2d+hilbert --ssm_attention_ratio all-ssm --ssm_state_dim 64 "${@:2}"
    ;;

  hybrid_dit+2d+zigzag_all_ssm)
    run_experiment hybrid_dit+2d+zigzag --ssm_attention_ratio all-ssm --ssm_state_dim 64 "${@:2}"
    ;;

  *)
    echo "Usage: $0 {ablation_name} [extra --args for training.py]"
    echo ""
    echo "Scan-order ablations:"
    echo "  simple_dit, simple_dit+hilbert, simple_dit+zigzag"
    echo "  hybrid_dit_3to1, hybrid_dit+hilbert_3to1, hybrid_dit+zigzag_3to1"
    echo "  hybrid_dit+hilbert_1to1, hybrid_dit+hilbert_all_ssm"
    echo "  hybrid_dit+zigzag_1to1"
    echo ""
    echo "Direction alpha (2D state fusion):"
    echo "  hybrid_dit+2d_3to1, hybrid_dit+2d+hilbert_3to1, hybrid_dit+2d+zigzag_3to1"
    echo "  hybrid_dit+2d_1to1, hybrid_dit+2d_all_ssm, hybrid_dit+2d+hilbert_all_ssm"
    exit 1
    ;;
esac
