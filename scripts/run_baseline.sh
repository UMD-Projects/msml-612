#!/bin/bash
set -euo pipefail

export PATH=$HOME/miniconda3/envs/flaxdiff/bin:$HOME/miniconda3/bin:$PATH
export TOKENIZERS_PARALLELISM=false

FLAXDIFF_DIR=$HOME/FlaxDiff

echo "=== Baseline: simple_dit+hilbert LDM (reproducing j8denkrd) ==="

python "$FLAXDIFF_DIR/training.py" \
    --architecture=simple_dit+hilbert \
    --autoencoder=stable_diffusion \
    --autoencoder_opts='{"modelname":"pcuenq/sd-vae-ft-mse-flax"}' \
    --dataset=oxford_flowers102 \
    --image_size=256 \
    --batch_size=64 \
    --emb_features=512 \
    --num_layers=16 \
    --num_heads=8 \
    --patch_size=2 \
    --mlp_ratio=4 \
    --learning_rate=0.0003 \
    --optimizer=adamw \
    --noise_schedule=edm \
    --dtype=float32 \
    --precision=default \
    --norm_groups=0 \
    --epochs=150 \
    --flash_attention=False \
    --distributed_training=False \
    --only_pure_attention=True \
    --use_self_and_cross=True \
    --activation=swish \
    --val_metrics clip \
    --best_tracker_metric=val/clip_similarity \
    --checkpoint_dir="$HOME/checkpoints" \
    --checkpoint_fs=local \
    --wandb_project=msml612-training \
    --wandb_entity=umd-projects \
    --GRAIN_WORKER_COUNT=8 \
    --GRAIN_READ_THREAD_COUNT=16 \
    --GRAIN_READ_BUFFER_SIZE=32 \
    --GRAIN_WORKER_BUFFER_SIZE=20
