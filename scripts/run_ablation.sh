#!/bin/bash
set -euo pipefail

export PATH=$HOME/miniconda3/envs/flaxdiff/bin:$HOME/miniconda3/bin:$PATH
export TOKENIZERS_PARALLELISM=false

FLAXDIFF_DIR=$HOME/FlaxDiff

COMMON_ARGS=(
    --autoencoder=stable_diffusion
    --autoencoder_opts='{"modelname":"pcuenq/sd-vae-ft-mse-flax"}'
    --dataset=oxford_flowers102
    --image_size=256
    --batch_size=64
    --emb_features=512
    --num_layers=16
    --num_heads=8
    --patch_size=2
    --mlp_ratio=4
    --learning_rate=0.0003
    --optimizer=adamw
    --noise_schedule=edm
    --dtype=float32
    --precision=default
    --norm_groups=0
    --epochs=150
    --flash_attention=False
    --distributed_training=False
    --only_pure_attention=True
    --use_self_and_cross=True
    --activation=swish
    --val_metrics clip
    --best_tracker_metric=val/clip_similarity
    --checkpoint_dir="$HOME/checkpoints"
    --checkpoint_fs=local
    --wandb_project=msml612-training
    --wandb_entity=umd-projects
    --GRAIN_WORKER_COUNT=8
    --GRAIN_READ_THREAD_COUNT=16
    --GRAIN_READ_BUFFER_SIZE=32
    --GRAIN_WORKER_BUFFER_SIZE=20
)

run_experiment() {
    local name="$1"
    shift
    echo ""
    echo "=============================================="
    echo "  ABLATION: $name"
    echo "=============================================="
    python "$FLAXDIFF_DIR/training.py" "${COMMON_ARGS[@]}" "$@"
}

# 1. Control: all-attention, no Hilbert
run_experiment "simple_dit" \
    --architecture=simple_dit

# 2. All-attention + Hilbert
run_experiment "simple_dit+hilbert" \
    --architecture=simple_dit+hilbert

# 3. Hybrid SSM 3:1, no Hilbert
run_experiment "hybrid_dit 3:1" \
    --architecture=hybrid_dit \
    --ssm_attention_ratio="3:1" \
    --ssm_state_dim=64

# 4. Hybrid SSM 3:1 + Hilbert (full novel method)
run_experiment "hybrid_dit+hilbert 3:1" \
    --architecture=hybrid_dit+hilbert \
    --ssm_attention_ratio="3:1" \
    --ssm_state_dim=64

# 5. Ratio ablation: 1:1
run_experiment "hybrid_dit+hilbert 1:1" \
    --architecture=hybrid_dit+hilbert \
    --ssm_attention_ratio="1:1" \
    --ssm_state_dim=64

# 6. Ratio ablation: 1:3
run_experiment "hybrid_dit+hilbert 1:3" \
    --architecture=hybrid_dit+hilbert \
    --ssm_attention_ratio="1:3" \
    --ssm_state_dim=64

# 7. All-SSM + Hilbert
run_experiment "hybrid_dit+hilbert all-ssm" \
    --architecture=hybrid_dit+hilbert \
    --ssm_attention_ratio="all-ssm" \
    --ssm_state_dim=64

echo ""
echo "=== All ablation experiments complete ==="
