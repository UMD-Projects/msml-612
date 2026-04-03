#!/bin/bash
set -euo pipefail

echo "=== MSML612 GPU Setup ==="

apt-get update && apt-get install -y --no-install-recommends \
    wget bash curl git ca-certificates libgl1 libglib2.0-0

if [ ! -d "$HOME/miniconda3/bin" ]; then
    mkdir -p $HOME/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O $HOME/miniconda3/miniconda.sh
    bash $HOME/miniconda3/miniconda.sh -b -u -p $HOME/miniconda3
    rm $HOME/miniconda3/miniconda.sh
fi

export PATH=$HOME/miniconda3/bin:$PATH

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

conda create -n flaxdiff python=3.12 -y 2>/dev/null || true
export PATH=$HOME/miniconda3/envs/flaxdiff/bin:$HOME/miniconda3/bin:$PATH

pip install --no-cache-dir "jax[cuda12]" flax[all]

pip install --no-cache-dir --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cpu

pip install --no-cache-dir \
    "diffusers==0.29.2" \
    orbax optax clu grain augmax \
    "albumentations==1.4.14" "albucore==0.0.16" \
    datasets "transformers==4.41.2" \
    opencv-python pandas tensorflow-cpu tensorflow-datasets \
    jupyterlab python-dotenv scikit-learn termcolor wrapt \
    wandb gcsfs decord video-reader-rs \
    colorlog importlib_resources einops matplotlib

FLAXDIFF_DIR=$HOME/FlaxDiff
if [ ! -d "$FLAXDIFF_DIR" ]; then
    git clone --depth 1 https://github.com/AshishKumar4/FlaxDiff.git "$FLAXDIFF_DIR"
fi
cd "$FLAXDIFF_DIR"
pip install --no-cache-dir -e .

python -c "
import jax
print('JAX version:', jax.__version__)
print('Backend:', jax.default_backend())
print('Devices:', jax.devices())
assert jax.default_backend() == 'gpu', 'ERROR: JAX not using GPU!'
print('GPU setup verified.')
"

echo "Preparing oxford_flowers102 dataset..."
python -c "
import tensorflow_datasets as tfds
builder = tfds.builder('oxford_flowers102', file_format='array_record')
builder.download_and_prepare()
print('Dataset ready at:', builder.data_dir)
"

ulimit -n 65535 2>/dev/null || true

grep -q 'miniconda3/envs/flaxdiff' ~/.bashrc 2>/dev/null || \
    echo 'export PATH=$HOME/miniconda3/envs/flaxdiff/bin:$HOME/miniconda3/bin:$PATH' >> ~/.bashrc
grep -q 'TOKENIZERS_PARALLELISM' ~/.bashrc 2>/dev/null || \
    echo 'export TOKENIZERS_PARALLELISM=false' >> ~/.bashrc

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate the environment and log in to wandb:"
echo "  source activate_env.sh"
echo "  wandb login"
echo "  bash scripts/run_baseline.sh"
