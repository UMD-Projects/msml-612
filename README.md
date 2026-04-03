# Locality-Preserving Hybrid SSM-Attention Diffusion Models for Text-to-Image Generation

**MSML612 Course Project** - Ashish Kumar Singh, Aman Pratap Singh

## Overview

We propose a hybrid architecture that interleaves O(n) State Space Model (S5) blocks with O(n^2) attention blocks inside a Diffusion Transformer (DiT), using Hilbert curve patch serialization to preserve spatial locality. This combines the efficiency of SSMs for local processing with the global composition ability of attention.

## Training Infrastructure

All primary training is conducted on **Google Cloud TPUs** (v4-8, v6e-4) via the [TRC (TPU Research Cloud)](https://sites.research.google/trc/) program. The GPU scripts below are provided for **reproducibility verification** on commodity hardware.

Final models will be trained on curated large-scale datasets (LAION-Aesthetics, CC12M, DiffusionDB, COYO) stored as ArrayRecord shards on GCS. The GPU reproduction scripts use Oxford Flowers 102 as a lightweight proxy dataset with identical architecture and hyperparameters.

All experiment runs are publicly tracked at [wandb.ai/umd-projects/msml612-training](https://wandb.ai/umd-projects/msml612-training).

## Reproducing Results (GPU)

### 1. Setup

On any machine with an NVIDIA GPU (tested on A100 80GB):

```bash
git clone https://github.com/UMD-Projects/msml-612.git
cd msml-612
bash scripts/setup_gpu.sh
source activate_env.sh
wandb login
```

This installs all dependencies (JAX CUDA, FlaxDiff, etc.), prepares the Oxford Flowers dataset in ArrayRecord format, and verifies GPU access.

### 2. Run Baseline

Trains a DiT+Hilbert latent diffusion model on Oxford Flowers (256px, SD-VAE):

```bash
bash scripts/run_baseline.sh
```

Config: `simple_dit+hilbert`, LDM (SD-VAE), batch 64, emb 512, 16 layers, 8 heads, patch 2, lr 3e-4, EDM noise schedule, 150 epochs.

### 3. Run Ablation Experiments

Runs all 7 architecture variants sequentially (same hyperparams, only architecture varies):

```bash
bash scripts/run_ablation.sh
```

| # | Architecture | What it tests |
|---|---|---|
| 1 | `simple_dit` | Control (all-attention, raster scan) |
| 2 | `simple_dit+hilbert` | Effect of Hilbert ordering alone |
| 3 | `hybrid_dit` 3:1 | Effect of SSM blocks alone |
| 4 | `hybrid_dit+hilbert` 3:1 | **Full novel method** |
| 5 | `hybrid_dit+hilbert` 1:1 | SSM:Attention ratio ablation |
| 6 | `hybrid_dit+hilbert` 1:3 | SSM:Attention ratio ablation |
| 7 | `hybrid_dit+hilbert` all-ssm | Pure SSM (no attention) |

### TPU Training

Primary training runs on Google Cloud TPUs via the TRC grant. See [tpu-tools](https://github.com/AshishKumar4/tpu-tools) for provisioning and `configs/ablation_baseline.sh` for the TPU experiment launcher.

## Repository Structure

```
activate_env.sh             # Source this to activate the conda environment
scripts/
  setup_gpu.sh              # One-command GPU environment setup
  run_baseline.sh            # Baseline DiT+Hilbert LDM training
  run_ablation.sh            # Full architecture ablation (7 experiments)
  convert_hf_to_arrayrecord.py  # Dataset conversion to ArrayRecord format
  evaluation_pipeline.py     # FID/CLIP evaluation
configs/
  ablation_baseline.sh       # TPU ablation launcher
  best_dit_config.json       # Reference DiT config
report/
  Project_Proposal.pdf       # Original proposal
  interim_report/            # Interim report
Dockerfile                   # Docker image for TPU training
```

## Key Code (in FlaxDiff)

- [`flaxdiff/models/ssm_dit.py`](https://github.com/AshishKumar4/FlaxDiff/blob/main/flaxdiff/models/ssm_dit.py) - S5 SSM layer, bidirectional S5, SSMDiTBlock, HybridSSMAttentionDiT
- [`flaxdiff/models/hilbert.py`](https://github.com/AshishKumar4/FlaxDiff/blob/main/flaxdiff/models/hilbert.py) - Hilbert curve patch serialization
- [`flaxdiff/models/simple_dit.py`](https://github.com/AshishKumar4/FlaxDiff/blob/main/flaxdiff/models/simple_dit.py) - Base DiT with AdaLN + RoPE
- [`training.py`](https://github.com/AshishKumar4/FlaxDiff/blob/main/training.py) - Training script with all architecture configs

## Dependencies

- **[FlaxDiff](https://github.com/AshishKumar4/FlaxDiff)** (v0.2.11+) - JAX/Flax diffusion library (our prior work)
- JAX with CUDA 12 (GPU) or TPU support
- Stable Diffusion VAE (`pcuenq/sd-vae-ft-mse-flax`) for latent diffusion
- CLIP (`openai/clip-vit-large-patch14`) for text conditioning and evaluation

## Experiment Tracking

- Training runs: [wandb.ai/umd-projects/msml612-training](https://wandb.ai/umd-projects/msml612-training)
- Data processing: [wandb.ai/umd-projects/msml612-data](https://wandb.ai/umd-projects/msml612-data)
