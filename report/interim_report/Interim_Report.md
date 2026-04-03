# Locality-Preserving Hybrid SSM-Attention Diffusion Models for Text-to-Image Generation

## MSML612 Project Interim Report

**By Ashish Kumar Singh, Aman Pratap Singh**

---

## Background

As described in our proposal, we are building a hybrid SSM-Attention Diffusion Transformer architecture that uses Hilbert curve ordering for patch serialization in SSM blocks. The core idea is to combine O(n) SSM blocks for efficient local spatial reasoning with sparse O(n^2) attention blocks for global composition. **No previous work has utilized Hilbert-ordered SSMs with attention in a hybrid diffusion model.**

This interim report covers the substantial progress made across four areas: (1) data curation and processing pipeline, (2) model architecture design and implementation, (3) baseline training results, and (4) infrastructure setup for large-scale distributed training on TPUs.

---

## Data Preparation and Curation

One of the most significant efforts in this project has been assembling, curating, and processing a large-scale, high-quality text-image dataset suitable for training diffusion models. Web-scraped image datasets present unique challenges: URL rot (studies show 66% of web links since 2013 have decayed), inconsistent image quality, watermarks, low-resolution images, poor text-image alignment, and inappropriate content. Careful curation is essential — training on low-quality data directly degrades model output quality.

### Dataset Sources

We curate from multiple complementary sources to ensure diversity and quality:

| Dataset | Raw Size | After Filtering | Filter Criteria |
|---------|----------|-----------------|-----------------|
| **LAION-2B-Aesthetic** | ~52M | ~37.3M | WIDTH/HEIGHT >= 256, CLIP similarity >= 0.27, watermark score <= 0.6, aesthetic score >= 4.2 |
| **COYO-700M** | ~747M | ~24.6M | CLIP similarity >= 0.26, aesthetic score >= 5.4, watermark <= 0.8, min 256px |
| **LAION-12M + MS-COCO** | ~12.7M | ~15M (with COCO 4x oversampling) | Concatenated with COCO oversampled 5x for caption quality |
| **CC12M** | ~12.4M | ~12.4M | Direct mapping, no filtering needed |
| **CC3M** | ~3.3M | ~3.3M | Direct mapping |
| **DiffusionDB** | ~2M | ~2M | Synthetic (Stable Diffusion generations with prompts) |

**Total curated data: ~95M+ image-text pairs**

### Quality Filtering Pipeline

Each dataset required tailored filtering. The key quality signals we filter on are:

- **CLIP similarity score** (text-image alignment): We require >= 0.26-0.27 depending on dataset. This removes images where the caption is unrelated to the visual content.
- **Aesthetic score** (visual quality): Ranges from 4.2 to 5.4 minimum depending on dataset. This filters out low-quality, blurry, or visually unappealing images.
- **Watermark probability**: We cap at 0.6-0.8 to remove heavily watermarked stock photos.
- **Resolution**: Minimum 256x256 pixels. Images below this threshold lack sufficient detail for 256px generation.
- **Aspect ratio**: Maximum 2.4:1 to avoid extreme panoramic/portrait images that degrade after padding.
- **Image variance**: Near-zero variance images (solid colors, blank frames) are rejected.

For the LAION-12M + MS-COCO composite, we oversample COCO captions 5x because COCO has the highest-quality human-written captions. This improves text-image alignment learning. The final composite is shuffled with seed 42 for reproducibility.

### Data Format and Processing

All datasets are converted to Google's **ArrayRecord** shard format for efficient distributed I/O on TPUs. The pipeline is:

1. **Download metadata** (parquet files with URLs and captions from HuggingFace)
2. **Filter** using dataset-specific quality thresholds
3. **Download images** using our modified fork of img2dataset (github.com/AshishKumar4/img2dataset) which we extended to output ArrayRecord format — the native format expected by Google's Grain data loading library in the JAX/Flax ecosystem
4. **Resize** to 256x256 with aspect-ratio-preserving resize and white-border padding
5. **Encode** as JPEG (quality 95) in ArrayRecord shards (80,000 samples per shard)
6. **Upload** to Google Cloud Storage for TPU access

Each ArrayRecord sample stores: image bytes (JPEG), caption text (UTF-8), a unique key, and JSON-encoded metadata. The format is designed for Grain's parallel, deterministic data loading with true global shuffling across the entire dataset.

For text encoding, we use a frozen **CLIP ViT-L/14** text encoder, producing 77-token embeddings of dimension 768. Captions are tokenized and encoded on-the-fly during training.

### URL Rot and Synthetic Data

A significant challenge we encountered is **URL rot** — the decay of web-hosted image URLs over time. LAION-2B was published in 2022, and we observed approximately 35-40% of URLs are now dead (HTTP 403/404/410 errors). Studies show that 66% of web links since 2013 have experienced link rot, with a typical half-life of 2-6 years for modern URLs.

To mitigate this, we supplemented our dataset with **DiffusionDB** (~2M Stable Diffusion-generated images with prompts). This synthetic dataset has zero URL decay issues and provides high-quality, diverse image-text pairs. We plan to also incorporate **JourneyDB** (4.4M Midjourney images) and **CommonCatalog** (14.6M CC-licensed images from Flickr) in the coming weeks.

---

## Model Architecture and Implementation

### FlaxDiff Framework

Our implementation is built on **FlaxDiff** (https://github.com/AshishKumar4/FlaxDiff), an open-source diffusion library we developed in JAX/Flax. FlaxDiff provides modular noise schedulers (EDM, Karras, cosine), noise predictors (epsilon, v-prediction, Karras preconditioning), ODE/SDE samplers (DDPM, DDIM, Euler, Heun, RK4, DPM-Solver), and a distributed training loop.

For this project, we extend FlaxDiff with:

### Hilbert Curve Patch Serialization

We implemented a complete Hilbert curve module (`flaxdiff/models/hilbert.py`, ~620 lines) that serializes 2D image patches into 1D sequences while preserving spatial locality. The Hilbert curve is a space-filling curve with the mathematical property of **minimum maximum distance** between spatially adjacent cells — making it ideal for sequential models that rely on local context.

Our implementation supports: index generation for arbitrary rectangular patch grids, forward and inverse patch reordering (fully JIT-compatible), and visualization utilities. The Hilbert ordering is integrated into our DiT, MM-DiT, and UViT architectures via a configurable `use_hilbert` flag.

*[Figure 1: Hilbert curve ordering vs. row-major ordering on a 32x32 grid — see hilbert_curve_mapping.png]*

*[Figure 2: Hilbert patch serialization on a real image — see hilbert_patch_demo.png]*

### S5 SSM Block Implementation

We implemented the **S5 (Simplified State Space)** layer as a drop-in replacement for attention in DiT blocks (`flaxdiff/models/ssm_dit.py`). The S5 layer uses diagonal state spaces with `jax.lax.associative_scan` for efficient parallel computation on TPUs. Key components:

- **Diagonal state matrix** (complex): Parameterized with learned log-negative real part (ensuring stability) and imaginary part, initialized with HiPPO-inspired frequencies.
- **ZOH discretization**: Zero-order hold for converting continuous-time dynamics to discrete steps.
- **Bidirectional scanning**: Forward and backward parallel scans, since image patches have no inherent directionality along the serialization curve.
- **AdaLN conditioning**: Identical to the attention DiT block — scale, shift, and gate parameters modulated by timestep and text embeddings.

The `SSMDiTBlock` maintains the **exact same interface** as the standard `DiTBlock`:

```
__call__(self, x, conditioning, freqs_cis) → x
```

This allows seamless interleaving of SSM and attention blocks.

### Hybrid SSM-Attention Architecture

The `HybridSSMAttentionDiT` model (`flaxdiff/models/ssm_dit.py`) interleaves SSM blocks with attention blocks in a configurable ratio. The block pattern is specified as a ratio string:

- `"3:1"` → [SSM, SSM, SSM, Attn, SSM, SSM, SSM, Attn, ...]
- `"1:1"` → [SSM, Attn, SSM, Attn, ...]
- `"all-ssm"` → All SSM blocks (pure O(n) model)
- `"all-attn"` → Standard DiT baseline

Both block types share the same AdaLN conditioning, allowing fair comparison. The model has been tested on TPU v4 and produces correct output shapes for all configurations, including with Hilbert ordering and text conditioning.

At 12 layers with 768 embedding dimension, the hybrid 3:1 model has ~131M parameters — comparable to DiT-B scale.

### Noise Schedules and Training Objective

We use the **EDM (Elucidating Diffusion Models)** noise schedule with Karras preconditioning (sigma_max=80, rho=7, sigma_data=0.5), which provides improved training dynamics and sampling quality compared to the original DDPM cosine schedule. We train with the Karras prediction transform and use the Euler Ancestral sampler (200 steps) for inference.

---

## Baseline Training Results

We have trained baseline models to validate our training infrastructure and establish quality benchmarks. Training was conducted on **TPU v4-8** pods (4 TPU v4 chips) using distributed data-parallel training via JAX's `shard_map`.

### UNet Baseline on Oxford Flowers

Our best UNet baseline achieves a training loss of **0.053** after 296K steps on Oxford Flowers 102 at 128x128 resolution. Configuration: emb_features=256, feature_depths=[64, 128, 256, 512], num_res_blocks=2, EDM noise schedule, AdamW optimizer with lr=2e-4.

*[Figure 3: Generated flower samples using Euler Ancestral sampler — see text2img euler ancestral 1.png]*

*[Figure 4: 64 unconditional samples with Heun sampler — see heun.png]*

### UNet Baseline on LAION+COCO (Large-Scale)

On the full LAION-Aesthetics + CC12M + MS-COCO + COYO combined dataset (~10.7M images), our UNet baseline achieves a training loss of **0.029** after 562K steps at 256x256 resolution with latent diffusion (Stable Diffusion VAE). This model uses batch size 32, emb_features=512, feature_depths=[128, 128, 256, 512], and 3 residual blocks per level.

*[Figure 5: Text-conditioned samples on the large-scale dataset — see text2img euler ancestral 2.png]*

These baselines establish the quality floor that our hybrid SSM-Attention model aims to match or exceed with significantly reduced computational cost.

---

## Infrastructure and Training Setup

### Google TPU Research Cloud

We secured a **Google TPU Research Cloud (TFRC)** grant providing:
- 32 on-demand Cloud TPU v4 chips in us-central2-b
- 64 spot TPU v5e chips across multiple zones
- 32 spot TPU v4 chips in us-central2-b
- 64 spot TPU v6e chips across multiple zones

This provides access to v4-32 pod slices (32 chips, 4 hosts) for distributed multi-host training, along with $300 in Google Cloud credits for storage and networking.

### GCP Infrastructure

We provisioned:
- **GCS bucket** (`gs://msml612-diffusion-data/`) in us-central2 (same region as TPU v4 for zero-egress-cost data access)
- **250GB persistent SSD** attached to the TPU VM for checkpoints and temporary data processing
- **TPU management tooling** (https://github.com/AshishKumar4/tpu-tools): Custom shell scripts for TPU creation, setup, SSH management, multi-TPU orchestration via tmux, and spot TPU resilience with queued resources

### Challenges

- **TPU capacity exhaustion**: On-demand v4 capacity in us-central2-b was intermittently unavailable, requiring us to use GCP's queued resources API which waits in a queue until capacity frees up. Spot TPUs are more available but can be preempted mid-training.
- **URL rot**: ~35-40% of LAION-2B URLs are now dead, significantly slowing data preparation. We addressed this by incorporating pre-downloaded datasets (DiffusionDB) and developing parallel converters.
- **img2dataset compatibility**: The upstream img2dataset library does not support Google's ArrayRecord output format. We forked and modified it (github.com/AshishKumar4/img2dataset) to produce ArrayRecord shards compatible with Grain's data loading pipeline.

---

## Code and Reproducibility

All code is publicly available:

- **FlaxDiff**: https://github.com/AshishKumar4/FlaxDiff — Diffusion model library (models, schedulers, samplers, trainers, data pipelines)
- **TPU Tools**: https://github.com/AshishKumar4/tpu-tools — TPU provisioning and management
- **img2dataset fork**: https://github.com/AshishKumar4/img2dataset — ArrayRecord output support

Key files for this project:
- `flaxdiff/models/ssm_dit.py` — S5 SSM block and Hybrid SSM-Attention DiT architecture
- `flaxdiff/models/hilbert.py` — Hilbert curve patch serialization
- `flaxdiff/models/simple_dit.py` — DiT baseline architecture
- `flaxdiff/trainer/general_diffusion_trainer.py` — Distributed training loop
- `training.py` — Main training entry point with 150+ configurable arguments
- `datasets/convert_parallel.py` — Parallelized dataset converter for HuggingFace → ArrayRecord

Training is fully reproducible via:
```
python training.py --architecture hybrid_dit+hilbert --ssm_attention_ratio 3:1 \
  --dataset oxford_flowers102 --batch_size 16 --image_size 128 \
  --noise_schedule edm --learning_rate 0.0002 --optimizer adamw \
  --wandb_project msml612-training --wandb_entity umd-projects
```

Experiment tracking: https://wandb.ai/umd-projects/msml612-training

---

## Current Status and Next Steps

| Component | Status |
|-----------|--------|
| Data curation pipeline (95M+ pairs) | Complete |
| ArrayRecord conversion to GCS | In progress (~37M LAION + 2M DiffusionDB currently processing) |
| FlaxDiff framework extensions | Complete |
| Hilbert curve serialization | Complete |
| S5 SSM block implementation | Complete |
| Hybrid SSM-Attention DiT architecture | Complete |
| UNet baseline training | Complete (loss 0.029 on LAION, 0.053 on flowers) |
| DiT baseline training | In progress |
| Hybrid SSM model training | Starting (pending TPU capacity) |
| Flow matching training objective | Planned |
| Serialization ablations (Hilbert vs Z-order vs raster vs zigzag) | Planned |
| FID/CLIP evaluation on COCO-30K | Planned |

### Remaining Work (4 weeks)

- **Week 1**: Complete data uploads. Train DiT and Hybrid SSM baselines on Oxford Flowers and LAION.
- **Week 2**: Implement Z-order and zigzag serialization. Begin serialization ablations.
- **Week 3**: Run full ablation grid (SSM/attention ratios x serialization orders). Compute FID and CLIP scores.
- **Week 4**: Compile results, generate comparison figures, write final report.

---

## References

**Recent Diffusion Models & Related Architectures**

1. Peebles & Xie. "Scalable Diffusion Models with Transformers." ICCV 2023.
2. Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." ICML 2024.
3. Gu & Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." COLM 2024.
4. Teng et al. "DiM: Diffusion Mamba for Efficient High-Resolution Image Synthesis." 2024.
5. Hu et al. "ZigMa: A DiT-style Zigzag Mamba Diffusion Model." ECCV 2024.
6. Phung et al. "DiMSUM: Diffusion Mamba — A Scalable and Unified Spatial-Frequency Method." NeurIPS 2024.
7. "HilbertA: Hilbert Attention for Efficient Diffusion Models." 2025.
8. "FractalMamba++: Fractal Scanning Strategies for Vision Mamba Models." AAAI 2025.
9. Lipman et al. "Flow Matching for Generative Modeling." ICLR 2023.
10. Smith et al. "Simplified State Space Layers for Sequence Modeling." ICLR 2023.

**Foundational Diffusion Model Papers**

- Ho et al. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
- Song et al. "Denoising Diffusion Implicit Models." ICLR 2021.
- Nichol & Dhariwal. "Improved Denoising Diffusion Probabilistic Models." ICML 2021.
- Dhariwal & Nichol. "Diffusion Models Beat GANs on Image Synthesis." NeurIPS 2021.
- Song et al. "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR 2021.
- Karras et al. "Elucidating the Design Space of Diffusion-Based Generative Models." NeurIPS 2022.
- Lu et al. "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps." NeurIPS 2022.

**Datasets**

- Schuhmann et al. "LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models." NeurIPS 2022.
- Changpinyo et al. "Conceptual 12M: Pushing Web-Scale Image-Text Pre-Training." CVPR 2021.
- Byeon et al. "COYO-700M: Image-Text Pair Dataset." 2022.
- Lin et al. "Microsoft COCO: Common Objects in Context." ECCV 2014.
- Wang et al. "DiffusionDB: A Large-scale Prompt Gallery Dataset for Text-to-Image Generative Models." ACL 2023.
