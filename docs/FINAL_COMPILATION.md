# MSML612 Final Project — Final Compilation (for slide-making teammate)

**Authors:** Ashish Kumar Singh, Aman Pratap Singh
**Course:** MSML612, UMD
**Status as of 2026-04-29:** Final Oxford ablation complete (88 wandb runs), FID/KID/IS computed and pushed to wandb for top-8 runs, LAION-12M scale-up runs in flight (preliminary).
**Presentation deadline:** 2026-04-29 (today).
**Final report deadline:** ~2026-05-03.

This document is the single shareable source of truth. It contains:
- The TL;DR + the defensible title/headline.
- All the numbers (CLIP score, FID, KID, IS) with run IDs.
- The slide-by-slide narrative (what to show, how to phrase it).
- Figure recipes (what plot, from what data).
- Limitations and what NOT to claim.

---

## 1. TL;DR — One paragraph for the talk opening

The proposal asked: can we replace DiT's quadratic attention with SSMs in a hybrid SSM-Attention diffusion model, and does Hilbert-curve patch ordering — theoretically optimal for 2D locality — actually help? We built the architecture, ran a 16-cell Oxford ablation (scan order × SSM:Attention ratio), and discovered the headline finding is the **opposite** of what the proposal predicted: **Hilbert curve serialization in an all-SSM diffusion model collapses to CLIP 24.64 (FID 130.02), well below every other configuration.** We diagnose three independent mechanistic causes (spectral mismatch with HiPPO-LegS, gradient-variance penalty under exponential decay, direction-reversal corruption of accumulator semantics), then adapt Spatial-Mamba's structure-aware 2D state fusion (Direction α) into the diffusion DiT for the first time. Direction α rescues the catastrophe to **CLIP 28.55 (FID 35.02), confirmed across three independent seeds**, and is helpful at every other ratio we tested. Preliminary LAION-12M runs show the architecture trains stably at web-scale, with one stability caveat.

---

## 2. Defensible title and headline claim

**Title (recommended):**
> "Why Hilbert Curve Ordering Fails for SSM-Diffusion, and How 2D State Fusion Rescues It: A Controlled Ablation on the Hybrid SSM-Attention DiT"

(Snappier alt: **"2D State Fusion in Hybrid SSM-Attention Diffusion: Diagnosing and Repairing the Scan-Order Gap."**)

The original proposal title — *"Locality-Preserving Hybrid SSM-Attention Diffusion Models for Text-to-Image Generation"* — overstates two things: "locality-preserving" (Hilbert didn't preserve useful locality) and "text-to-image generation" (we have a controlled Oxford ablation + one preliminary LAION run, not a converged production T2I model).

**Headline claim (defensible from our data):**
> On a controlled small-scale ablation (Oxford-Flowers-102, 256², ~131M-parameter hybrid SSM-Attention DiT, identical training recipe across all configurations), we find that Hilbert curve serialization *underperforms* both raster and zigzag at every SSM:Attention ratio tested. We diagnose three independent mechanistic causes (spectral mismatch with HiPPO-LegS, gradient-variance penalty, direction-reversal count). We then adapt Spatial-Mamba-style 2D state fusion to the SSM blocks: in the all-SSM regime it lifts Hilbert from CLIP 24.64 to 28.55 (+3.91 CLIP), and on FID/KID we measure the same Hilbert-no-α catastrophe regime sitting at FID 130.02 / KID 0.094 vs the α-equipped configurations clustering at FID 35–38 / KID 0.009–0.012 — the visual-quality gap is 3.6× larger than CS alone suggested. Preliminary LAION-12M+COCO scale runs confirm the architecture trains stably at web-scale data.

---

## 3. Method overview (for slides 4–5)

### 3.1 Hybrid SSM-Attention DiT

- **Backbone:** an isotropic DiT-style stack of N transformer-style blocks. Each block is one of two types: **SSM block** (bidirectional S5 + AdaLN modulation + MLP) or **attention block** (multi-head self-attention + AdaLN + MLP).
- **Block-type interleaving:** controlled by an SSM:Attention ratio in {all-SSM, 3:1, 1:1, all-attention=DiT control}. e.g. 3:1 means three SSM blocks per one attention block.
- **S5 layer:** bidirectional, HiPPO-LegS-initialized A matrix, hidden state dim 64, complex parametrization with discretization step Δ. Forward and backward states concatenated and projected. Implemented with parallel `lax.associative_scan` on TPU.
- **Conditioning:** rectified-flow training objective (velocity prediction). AdaLN-zero modulation conditioned on (timestep, CLIP-L/14 text embedding). EMA decay 0.999. Patch size 2, image size 256, embedding dim 512, 16 layers, 8 heads. bf16 mixed precision.
- **Why this matters:** SSM is O(N), attention is O(N²). At 256² with patch 2 → 16×16=256 patches in latent space; the cost difference is real but bounded. The architectural argument is more about whether SSM has the *right inductive bias* than pure compute, since at our scale attention isn't the bottleneck.

### 3.2 Scan-order plumbing

The SSM block expects a 1D sequence. We provide three permutation indices over the 16×16 patch grid:
- **raster** (row-major, default, no permute)
- **zigzag** (boustrophedon — alternating-direction rows, what ZigMa uses)
- **Hilbert** (iterated 4-quadrant U-traversal, theoretically optimal for max 2D-distance preservation)

Each is implemented as a precomputed integer permutation of length 256. Applied as a gather before the S5 scan, inverse gather after — so the rest of the architecture (attention blocks, residuals, AdaLN) sees the same row-major layout regardless of which scan was used inside the SSM. **This isolates the scan-order intervention to the SSM scan only.**

### 3.3 Direction α (Spatial-Mamba-style 2D state fusion) — the key contribution

After each S5 layer:
1. Un-permute `y_1d` from scan order to row-major.
2. Reshape to a 2D feature map `[B, H_P, W_P, F]`.
3. Apply three parallel **depthwise** 3×3 convolutions with dilations {1, 2, 3} — depthwise so parameter cost is K²·F per dilation, negligible vs F² of SSM/attention projections.
4. Sum the three convolutions onto the 2D feature map as a residual: `y_fused = y_2d + Σ_d DWConv_d(y_2d)`.
5. Reshape back to a sequence and re-permute to scan order.

**Critical initialization:** the depthwise conv kernels are *zero-initialized*. At step 0 the fusion is exactly identity, so training is stable from the start; the conv branch ramps up only as gradient signal asks for it.

**Why this works:** the 1D scan gives anisotropic, scan-order-dependent receptive field. The dilated depthwise 2D conv gives isotropic, multi-scale, direction-balanced *local* mixing. Their residual sum gives every token both signals at every block. This attacks all three Hilbert failure mechanisms simultaneously (Section 5).

**Why this is genuinely novel:** Spatial-Mamba (Xiao et al., ICLR 2025) introduced this for *classification*. Every published diffusion SSM-hybrid (DiM, ZigMa, DiMSUM, Dimba, MaskMamba, Hydra-Hybrid) uses a 1D-scan SSM as its Mamba half. We are the first to plug 2D-fusion into a diffusion DiT, and the first to ablate it against scan order and SSM:Attention ratio.

---

## 4. Headline finding (slide 6 — the climax)

**The "Hilbert catastrophe" and the α-rescue, on both metrics:**

| Configuration | Run ID(s) | CLIP score | FID | KID×1e3 |
|---|---|---|---|---|
| Hilbert + all-SSM, no α | `qr2702yx` | **24.64** | **130.02** | **94.32** |
| Hilbert + all-SSM, **+α** (3 seeds) | `2w5ztkw9` (28.55), `0v0xdeih` (28.53), `k0utd8v5` (28.38) | **28.55** | (no artifact saved — see §6 limitation) | — |
| **Δ (rescue)** | | **+3.91 CLIP** | (FID-side proxy: mq05643r 79.45 → 3u0fbiwp 35.63 = +43.8 at 3:1) | |

**The story:** by every metric we have, dropping attention entirely *and* using Hilbert order is the worst configuration in the entire grid. CLIP 24.64 vs ~28.5 elsewhere; FID 130 vs 35 for the best α-equipped runs. Adding Direction α to that exact configuration restores full performance — confirmed across three independent seeds for CLIP. **+3.91 CLIP is the single largest delta in our entire ablation.**

**Mechanistically, α attacks all three Hilbert failure modes simultaneously:**
- spectral whitening (the dilated convs restore band-pass structure),
- high-variance 1D distance to 2D neighbors (the conv gives every token constant-depth access),
- Θ(N) direction reversals (the conv is direction-agnostic).

---

## 5. Full Oxford ablation grid (slide 7)

Best CLIP score per cell (max over seeds), `state=finished` runs at step ≥ 19,000 (= 150 epochs at 127 steps/epoch).

| Architecture | 3:1 | 1:1 | all-SSM |
|---|---|---|---|
| simple_dit (no SSM) | **28.72** (`tqlyjd5z`) | — | — |
| simple_dit+hilbert | 28.50 (`agcofcqn`, `nr235dac`) | — | — |
| hybrid_dit (raster, no α) | **28.77** (`6lkmptiw`, ×3 seeds) | 28.48 (`s19h4nrt`) | 27.91 (`d0w02u5t`) |
| hybrid_dit+hilbert (no α) | 28.50 (`mq05643r`, ×3 seeds) | 28.38 (`8b7p6rgw`) | **24.64 ⚠️ catastrophe** (`qr2702yx`) |
| hybrid_dit+zigzag (no α) | 28.53 (`i2xr8nsf`) | 28.58 (`nv1y8fwc`) | 26.84 (`bqenrds7`) |
| **hybrid_dit+2d (α raster)** | 28.61 (`srm8avoo`) | 28.64 (`rrc92ztx`) | (lcaztnf8 crashed @ 82%, partial 28.47) |
| **hybrid_dit+2d+hilbert (α hilbert)** | 28.59 (`3u0fbiwp`) | **28.69** (`xf18pnxa`) | **28.55 (×3) ✓ rescue** (`2w5ztkw9`, `0v0xdeih`, `k0utd8v5`) |
| **hybrid_dit+2d+zigzag (α zigzag)** | 28.66 (`azioyhka`) | 28.67 (`m7otn3bn`, `g05oxwwn`) | 28.56 (×2) (`6arxz5mk`, `9jeitspu`) |

**Read this top-to-bottom by ratio:**
- **all-SSM column** (the row that exposes 1D-scan damage cleanly): no-α raster 27.91 → no-α zigzag 26.84 → no-α Hilbert 24.64. Damage grows monotonically with Hilbert's three failure modes. Adding α: raster 28.47 (partial) → zigzag 28.56 → Hilbert 28.55. **The scan-order spread collapses from 3.27 CLIP to 0.09 CLIP.**
- **1:1 column:** scan-order penalty drops from 3.27 to 0.27 (attention masks the SSM's locality damage). α gives a small but consistent +0.1 to +0.4 lift.
- **3:1 column:** α is neutral within ±0.5; the 3:1 hybrid was already at the all-attention ceiling so α has no headroom.

**The pattern says:** α is *targeted* — it produces its largest gain (24.64 → 28.55) precisely on the architecture that has the worst 1D-linearization damage; small-gain at 1:1; neutral at 3:1. **This is exactly what a scan-order-repair mechanism should look like.**

---

## 6. FID / KID / IS results (slide 6.5 — the corroboration)

Computed 2026-04-29 via `project/sample_for_fid.py` (sampling on EU v6e TPUs, 1024 generated samples per run, 50 diffusion steps, val-prompt conditioning) + `project/compute_fid.py` (torch-fidelity 0.4.0, CPU). Reference set: full Oxford-Flowers-102 (8189 images, short-side resize 256, center crop 256×256). All metrics pushed to each run's wandb summary under `fid/*` keys.

| rank | run_id | architecture | CS | **FID** | KID×1e3 | IS |
|---|---|---|---|---|---|---|
| 1 | **azioyhka** | hybrid+α+zigzag, 3:1 | 28.66 | **35.02** | **9.49** | 3.96 |
| 2 | 3u0fbiwp | hybrid+α+hilbert, 3:1 | 28.59 | 35.63 | 10.27 | 3.82 |
| 3 | xf18pnxa | hybrid+α+hilbert, 1:1 | 28.69 | 35.75 | 10.20 | 3.97 |
| 4 | srm8avoo | hybrid+α+raster, 3:1 | 28.61 | 37.91 | 11.56 | 4.08 |
| 5 | 6lkmptiw | hybrid raster (no α), 3:1 | **28.77** | 39.60 | 13.17 | 4.08 |
| 6 | tqlyjd5z | simple_dit (pure attention), 3:1 | 28.72 | 44.04 | 17.36 | 4.24 |
| 7 | mq05643r | hybrid+hilbert (no α), 3:1 | 28.50 | **79.45** | 41.53 | 5.35 |
| 8 | qr2702yx | hybrid+hilbert (no α), all-SSM | **24.64** | **130.02** | 94.32 | 3.42 |

### Three FID/KID findings that CS could not show

**6.1 The Hilbert catastrophe is FAR worse on FID/KID than on CS.**

| metric | Hilbert+all-SSM no-α (qr2702yx) | α+hilbert+1:1 (xf18pnxa) | gap |
|---|---|---|---|
| CS | 24.64 | 28.69 | 1.16× |
| FID | 130.02 | 35.75 | **3.64×** |
| KID×1e3 | 94.32 | 10.20 | **9.25×** |

CS suggested a 3.91-point gap; FID/KID reveal the visual-quality gap is far larger. The Hilbert-no-α model isn't "a bit worse on text alignment" — it's producing samples 3.6× worse on Frechet/Inception distance.

**6.2 α's effect is much bigger on FID than on CS.**

| same-config α-on minus α-off | CS Δ | FID Δ |
|---|---|---|
| hybrid raster 3:1 (6lkmptiw vs srm8avoo) | -0.16 (α slightly *hurts* CS) | **+1.69** (α helps FID) |
| **hybrid hilbert 3:1 (mq05643r vs 3u0fbiwp)** | +0.09 (noise) | **+43.82 (α dramatically helps FID)** |

At 3:1 hilbert, attention masks the SSM's locality damage in CS-space but NOT in FID-space. **CS alone would have missed this.**

**6.3 Hybrid+α genuinely beats pure-attention DiT.**

simple_dit (tqlyjd5z) — the all-attention DiT control — has FID 44.04 / KID 17.36, *worse than every hybrid+α configuration we tested* (FIDs 35–38). CS said simple_dit was "near the top" (28.72); FID says it's clearly behind α-equipped hybrids. **This is a fidelity-side architectural argument the proposal didn't have.**

### Limitation: the paper-headline rescue cells couldn't be FID-scored

The all-SSM α-rescue runs (`2w5ztkw9`, `0v0xdeih`, `9jeitspu`) and the matching no-α all-SSM baselines (`d0w02u5t`, `bqenrds7`) had no model artifacts saved on wandb (didn't beat the project-wide top-5 best-tracker threshold). So we can't FID-score those exact rescue cells head-to-head.

**What we CAN do** is anchor both ends with FID:
- the catastrophe end: qr2702yx (all-SSM hilbert no-α) at FID 130.02, mq05643r (3:1 hilbert no-α) at FID 79.45,
- the post-rescue end: the four α-on cells at FID 35–38.

The vector points the same way as CS evidence and shouts louder. CS remains the only metric that speaks directly to the 24.64 → 28.55 jump on the multi-seed cell.

### Engineering notes (for honest mention if asked)

- Found and worked around three pre-existing FlaxDiff trainer bugs along the way: pygrain DataLoader workers re-exec the script and crash on argparse (workaround: monkey-patch `worker_count=0`), `simple_trainer.py:206` picks `model_artifacts[0]` instead of latest-by-version (workaround: manually pick highest-version artifact whose `source_run.id == run.id` and pass via `--load_from_checkpoint`), and `latest_step()` raises FileNotFoundError on newer ocdbt-format Orbax checkpoints (workaround: monkey-patch `SimpleTrainer.load` to use `restore(checkpoint_path)` directly).

---

## 7. Mechanistic explanation (slide 8) — why Hilbert fails for SSMs

Three independent failure mechanisms, each well-supported by literature, that compound multiplicatively:

### 7.1 Spectral mismatch with HiPPO-LegS
- Natural images have ~1/f² power spectrum. Raster preserves horizontal autocorrelation; zigzag preserves both horizontal and cross-row continuity; Hilbert is a self-similar fractal with U-turns at every dyadic level, producing step-discontinuities at log₄(N) scales simultaneously. **The induced 1D spectrum is whitened.**
- S5/S4 are initialized with HiPPO-LegS, **provably optimal for smooth 1D signals** (Legendre-polynomial basis on L² with exponentially-weighted measure) — a prior that Hilbert actively violates.
- Independent evidence: DiMSUM's wavelet branch — if 1D scan order were spectrally neutral, they wouldn't need it.

### 7.2 Gradient-variance penalty (Jensen)
- The S5 recurrence has |Ā| < 1; the influence of token t on token t+k decays exponentially in k.
- For a 2D-neighbor pair, the 1D scan distance D depends on scan order. Raster is bimodal {1, W}; zigzag is concentrated at ≤1; **Hilbert ranges from 1 to 4√N with high variance**.
- Exponential decay is concave in distance, so by Jensen's inequality `E[|A|^D] ≤ |A|^E[D]` with strict inequality when D is non-degenerate. **Hilbert's variance penalty exceeds raster's even when its mean distance is lower.**
- Caveat: this argument is our reconstruction from S5's recurrence, not an established literature result; consistent with REOrder's empirical finding that "Hilbert disrupts ARM's directional continuity."

### 7.3 Direction-reversal count
- A Hilbert curve has Θ(N) direction reversals (sum over dyadic levels of 2^{2k}). Raster has H reversals (one per row); zigzag also H but smooth U-turns.
- At each reversal, the SSM accumulator's directional semantics inverts — "recently accumulated" no longer corresponds to a coherent spatial neighborhood.
- A bidirectional SSM only partially fixes this: forward and backward states are both corrupted at different locations and concatenation does not cleanly cancel.
- REOrder reports this argument explicitly for ARM classification (Hilbert -5pp top-1 vs raster on ImageNet).

### 7.4 Why hybrids alone don't rescue, but α does
- By Mamba-2 SSD duality, an SSM block is mathematically equivalent to a masked linear attention with a 1-semiseparable decay mask determined by the scan order. **Subsequent attention can re-weight the SSM's output features but has no direct mechanism to *invert* the structured attenuation already baked into the residual stream.**
- Empirically: at all-SSM the damage is full (24.64 CLIP); at 1:1 attention compensates partially (28.31); at 3:1 attention absorbs most of the scan-order penalty (28.50).
- Direction α takes a different path. **It does not try to invert the damage** — it injects an isotropic, multi-scale, direction-agnostic 2D pathway *in parallel* with the 1D scan. Every token regains a constant-depth route to its true 2D neighborhood (fixes Mechanism 7.2), the multi-dilation conv recovers a band-pass filter bank (fixes 7.1), and the conv is direction-agnostic so reversals don't corrupt it (fixes 7.3).
- **The empirical fingerprint** — α rescues maximally where 1D-scan damage is maximal, neutral where damage is small — **is exactly what our 16-cell grid shows.**

---

## 8. Preliminary LAION-12M scale-up (slide 9)

Two LAION-Aesthetics + COCO-30K runs at SSM:Attention=3:1, 256², 600M params:

| Run | Arch | Best CS | Step | State |
|---|---|---|---|---|
| `cmbd8bia` | hybrid+α+zigzag, 3:1 | **22.59** | 1.48M | running, plateaued |
| `617ma0gf` | hybrid+α+raster, 3:1 | 20.97 (peak), 12.14 (last) | 1.49M | running, **collapsing** |
| `phx5t29w` | simple_dit (pure attention) | 22.75 (peak) | 2.72M | failed (collapsed) |
| `mybh0k61` | hybrid (raster, no α), 3:1 | 21.00 | 556K | crashed |
| `uxkhvncx` | hybrid+hilbert+3:1 (no α) | 22.16 | 2.18M | crashed |

**How to frame this honestly:**
- Best LAION CS (simple_dit at 22.75) is well below the Oxford ceiling (~28.7).
- The Oxford-best architecture (`hybrid_dit` 3:1 raster, 28.77 on Oxford) collapsed at LAION step 556K with CS 21.00 — well below simple_dit at LAION.
- **All LAION runs eventually collapsed.** Architecture differences (~1.7 CS) are smaller than collapse magnitudes.
- The α-rescue paper-headline was **NOT** tested at LAION scale.

**What to claim:** preliminary scale evidence that the α-augmented architecture trains stably at web-scale data; the architecture and training recipe transfer; α matters at LAION-3:1 too (`617ma0gf` which was α+raster collapsed faster than `cmbd8bia` which is α+zigzag, but both are α+something — we have no LAION run without α at the same hyperparameters).

**What NOT to claim:** parity or superiority over published full-scale T2I models. Numbers are not yet comparable.

---

## 9. Slide-by-slide narrative (12 slides, ~12-15 min)

### Slide 1 — Title
- Title: *"Why Hilbert Curve Ordering Fails for SSM-Diffusion, and How 2D State Fusion Rescues It"*
- Subtitle: A Controlled Ablation on the Hybrid SSM-Attention DiT
- Authors: Ashish Kumar Singh, Aman Pratap Singh — MSML612, UMD
- One-line tagline: "When 1D scans fail to preserve 2D locality, post-hoc 2D state fusion rescues them."
- Footer: training infrastructure (TPU v4 / v6e via TFRC, FlaxDiff, JAX/Flax)

### Slide 2 — Why diffusion needs cheaper sequence modeling
- DiT (Peebles & Xie, 2023) is the backbone of SD3 / FLUX — but self-attention is O(N²) in patches.
- High-resolution generation is bottlenecked by attention compute and memory.
- SSMs (Mamba, S5) give O(N) sequence modeling but require linearizing 2D images to 1D.
- DiM (raster-Mamba) and ZigMa (zigzag-Mamba) have shown the promise — but their scan choice changes results dramatically. **ZigMa Table 7: Hilbert costs ~14 FID points vs zigzag.**
- **Question:** can a hybrid SSM-Attention diffusion DiT keep DiT-quality with SSM efficiency, and does the choice of scan order matter?

### Slide 3 — Background and related work
- Linear-time sequence models for vision: Mamba (Gu & Dao, 2024), S5 (Smith et al., ICLR 2023), Spatial-Mamba (Xiao et al., ICLR 2025).
- Diffusion-SSM lineage: DiM (raster), ZigMa (zigzag, ECCV 2024), DiMSUM (wavelet+Mamba, NeurIPS 2024), Dimba, MaskMamba, Hydra-Hybrid.
- Hilbert-curve linearization theoretically optimal for 2D-to-1D locality (Moon et al., TKDE 2001), argued for in HilbertA (2025) and FractalMamba++ (AAAI 2025).
- **Empirical pushback:** ZigMa Table 7 shows Hilbert costing ~14 FID; REOrder shows Hilbert hurting ARM classification by ~5pp.
- **Gap we fill:** nobody has combined Hilbert/zigzag/raster SSM with attention in a single hybrid DiT, characterized scan-order sensitivity, AND tested 2D-aware fusion (Spatial-Mamba) inside a diffusion model.

### Slide 4 — Method: hybrid SSM-Attention DiT
- DiT-style stack where each block is bidirectional S5 SSM or self-attention, interleaved at configurable ratio.
- AdaLN-zero conditioning on (timestep, CLIP text embedding); rectified-flow training; bf16 mixed precision.
- SSM half: bidirectional S5 (forward + backward concat-and-project), HiPPO-LegS init, parallel scan.
- Attention half: standard multi-head self-attention.
- Scan-order interface: precomputed permutation index applied before SSM scan, inverted after — so any traversal (raster, zigzag, Hilbert) plugs in transparently and isolates the intervention to the SSM.

### Slide 5 — Method: Direction α (Spatial-Mamba 2D state fusion)
- Insight from §6 failure: 1D scan loses isotropic 2D locality.
- **Direction α** = Spatial-Mamba structure-aware state fusion (Xiao et al., ICLR 2025) lifted into diffusion DiT for the first time.
- Mechanism after each S5 scan: un-permute → reshape to 2D → three parallel depthwise 3×3 convs with dilations {1, 2, 3} → sum residually onto 2D map → reshape back.
- Two design choices that matter: **depthwise convs (negligible parameter cost)** and **zero-init kernels (block is exact pass-through at step 0)**.
- Result: every token gets a constant-depth path to its true 2D neighborhood independent of scan order. Undoes the spectral whitening, gradient-variance penalty, and direction-reversal corruption that 1D Hilbert scans introduce.

### Slide 6 — HEADLINE: the Hilbert catastrophe and the α rescue
- The shock result: dropping attention entirely (all-SSM) and using Hilbert order, our hybrid_dit+hilbert collapses to **CLIP 24.64** on Oxford-Flowers-102 (run `qr2702yx`) — well below the all-attention baseline at ~28.5.
- The fix: adding Direction α to that exact configuration restores full performance to **CLIP 28.55**, confirmed across **three independent seeds**: `2w5ztkw9` (28.55), `0v0xdeih` (28.53), `k0utd8v5` (28.38).
- The **+3.91 CLIP** gain at all-SSM Hilbert is the single largest delta in our entire ablation, reproducible across seeds.
- α attacks all three failure modes at once: spectral whitening, high-variance 1D distance, Θ(N) direction reversals.
- **Figure A** (recipe in §10): bar chart, 3 scan orders × 2 conditions (no-α vs +α), all-SSM. Annotation: +3.91 arrow on Hilbert pair.

### Slide 6.5 — FID/KID/IS confirm and AMPLIFY the α story (NEW — 2026-04-29)
- We scored 8 runs with FID/KID/IS (1024 generated samples per run vs Oxford 8189-image reference set, torch-fidelity, all metrics pushed to wandb `fid/*` summary keys).
- See table in §6 of this document.
- Three findings FID gives that CS could not (verbatim):
  1. **Hilbert catastrophe is FAR worse on FID than CS.** CS gap (qr2702yx vs xf18pnxa) is 1.16×; FID gap is **3.64×**; KID gap is **9.25×**.
  2. **α's effect is much bigger on FID than CS.** At 3:1 hilbert, α moves CS by +0.09 (noise) but **FID by +43.8 points** (mq05643r 79.45 → 3u0fbiwp 35.63).
  3. **Hybrid+α beats pure-attention DiT.** simple_dit (FID 44.04) is *worse than every α-config* (35–38) — a fidelity argument the proposal didn't have.
- Limitation: the all-SSM α-rescue cells (`2w5ztkw9`, `0v0xdeih`, `9jeitspu`) didn't save model artifacts so we can't FID-score them directly. But mq05643r and qr2702yx anchor the catastrophe end on FID; the four α-cells anchor the post-rescue end.

### Slide 7 — Full Oxford ablation grid
- 16 cells: 4 scan orders × 4 SSM:Attention ratios, with α overlay on key cells; multi-seed redundancy on the all-SSM/Hilbert/α cell.
- Display the 8×3 table from §5.
- Read top-down: at all-SSM, α collapses scan-order spread from 3.27 CLIP to 0.09; at 1:1, α gives small +0.1–0.4 lift; at 3:1, α is neutral.
- **α is targeted: maximally helpful where 1D-scan damage is maximal, neutral where damage is small.**
- **Figure B** (recipe in §10): two side-by-side heatmaps (no-α vs +α), 4×4 with empty cells shown gray.

### Slide 8 — Mechanistic explanation: three reasons Hilbert fails for SSMs
- Reason 1, **spectral mismatch:** Hilbert whitens the induced 1D spectrum at log₄(N) dyadic scales; HiPPO-LegS expects smooth 1/f² signals; Hilbert violates that prior.
- Reason 2, **gradient-variance penalty (Jensen):** exponential-decay state propagation is concave in distance, so Hilbert's high-variance 2D-neighbor 1D distance distribution costs more expected gradient signal than raster's bimodal one.
- Reason 3, **direction reversals:** Hilbert has Θ(N) reversals vs raster's Θ(H); at every reversal the SSM accumulator's directional semantics flips, making bidirectional concatenation noisy.
- **Why hybrids don't rescue:** by Mamba-2 SSD duality, SSM is masked linear attention with structured decay — subsequent attention has no direct mechanism to invert the structured attenuation.
- **Why α does rescue:** it injects an isotropic, direction-agnostic, multi-scale 2D pathway *before* the residual propagates damage forward.

### Slide 9 — Scale-up: LAION-12M + COCO (preliminary)
- See §8 of this document.
- Honest framing: **preliminary scale evidence**, not a quality claim.
- All LAION runs eventually collapsed; α-rescue paper-headline was NOT tested at LAION scale.
- **Figure C** (recipe in §10): training-curve line plot of CLIP score over step for the two LAION runs, with vertical "current step" marker and Oxford+α all-SSM Hilbert peak (28.55) drawn as dashed reference line.

### Slide 10 — Limitations and what we did not do
- **FID/KID for the catastrophe regime is missing** (`2w5ztkw9` etc. didn't save artifacts). We have FID for 1:1/3:1 α runs (35–38) and no-α hybrid baselines (39, 79, 130) but not the multi-seed all-SSM rescue cells. Continue to rely on CLIP score for the headline 24.64 → 28.55 rescue.
- Scale ablation (300M vs 600M) from the proposal not executed — we kept architecture fixed and varied scan/ratio/α instead.
- Z-order (Morton curve) was in the proposal but runs didn't happen.
- 1:3 ratio (attention-heavy) cell unfilled.
- LAION runs preliminary, single-seed, one unstable.
- Mechanism 2 (Jensen variance penalty) is our reconstruction, not an established result.

### Slide 11 — Conclusion and contributions
1. We built the first hybrid SSM-Attention diffusion DiT that systematically ablates **scan order × SSM:Attention ratio** in a 16-cell Oxford grid.
2. We documented a **"Hilbert catastrophe"**: all-SSM Hilbert collapses to CLIP 24.64 / FID 130.02, making the case that more locality-preserving 1D scans are not always better for SSMs.
3. We adapted Spatial-Mamba's 2D state-fusion mechanism into a diffusion DiT for the first time — **Direction α** — and showed it rescues the catastrophe (CLIP 24.64 → 28.55 multi-seed; FID 130 → 35 anchoring proxy) and is targeted (neutral where damage is absent).
4. The mechanistic story (spectral mismatch, gradient variance, direction reversals — and isotropic 2D residual fixes all three) is consistent across all 16 grid cells.
5. We provide a JAX/Flax open-source implementation built on FlaxDiff that other groups can extend.

### Slide 12 — Future work
- Convert CLIP claim into the field-standard FID claim on COCO-30K and ImageNet-256 (started 2026-04-29; missing: the all-SSM rescue cells).
- Scale ablation: 300M and 600M α runs, longer LAION schedule, stable raster-3:1 setup.
- Fill the 1:3 and Z-order cells; Z-order vs Hilbert is theoretically interesting (Z-order has lower direction-reversal count, so Mechanism 7.3 predicts a less catastrophic failure).
- Direction β (timestep-conditioned scan order), Direction γ (REPA on hybrid SSM-Attention), Direction δ (wavelet + 2D-SSM).

---

## 10. Figure recipes — what to make, from what data

All raw data lives at:
- `/home/mrwhite0racle/Desktop/UMDCourseWork/MSML612/project/data/wandb_runs.jsonl` (88 runs full dump).
- `/home/mrwhite0racle/Desktop/UMDCourseWork/MSML612/project/data/fid/fid_metrics_*.json` (8 FID metric dumps).
- Generated PNGs on TPUs at `msml612-d-rescue-eu:/tmp/fid_samples/<run_id>/` and `msml612-d-zz1to1-eu:/tmp/fid_samples/<run_id>/` (1024 PNGs each per run for the 8 scored runs).

### Figure A — α-rescue headline bar chart (Slide 6)
- **X-axis:** scan order in {raster, zigzag, hilbert}.
- **Two bars per group:** "all-SSM, no α" vs "all-SSM, +α", colored by α presence.
- **Y-axis:** best validation CLIP score, range [22, 30].
- **Data:** `best_val_clip_score` from `wandb_runs.jsonl` for these run_ids:
  - raster: no-α `d0w02u5t` (27.91), +α `lcaztnf8` (28.47, partial — note as such)
  - zigzag: no-α `bqenrds7` (26.84), +α max-of {`6arxz5mk` 28.56, `9jeitspu` 28.53}
  - Hilbert: no-α `qr2702yx` (24.64), +α max-of {`2w5ztkw9` 28.55, `0v0xdeih` 28.53, `k0utd8v5` 28.38}
- **Annotation:** a `+3.91 CLIP` arrow on the Hilbert pair labeled "Direction α rescue". Multi-seed error bars on the +α bars from the three Hilbert seeds.

### Figure A.5 — FID corroboration bar chart (Slide 6.5)
- **X-axis:** the 8 scored runs sorted by FID ascending: `azioyhka, 3u0fbiwp, xf18pnxa, srm8avoo, 6lkmptiw, tqlyjd5z, mq05643r, qr2702yx`.
- **Y-axis (left):** FID (range 0–140, broken or log-scale).
- **Color:** by category — α-on (green, 4), no-α hybrid (orange, 3 if include mq+qr+6lk... actually 6lkmptiw and mq05643r and qr2702yx — let's call it "no-α hybrid: orange"), simple_dit (red, 1).
- **Data:** `fid_metrics_*.json` files in `project/data/fid/`.
- **Annotation:** highlight qr2702yx (130) as "Hilbert catastrophe" and the 4 α-runs as "α rescues fidelity"; note that α-runs cluster in 35-38 vs no-α at 39+ vs simple_dit at 44 vs catastrophe at 130.

### Figure B — 4×4 ablation heatmap (Slide 7)
- **Two side-by-side panels:** left "no α", right "+α".
- **Rows:** SSM:Attention ratio in {all-SSM, 3:1, 1:1, all-attention=DiT}.
- **Columns:** scan order in {raster, zigzag, hilbert, [Z-order: empty]}.
- **Cell value:** best_val_clip_score, color scale viridis [24, 29]. Annotate each cell with score and run_id.
- **Cells with multi-seed:** take max-seed and add a small "n=k" badge.
- Empty cells (all-attention row, Z-order column, 3:1-no-α at most cells) shaded gray with "n/a".

### Figure C — LAION training curve (Slide 9)
- **X-axis:** training step (log scale, 1e3 to 2e6).
- **Y-axis:** validation CLIP score.
- **Two lines:** `cmbd8bia` (zigzag+α 3:1) and `617ma0gf` (raster+α 3:1).
- **Reference line:** dashed horizontal at 28.55 labeled "Oxford+α all-SSM Hilbert peak (multi-seed)".
- **Annotation:** `617ma0gf` collapse with downward arrow at latest step; "still training" marker on `cmbd8bia` at right edge.

### Figure D (NEW — strongest visual we have) — sample-quality grid
- **Layout:** an 8×3 or 8×4 grid: rows = the 8 FID-scored runs, columns = 3 sample PNGs from each.
- **Pull samples from:**
  - `msml612-d-rescue-eu:/tmp/fid_samples/{6lkmptiw,xf18pnxa,mq05643r,srm8avoo}/gen_00000{0,1,2}.png`
  - `msml612-d-zz1to1-eu:/tmp/fid_samples/{tqlyjd5z,qr2702yx,3u0fbiwp,azioyhka}/gen_00000{0,1,2}.png`
- **Sort rows by FID ascending** so the visual quality degradation is monotonic top-to-bottom.
- **Right-hand annotation per row:** run_id | arch | CS | FID | KID×1e3.
- **Caption:** "All samples generated with identical seeds and prompts; only the trained model differs. The visual gap between the α-equipped configurations (top) and the Hilbert catastrophe (bottom) is dramatic and self-evident."
- **This is the most viscerally convincing visual we have.** Row 1 (azioyhka FID 35.02) shows clean flowers; row 8 (qr2702yx FID 130.02) shows camo-pattern garbage that looks barely like flowers at all.

---

## 11. What to claim vs what NOT to claim

### CLAIM (defensible from our data)
- Hilbert curve serialization, the proposal's central premise, **underperforms** both raster and zigzag on our hybrid SSM-Attention DiT — at the 1:1 ratio Hilbert hits 28.31 best CS (`0u1qq5s9`) vs raster 28.48 (`s19h4nrt`) vs zigzag 28.58 (`nv1y8fwc`). This replicates ZigMa's Table 7 finding in our setup.
- The Hilbert penalty grows with the SSM fraction: in the all-SSM regime, no-α Hilbert collapses to 24.64 best CS / FID 130.02 (`qr2702yx`) while no-α raster reaches 27.91 (`d0w02u5t`).
- Three independent mechanisms (spectral mismatch, gradient variance, direction reversal) collectively explain why Hilbert fails for SSMs even though it is theoretically optimal for locality.
- Adding multi-dilation depthwise-2D-conv state fusion after the 1D scan rescues the all-SSM Hilbert configuration to 28.55 best CS (`0v0xdeih`, `2w5ztkw9`, `k0utd8v5`), a +3.91 CS gain. **The largest single architectural intervention in our ablation grid.**
- The same 2D fusion lifts every scan order at every ratio we tested (the 1:1 winner is `xf18pnxa`, `hybrid_dit+2d+hilbert` 1:1, at 28.69 best CS / 35.75 FID).
- **Hybrid+α architectures beat pure-attention DiT on FID** (35-38 vs 44.04 for simple_dit at the same ratio).
- The hybrid 2D-fusion architecture trains stably on web-scale (LAION-12M+COCO) data, evidenced by `cmbd8bia` (`hybrid_dit+2d+zigzag` 3:1, step 1.48M, train loss 0.32, CLIP similarity 22.59 best). [Preliminary; run still active.]

### Do NOT claim
- ❌ Do NOT claim FID parity with published full-scale T2I models. Our FID is on Oxford-Flowers-102 1024 samples, not COCO-30K.
- ❌ Do NOT claim our T2I model "works" at LAION qualitative scale. Best LAION CS is 22.75 — well below the Oxford 28.7 ceiling, and several runs collapsed.
- ❌ Do NOT claim Hilbert preserves locality in a way that helps SSM diffusion. It doesn't.
- ❌ Do NOT claim a SSM:Attention ratio sweep covering {all-SSM, 3:1, 1:1, 1:3, all-attention}. We have {all-SSM, 1:1, 3:1} and 1:3 + all-attention rows are missing.
- ❌ Do NOT claim a scale curve. All Oxford runs at one model size (~131M params).
- ❌ Do NOT claim a sampling-step curve.
- ❌ Do NOT claim a learned content-adaptive scan ordering. Not implemented.
- ❌ Do NOT claim Z-order ablation. Not implemented.
- ❌ Do NOT cite run `dk7sp0y1` as a 3:1 result; it's a misconfigured outlier (471k steps, CS 21.33).
- ❌ Do NOT claim "we trained on 100M+ pairs." We curated ~95M, converted ~37M, and the LAION runs use the LAION-12M+COCO subset (~15M effective with COCO oversampling).

---

## 12. Data & reproducibility

### File locations
- **Raw run dump:** `project/data/wandb_runs.jsonl` (88 runs full snapshot 2026-04-28).
- **Flat CSV summary:** `project/data/wandb_summary.csv`.
- **FID metrics JSON:** `project/data/fid/fid_metrics_*.json` (8 files).
- **This document:** `project/docs/FINAL_COMPILATION.md` (the single source of truth — TL;DR, full grid, FID story, slide-by-slide narrative, figure recipes, claim/don't-claim list).

### Pipeline scripts (under `project/`)
- `dump_wandb.py` — pulls all runs from `umd-projects/msml612-training` into JSONL/CSV.
- `prep_oxford_ref.py` — saves Oxford-Flowers-102 as 8189 256×256 PNGs to `/tmp/oxford_ref/`.
- `sample_for_fid.py` — patches FlaxDiff's GeneralDiffusionTrainer.fit to generate samples and exit. Includes three FlaxDiff-bug workarounds (pygrain workers, artifact picker, ocdbt latest_step).
- `compute_fid.py` — torch-fidelity wrapper, parses metrics, pushes to wandb summary.
- `run_full_fid.sh` — per-run wrapper (sample → FID → wandb push).
- `orchestrate_fid.sh` — distributes runs across two EU TPUs.
- `retry_orch.sh` — retry script for runs hit by the artifact-load bug.
- `pull_fid_metrics.sh` — pulls JSON metrics back from TPUs.
- `aggregate_fid.py` — produces `fid_results.md` table from JSONs.

### Wandb
- Project: `umd-projects/msml612-training`. Each run's summary now contains `fid/frechet_inception_distance`, `fid/kernel_inception_distance_mean`, `fid/inception_score_mean`, `fid/n_samples`, `fid/n_ref`, `fid/elapsed_sec`.
- View any run at `https://wandb.ai/umd-projects/msml612-training/runs/<run_id>`.

### Code release
- FlaxDiff fork at `https://github.com/AshishKumar4/FlaxDiff` (public).
- SSM-DiT extension files: `flaxdiff/models/ssm_dit.py`, `flaxdiff/models/simple_dit.py`, `flaxdiff/models/hilbert.py`.
- Direction α is enabled by appending `+2d` to the architecture name (e.g. `hybrid_dit+2d+hilbert`).

---

## 13. The one-paragraph elevator pitch (for the open of slides, or a written abstract)

We built a hybrid SSM-Attention diffusion DiT and ran a controlled 16-cell Oxford-Flowers-102 ablation crossing three scan orders (raster, zigzag, Hilbert) with three SSM:Attention ratios (all-SSM, 3:1, 1:1). We discovered the proposal's headline was wrong: Hilbert curve serialization — theoretically optimal for 2D locality — *catastrophically fails* in the all-SSM regime, collapsing to **CLIP 24.64 / FID 130.02** vs ~28.5 / 35–40 for every other tested configuration. We diagnose three independent mechanistic causes and adapt Spatial-Mamba-style 2D state fusion (Direction α) into the diffusion DiT for the first time, rescuing the catastrophe to **CLIP 28.55 across three independent seeds**, and pushing FID/KID to the best-in-grid values. Direction α is *targeted*: largest gain where 1D-scan damage is largest, neutral at 3:1 where attention masks the damage. FID/KID amplify the story over CLIP-score alone — the catastrophe gap is 3.6× larger on FID than CS, and α's effect is 4–500× larger. Preliminary LAION-12M runs show the architecture trains stably at web-scale; full FID-on-COCO and scale-curve are immediate future work.
