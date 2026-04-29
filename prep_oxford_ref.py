"""Save all Oxford Flowers 102 images as 256x256 PNGs for FID reference."""
from pathlib import Path
import tensorflow_datasets as tfds
from PIL import Image as PI

ref = Path("/tmp/oxford_ref")
existing = list(ref.glob("*.png")) if ref.exists() else []
if len(existing) > 5000:
    print(f"already have {len(existing)} ref images, skipping")
else:
    ref.mkdir(parents=True, exist_ok=True)
    ds = tfds.data_source("oxford_flowers102", split="all", try_gcs=False)
    n = 0
    for ex in ds:
        im = ex["image"]
        pil = PI.fromarray(im)
        w, h = pil.size
        s = 256 / min(w, h)
        nw, nh = int(round(w * s)), int(round(h * s))
        pil = pil.resize((nw, nh), PI.LANCZOS)
        l = (nw - 256) // 2
        t = (nh - 256) // 2
        pil = pil.crop((l, t, l + 256, t + 256))
        pil.save(ref / f"ref_{n:05d}.png")
        n += 1
        if n % 1000 == 0:
            print(f"  saved {n}")
    print(f"saved {n} ref images to {ref}")
