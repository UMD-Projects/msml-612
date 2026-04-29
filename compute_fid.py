"""Compute FID + KID + IS for a generated-sample dir against an Oxford ref dir,
parse the metrics, optionally push to the originating wandb run summary.

Usage:
    python compute_fid.py --wandb_id xf18pnxa \
        --samples_dir /tmp/fid_samples/xf18pnxa \
        --ref_dir /tmp/oxford_ref \
        --push_wandb
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--wandb_id", required=True)
p.add_argument("--samples_dir", required=True)
p.add_argument("--ref_dir", required=True)
p.add_argument("--push_wandb", action="store_true")
p.add_argument("--out_json", default=None,
               help="Optional path to dump metrics JSON locally.")
args = p.parse_args()

samples = Path(args.samples_dir)
ref = Path(args.ref_dir)
n_samples = len(list(samples.glob("*.png")))
n_ref = len(list(ref.glob("*.png")))
print(f"[fid] wandb_id={args.wandb_id} samples={n_samples} ref={n_ref}")
if n_samples < 50:
    print("[fid] FATAL: too few samples", file=sys.stderr); sys.exit(2)
if n_ref < 50:
    print("[fid] FATAL: too few ref images", file=sys.stderr); sys.exit(2)

# KID subset size must be <= min(n1, n2). Default 1000 fails for small smoke tests.
kid_subset = min(1000, n_samples, n_ref)
cmd = [
    "fidelity",
    "--fid", "--kid", "--isc",
    "--input1", str(samples),
    "--input2", str(ref),
    "--samples-find-deep", "0",
    "--no-cuda",
    "--kid-subset-size", str(kid_subset),
]
print("[fid] cmd:", " ".join(cmd))
env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = ""
t0 = time.time()
res = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)
elapsed = time.time() - t0
print(f"[fid] elapsed {elapsed:.1f}s exit={res.returncode}")
print("---- stdout ----")
print(res.stdout)
print("---- stderr ----")
print(res.stderr[-2000:] if res.stderr else "")

if res.returncode != 0:
    sys.exit(res.returncode)

# Parse metrics. torch-fidelity emits lines like:
#   frechet_inception_distance: 27.4123
#   kernel_inception_distance_mean: 0.0123
#   kernel_inception_distance_std: 0.0021
#   inception_score_mean: 5.42
#   inception_score_std: 0.31
metrics = {}
for line in res.stdout.splitlines():
    m = re.match(r"^([a-z_]+):\s+([-+0-9.eE]+)\s*$", line.strip())
    if m:
        metrics[m.group(1)] = float(m.group(2))
print("[fid] parsed:", json.dumps(metrics, indent=2))

if args.out_json:
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(
        {"wandb_id": args.wandb_id, "n_samples": n_samples, "n_ref": n_ref,
         "elapsed_sec": elapsed, "metrics": metrics}, indent=2))
    print(f"[fid] wrote {args.out_json}")

if args.push_wandb and metrics:
    import wandb
    print(f"[fid] resuming wandb run {args.wandb_id}...")
    run = wandb.init(project="msml612-training", entity="umd-projects",
                     id=args.wandb_id, resume="must")
    for k, v in metrics.items():
        run.summary[f"fid/{k}"] = v
    run.summary["fid/n_samples"] = n_samples
    run.summary["fid/n_ref"] = n_ref
    run.summary["fid/elapsed_sec"] = elapsed
    run.summary.update()
    run.finish()
    print(f"[fid] pushed to wandb run {args.wandb_id}")
