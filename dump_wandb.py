"""Dump all wandb runs from umd-projects/msml612-training to local files.

Outputs:
  project/data/wandb_runs.jsonl       — one line per run with config + summary
  project/data/wandb_summary.csv      — flat table for quick analysis
  project/data/history/<run_id>.csv   — per-step training history per run

Robust to wandb's lazy-config-load by re-fetching each run via api.run().
"""
import json
import csv
import os
from pathlib import Path
import wandb

OUT_DIR = Path("/home/mrwhite0racle/Desktop/UMDCourseWork/MSML612/project/data")
HISTORY_DIR = OUT_DIR / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

api = wandb.Api()
runs = list(api.runs("umd-projects/msml612-training", order="-created_at", per_page=200))
print(f"Total runs: {len(runs)}")

JSONL_PATH = OUT_DIR / "wandb_runs.jsonl"
CSV_PATH = OUT_DIR / "wandb_summary.csv"

csv_rows = []
with open(JSONL_PATH, "w") as jf:
    for i, r in enumerate(runs):
        # Re-fetch for reliable config
        try:
            rr = api.run(f"umd-projects/msml612-training/{r.id}")
        except Exception as e:
            print(f"  {r.id}: api.run failed: {e}")
            continue
        cfg = dict(rr.config) if isinstance(rr.config, dict) else {}
        args = cfg.get("arguments", {}) or {}
        if not isinstance(args, dict):
            args = {}
        model_cfg = cfg.get("model", {}) or {}
        if not isinstance(model_cfg, dict):
            model_cfg = {}
        summary = dict(rr.summary) if rr.summary else {}

        rec = {
            "id": rr.id,
            "name": rr.name,
            "state": rr.state,
            "created_at": str(rr.created_at) if hasattr(rr, "created_at") else None,
            "tags": list(rr.tags) if hasattr(rr, "tags") else [],
            "url": rr.url,
            # Architecture/config-level
            "arch": args.get("architecture") or cfg.get("architecture"),
            "ratio": args.get("ssm_attention_ratio") or model_cfg.get("ssm_attention_ratio"),
            "dataset": args.get("dataset") or cfg.get("dataset"),
            "epochs": args.get("epochs"),
            "steps_per_epoch": args.get("steps_per_epoch"),
            "batch_size": args.get("batch_size"),
            "image_size": args.get("image_size"),
            "learning_rate": args.get("learning_rate"),
            "use_2d_fusion": args.get("use_2d_fusion"),
            "use_hilbert": args.get("use_hilbert"),
            "use_zigzag": args.get("use_zigzag"),
            "ssm_state_dim": args.get("ssm_state_dim"),
            "emb_features": args.get("emb_features"),
            "num_layers": args.get("num_layers"),
            "num_heads": args.get("num_heads"),
            "patch_size": args.get("patch_size"),
            "noise_schedule": args.get("noise_schedule"),
            "ema_decay": args.get("ema_decay"),
            "augmentation_mode": args.get("augmentation_mode"),
            # Summary metrics
            "step": summary.get("_step"),
            "runtime_sec": summary.get("_runtime"),
            "best_val_clip_score": summary.get("best_val/clip_score"),
            "val_clip_score": summary.get("val/clip_score"),
            "best_val_clip_similarity": summary.get("best_val/clip_similarity"),
            "val_clip_similarity": summary.get("val/clip_similarity"),
            "best_val_loss": summary.get("best_val/loss"),
            "val_loss": summary.get("val/loss"),
            "train_loss": summary.get("train/loss") or summary.get("train/avg_loss"),
            "train_best_loss": summary.get("train/best_loss"),
            "train_avg_loss": summary.get("train/avg_loss"),
            # Full snapshots
            "summary_full": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)[:200]) for k, v in summary.items()},
        }
        jf.write(json.dumps(rec, default=str) + "\n")
        csv_rows.append(rec)

        # Pull history (key training/val metrics over time)
        try:
            hist = list(rr.scan_history(keys=["_step", "train/loss", "train/avg_loss", "val/clip_score", "val/loss", "best_val/clip_score", "_runtime"]))
            if hist:
                with open(HISTORY_DIR / f"{rr.id}.csv", "w", newline="") as hf:
                    writer = csv.DictWriter(hf, fieldnames=["_step", "train/loss", "train/avg_loss", "val/clip_score", "val/loss", "best_val/clip_score", "_runtime"])
                    writer.writeheader()
                    for row in hist:
                        writer.writerow({k: row.get(k) for k in writer.fieldnames})
        except Exception as e:
            print(f"  {rr.id}: history err: {e}")

        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{len(runs)}")

# Flat CSV summary
csv_keys = [k for k in csv_rows[0].keys() if k != "summary_full"] if csv_rows else []
with open(CSV_PATH, "w", newline="") as cf:
    writer = csv.DictWriter(cf, fieldnames=csv_keys)
    writer.writeheader()
    for row in csv_rows:
        writer.writerow({k: row.get(k) for k in csv_keys})

print(f"\nDumped {len(csv_rows)} runs.")
print(f"  jsonl: {JSONL_PATH}")
print(f"  csv: {CSV_PATH}")
print(f"  history dir: {HISTORY_DIR} ({len(list(HISTORY_DIR.glob('*.csv')))} files)")
