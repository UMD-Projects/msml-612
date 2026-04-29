"""Aggregate FID metrics JSON files from one or more TPUs into a CSV/markdown table.

Usage:
    python3 aggregate_fid.py /path/to/fid_metrics_*.json -o report.md
"""
import argparse, glob, json, sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="JSON files or globs")
    p.add_argument("-o", "--out", default="-", help="Output path (default stdout)")
    args = p.parse_args()

    files = []
    for pat in args.inputs:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        print(f"no files matched {args.inputs}", file=sys.stderr); sys.exit(1)

    rows = []
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
            m = d.get("metrics", {})
            rows.append({
                "wandb_id": d.get("wandb_id"),
                "n_samples": d.get("n_samples"),
                "fid": m.get("frechet_inception_distance"),
                "kid_mean": m.get("kernel_inception_distance_mean"),
                "kid_std": m.get("kernel_inception_distance_std"),
                "is_mean": m.get("inception_score_mean"),
                "is_std": m.get("inception_score_std"),
                "elapsed": d.get("elapsed_sec"),
            })
        except Exception as e:
            print(f"  skip {f}: {e}", file=sys.stderr)

    rows.sort(key=lambda r: r["fid"] if r["fid"] is not None else 1e9)

    lines = []
    lines.append("| run | N | FID | KID×1e3 ± std | IS ± std | sec |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        kid_str = f"{r['kid_mean']*1000:.2f} ± {r['kid_std']*1000:.2f}" if r['kid_mean'] is not None else "—"
        is_str = f"{r['is_mean']:.2f} ± {r['is_std']:.2f}" if r['is_mean'] is not None else "—"
        lines.append(f"| {r['wandb_id']} | {r['n_samples']} | {r['fid']:.2f} | {kid_str} | {is_str} | {r['elapsed']:.0f} |")

    out = "\n".join(lines)
    if args.out == "-":
        print(out)
    else:
        Path(args.out).write_text(out + "\n")
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
