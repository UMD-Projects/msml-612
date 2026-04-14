"""Spot TPU experiment supervisor.

Reads a manifest of ablation experiments and:
1. For each "queued" experiment without an active TPU, submits a queued-resource
   request via gcloud (the TPU will boot, run bootstrap.sh, and start training).
2. For each "active" experiment, polls wandb to check progress and writes the
   wandb run ID back to GCS so the next preemption recovery can resume.
3. For experiments whose TPU is in FAILED state, re-queues them.
4. For experiments whose wandb run is `finished`, marks them done.

Designed to be run inside tmux/screen on msml612-data (an on-demand TPU that
won't get preempted) so it can supervise the spot v6e fleet without dying when
your laptop disconnects.

Usage:
    python3 supervisor.py --manifest manifest.json [--once]

    --once   : Do a single pass and exit (useful for cron / one-off triggers).
    --no-act : Print actions without executing (dry run).
    --interval N : Sleep N seconds between passes (default 300 = 5 min).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ----- Shell helpers ---------------------------------------------------------

def run(cmd: list[str], capture: bool = True, check: bool = False) -> tuple[int, str]:
    """Run a subprocess and return (returncode, stdout+stderr)."""
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.returncode, (result.stdout + result.stderr)


def gcloud_qr_describe(qr_name: str, zone: str) -> Optional[dict]:
    """Return the queued-resource description as a dict, or None if not found."""
    code, out = run([
        "gcloud", "compute", "tpus", "queued-resources", "describe", qr_name,
        f"--zone={zone}", "--format=json",
    ])
    if code != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def gcloud_qr_state(qr_name: str, zone: str) -> str:
    """Return the queued resource state, or 'NOT_FOUND' if missing."""
    desc = gcloud_qr_describe(qr_name, zone)
    if desc is None:
        return "NOT_FOUND"
    return desc.get("state", {}).get("state", "UNKNOWN")


def gcloud_tpu_state(tpu_name: str, zone: str) -> str:
    """Return the TPU VM state, or 'NOT_FOUND' if missing."""
    code, out = run([
        "gcloud", "compute", "tpus", "tpu-vm", "describe", tpu_name,
        f"--zone={zone}", "--format=value(state)",
    ])
    if code != 0:
        return "NOT_FOUND"
    return out.strip() or "UNKNOWN"


# ----- Experiment data model -------------------------------------------------

@dataclass
class Experiment:
    name: str
    description: str
    tpu_name: str
    zone: str
    accelerator: str
    expected_steps: int
    status: str = "queued"  # queued | provisioning | running | preempted | finished | failed
    wandb_id: Optional[str] = None

    @property
    def qr_name(self) -> str:
        return f"qr-{self.tpu_name}"


# ----- Manifest I/O ----------------------------------------------------------

class Manifest:
    def __init__(self, path: Path):
        self.path = path
        with open(path) as f:
            data = json.load(f)
        self.gcs_bucket: str = data["gcs_bucket"]
        self.wandb_project: str = data["wandb_project"]
        self.experiments: list[Experiment] = [
            Experiment(**{k: v for k, v in e.items() if not k.startswith("_")})
            for e in data["experiments"] if not e.get("name", "").startswith("_")
        ]
        self._extra = {k: v for k, v in data.items() if k not in ("gcs_bucket", "wandb_project", "experiments")}

    def save(self):
        out = {
            **self._extra,
            "gcs_bucket": self.gcs_bucket,
            "wandb_project": self.wandb_project,
            "experiments": [asdict(e) for e in self.experiments],
        }
        with open(self.path, "w") as f:
            json.dump(out, f, indent=2)


# ----- Wandb integration -----------------------------------------------------

def find_wandb_run_for_experiment(project: str, experiment_name: str) -> Optional[dict]:
    """Find the most recent wandb run matching the experiment name."""
    try:
        import wandb
    except ImportError:
        return None
    api = wandb.Api()
    runs = list(api.runs(project, order="-created_at"))
    pattern = re.escape(experiment_name).replace(r"\+", r"\+")
    for r in runs:
        name = r.name or ""
        if name.startswith("old-"):
            continue
        if re.search(rf"arch-{pattern}\b", name) or experiment_name.replace("+", "_") in name:
            return {
                "id": r.id,
                "state": r.state,
                "name": name,
                "summary": dict(r.summary or {}),
            }
    return None


# ----- Supervisor logic ------------------------------------------------------

class Supervisor:
    def __init__(self, manifest: Manifest, dry_run: bool = False, project_root: Optional[Path] = None):
        self.manifest = manifest
        self.dry_run = dry_run
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent
        self.launch_script = Path(__file__).parent / "launch_experiment.sh"

    def _launch(self, exp: Experiment) -> bool:
        """Submit a queued resource request for an experiment via launch_experiment.sh."""
        cmd = [
            "bash", str(self.launch_script),
            "--experiment", exp.name,
            "--tpu-name", exp.tpu_name,
            "--zone", exp.zone,
            "--accelerator", exp.accelerator,
            "--gcs-bucket", self.manifest.gcs_bucket,
        ]
        print(f"  $ {' '.join(cmd)}")
        if self.dry_run:
            return True
        code, out = run(cmd)
        if code != 0:
            print(f"  Launch failed: {out[:300]}")
            return False
        return True

    def _delete_qr(self, exp: Experiment) -> bool:
        """Delete a queued resource (used when re-queuing after failure)."""
        cmd = ["gcloud", "compute", "tpus", "queued-resources", "delete", exp.qr_name,
               f"--zone={exp.zone}", "--quiet"]
        print(f"  $ {' '.join(cmd)}")
        if self.dry_run:
            return True
        code, _ = run(cmd)
        return code == 0

    def step(self):
        """Single supervision pass."""
        print(f"\n[{time.strftime('%H:%M:%S')}] Supervisor pass")
        for exp in self.manifest.experiments:
            self._step_one(exp)
        self.manifest.save()

    def _step_one(self, exp: Experiment):
        """Decide what to do with one experiment based on its current state."""
        qr_state = gcloud_qr_state(exp.qr_name, exp.zone)
        wandb_info = find_wandb_run_for_experiment(self.manifest.wandb_project, exp.name)
        wandb_state = wandb_info["state"] if wandb_info else None
        wandb_id = wandb_info["id"] if wandb_info else None

        # Update wandb_id if found (so future resume works)
        if wandb_id and exp.wandb_id != wandb_id:
            exp.wandb_id = wandb_id

        # State machine
        if wandb_state == "finished":
            if exp.status != "finished":
                print(f"[{exp.name}] FINISHED (wandb run {wandb_id})")
                exp.status = "finished"
            return

        if qr_state == "NOT_FOUND":
            if exp.status in ("queued", "preempted", "failed"):
                print(f"[{exp.name}] No active queued resource → submitting")
                if self._launch(exp):
                    exp.status = "provisioning"
            return

        if qr_state in ("WAITING_FOR_RESOURCES", "ACCEPTED", "PROVISIONING", "CREATING"):
            if exp.status != "provisioning":
                print(f"[{exp.name}] queued resource {qr_state}")
                exp.status = "provisioning"
            return

        if qr_state == "ACTIVE":
            tpu_state = gcloud_tpu_state(exp.tpu_name, exp.zone)
            if tpu_state == "READY":
                if wandb_state == "running":
                    if exp.status != "running":
                        print(f"[{exp.name}] RUNNING (wandb {wandb_id}, TPU READY)")
                        exp.status = "running"
                else:
                    # TPU is up but no wandb run yet — bootstrap.sh is probably still in setup
                    if exp.status != "running":
                        print(f"[{exp.name}] TPU READY, waiting for wandb run...")
                        exp.status = "provisioning"
            elif tpu_state in ("PREEMPTED", "STOPPED", "TERMINATED"):
                print(f"[{exp.name}] PREEMPTED → deleting QR and re-launching")
                self._delete_qr(exp)
                exp.status = "preempted"
            elif tpu_state == "NOT_FOUND":
                print(f"[{exp.name}] TPU disappeared → deleting QR and re-launching")
                self._delete_qr(exp)
                exp.status = "preempted"
            return

        if qr_state in ("FAILED", "SUSPENDED"):
            print(f"[{exp.name}] queued resource {qr_state} → deleting and re-queuing")
            self._delete_qr(exp)
            exp.status = "failed"
            return

        print(f"[{exp.name}] qr_state={qr_state} (no action)")


# ----- CLI -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-act", dest="dry_run", action="store_true")
    ap.add_argument("--interval", type=int, default=300, help="Seconds between passes (default 300)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        sys.exit(1)

    manifest = Manifest(manifest_path)
    sup = Supervisor(manifest, dry_run=args.dry_run)

    print(f"Supervisor started. Manifest: {manifest_path}")
    print(f"  GCS bucket: {manifest.gcs_bucket}")
    print(f"  Wandb project: {manifest.wandb_project}")
    print(f"  Experiments: {len(manifest.experiments)}")
    print(f"  Dry run: {args.dry_run}")

    while True:
        try:
            sup.step()
        except Exception as e:
            print(f"  Supervisor error (continuing): {e}")
        if args.once:
            break
        print(f"  Sleeping {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
