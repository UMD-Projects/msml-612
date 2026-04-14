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


# ----- Quota tracking (defense-in-depth) -------------------------------------

# TFRC quota grant. Keys are (zone, family, is_spot). family is one of
# {"v4", "v5e", "v6e", "v5p"}. Values are total chip count for that bucket.
# Source: TFRC grant email — keep in sync if the user's quota changes.
TFRC_QUOTA = {
    ("us-central2-b",   "v4",  False): 32,  # on-demand v4
    ("us-central2-b",   "v4",  True):  32,  # spot v4
    ("europe-west4-a",  "v6e", True):  64,  # spot v6e
    ("us-east1-d",      "v6e", True):  64,  # spot v6e
    ("europe-west4-b",  "v5e", True):  64,  # spot v5e
    ("us-central1-a",   "v5e", True):  64,  # spot v5e
}


def parse_chip_count(accelerator_type: str) -> int:
    """Return the number of chips in a TPU accelerator type string.

    Examples:
        v4-8 -> 8
        v4-32 -> 32
        v6e-4 -> 4
        v5litepod-8 -> 8
    """
    m = re.search(r"-(\d+)$", accelerator_type)
    if not m:
        return 0
    return int(m.group(1))


def parse_family(accelerator_type: str) -> str:
    """Return the chip family for an accelerator type string."""
    if accelerator_type.startswith("v6e"):
        return "v6e"
    if accelerator_type.startswith("v5p"):
        return "v5p"
    if accelerator_type.startswith("v5e") or accelerator_type.startswith("v5litepod"):
        return "v5e"
    if accelerator_type.startswith("v4"):
        return "v4"
    if accelerator_type.startswith("v3"):
        return "v3"
    return "unknown"


def list_chips_in_zone(zone: str) -> dict:
    """Return chips currently used in a zone, broken down by (family, is_spot).

    Counts both ACTIVE TPU VMs and queued resources that have or will allocate
    chips. Chips in FAILED/SUSPENDED/preempted states do NOT count (GCP has
    released them already).

    Returns dict like {("v6e", True): 8, ("v4", False): 16}.
    """
    used = {}

    # 1. Direct TPU VMs in the zone
    code, out = run([
        "gcloud", "compute", "tpus", "tpu-vm", "list",
        f"--zone={zone}", "--format=json",
    ])
    if code == 0:
        try:
            tpus = json.loads(out)
            for t in tpus:
                state = t.get("state", "")
                if state in ("DELETING", "PREEMPTED", "TERMINATED", "STOPPED", "SUSPENDED", "FAILED"):
                    continue
                acc = t.get("acceleratorType", "")
                family = parse_family(acc)
                chips = parse_chip_count(acc)
                # spec.preemptible / preemptible flag varies by API version; check both
                is_spot = bool(t.get("schedulingConfig", {}).get("preemptible") or
                               t.get("schedulingConfig", {}).get("spot") or
                               t.get("preemptible", False))
                key = (family, is_spot)
                used[key] = used.get(key, 0) + chips
        except json.JSONDecodeError:
            pass

    # 2. Queued resources in the zone (active ones consume quota even before
    #    the underlying VM exists). Note that queued resources in
    #    WAITING_FOR_RESOURCES do NOT yet hold quota — only ACTIVE/PROVISIONING
    #    do. But we conservatively count all non-terminal queued resources.
    code, out = run([
        "gcloud", "compute", "tpus", "queued-resources", "list",
        f"--zone={zone}", "--format=json",
    ])
    if code == 0:
        try:
            qrs = json.loads(out)
            for qr in qrs:
                state = qr.get("state", {}).get("state", "")
                if state in ("FAILED", "SUSPENDED", "DELETING"):
                    continue
                # Skip queued resources whose underlying TPU we already counted above
                node_specs = (qr.get("tpu", {}) or {}).get("nodeSpec", []) or []
                for spec in node_specs:
                    node = spec.get("node", {})
                    node_id = spec.get("nodeId", "")
                    acc = node.get("acceleratorType", "")
                    if not acc:
                        continue
                    family = parse_family(acc)
                    chips = parse_chip_count(acc)
                    is_spot = bool(node.get("schedulingConfig", {}).get("preemptible") or
                                   node.get("schedulingConfig", {}).get("spot"))
                    # Don't double-count: only count queued resources whose
                    # corresponding TPU VM doesn't exist yet (state != ACTIVE means
                    # there's no VM, but ACTIVE means the VM was already counted above).
                    if state == "ACTIVE":
                        continue
                    key = (family, is_spot)
                    used[key] = used.get(key, 0) + chips
        except json.JSONDecodeError:
            pass

    return used


def quota_status(zone: str, family: str, is_spot: bool) -> tuple[int, int, int]:
    """Return (used, quota, free) for a (zone, family, spot) bucket.

    Returns (-1, -1, -1) if no TFRC quota record exists for this combination.
    """
    quota = TFRC_QUOTA.get((zone, family, is_spot), -1)
    if quota < 0:
        return (-1, -1, -1)
    used_dict = list_chips_in_zone(zone)
    used = used_dict.get((family, is_spot), 0)
    return (used, quota, quota - used)


def can_launch(zone: str, accelerator_type: str, is_spot: bool, log_prefix: str = "") -> tuple[bool, str]:
    """Return (allowed, reason) for whether launching `accelerator_type` in `zone` is within quota.

    Hard refuses if the request would push usage over TFRC_QUOTA. Returns True
    with a reason like "ok: 8/64 used in europe-west4-a" if it fits.
    """
    family = parse_family(accelerator_type)
    chips_needed = parse_chip_count(accelerator_type)
    if chips_needed == 0:
        return False, f"could not parse chip count from accelerator '{accelerator_type}'"

    used, quota, free = quota_status(zone, family, is_spot)
    if quota < 0:
        return False, f"no TFRC_QUOTA record for ({zone}, {family}, spot={is_spot}); refusing as safety measure"

    if chips_needed > free:
        return False, (f"would exceed quota: need {chips_needed} {family} chips in {zone} "
                       f"({'spot' if is_spot else 'on-demand'}), only {free}/{quota} free "
                       f"(used {used}/{quota})")

    return True, f"ok: would use {used + chips_needed}/{quota} {family} chips in {zone}"


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
        # Hard cap on total concurrently launching/running experiments. Default 6.
        self.max_concurrent: int = int(data.get("max_concurrent", 6))
        self.experiments: list[Experiment] = [
            Experiment(**{k: v for k, v in e.items() if not k.startswith("_")})
            for e in data["experiments"] if not e.get("name", "").startswith("_")
        ]
        self._extra = {k: v for k, v in data.items()
                       if k not in ("gcs_bucket", "wandb_project", "experiments", "max_concurrent")}

    def save(self):
        out = {
            **self._extra,
            "gcs_bucket": self.gcs_bucket,
            "wandb_project": self.wandb_project,
            "max_concurrent": self.max_concurrent,
            "experiments": [asdict(e) for e in self.experiments],
        }
        with open(self.path, "w") as f:
            json.dump(out, f, indent=2)


# ----- Wandb integration -----------------------------------------------------

def find_wandb_run_for_experiment(project: str, experiment_name: str) -> Optional[dict]:
    """Find the most recent (non-`old-`) wandb run matching this experiment.

    Matching strategy:
    - Map the manifest experiment name to the (architecture, ssm_attention_ratio)
      pair that the training run will log.
    - Find the most recent run whose `arch-<architecture>` substring matches
      AND whose ssm_attention_ratio matches (when applicable).
    - Use the run's `_step` to confirm it's actually progressing (not just
      created and crashed immediately).
    """
    try:
        import wandb
    except ImportError:
        return None

    # Map manifest experiment name → (arch path component, ssm ratio or None)
    name_to_arch = {
        "simple_dit":                 ("simple_dit",          None),
        "simple_dit+hilbert":         ("simple_dit+hilbert",  None),
        "hybrid_dit_3to1":            ("hybrid_dit",          "3:1"),
        "hybrid_dit+hilbert_3to1":    ("hybrid_dit+hilbert",  "3:1"),
        "hybrid_dit+hilbert_1to1":    ("hybrid_dit+hilbert",  "1:1"),
        "hybrid_dit+hilbert_all_ssm": ("hybrid_dit+hilbert",  "all-ssm"),
    }
    if experiment_name not in name_to_arch:
        return None
    target_arch, target_ratio = name_to_arch[experiment_name]

    api = wandb.Api()
    runs = list(api.runs(project, order="-created_at"))

    for r in runs:
        name = r.name or ""
        if name.startswith("old-"):
            continue

        # Match `arch-<exact_arch>/` (use trailing slash since name uses path encoding)
        # so `arch-simple_dit/` doesn't match `arch-simple_dit+hilbert/`.
        if re.search(rf"arch-{re.escape(target_arch)}/", name) is None:
            continue

        # If the architecture has an SSM ratio, verify it matches via run config.
        if target_ratio is not None:
            cfg_ratio = (r.config or {}).get("ssm_attention_ratio")
            if cfg_ratio is None:
                cfg_ratio = (r.config or {}).get("model", {}).get("ssm_attention_ratio")
            if cfg_ratio != target_ratio:
                continue

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
        # Track chips committed in the CURRENT pass (gcloud queued-resources list
        # may take a few seconds to reflect newly-submitted requests, so we add
        # in-pass commitments to the quota check to avoid double-spending).
        # Key: (zone, family, is_spot) -> chip count
        self._pass_committed: dict = {}

    def _count_active_experiments(self) -> int:
        """Number of manifest experiments currently in provisioning/running state."""
        return sum(1 for e in self.manifest.experiments
                   if e.status in ("provisioning", "running"))

    def _quota_check(self, exp: Experiment) -> tuple[bool, str]:
        """Hard quota check before launching an experiment. Returns (ok, reason).

        Combines:
          - max_concurrent cap from manifest
          - TFRC quota by zone × family × spot (live from gcloud)
          - In-pass commitments (chips we've already submitted this supervisor pass)
        """
        # Layer 1: max_concurrent cap from the manifest
        active = self._count_active_experiments()
        if active >= self.manifest.max_concurrent:
            return False, (f"max_concurrent={self.manifest.max_concurrent} reached "
                           f"({active} provisioning/running)")

        # Layer 2: TFRC quota by zone × family × spot
        is_spot = True
        family = parse_family(exp.accelerator)
        chips_needed = parse_chip_count(exp.accelerator)
        if chips_needed == 0:
            return False, f"could not parse chip count from '{exp.accelerator}'"

        used, quota, free = quota_status(exp.zone, family, is_spot)
        if quota < 0:
            return False, f"no TFRC_QUOTA record for ({exp.zone}, {family}, spot={is_spot})"

        # Layer 3: include in-pass commitments (haven't propagated to gcloud list yet)
        in_pass = self._pass_committed.get((exp.zone, family, is_spot), 0)
        effective_used = used + in_pass
        effective_free = quota - effective_used

        if chips_needed > effective_free:
            return False, (f"would exceed quota: need {chips_needed} {family} chips in {exp.zone} "
                           f"(spot={is_spot}), only {effective_free}/{quota} free "
                           f"(gcloud_used={used} + in_pass={in_pass})")

        return True, (f"ok: would use {effective_used + chips_needed}/{quota} {family} chips in {exp.zone} "
                      f"(gcloud_used={used} + in_pass={in_pass} + new={chips_needed})")

    def _launch(self, exp: Experiment) -> bool:
        """Submit a queued resource request for an experiment via launch_experiment.sh.

        Pre-flight: enforces max_concurrent, TFRC quota, and in-pass commitments.
        Refuses to submit if the launch would exceed any limit.
        """
        ok, reason = self._quota_check(exp)
        if not ok:
            print(f"  REFUSED: {reason}")
            return False
        print(f"  OK: {reason}")

        cmd = [
            "bash", str(self.launch_script),
            "--experiment", exp.name,
            "--tpu-name", exp.tpu_name,
            "--zone", exp.zone,
            "--accelerator", exp.accelerator,
            "--gcs-bucket", self.manifest.gcs_bucket,
        ]
        print(f"  $ {' '.join(cmd)}")

        if not self.dry_run:
            code, out = run(cmd)
            if code != 0:
                print(f"  Launch failed: {out[:300]}")
                return False

        # Reserve chips in the in-pass committed counter so subsequent quota
        # checks in this same pass don't double-spend.
        family = parse_family(exp.accelerator)
        chips = parse_chip_count(exp.accelerator)
        key = (exp.zone, family, True)  # is_spot=True
        self._pass_committed[key] = self._pass_committed.get(key, 0) + chips
        return True

    def _print_quota_status(self):
        """Log current TFRC quota usage on every pass."""
        print("  Quota status:")
        seen_zones = set()
        for exp in self.manifest.experiments:
            if exp.zone in seen_zones:
                continue
            seen_zones.add(exp.zone)
            family = parse_family(exp.accelerator)
            used, quota, free = quota_status(exp.zone, family, is_spot=True)
            if quota >= 0:
                print(f"    {exp.zone}/{family} (spot): {used}/{quota} chips used, {free} free")
            else:
                print(f"    {exp.zone}/{family} (spot): no TFRC_QUOTA record")

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
        # Reset in-pass commitment counter at the start of every pass — gcloud
        # will reflect committed chips by the next pass.
        self._pass_committed = {}
        self._print_quota_status()
        active = self._count_active_experiments()
        print(f"  Active experiments: {active}/{self.manifest.max_concurrent}")
        for exp in self.manifest.experiments:
            self._step_one(exp)
        if not self.dry_run:
            self.manifest.save()
        else:
            print("  (dry run: manifest NOT saved)")

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
