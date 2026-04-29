"""Run training.py's init + load checkpoint, but instead of training, generate
N samples conditioned on Oxford class labels and save as PNG.

This script handles BOTH the parent process (which sets up FID sampling) and
respawned children (which inherit the parent's modified sys.argv pointing at
training.py-style args). When run as a child without --wandb_id we just
delegate to training.main() so the worker does its normal initialization.

Usage:
    python sample_for_fid.py --wandb_id xf18pnxa --n_samples 2048 --out_dir /tmp/fid_samples/xf18pnxa
"""
import os, sys, argparse
from pathlib import Path

# Probe sys.argv for --wandb_id BEFORE doing anything else, because dataset
# prefetch workers / jax.distributed children may re-execve this script with
# the parent's crafted training.py-style argv (no --wandb_id).
_HAS_WANDB_ID = any(a == "--wandb_id" or a.startswith("--wandb_id=") for a in sys.argv[1:])


def _build_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--wandb_id", required=True)
    p.add_argument("--n_samples", type=int, default=2048)
    p.add_argument("--diffusion_steps", type=int, default=50)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--research_dir", default="/home/mrwhite0racle/research")
    return p.parse_known_args()[0]


def _make_training_argv(preargs, rargs):
    return [
        "training.py",
        "--autoencoder", rargs.get("autoencoder", "stable_diffusion"),
        "--autoencoder_opts", rargs.get("autoencoder_opts", '{"modelname":"pcuenq/sd-vae-ft-mse-flax"}'),
        "--dataset", rargs.get("dataset", "oxford_flowers102"),
        "--dataset_path", "/home/mrwhite0racle/gcs_mount",
        "--batch_size", str(rargs.get("batch_size", 64)),
        "--image_size", str(rargs.get("image_size", 256)),
        "--epochs", "1",
        "--steps_per_epoch", "1",
        "--val_steps_per_epoch", "1",
        "--noise_schedule", rargs.get("noise_schedule", "edm"),
        "--learning_rate", str(rargs.get("learning_rate", 0.0003)),
        "--optimizer", rargs.get("optimizer", "adamw"),
        "--dropout_rate", str(rargs.get("dropout_rate", 0.1)),
        "--ema_decay", str(rargs.get("ema_decay", 0.999)),
        "--augmentation_mode", rargs.get("augmentation_mode", "flip_jitter"),
        "--emb_features", str(rargs.get("emb_features", 512)),
        "--num_layers", str(rargs.get("num_layers", 16)),
        "--num_heads", str(rargs.get("num_heads", 8)),
        "--patch_size", str(rargs.get("patch_size", 2)),
        "--mlp_ratio", str(rargs.get("mlp_ratio", 4)),
        "--norm_groups", str(rargs.get("norm_groups", 0)),
        "--dtype", rargs.get("dtype", "float32"),
        "--precision", rargs.get("precision", "default"),
        "--only_pure_attention", str(rargs.get("only_pure_attention", True)),
        "--distributed_training", "True",
        "--val_metrics", "clip", "clip_score",
        "--best_tracker_metric", "val/clip_score",
        "--wandb_project", "msml612-training",
        "--wandb_entity", "umd-projects",
        "--architecture", rargs.get("architecture"),
        "--ssm_attention_ratio", str(rargs.get("ssm_attention_ratio", "3:1")),
        "--ssm_state_dim", str(rargs.get("ssm_state_dim", 64)),
        "--use_2d_fusion", str(rargs.get("use_2d_fusion", False)),
        "--resume_last_run", preargs.wandb_id,
    ]


def _patched_fit_factory(out_dir, n_samples_target, diffusion_steps):
    import jax
    import numpy as np
    from PIL import Image as PI

    def patched_fit(self, data, training_steps_per_epoch, epochs, sampler_class=None,
                    sampling_noise_schedule=None, val_steps_per_epoch=1):
        print(f"\n[FID Pipeline] Hijacked fit(). Generating {n_samples_target} samples -> {out_dir}", flush=True)
        val_step_fn = self._define_validation_step(sampler_class, sampling_noise_schedule)
        val_state = self.get_state() if hasattr(self, 'get_state') else self.state
        val_iter = iter(data["val"]())
        saved = 0
        batch_idx = 0
        while saved < n_samples_target:
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(data["val"]())
                batch = next(val_iter)
            samples = val_step_fn(val_state, batch, diffusion_steps)
            samples = np.asarray(jax.device_get(samples))
            samples = ((samples + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
            for i in range(samples.shape[0]):
                if saved >= n_samples_target:
                    break
                PI.fromarray(samples[i]).save(out_dir / f"gen_{saved:06d}.png")
                saved += 1
            batch_idx += 1
            if batch_idx % 2 == 0:
                print(f"[FID Pipeline]   saved {saved}/{n_samples_target}", flush=True)
        print(f"[FID Pipeline] Done. Wrote {saved} samples to {out_dir}", flush=True)
        sys.exit(0)
    return patched_fit


def _run_as_child():
    """Respawn entry: a multiprocessing 'spawn' child re-exec'd this script
    inheriting the parent's training.py-style sys.argv. Whatever it was meant
    to do, the parent has the TPU lock and the patched fit() is doing real
    work; children fighting for TPU just break the run. Exit silently.
    """
    sys.exit(0)


def _run_as_parent():
    preargs = _build_args()
    out_dir = Path(preargs.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, preargs.research_dir)

    import wandb
    api = wandb.Api()
    run = api.run(f"umd-projects/msml612-training/{preargs.wandb_id}")
    cfg = dict(run.config)
    rargs = cfg.get("arguments", {})
    print(f"[FID Pipeline] Parent: run {preargs.wandb_id} arch={rargs.get('architecture')} "
          f"ratio={rargs.get('ssm_attention_ratio')} ds={rargs.get('dataset')}", flush=True)

    # Workaround for FlaxDiff trainer bug (simple_trainer.py:206 picks
    # model_artifacts[0], often an old/wrong artifact). Find the latest
    # VALID artifact owned by this run — meaning highest version whose
    # downloaded directory contains an orbax-compatible checkpoint (i.e.,
    # at least one numeric subdirectory representing a step).
    model_arts = [a for a in run.logged_artifacts() if a.type == "model"]
    own_arts = []
    for a in model_arts:
        try:
            src = a.source_run
            if src is not None and src.id == preargs.wandb_id:
                own_arts.append(a)
        except Exception:
            pass
    if not own_arts:
        own_arts = model_arts  # fallback
    own_arts.sort(key=lambda a: int(a.version.lstrip("v")) if a.version.startswith("v") and a.version[1:].isdigit() else -1, reverse=True)

    preargs._ckpt_dir = None
    from pathlib import Path as _P
    for a in own_arts[:3]:  # try top-3 latest
        print(f"[FID Pipeline] Trying artifact {a.name} (version {a.version})...", flush=True)
        try:
            d = a.download()
            # Valid orbax checkpoint dir has either:
            #   - step-numbered subdirs (legacy CheckpointManager layout), OR
            #   - direct default/ subdir with ocdbt content (newer orbax layout)
            children = {c.name for c in _P(d).iterdir() if c.is_dir() or c.is_file()}
            ocdbt_path = _P(d) / "default" / "manifest.ocdbt"
            num_dirs = [c for c in _P(d).iterdir() if c.is_dir() and c.name.isdigit()]
            if num_dirs or ocdbt_path.exists():
                fmt = "step-subdirs" if num_dirs else "ocdbt-direct"
                print(f"[FID Pipeline] OK ({fmt}): {a.name} at {d}", flush=True)
                preargs._ckpt_dir = d
                break
            else:
                print(f"[FID Pipeline]  -> not a valid checkpoint (children={list(children)[:5]}); trying older", flush=True)
        except Exception as e:
            print(f"[FID Pipeline]  -> download/check failed: {e}", flush=True)

    if preargs._ckpt_dir is None:
        print(f"[FID Pipeline] FATAL: no valid checkpoint artifact for {preargs.wandb_id}", flush=True)
        sys.exit(2)

    sys.argv = _make_training_argv(preargs, rargs)
    sys.argv.extend(["--load_from_checkpoint", str(preargs._ckpt_dir)])

    # Disable pygrain multiprocessing — workers re-exec this script and break.
    # We only need val data for conditioning, not throughput.
    import grain.python as pygrain
    _original_loader = pygrain.DataLoader
    def _no_workers_loader(*args, **kwargs):
        kwargs["worker_count"] = 0
        return _original_loader(*args, **kwargs)
    pygrain.DataLoader = _no_workers_loader
    print("[FID Pipeline] Patched pygrain.DataLoader to worker_count=0", flush=True)

    # Monkey-patch SimpleTrainer.load to handle ocdbt-format checkpoints where
    # latest_step() raises FileNotFoundError. Force load_directly_from_dir=True.
    from flaxdiff.trainer.simple_trainer import SimpleTrainer as _ST
    import orbax.checkpoint as _orbax
    def _patched_load(self, checkpoint_path, checkpoint_step=None, load_directly_from_dir=False):
        print(f"[FID Pipeline] _patched_load: forcing load_directly_from_dir=True for {checkpoint_path}", flush=True)
        checkpointer = _orbax.PyTreeCheckpointer()
        options = _orbax.CheckpointManagerOptions(max_to_keep=4, create=False)
        manager = _orbax.CheckpointManager(checkpoint_path, checkpointer, options)
        # Use restore() directly without calling latest_step() (which can crash on ocdbt).
        try:
            ckpt = manager.restore(checkpoint_path)
        except Exception:
            # Some orbax versions need a step number; try latest_step then restore(step)
            try:
                step = manager.latest_step()
                ckpt = manager.restore(step)
            except Exception as e:
                print(f"[FID Pipeline] _patched_load: BOTH paths failed: {e}", flush=True)
                raise
        state = ckpt['state']
        best_state = ckpt['best_state']
        rngstate = ckpt['rngs']
        self.best_loss = ckpt['best_loss']
        if self.best_loss == 0:
            self.best_loss = 1e9
        print(f"[FID Pipeline] _patched_load: loaded successfully, best_loss={self.best_loss}", flush=True)
        return None, state, best_state, rngstate
    _ST.load = _patched_load
    print("[FID Pipeline] Patched SimpleTrainer.load to bypass latest_step()", flush=True)

    from flaxdiff.trainer.general_diffusion_trainer import GeneralDiffusionTrainer
    GeneralDiffusionTrainer.fit = _patched_fit_factory(out_dir, preargs.n_samples, preargs.diffusion_steps)

    import training
    training.main(training.parser.parse_args())


if __name__ == "__main__":
    if _HAS_WANDB_ID:
        _run_as_parent()
    else:
        _run_as_child()
