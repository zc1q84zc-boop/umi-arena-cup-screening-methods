#!/usr/bin/env python3
"""Isolated OpenPI pi0.5 recipe; do not modify the first run or shared OpenPI."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--openpi-root", type=Path, required=True)
    p.add_argument("--evaluation-root", type=Path, required=True)
    p.add_argument("--mode", choices=["data-smoke", "gradient-smoke", "train", "verify-checkpoint"], required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    for path in [args.openpi_root/"src", args.evaluation_root]: sys.path.insert(0, str(path))
    from train_cup_clean_pi05 import install_lerobot_compat
    install_lerobot_compat()
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import numpy as np
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config, data_loader, optimizer, sharding, weight_loaders
    from umi_arena.yubi.config import LeRobotYubiDataConfig
    from dataset import CupDataset

    assert jax.device_count() == 2, jax.devices()
    assert all("A100" in d.device_kind for d in jax.devices()), jax.devices()
    smoke = args.mode in ("gradient-smoke", "verify-checkpoint")
    train_config = config.TrainConfig(
        name="pi05_cup_intersection", project_name="umi-arena-track1", exp_name="pi05_cup_intersection_v2",
        model=Pi0Config(pi05=True, action_dim=32, action_horizon=32),
        data=LeRobotYubiDataConfig(repo_id="yubi-cup-intersection-independent-v2",
                                  assets=config.AssetsConfig(asset_id="intersection_v2")),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(Path.home()/".cache/openpi/openpi-assets/checkpoints/pi05_base/params")),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=5e-5, decay_steps=30000, decay_lr=5e-6),
        assets_base_dir=str(args.work_dir/"assets"),
        checkpoint_base_dir=str(args.work_dir/("smoke_checkpoints" if smoke else "checkpoints")),
        batch_size=32, num_workers=4, num_train_steps=2 if smoke else 30000,
        log_interval=1 if smoke else 100, save_interval=1, keep_period=10000,
        overwrite=False, resume=args.resume, wandb_enabled=False, fsdp_devices=2, seed=42,
        policy_metadata={"dataset_manifest":str(args.work_dir/"prepared/manifest.json"),
                         "action_hz":10, "no_pairing":True, "validation":"UUID-disjoint, train-only normalization"})
    prepared = args.work_dir/"prepared"
    data_loader.create_torch_dataset = lambda dc, horizon, model: CupDataset(prepared)
    manifest = json.loads((prepared/"manifest.json").read_text())
    assert manifest["summary"]["episodes"] == 2781 and manifest["no_pairing"] is True
    print("RUN_CONFIG", json.dumps({"mode":args.mode, "devices":[str(x) for x in jax.devices()],
          "steps":train_config.num_train_steps,"batch":32,"checkpoint_steps":[10000,20000,30000],
          "action_hz":10,"summary":manifest["summary"]}), flush=True)

    if args.mode == "data-smoke":
        started = time.monotonic()
        loader = data_loader.create_data_loader(train_config, shuffle=True, num_batches=3)
        for observation, actions in loader:
            assert actions.shape == (32,32,32) and observation.state.shape == (32,32)
            assert np.isfinite(np.asarray(actions)).all() and np.isfinite(np.asarray(observation.state)).all()
            assert not np.asarray(observation.image_masks["base_0_rgb"]).any()
            assert all(np.asarray(observation.image_masks[k]).all() for k in ["left_wrist_0_rgb","right_wrist_0_rgb"])
        print("DATA_SMOKE_OK", {"samples":96,"seconds":time.monotonic()-started}, flush=True)
        return

    if args.mode == "verify-checkpoint":
        from openpi.policies import policy_config
        checkpoint = train_config.checkpoint_dir/"2"
        assert (checkpoint/"_CHECKPOINT_METADATA").is_file()
        policy = policy_config.create_trained_policy(train_config, checkpoint)
        for split in ("train", "val"):
            ds = CupDataset(prepared, split)
            prompts = sorted({r["prompt"] for r in ds.manifest["episodes"].values() if r["split"] == split})
            assert len(prompts) == 2
            for prompt in prompts:
                index = next(i for i,(idx,t) in enumerate(ds.samples) if t == 0 and ds.manifest["episodes"][idx]["prompt"] == prompt)
                sample = ds[index]
                for k in ["observation.pose.left_hand_root.relative", "observation.pose.right_hand_root.relative", "action.joint_states"]: sample.pop(k)
                output = np.asarray(policy.infer(sample)["actions"])
                assert output.shape == (32,16) and np.isfinite(output).all()
                print("CHECKPOINT_INFER_OK", split, prompt, output.shape, flush=True)
        return

    spec = importlib.util.spec_from_file_location("openpi_train_upstream", args.openpi_root/"scripts/train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    original_save = train._checkpoints.save_state
    val_batches = []
    val_fn = None
    mesh = sharding.make_mesh(2)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    fixed_count = 64 if smoke else 256
    val_ids = CupDataset(prepared, "val", fixed_count).samples
    (args.work_dir/("smoke_validation_samples.json" if smoke else "validation_samples.json")).write_text(json.dumps(val_ids)+"\n")

    def evaluate(state, completed):
        nonlocal val_fn
        if not val_batches:
            original_create = data_loader.create_torch_dataset
            data_loader.create_torch_dataset = lambda dc,horizon,model: CupDataset(prepared,"val",fixed_count)
            try:
                dc = train_config.data.create(train_config.assets_dirs, train_config.model)
                loader = data_loader.create_torch_data_loader(dc, train_config.model, 32, 32,
                          sharding=data_sharding, num_batches=fixed_count//32, num_workers=0, seed=42)
                val_batches.extend(loader)
            finally:
                data_loader.create_torch_dataset = original_create
        if val_fn is None:
            def compute(params, batch, rng):
                model = nnx.merge(state.model_def, params)
                model.eval()
                return jnp.mean(model.compute_loss(rng, batch[0], batch[1], train=False))
            val_fn = jax.jit(compute)
        params = state.ema_params if state.ema_params is not None else state.params
        with sharding.set_mesh(mesh):
            values = [float(val_fn(params,b,jax.random.fold_in(jax.random.key(4242),i))) for i,b in enumerate(val_batches)]
        assert np.isfinite(values).all(), values
        row = {"completed_step":completed,"ema_flow_matching_loss":float(np.mean(values)),
               "samples":fixed_count,"fixed_seed":4242,"is_task_success_metric":False,"time":time.time()}
        with (args.work_dir/("smoke_validation.jsonl" if smoke else "validation.jsonl")).open("a") as f: f.write(json.dumps(row)+"\n")
        print("HELDOUT_VALIDATION",json.dumps(row),flush=True)

    def save_and_validate(manager, state, loader, zero_step):
        completed = zero_step+1
        validation_steps = (2,) if smoke else (1000,5000,10000,15000,20000,25000,30000)
        if completed in validation_steps:
            assert int(state.step) == completed
            evaluate(state, completed)
        if completed in ((2,) if smoke else (10000,20000,30000)):
            print("CHECKPOINT_COMPLETED_STEP", completed, flush=True)
            return original_save(manager,state,loader,completed)
    train._checkpoints.save_state = save_and_validate
    train.main(train_config)
    print("TRAIN_FINISHED",train_config.num_train_steps,flush=True)


if __name__ == "__main__": main()
