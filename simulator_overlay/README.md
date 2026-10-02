# Reproducible simulator source overlay

This overlay carries the local Dual Franka/YUBI simulator changes used by the Track 1 online console. It targets the public upstream commit `609e6da` of [dual-franka-yubi-isaac-sim](https://github.com/StevenLiudw/dual-franka-yubi-isaac-sim). The upstream project is MIT licensed; its license and third-party notices are retained here.

The overlay includes source for the 180° wrist-camera optical roll, updated camera validation, continuous 60 Hz joint targets with trajectory preview and damping, bounded IK, a two-stage plate-and-return task evaluator, and CPU-only tests. It deliberately excludes the rebuilt `franka_yubi_panda.usdc`, NVIDIA's stock Panda files, recorded video, model weights, and any private replay data. The composed USD must be rebuilt on a licensed Isaac Sim 5.1 machine.

```bash
git clone https://github.com/zc1q84zc-boop/umi-arena-cup-screening-methods.git
git clone https://github.com/StevenLiudw/dual-franka-yubi-isaac-sim.git
cd dual-franka-yubi-isaac-sim
git checkout 609e6da
python3 ../umi-arena-cup-screening-methods/simulator_overlay/apply_overlay.py .
```

The script refuses a different commit or a dirty checkout, and copies only the listed source files. It does not contact a shared GPU host. To build and validate, follow the [upstream asset-provisioning instructions](https://github.com/StevenLiudw/dual-franka-yubi-isaac-sim#provision-assets-and-create-the-scene) with a locally licensed Panda asset, then run:

```bash
bash yubi_isaac_sim_env/rebuild_assets.sh
bash yubi_isaac_sim_env/pxr_python.sh -m yubi_isaac_sim_env.validate_assets
python3 -m unittest -v tests.test_online_smoothing tests.test_two_stage_task
```

The CPU tests verify the controller and task metric; the USD/RTX checks require an Isaac Sim GPU runtime. This overlay represents the local source snapshot, not a claim that the inaccessible shared squirrel deployment has already been updated to this exact snapshot. Compare its current checkout and private `adapters/` before deploying there.
