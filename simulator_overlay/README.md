# Reproducible simulator source overlay

This overlay carries the local Dual Franka/YUBI simulator changes used by the Track 1 online console. It targets public upstream commit `29652dc` on branch `simulation-replay-259632-259633` of [dual-franka-yubi-isaac-sim](https://github.com/StevenLiudw/dual-franka-yubi-isaac-sim). The upstream project is MIT licensed; its license and third-party notices are retained here.

The 2026-10-09 overlay includes current-pose wrist cameras, visual alignment v3, mirrored jaw control, CAD aperture mapping, bounded continuous targets, a two-stage plate-and-return evaluator, detailed contact layers, deformable-cup probes, and CPU tests. It includes a procedural cloth texture. The rebuilt `franka_yubi_panda.usdc`, NVIDIA's stock Panda files, recordings, weights, and private replay data are excluded. The composed USD must be rebuilt on a licensed Isaac Sim 5.1 machine. The pinned upstream supplies the separately licensed YUBI CAD and meshes; preserve its complete source and notices.

```bash
git clone https://github.com/zc1q84zc-boop/umi-arena-cup-screening-methods.git
git clone https://github.com/StevenLiudw/dual-franka-yubi-isaac-sim.git
cd dual-franka-yubi-isaac-sim
git checkout 29652dc
python3 ../umi-arena-cup-screening-methods/simulator_overlay/apply_overlay.py .
```

The script refuses a different commit or a dirty checkout, and copies only the listed source files. It does not contact a shared GPU host. To build and validate, follow the [upstream asset-provisioning instructions](https://github.com/StevenLiudw/dual-franka-yubi-isaac-sim#provision-assets-and-create-the-scene) with a locally licensed Panda asset, then run:

```bash
bash yubi_isaac_sim_env/rebuild_assets.sh
bash yubi_isaac_sim_env/pxr_python.sh -m yubi_isaac_sim_env.validate_assets
python3 -m unittest -v tests.test_online_smoothing tests.test_two_stage_task
```

The CPU tests verify the controller and task metric; the USD/RTX checks require an Isaac Sim GPU runtime. This overlay represents the local source snapshot, not a claim that the inaccessible shared squirrel deployment has already been updated to this exact snapshot. Compare its current checkout and private `adapters/` before deploying there.
