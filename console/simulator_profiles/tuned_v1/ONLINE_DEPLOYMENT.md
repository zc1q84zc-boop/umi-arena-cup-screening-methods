# Trained-model online deployment

`tuned_online_v1` reuses the successful replay's versioned simulation assets,
but is a separate trained-policy path. It does **not** replay the demonstration,
retrain or modify checkpoints, or deploy physical-robot commands.

## Shared changes

- Two explicit, opposed YUBI jaw drives and the asset's CAD-supported stroke.
- One fixed source-angle → CAD distal gap → motor-angle curve. Current gripper
  state uses its inverse before being sent to each trained policy. Closed-gap
  plateaus and saturated stroke cannot recover a unique encoder angle.
- Tuned detailed collision/contact layers, including the cup's compliant-contact
  trial. These remain simulation estimates, not measured hardware parameters.
- The anatomical left camera follows `LeftMount`, and right follows `RightMount`.
  Each sensor is positioned from the **current GPU gripper-base pose** before
  rendering. Near clipping is 10 mm; the same live sensors supply policy images
  and the synchronized result videos. Intrinsics/extrinsics remain approximate.
- Existing online semantics: 10 Hz model requests, all three returned actions
  executed sequentially at 30 Hz, physics at 60 Hz. The continuous command
  governor retains 0.8 rad/s and 1.5 rad/s² limits and USD joint bounds. The
  second jaw is mirrored after governing the single jaw command.

The fixed source-hand/table transform and approximate frame-0 reset fit are
explicit simulation registration priors. The replay's 70 mm left displacement,
10 mm episode height shift, jaw bias and frame-dependent closure correction
are excluded. No recorded future frames, cup-position feedback or forced grasp
timing enters the baseline model payload or returned-action conversion.

## Runtime and validation

The private squirrel deployment is
`/home/lrl/dual-franka-yubi-isaac-sim-console-tuned-v1`. Its launcher is
`run_tuned_online.sh`, and its isolated adapters live in `adapters/`. The old
customized simulator remains available unchanged for historical diagnostics.
Model weights stay under the existing private model workspace.

`deploy_servers/validate_tuned_models.py` sequentially loads identity-checked
dedicated model units, launches a bounded live probe only on an otherwise free
GPU0, copies small result artifacts and runs `verify_tuned_online.py`. It then
stops only the task's simulator/model units and checks GPU0 is free. Reports
live under `sim_validation/tuned_online_v1_*_verified`.

Verification includes finite causal actions, matching camera poses in model
input and video audits, three aligned 30 fps streams, mirrored jaw commands,
and command velocity/acceleration. A short passed integration probe is **not**
a full-task success result. Cup-on-plate then released return within 35 mm is
scored separately from observed simulator state; the evaluator never controls
or teleports the cup.

The console selects the upgraded path only for an entry explicitly marked
`simulator_profile: tuned_online_v1`. Assisted diagnostic policies retain their
old isolated paths and are never counted as baseline model successes.
