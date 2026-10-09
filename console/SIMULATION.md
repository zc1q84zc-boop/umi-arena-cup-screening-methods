# Isaac Sim console integration

The simulation page is embedded in the **双臂仿真 · 在线模型** tab of the
main workbench at <http://127.0.0.1:8768/#simulation>; its standalone address
is <http://127.0.0.1:8772/>. The public source is
[dual-franka-yubi-isaac-sim](https://github.com/StevenLiudw/dual-franka-yubi-isaac-sim),
cloned locally at /home/hkclr/Documents/ChatGPT/ATEC/dual-franka-yubi-isaac-sim.

## Current deployment (2026-09-28)

The live console now targets `squirrel_5090`, physical GPU 0, at
`/home/lrl/dual-franka-yubi-isaac-sim-deploy`. Its repository is at public
commit `221a001` plus local console changes for continuous runs, the two-stage
task check, and the corrected +1 jaw mimic gearing. The rebuilt USD passed
offline validation. The A100 model endpoints use private reverse tunnels to
squirrel; LingBot 10k and π0.5 10k health checks passed through those tunnels.
The previous `lightyear-cuda` runs are retained as historical records. Squirrel
has one GPU. Its locally migrated π0.5 10k, 20k, and 30k exports each passed
model-load and three-step causal Isaac smoke tests while sharing GPU 0 with
Isaac Sim. The console now offers each as a local inference backend, checks
process ownership before launch, and stops its dedicated model service after
each run. LingBot and OpenWAM local inference remain unavailable pending
separate runtime validation.
Do not copy NVIDIA's stock Panda files into this console or a public repository.

## Execution path

sim_console.py is a separate loopback-only HTTP service. A request selects
one policy from sim_policy_registry.json, a deterministic random layout,
seed, a 10 Hz run mode, and one recorded camera. Online models default to
running until the selected task objective succeeds. The simulator's scene
step truncation and the console's 45-minute command timeout are disabled for
that mode; the page can request a graceful stop, which closes the video and
records `stop_reason: user_stop`. Fixed-step diagnosis remains available.
The service checks
that physical RTX 5090 GPU 0 has no unmanaged compute process (allowing only
the selected, identity-checked local model service), then launches the
repository's run_headless.sh over SSH under a remote lock. Each run gets a
new, non-overwritten remote directory. Only its MP4, JPEG preview, joint CSV,
manifest, report, and adapter audit are copied to private sim_runs/<run-id>/
for display. The
browser cannot supply a shell command or arbitrary script path. POST
requests require same-origin JSON; the service binds to 127.0.0.1 only.

The runner now also atomically writes a JPEG from the selected recording
camera after every 10 Hz control step. The console copies that small preview
while the simulator is running and shows it on the first screen. The MP4 is
not playable until ffmpeg closes it at the end of the episode; after that,
the JPEG is the video's visible poster and the play button opens the 30 fps
recording. The video file is H.264/yuv420p, not an interactive Isaac viewport.

## Cup-task success metric

The simulator's original `is_success` means only that the cup rested on the
plate for five control steps. It does **not** cover the Track 1 instruction to
put the cup back. New runs can select `plate_return`: after the original plate
criterion, the cup must return within 35 mm XY and 8 mm height of its reset
pose, remain upright and nearly stationary, and both grippers must be open
for five consecutive control steps. The report retains the original
`is_success` and records `task_stage`, `plate_placed`, `full_task_success`,
the cup pose and gripper opening at every policy step. Old runs retain their
one-stage meaning; do not relabel them as full-task successes. This is a
simulator metric, not evidence that the real robot can perform the task.

The current online model runs are **interface checks**, not successful
grasp/placement trials. In the available 30-step π0.5 30k run, nearest
tool-to-cup distance changed from 0.406 m to 0.492 m; in OpenWAM 10k's
60-step run it changed from 0.406 m to 0.607 m, while the cup moved less than
1 mm. LingBot 10k's 10-step run remained around 0.46 m. Those observations
alone cannot distinguish visual distribution shift, task prompting, action
coordinates, or controller behavior. No checkpoint is marked as having
completed the two-stage task. A full-task π0.5 30k website trial on 2026-09-28
(`8c8bec65d744`, fixed 600-step cap) produced 585 live control observations
and a 1,753-frame video, but did not lift the cup. It stopped when the
simulator rejected a right-arm command outside the USD joint limits. The
failure video and exact safety error are available on the run page; simply
increasing the earlier 150-step limit does not solve this coordinate/action
problem. No joint limit was bypassed.

The systemd user service is track1-simulation-console.service on port 8772;
the existing practice replay service on 8768 now embeds this page.

Three console-launched checks passed on 2026-09-27:

- Hold, random:0, head camera, 2 steps: completed; 7 video frames and
  7 joint samples.
- Hold trajectory, random:1, overview camera, 2 steps: completed with
  world_tool_waypoint_chunks; 7 video frames and 7 joint samples.
- Push demo, random:0, overview camera, 3 steps: completed with joint_targets;
  10 video frames and 10 joint samples.

These are interface and recording checks, not manipulation successes. The
page also exposes the repository's oscillate and push_demo examples.
π0.5 10k/20k/30k, LingBot VLA2 5k/10k, and OpenWAM Alpha 5k/10k now have separate
experimental online adapters. LingBot consumed fresh left/right wrist renders
for a completed 10-step run; OpenWAM consumed fresh head renders for a completed
10-step run. π0.5 10k and 20k each completed 10-step live-wrist runs with
31 aligned frames/samples and distinct images at every step. None achieved
a grasp in those short runs. Their VR hand-root
to simulator tool mappings are provisional first-frame alignments, not measured
physical extrinsics. Other imported checkpoint entries remain visible but
disabled until independently verified online. Earlier offline replays used
recorded training observations and cannot prove closed-loop performance.

## π0.5 online path and measured result (2026-09-28)

- A100: `/mnt/data/benyun/workspace/pi05_cup_clean_20260923/scripts/pi05_online_server.py`
  loads the verified clean-cup 30k checkpoint via the official evaluation
  Policy adapter on A100 GPU 6. It listens only on 127.0.0.1:18781.
  The user service `pi05-online-a100.service` is **not enabled at boot**.
  It is currently stopped after validation to release its idle GPU memory;
  the console's explicit start button starts the identity-checked service
  and private bridges together.
- Local SSH services `pi05-a100-forward.service` and
  `pi05-lightyear-reverse.service` join the two loopback-only endpoints.
  Model weights and training data remain on the private A100. Only current
  simulated RGB frames/state and the returned actions cross the tunnel.
- The inference-location selector distinguishes the existing A100 endpoint,
  a dedicated RTX 5090 endpoint on GPU 3, and the current Jetson Orin.
  A backend is selectable for execution only if its exact private model
  service and bridge are healthy. Orin is explicitly unavailable until a
  model copy and compatible, tested local runtime exist; it is never silently
  mapped to A100. The 5090 weights-only inference export is separate from
  the 42 GiB training checkpoint. All 20 inference files matched the A100
  source by path, size and SHA-256 before its start button was enabled.
  Inference units are never boot-enabled.
  The **退出推理服务** button stops only the selected, identity-checked managed
  unit and its own private SSH bridges, and refuses to act while a simulation
  run is active. It does not terminate unrelated GPU processes or the shared
  Isaac console service. After their experiments, π0.5 10k/20k/30k,
  LingBot 5k/10k and OpenWAM 5k/10k inference units and bridges were stopped;
  A100 GPUs 5–7 and 5090 GPUs 3/6 had no remaining task compute processes.
  LingBot's 10k training was left intact until its final checkpoint and HF
  export finished normally.
- The 5090 adapter `adapters/pi05_isaac_online_adapter.py` reads current
  640×480 left/right wrist renders and current tool/jaw states at each
  10 Hz step. It computes `left^-1 * right` in the common simulator world
  frame, maps gripper state to the clean-cup source range, integrates the
  first three 30 Hz body-relative actions, and sends one 10 Hz world-frame
  YUBI tool waypoint and bounded gripper command to dual-arm IK. No future
  or prerecorded observations enter the model request.
- Fixed-asset smoke run `b7330aa7f42d`: 30/30 online observations, 91
  aligned video/joint samples, 30 distinct rendered frames per wrist,
  mean inference 116 ms, max arm-joint motion 0.961 rad. Measured driven
  jaw angles left 0.596→0.365 rad and right 0.596→0.507 rad, with mimic
  opposition error below 0.003 rad. The episode did **not** complete the
  cup task. Input images, MP4, JSONL audit and CSV are in
  `sim_runs/b7330aa7f42d/` and visible in the workbench.
- RTX 5090 inference run `4574c69511ea`: the same π0.5 30k policy loaded on
  GPU 3 and completed 10/10 live wrist-image steps in the GPU 6 Isaac scene.
  All ten left and ten right input images had distinct hashes, all reported
  numeric action/pose/gripper values were finite, and the H.264 video has 31
  frames aligned with 31 joint samples. Mean model latency was 68.6 ms;
  the cup task did not succeed. The dedicated 5090 inference service and
  simulator exited after validation; GPU 3 and GPU 6 had no remaining compute
  processes. Audit, video and input samples are in `sim_runs/4574c69511ea/`.
- Non-model 20-step range probe `7197f52fcd88` verified both driven jaws
  travel from ~0 to 0.596 rad and both mimic jaws from ~0 to −0.595 rad.
  The previous USD's PhysX mimic gearing sign was reversed, preventing
  actuation. The old USD is preserved as
  `assets/franka_yubi_panda.pre_mimic_fix_20260928.usdc` on the 5090 host.
  The new USD passed offline validation. NVIDIA defines the constraint as
  `mimic + gearing * reference + offset = 0`, so opposed jaws need +1.

This is an experimental integration: simulated YUBI tool = UMI hand root
is a provisional identity extrinsic; source gripper endpoints −0.45 and
0.78 rad come from training q01/q99, not a physical calibration. Camera
mounts, table, object geometry and contact parameters are nominal. Thus
the run verifies online inference, coordinate arithmetic, gripper motion
and recording—not task success or real-robot transfer.

LingBot 5k/10k run separately through A100 GPU 7 and private ports 18784/18802;
selecting an online model and starting a simulation now switches between the
console-managed checkpoints on the same GPU, starts its private bridges, and
waits for the selected health endpoint before launching Isaac. The switch
checks each service PID against its expected script and checkpoint and refuses
to stop or share the GPU with an unrecognized compute process. Opening the
page or browsing model choices does not allocate a GPU.
The LingBot 10k ten-step run `ad622a08d8c5` consumed ten distinct current
left and right wrist renders, produced finite bounded actions and 31 aligned
video/joint samples. The task did not succeed. Afterward its dedicated inference
unit, private SSH bridges and simulator process exited. OpenWAM Alpha
10k through A100 GPU 5 and private port 18787. Each has its own loopback-only
bridge to the 5090 simulator and a model-specific adapter. OpenWAM's first
single-left-arm alignment yielded an unsafe 0.56 m right-arm jump, correctly
rejected by the 4 cm/30 Hz safety gate. The subsequent **per-arm** initial-pose
alignment yielded a completed 10-step run with ten distinct current head
renders, 31 synchronized video/joint samples and maximum 10 Hz displacement
about 19 mm. A longer **60-step console run** (`13094ace95e6`) produced 60
distinct live head images and 181 aligned video/joint samples; both measured
jaws closed near zero radians, but the cup moved only about 0.4 mm and the
task failed. Its final tools remained more than 0.6 m from the cup. A separate
three-step run (`6b579c9c3101`) records and displays the actual first head
image while clearly labeling the wrist images as audit-only. This per-arm
mapping is not one measured common VR-to-world transform; its use is
restricted to simulator diagnosis.

## Controlled grasp and contact calibration (2026-09-28)

See [the diagnostic report](GRASP_CALIBRATION_20260928.md) and
[`sim_contact_calibration.json`](sim_contact_calibration.json). A non-model
probe used the observed simulator world pose of the cup and a bounded
align–descend–close–lift sequence on the closer left arm. A 6D IK attempt
with the home wrist orientation hit Franka joint 4's lower limit before
descending; position-only IK reached the cup. At jaw opening ~0.364 rad,
transformed jaw meshes contacted opposite sides of the **rim**, near
73–75 mm above the table, rather than the lower cup wall. The contact
midpoint was ~`(0,-0.0084,-0.0165) m` from the authored `yubi_tool` at that
opening; this conditional geometric offset is not a measured UMI/robot
extrinsic or a force-sensor reading.

The original-scene probe touched but did not lift the cup stably. An
isolated, **non-hardware-validated** drive/friction variant plus slow lift
raised it transiently 47.6 mm before slip and ~80° final tilt. Both
recordings are available in the console as non-model tests, with per-step
traces and explicit `stable_grasp_verified=false`. The original scene and
model adapter were not changed. Until a repeatable stable grasp passes,
trained-policy success in this scene is not a reliable model comparison.
The selectable **杯子接近—夹持探针 · 非模型** repeats the bounded diagnostic
in the original scene and exports its stepwise trace and pass/fail summary;
allow enough 10 Hz steps for the approach, grasp and lift phases.

## Adding a trained model

An adapter must be a trusted Python script installed in the remote repo/
or adapters/ directory and registered by an administrator in
sim_policy_registry.json. For an end-effector model, it exports
predict(observation, step, episode) and returns absolute simulator-world-frame
YUBI tool poses in metres, wxyz quaternions, normalized gripper opening in
[0, 1], and action_dt_s: 0.1. The repository's policy_adapter.py validates
and executes that contract using bounded differential IK. A joint-space
model may instead export act(...) and use kind joint_script.

Before setting ready: true, verify checkpoint provenance and completeness,
online image ordering/crop/normalization, proprio history, task prompt,
body/camera/tool/world transforms, 30 Hz-to-10 Hz action resampling, and
gripper angle-to-open-fraction mapping. Prove inference uses only current
and past simulator observations, never future recorded frames. Smoke-test
one short episode, inspect input and output transforms, and confirm finite
actions, joint bounds, collision behaviour, video/CSV alignment, and the
report. Simulated success still does not establish real-robot transfer:
scene geometry, camera extrinsics, friction, object mass, and YUBI dynamics
are provisional estimates.

Keep raw demonstrations and model weights on their private machines. The
console registry contains only labels and adapter paths, not weights.
