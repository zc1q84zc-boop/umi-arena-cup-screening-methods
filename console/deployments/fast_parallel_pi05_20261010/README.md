# Faster full-task π0.5 evaluation on dual RTX 4090

This deployment snapshot requires the licensed Isaac Sim runtime, provisioned
tuned simulator, private model assets and services, and the existing isolated
`jaw_margin_20261010` Python environment. It is not a CPU-only clone-and-run demo.

`full_task_sequential.py` decorates the runner in its own process. Manipulation
remains model driven; release, 12 cm retreat, verified right-arm homing and
holding the inactive arm are recorded controller interventions. The right-home
gate must pass before the first left-return request. There is no fixed step or
wall-clock cutoff.

`run_evaluation_intersection.sh` is the deployment template for new 10k/20k
checkpoints: arm/jaw gains 6/4, velocity 1.2 rad/s, acceleration 2.4 rad/s²,
240 Hz physics, 128 solver iterations, 0.005 rad left closure margin and three
30 fps videos. Configure its target workspace, Python environment and named
services. Parallel mode verifies exact known service identities and GPU locks.

`intersection_step_server.py` shares the final intersection input/output
contract and accepts 10k, 20k or 30k checkpoint identities. Deploy beside the
existing `official_cup_prompts.py` and training/evaluation contract dependencies.
The new wrappers use the shared adapter, base π0.5 adapter, calibration and
prompt modules in `simulator_profiles/tuned_v1/adapters/`. Private weights and
normalization assets are separate prerequisites. Native console templates are
in the adjacent `migration_4090_20261009` directory.

`preview_recording.py` exports a read-only prefix of an open recording without
stopping the task. It uses H.264 parameter sets from a verified recording made
by the same writer, decodes existing frames and omits the incomplete tail. Use
the minimum decoded prefix length with `--max-frames` for all three views.
Label these outputs **in progress**, not final task recordings. No scene frames
are invented and originals stay untouched. The first 90 recovered frames had
identical decoded pixel hashes and frame order. FFmpeg and ffprobe are required.

See [deployment parameters and current results](../../../FAST_PARALLEL_PI05_20261010.md).
