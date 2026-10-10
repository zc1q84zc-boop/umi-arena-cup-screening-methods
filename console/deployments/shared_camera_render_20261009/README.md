# Shared camera rendering deployed 2026-10-09

The 4090 console's `run_visual_aligned` entry point enables `shared_camera_render_v1` by default. The original runner remains available for an A/B comparison; `UMI_SHARED_CAMERA_RENDER=0` disables this decoration.

All three render products update during one shared batch of five RTX warmup renders plus one acquisition render. Acquisition reads every rendered sensor frame (`frequency=-1`), preventing Isaac's default 120 Hz sensor metadata throttle from retaining a previous timestamp. Every camera must supply a valid RGB frame whose native rendering time matches the requested physics state. Rendering is checked not to advance physics.

The group is cached by world identity, physics time, camera identity, image resolution, flip setting, and actual camera pose. A reset invalidates it. Recording and the next model observation reuse the same owned RGB buffers. Policy input copies prevent callers from modifying the recording cache. Camera or physics changes force a new group. The current pi0.5 adapter sends the two wrist views to its server; the head view is synchronized and recorded as before.

`shared_camera_render.jsonl` audits group IDs, raw RGB hashes, rendering times, and reuse. Model input metadata includes group ID and reuse status. `report.json` and `manifest.json` include counters. Audits flush after each group, including the final post-action frames.

## Verification

- 19 CPU tests passed (10 cache/synchronization checks and 9 existing image-input checks).
- RTX same-state A/B: 216 render calls became 54, a 75% reduction. Capture/observation time was 2.404 s versus 0.827 s, a 2.91× speedup for this part of the pipeline. Physics and inference are excluded from that measurement.
- Small real joint movements tested cache invalidation and current wrist poses. Maximum mean RGB difference between the old and new capture paths was 1.643/255, with RTX temporal accumulation active.
- Native console integration run `68f64c84ae71`: 6 model requests, 18 executed servo actions, 19 frames in each video; all 6 requests reused their recorded group with matching native camera/physics times. All three videos decoded and the console action audit passed. First model inference took 9736 ms for initialization; subsequent inference averaged 83.0 ms.
- Observed online throughput after warmup: 1.369 s/request before versus 0.755 s/request after (99 and 70 intervals), approximately 1.81× faster. These runs follow different evolving states; the controlled same-state A/B above isolates rendering work.
- Full evaluation `d6ff3306df40` was started through the native API with run-until-success enabled. Its outcome is separate from integration validation.

Physics remains 240 Hz with 128 FEM solver iterations. The 3 GPa, 1 mm shell, model/control parameters, camera geometry/appearance, and 45 mm plate-center threshold remain unchanged. Hash checks before deployment protected production files and unchanged physics inputs. Previous run `7c4df6502f53` was gracefully stopped and its recordings preserved.

See `gpu_probe_result.json`, `integration_verification.json`, and `deployment_audit.json` for measurements, native checks, and rollback paths.
