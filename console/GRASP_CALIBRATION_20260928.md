# Dual-YUBI cup grasp calibration, 2026-09-28

This is a simulator-internal diagnostic, **not** a UMI-to-real-robot calibration or a successful task evaluation. All runs used the deterministic `random:0`, seed 42 cup setup on an otherwise free RTX 5090 GPU 6. The original scene, demonstrations, model checkpoints, and training manifest were not changed. Experimental drive/friction settings were isolated in a separate USD layer.

## Coordinate and contact measurements

The simulator uses a common Z-up world frame. Its illustrative table top is `z=0.750 m`; its left/right Franka mount translations are `(-0.530,-0.440,0.750)` and `(-0.530,+0.440,0.750) m`, with yaw `+0.35/-0.35 rad`. In the recorded reset, the cup base was `(-0.254269,-0.208095,0.750000) m`; the left YUBI tool was `(-0.069386,-0.271862,1.105647) m`. The probe selected the closer left arm and used *observed world poses*, rather than pixel-to-metre guesses, to command a bounded approach.

The authored `yubi_tool` is a CAD-estimated pinch frame at `(0,0.018,0.145) m` relative to the YUBI base. Transforming the actual jaw STL vertices by the observed finger poses in the 2 Nm reference probe (`grasp_calibration_20260928_e`, step 30, jaw `0.364 rad`) put the nearest left/right points at approximately `(±0.0408,-0.0084,-0.0165) m` in `yubi_tool` coordinates. Thus the **conditional geometric contact midpoint** was `(0,-0.0084,-0.0165) m` from `yubi_tool`, at this jaw opening. Both nearest points lay at the cup rim, about `73–75 mm` above the table, within `1–2 mm` of the modeled sidewall. This is a mesh-geometry proxy, not a measured contact force or a fixed transform for every jaw angle.

The dataset's VR hand-root-to-simulator-tool transform and real camera extrinsics remain **unmeasured**. The above world/mount/contact values cannot determine them. The online π0.5 adapter still marks its hand-root=tool map as provisional; these tests did not silently change it.

## Controlled validation

The diagnostic policy proceeds only after a bounded approach converges: align above cup, descend while open, close, then lift. It logs each live cup/tool/finger pose and jaw feedback to `grasp_calibration.jsonl`. It never teleports the cup or directly moves the other arm.

| Probe | Change from original scene | Observation |
| --- | --- | --- |
| `a` | Original 6D IK, original wrist orientation | Aborted before descent at ~0.12 m tool error; Franka joint 4 reached its lower limit. |
| `d` | Position-only IK, original physics | Both jaws approached the rim; cup lifted at most ~4.8 mm and remained on the table. |
| `e` | Isolated 2 Nm / stiffer jaw drive | Jaw opening improved, but the cup still slipped during lifting. |
| `h` | Isolated 2 Nm drive, static/dynamic friction 2.0/1.5, 5 mm/step lift | Cup rose transiently ~47.6 mm, then slipped and tipped; final tilt ~80.4°. **No stable grasp.** |
| `j` | Same experimental scene, stricter alignment and deeper target | Still made rim-first contact and shifted the cup ~86 mm; no stable grasp. |

The 2 Nm/friction values are **diagnostic hypotheses**, not certified YUBI motor/pad limits. We retained all results, including failed probes; no run is counted as successful merely because the arm moved, both jaws touched the cup, or the cup briefly rose.

The console contains playable non-model records `aa5acf364692` (original physics) and `5b0151ea6b08` (experimental high-friction slow lift), with their stepwise traces and summaries. Their `status=completed` means the simulator process and recording finished; `task success=false` and `stable_grasp_verified=false` are the actual outcome.

The selectable diagnostic was also launched end-to-end through the console as run `c2efa426e456` on the **original** scene: 100/100 policy steps, 301 aligned video/joint samples, and a downloadable live trace. It reached contact but lifted the cup only transiently 27.4 mm, then tipped it ~80.4°. The console's failure label and summary matched the underlying report. GPU 6 was idle after the run.

## Additional lower-sidewall trials

Seven further isolated trials are recorded under `calibration_runs/grasp_lower_wall_20260928_{a,b,c,d,e,f,g}`. The local probe supports bounded offsets up to 55 mm, incremental descent, and a measured jaw-closure gate before lifting. None passed the stable-grasp criterion. On original physics, the open jaw reached only about 0.495 rad and the probe stopped before descent. Under the separate high-friction/2 Nm USD layer, a direct lower target displaced the cup by about 90 mm. Incremental descent reached the close phase, but a prior ungated trial only lifted the cup transiently about 13 mm before slip. With the measured closure gate, the jaw stayed around 0.60 rad for 45 close steps and the probe safely aborted instead of counting an open-jaw lift. Offsets of 32 mm and 20 mm below the earlier target also jammed near the rim; holding wrist orientation made descent IK fail. The likely bottleneck is rim-first collision rather than insufficient lift speed. These trials are negative evidence, not a recipe for increasing force on the real robot.

## Follow-up with a measured closure gate

Two extra `-10 mm` target-height trials used an older deployed probe:
`grasp_offset_minus10mm_20260928` and `grasp_centered_minus10mm_20260928`.
They briefly lifted the cup 52.9 mm and 35.2 mm respectively, but both ended
with an ~80° tip. That probe advanced to lifting after a fixed 15 close steps,
without checking measured closure. Those peaks are retained for audit, **not**
counted as valid grasp successes.

After deploying the measured-closure probe, the isolated
`grasp_gated_minus10mm_20260928` trial approached the cup but the jaw stopped
around 0.418 rad during 45 close steps, short of the 0.40 rad gate. It reopened
to about 0.577 rad after the safe abort; the earlier interpretation that it
stayed near 0.577 throughout closure was incorrect. It aborted without
lifting; cup height changed under 0.4 mm. At this target height the simulated
finger/drive setup did not close far enough around the cup. The high-friction 2 Nm scene
remains a diagnostic layer, not a real-hardware specification. The new
two-stage metric correctly reported neither plate placement nor return.
The simulator process exited and GPU 6 was idle afterward.

## Validation gate before comparing trained models

The current cup grasp configuration is **not yet validated**. A passing diagnostic would need a repeatable approach without premature cup displacement, simultaneous two-sided contact below or securely on the rim, and an upright cup lifted at least `50 mm` and held for at least 10 policy steps without slip. Then repeat over several seeds. Only after that should we assess model task success. To make the result predictive of the real setup, separately measure the VR hand-root/tool transform, camera extrinsics, jaw pad geometry, grip force/friction, and robot base placement. The original simulator's success predicate checked only cup-on-plate; the new opt-in `plate_return` metric separately checks the complete task without changing the original flag.

## Convex-jaw and measured-contact follow-up

The original collision setting represented each concave YUBI jaw as a single
convex hull. Its hull volume was about 2.06×/2.15× the corresponding visible
STL volume; at one blocked pose a cup-wall point was inside the hull but about
5.3 mm from the visible jaw. The isolated
`variants/grip_convex_decomposition_20260928.usda` layer changes only the four
jaw collision approximations to `convexDecomposition`; it still sublayers the
**unverified** 2 Nm / high-friction diagnostic layer. The separate
`variants/grip_convex_decomposition_default_20260928.usda` keeps original
drive and friction values. Neither changes the default USD scene or asset.

With convex decomposition, a fast-close, rim-height probe in
`grasp_decomp_rim_orientation_hold_20260928` lifted the cup about 132 mm and
kept it elevated, but it settled at about 23° tilt. The tool was held within
about 0.7° throughout the lift; the tilt came from cup/gripper contact, not
the wrist-lift command. The jaw mesh proxy suggested one-sided rim support:
one side lay within roughly 1 mm of the rim while the other was roughly 8 mm
away. This is **not** a stable pinch-grasp or a task success.

The prior `jaw <= 0.40 rad` lift gate was too weak to prove contact. In a
slow-close run (`0.015 rad` per 10 Hz step), at `jaw=0.374 rad` the visible jaw
meshes were still approximately 2.2 and 4.8 mm from the cup sidewall; the
cup rose only 2.8 mm. The probe now offers an opt-in slow close, bounded jaw
preload, and an opt-in filtered left/right cup contact-force gate. With
`YUBI_DIAGNOSTIC_CUP_CONTACTS=1`, Isaac's pairwise contact-force view reports
both fingers separately. The default simulator does not enable this tracking.
The force-gated probe requires both fingers at >=0.05 N for three consecutive
steps before lifting and aborts if either drops below 0.02 N for three steps
during lifting or holding. The 1-step contact-sensor smoke test passed.

At a target 10 mm below the former rim target, measured dual-sided contact
appeared only briefly: the 0.03 rad preload trial had <1 mm peak cup lift;
the stronger, bounded 0.08 rad preload trial lost contact and safely aborted
after 2.5 mm peak lift. A predeclared 3×3 horizontal grid at that height,
covering ±5 mm in X and Y with the same speed, force and scene, produced nine
complete 100-step episodes. Eight lost two-sided contact during lift and
aborted; the ninth was still in lift at step 100 with only 0.3 mm peak lift.
Across the grid, peak lift was at most 3.9 mm. Detailed per-episode traces
and the simulator report are in `calibration_runs/grasp_decomp_force_grid_20260928/`.

With *original* drive and friction plus only the convex-decomposed jaw, the
open joint stayed near 0.496 rad rather than the required 0.56 rad. The
controller consequently held the arm above the cup instead of descending;
there was no two-finger contact or meaningful lift. Thus neither the original
scene nor the high-friction diagnostic layer has passed the known-target
grasp baseline. A model failure in this scene cannot yet be attributed to
the model's visual policy or coordinate map. Model image/extrinsic tuning is
deliberately deferred until a force-verified, repeatable grasp is possible.

## Video-led follow-up: the physical grasp is still not validated

I inspected the actual 30 Hz simulator videos, including enlarged frames of
the approach, closure, and lift, rather than inferring success from joint
angles. The rim-height run visibly hooks the cup at its upper edge. Slowing
closure leaves the cup on the table. Moving the same top-down wrist lower
puts the black gripper body into the open cup before the fingers can pinch its
outside wall. These clips are under `calibration_runs/video_inspection_20260928/`.

The original controller targeted the authored `yubi_tool` origin. An opt-in
probe now compensates the *CAD-derived* pinch offset
`(0,-0.0084,-0.0165) m`, rotated by the **live** tool quaternion each step.
This changed the test from a single rim touch to genuine simultaneous
left/right filtered cup contact, but the cup was still ejected as the jaws
continued closing. The change is diagnostic only; it is not a measured
hardware hand-to-tool calibration or a hidden correction to model output.

I then tested horizontal entry and recorded a lateral camera view. Entry
from the far side lets a blue fingertip enter the cup opening and move the
cup before enclosure. The better entry path reached two-sided contact for
about 40 consecutive 10 Hz steps and raised the cup only **11 mm** before it
slipped laterally and lost contact. Its trace and video are in
`calibration_runs/grasp_side_entry_preload040_20260928/`. Transforming the
actual STL vertices at initial lift found one jaw near **74–75 mm** above
the table (the rim), the other near **55–56 mm** (sidewall). Thus nominal
two-sided force is **not** a same-height, stable pinch. Following the cup's
simulator ground-truth XY during lift made lateral escape worse, so it stays
off by default and is forbidden as a model-inference input.

Moving the contact target lower, rotating the wrist 10–15 degrees to equalize
jaw height, or increasing the jaw servo stiffness in separate USD layers each
caused premature cup displacement or lost contact. Approaching from the arm's
base side at a horizontal wrist orientation did not converge in 6D IK at
the original estimated base pose (about **21 mm** residual). A diagnostic
50–100 mm base shift reduced but did not remove the residual or deliver a
clean entry. These base-shift and higher-stiffness layers remain isolated in
`variants/`; **none** are accepted as the real robot pose or production
physics. The unchanged no-flag scene still passed a one-step smoke run.

Consequently, the next physical fix is to measure or obtain (1) the actual
left/right base pose relative to the cup/table, (2) YUBI open aperture and
tip-pad contact geometry/force curve, and (3) the cup's true external shape
and friction. Then choose a collision-free *outside-wall* approach with both
contact patches at comparable height, validate 6D IK at every waypoint, and
require a >=50 mm upright lift held for >=10 policy steps over several
setups. Until that baseline passes, model image/extrinsic tuning cannot be
credited with repairing this failure and the console must not label this
probe as a successful grasp.
