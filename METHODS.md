# Screening rules and decision standard

## Scope and data integrity

The input allowlist contains only successful episodes of one exact cup task.
For each selected episode, rows are ordered by `frame_index`. A packed-shard
metadata boundary error can place rows in an adjacent Parquet shard; the
scanner checks neighboring shards before declaring missing rows. It verifies
the expected episode length, contiguous local frame indices, strictly
increasing timestamps, and approximately 30 Hz spacing (deviation >2 ms is
flagged). Non-finite motion/state data is flagged.

## Pose and action consistency

The screen checks quaternion norm error (>0.01), zero quaternions, and whether
the observed common-frame left-to-right transform agrees with the supplied
relative transform (>1 mm translation or >1° rotation). It computes each
hand's frame-to-frame translation and rotation. A relative action is compared
with the change between adjacent observed poses; >1 cm translation or >5°
rotation disagreement is flagged. Recorded and action gripper-angle minima
and maxima are reported, but are **not** mapped to a specific target gripper
or certified against its limits.

## Motion review thresholds

These are deliberately loose recording-space triage thresholds, **not** robot
joint or controller limits:

| Candidate trigger | Threshold |
| --- | ---: |
| Hand-root translation in one tick | >0.10 m |
| Hand-root linear speed | >2.0 m/s |
| Vector linear acceleration | >20.0 m/s² |
| Hand-root angular speed, FR3 reference comparison | >2.55 rad/s |
| Action/observation translation disagreement | >0.01 m |
| Action/observation rotation disagreement | >5° |

The code also flags a conditional FR3 TCP-speed conflict when a geometric
lower bound exceeds 3.05 m/s. It uses the observed root displacement, rotation
and time interval and assumes a fixed rigid hand-root-to-TCP offset no longer
than 0.5 m. The 3.0 m/s FR3 reference limit is from
[Franka's specifications](https://frankarobotics.github.io/docs/robot_specifications.html#limits-for-franka-research-3-fr3).
This is not a measured arena calibration. The 2.55 and 3.05 thresholds include
small margins around the nominal 2.5 rad/s and 3.0 m/s references.

## Evidence tiers and reversible decision

- `human_confirmed`: user-marked action jump. Some were individually reviewed;
  others were labeled as a group. The method retains this distinction in the
  restricted internal labels, not in this package.
- `machine_candidate_only`: at least one automatic rule fired, with no
  corresponding human mark. This is a **review candidate**, not a verified
  physical impossibility.
- `no_rule_trigger`: none of the implemented rules fired. This is not a
  guarantee of executable motion or successful manipulation.

The clean training view removes the union of the first two tiers at the
**episode** level. The 552 removed episodes remain recoverable in a separate
restricted view; source data is neither edited nor deleted. Full calibrated
continuous IK, joint positions/velocities/accelerations, self/dual-arm/table
collision, gripper mapping, and scene collision were not established for all
three target robots, so the package makes no hard-feasibility claim.

The underlying dataset and derived-data sharing restrictions are described in
the [AIRoA dataset terms](https://huggingface.co/datasets/airoa-org/yubi-corl2026-umi-arena).
