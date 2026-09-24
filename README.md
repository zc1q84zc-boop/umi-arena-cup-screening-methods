# UMI Arena cup-task screening: methods and aggregate results

[中文简介](README_zh.md)

This **local presentation package** explains a reversible, episode-level quality
screen for the successful `Place the cup on the plate, then put it back to its
original position` task. It contains the screening code and aggregate counts,
but **no raw recordings, Parquet rows, video, model checkpoints, per-episode
identifiers, manifests, or human-label records**.

The underlying [AIRoA dataset](https://huggingface.co/datasets/airoa-org/yubi-corl2026-umi-arena)
has limited-access terms. Do not add generated per-episode reports, clean or
quarantine manifests, media, or trained models to this presentation package.
This package has not been uploaded or assigned an open-source license.

## What was done

1. Select only successful cup-task demonstrations: 3,975 episodes, 827,323
   frames at 30 Hz. No failed demonstrations or other tasks enter the screen.
2. Run `src/hard_constraint_prefilter.py` over motion/state/timestamp columns.
   Its thresholds produce **review candidates**, not calibrated robot
   infeasibility verdicts. See [METHODS.md](METHODS.md) for every rule.
3. Preserve 14 user-marked action-jump episodes as a separate human evidence
   tier. `src/finalize_cup_review.py` creates the internal review ledger.
4. Apply the requested conservative, reversible policy: quarantine the union
   of human-marked episodes and all machine-flagged episodes. The source
   recordings stay unchanged. `src/build_clean_cup_view.py` creates internal
   index-backed selection views; the allowlist manifest is required when
   training, because linked packed source files still contain other episodes.

## Aggregate result

| Category | Episodes | Frames |
| --- | ---: | ---: |
| Successful cup-task input | 3,975 | 827,323 |
| Retained in conservative training view | 3,423 | 746,914 |
| Quarantined for review | 552 | 80,409 |

The machine screen flagged 551 episodes; 14 were user-marked action jumps.
Their overlap was 13 episodes, giving 552 in the union. There were 1,580
candidate frame events. Detailed aggregate counts are in
[`results/aggregate_summary.json`](results/aggregate_summary.json).

**Interpretation:** quarantine is a training-quality decision, not proof that
an episode cannot run on all target robots. The screen did not complete
calibrated three-robot continuous IK, joint-limit, collision, gripper-retargeting,
and joint-dynamics certification. A no-trigger result is likewise not a
feasibility certificate. One FR3 TCP-speed conflict is conditional on the
recorded 30 Hz timing and an assumed hand-root-to-TCP offset of at most 0.5 m.

## Reproduction with separately authorized inputs

The code requires an authorized local copy of the dataset, a successful-cup
selection manifest, and the team's human-label JSONL. Those inputs and the
per-episode outputs are deliberately absent here. Dependencies are listed in
[`requirements.txt`](requirements.txt). The original code uses a fixed
3,975-episode/827,323-frame scope check to prevent accidental task mixing.
The three scripts take `--help` for their exact arguments.

This package is for reviewing the method and aggregate outcome. Anyone
re-running it must obtain dataset access from AIRoA and comply with the
applicable terms and confidentiality pledge.
