# UMI Arena cup-task screening: methods and aggregate results

## Latest update · 2026-10-10

[Faster π0.5 checkpoint comparison](FAST_PARALLEL_PI05_20261010.md) adds the same-training-run 10k/20k checkpoints, dual-4090 evaluation and `natural_fast_v1`. Current three-camera clips are marked in progress; full-task success has not been confirmed.

[Recent simulator and control updates](RECENT_UPDATES_20261010.md) include shared three-camera rendering, the 240 Hz / 128-iteration cup profile, smoothing comparisons, the 0.005 rad left-jaw margin, explicit right-home-before-left sequencing, and four new task environments. [Training-data screening](TRAINING_DATA_SCREENING_20261010.md) adds the completed smartphone / chain-SPS quality scan and the pending original-reference IK stage. [Watch or download the latest three-camera trial](https://umi-recording-20261010-home005.steven-robotics-ai.chatgpt.site).

[中文简介](README_zh.md)

## Latest simulator and model update — 2026-10-09

See [the latest source and deployment handoff](RECENT_UPDATES_20261009.md) for
the tuned simulation, wrist-camera visual alignment, explicit mirrored jaws,
contact/deformable-cup probes, official stage prompts, dual-4090 console, and
new intersection training pipelines. π0.5 completed 30,000 steps; OpenWAM
completed 5,069 updates and verified consumption of all 160,338 training
windows. [Aggregate training results](results/latest_training_20261009.json)
and CPU tests are included; checkpoints, restricted records, runtime outputs,
and licensed NVIDIA assets remain external.

This **code and methods package** explains a reversible, episode-level quality
screen for the successful `Place the cup on the plate, then put it back to its
original position` task. It contains the screening code and aggregate counts,
but **no raw recordings, Parquet rows, video, model checkpoints, selection
manifests, or human-label records**. A few official practice episode IDs appear
in the optional simulator controller as configuration; no episode content is included.

The underlying [AIRoA dataset](https://huggingface.co/datasets/airoa-org/yubi-corl2026-umi-arena)
has limited-access terms. Do not add generated per-episode reports, clean or
quarantine manifests, media, or trained models to this presentation package.
The code and aggregate-only materials are in a public GitHub repository;
restricted data is excluded. No open-source license has been assigned.

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

## Optional SSH-backed episode viewer

[`viewer/`](viewer/README.md) follows the code-only GitHub / A100-data pattern
used by [YUBI_Visualization](https://github.com/Kostov0129/YUBI_Visualization).
It lists the 552 quarantined cup episodes from A100, then persists **only a
selected episode's** motion JSON and requested camera clips in a local
`.cache/` directory. The cache is excluded from Git, and the server listens
only on `127.0.0.1`. No checkpoint is needed or transferred.

## Model debugging and online Isaac Sim console

For the π0.5, LingBot, and OpenWAM training recipes, private A100 checkpoint
and squirrel RTX 5090 inference paths, and the authorized-team
clone/start/model-registration workflow, see
[`TRAINING_AND_INFERENCE.md`](TRAINING_AND_INFERENCE.md). A repository clone
does not include model weights or grant server access.

Start with the [Chinese model-debugging handoff](MODEL_DEBUGGING.md): server
roles and SSH prerequisites, exact private checkpoint locations, clone/run
commands, the online experiment workflow, known model readiness, and the
camera/pose/control/success-metric improvements. It links to the
[simulator source overlay](simulator_overlay/README.md), which reproduces the
local Isaac Sim changes from a pinned MIT-licensed upstream commit without
redistributing NVIDIA's Panda asset. The shared squirrel host, private
checkpoints, and authorized datasets are separate prerequisites.

[`console/`](console/README.md) contains the Track 1 trajectory console code:
three synchronized camera views, two-hand pose and gripper curves, and
π0.5, LingBot, and OpenWAM checkpoints selectable with per-action-chunk
playback. It also contains the 8772 online Isaac Sim page, SSH scheduler,
model adapters, inference-service scripts, and tests. It is **code only**;
practice records, generated runs, and weights are not included.
