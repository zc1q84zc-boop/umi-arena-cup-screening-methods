# OpenWAM: paired cup intersection, official YUBI inputs

Private A100 workspace:
`/mnt/data/benyun/workspace/openwam_cup_merged_official_20261008`.
No source videos, Parquet, weights, or raw trajectories are published.

The authoritative prepared selection is `prepared_v2/`. It contains 1,186
merged recordings / 2,372 source episodes / 538,430 raw frames, all from the
2,781-episode quality/IK intersection. The UUID split is 1,068 training and
118 validation recordings. There are 160,338 unique training control windows,
162,196 training samples after phase balancing, and 17,366 validation windows.
The earlier `prepared/` directory was a draft with a stricter position cutoff
used only for an initial real-gradient batch-2 probe; it is not the formal
training selection.

Pairing requires adjacent retained episodes from the same UUID in the exact
right-hand-place then left-hand-return prompt order. The intersection has
319 single-episode recordings; 12 other pairs have a filtered intermediate
episode. Another 33 pairs exceed the prior quality review's 2 m/s translation
or 2.55 rad/s angular proxy at the assumed contiguous 30 Hz boundary. These
are isolated with reasons, without deleting data. The IK-source membership
also includes 61 records with no retained episode, outside the intersection.
`pairing_decisions.jsonl` is the complete audit. These are reference quality
checks, not calibrated collision or continuous-IK certificates.

## Official inputs and targets

Verified against the official pages on 2026-10-08:

- https://umi-arena.airoa.io/submission-format
- https://umi-arena.airoa.io/evaluation

Only current left/right wrist images (native uint8 RGB HWC 480x640), current
inter-hand pose (7, xyzw), finger joints (2), and current `prompt` are policy
inputs. The two wrist views are deterministically tiled left then right into
a 640x256 image for the single-image OpenWAM backbone. Absolute hand poses
are read offline only to audit merge boundaries, never as model inputs.
The state is 9 values padded to the pretrained 80 slots; only those 9 are
enabled by the state mask. Actions are official left delta pose(7), right
delta pose(7), and absolute finger targets(2), padded to 80 with only 16
active slots. There is no center-camera dependency.

The 30 Hz source deltas are composed as SE(3) transforms over three source
ticks for causal 10 Hz control. At raw frame t, the next three pose-delta
rows t+1..t+3 provide the t-to-t+3 action; finger targets come from t+2.
This avoids playing a 30 Hz delta as a 10 Hz command. The actual model
chunk is 32 control steps, with nine video targets sampled every four
control steps. Future video frames are training targets; only the current
first image is a clean inference condition.

Both episodes are merged under one UUID with a preserved phase boundary.
At each training observation only its current phase prompt is used. Action
and video loss masks stop at a prompt change; the model is never trained to
execute second-phase actions under the first-phase prompt. Complete source
ordering does not by itself give this current-frame model long-term memory.
Remembering an unobservable original cup location requires a separate causal
history mechanism if held-out experiments demonstrate that need.

Normalization is computed on training UUIDs only. Phase balancing operates
on training windows only. `check_data.py` checks both real wrist streams,
both prompts, phase-tail loss masks, UUID disjointness, and SE(3) composition.

## Execution and budget

`pipeline.py` uses GPU 1 only, checks that it is idle before each stage, and
never stops other processes. GPU 0 is excluded. It waits for the initial
batch-2 probe, checks the final data, measures feasible batches 4/8 (or 2),
selects throughput with memory headroom, and sets accumulation for effective
batch 32. A two-update smoke run must produce positive finite gradients and
a finite saved checkpoint with changed action weights before formal training.

The first pilot is at most one balanced training pass, bounded in planning
by 24 GPU-hours with a 15% allowance for initialization, saves, and variance.
The exact sample count, optimizer updates, measured throughput and checkpoint
interval appear in `training_plan.json`. This is an initial empirical budget,
not a claim that one pass or 24 hours is sufficient. Formal training starts
from foundation weights; benchmark and smoke updates are discarded.

Video backbone is frozen; action/proprio fine-tuning uses BF16, ZeRO-2,
CPU optimizer offload, gradient checkpointing, LR 1e-5, seed 42. Milestones
save deployable weights and normalization assets. Full optimizer snapshots
are disabled because of their shared-NFS cost; exact optimizer-state resume
is therefore unavailable. Original runs and checkpoints are untouched.

Follow `pipeline_status.json`, `training_plan.json`, `training.log`, and
`training/steps.jsonl`. A failed prerequisite stops the pipeline before formal
training. Held-out replay is diagnostic; complete closed-loop task success is
still required for model selection. Compatibility with the official fixed
runtime and RTX 5070 Ti 16 GB has not yet been verified: the uncompressed
OpenWAM model exceeds that VRAM capacity and needs a deployment adaptation.
