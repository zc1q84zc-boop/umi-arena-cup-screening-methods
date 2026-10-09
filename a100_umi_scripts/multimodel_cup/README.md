# Clean cup multi-model comparison workspace

Scope: the audited, successful cup task only. Both LingBot-VLA 2.0 and
OpenWAM must train on the exact 3,423-episode / 746,914-frame allowlist that
the π0.5 run used. The source LeRobot v3 dataset is packed: a symlink to its
Parquet and video files is **not** a filtered dataset. Any training loader must
apply `selection/manifest.json` episode IDs and prove the resulting count.

Before training, run `verify_clean_inputs.py` against the clean manifest,
quarantine ledger, and source dataset `meta/info.json`. It records a manifest
hash and the exact allowlist in the private A100 workspace. Do not publish
the output, checkpoints, replay trajectories, or video in the public methods
repository.

The comparison console will display offline predictions on the same
recorded-observation timestamps as π0.5. It must not use future observation
frames as an inference condition, and it must label training-overlap episodes
as qualitative checks rather than independent evaluation. OpenWAM's training
video target spans t..t+32 (stride 4) alongside future actions; only its
current t image is supplied as a clean condition at inference. Its 20 raw
bimanual EEF dimensions are mapped into the pretrained 80-slot layout:
left 0–9 and right 34–43 (including the gripper slot per arm).

Data-location caveat: ten clean episodes have a source metadata/manifest
`data/file_index` that points one packed Parquet file early. Their complete
motion rows are in the adjacent `file_index + 1` shard. The official LeRobot
episode filter loads all 3,423 episodes and 746,914 frames correctly; custom
normalization/OpenWAM preparation reads the adjacent shard only when the
declared file is short, and fails unless per-episode frame counts match.

The current comparison recipe uses LingBot-VLA 2.0 (four A100s, 32-action
chunks) and OpenWAM-Alpha action/proprio fine-tuning (one A100; video backbone
frozen). Both runs require a real gradient/update/save-readback smoke test
before long training. Checkpoints and replay results stay private in this
workspace. Ten thousand steps and 5k/10k saves are the initial budget pending
the user's final preference.

## Execution and console

Remote workspace: `/mnt/data/benyun/workspace/umi_cup_multimodel_20260925` on
`openwam-a100`. LingBot's one-step smoke check was verified; its formal run
uses GPU 1–4 and logs to `lingbot/train.log`. OpenWAM's two-step smoke check
produced finite action gradients and a step-2 weights file on GPU 5; its
resumable optimizer state must finish saving and the process exit before
`verify_openwam_smoke.py` marks it safe for the formal run. GPU 0 belongs to
other users and must not be used.

The OpenWAM formal 5k/10k saves are deployable **weights-only** checkpoints.
Its smoke run showed that each full optimizer snapshot writes tens of GB to
shared NFS and can stall unrelated I/O; the formal launcher disables those
snapshots. This saves storage traffic, but a run interrupted between milestones
cannot resume with bit-for-bit identical optimizer state.

The local console is `track1_console/` at `http://127.0.0.1:8768/`. It already
supports a selectable second checkpoint trajectory and single action-chunk
playback/export on the same recorded-video timeline. Add a new entry to
`track1_console/checkpoints.json` only after the corresponding complete
5k/10k weights and offline replay report exist. The console receives small
prediction reports, never the model weights or source videos from A100.
