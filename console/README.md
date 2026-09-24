# UMI Arena Track 1 training-model console

This code-only console visualizes authorized practice-suite demonstrations and
offline π0.5 predictions. It synchronizes the center and two wrist cameras
with left/right hand poses and gripper-joint curves. Switch between the
registered 10k, 20k, and 30k checkpoints; select a single action chunk to
compare its trajectories and errors or export that chunk as JSON locally.
The model traces are offline predictions on recorded observations, **not**
closed-loop robot execution or an independent evaluation. The displayed cup
examples overlap training data.

No practice-suite records, videos, per-episode predictions, model weights, or
clean-selection manifests are published in this repository. `data/`,
`videos/`, and `replay_results/` are Git-ignored. They must be populated only
by a user with separate authorization under the
[AIRoA dataset terms](https://huggingface.co/datasets/airoa-org/yubi-corl2026-umi-arena).

## Prepare authorized inputs

Install the [official evaluation repository](https://github.com/airoa-org/umi-arena-evaluation)
and its dependencies where an authorized dataset copy is available. From this
directory, export the practice-suite motion records and camera clips:

```bash
PYTHONPATH=/path/to/umi-arena-evaluation python3 export_practice.py \
  --dataset /path/to/authorized-lerobot-dataset \
  --suite /path/to/umi-arena-evaluation/suites/practice-cup-smartphone.json \
  --output data

PYTHONPATH=/path/to/umi-arena-evaluation python3 export_videos.py \
  --dataset /path/to/authorized-lerobot-dataset \
  --suite /path/to/umi-arena-evaluation/suites/practice-cup-smartphone.json \
  --output videos
```

`export_videos.py` requires FFmpeg and FFprobe. The console starts without
model results; a registered checkpoint shows as pending until an authorized
`replay_results/<checkpoint-id>/report.json` is supplied. The optional
`replay_pi05_cup.py --help` script can generate visualization replays on a
machine that has the original checkpoint, clean-training manifest, dataset,
OpenPI, and official evaluation code. These generated files are also
restricted; do not commit them. The three registered IDs in
`checkpoints.json` are metadata only, not model weights.

## Run locally

```bash
python3 server.py
```

Open `http://127.0.0.1:8768/`. To use another replay directory, pass
`--replay /path/to/authorized-replay-results`. The default bind is localhost.
The bundled Three.js files are distributed under their MIT license in
`vendor/LICENSE`.

Run the no-data unit tests with `python3 -m unittest -v test_server.py`.
