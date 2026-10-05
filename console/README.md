# UMI Arena Track 1 training-model console

This code-only console visualizes authorized practice-suite demonstrations and
offline π0.5, LingBot, and OpenWAM predictions. It synchronizes the center and two wrist cameras
with left/right hand poses and gripper-joint curves. Switch between the
registered checkpoints; select a single action chunk to
compare its trajectories and errors or export that chunk as JSON locally.
The model traces are offline predictions on recorded observations, **not**
closed-loop robot execution or an independent evaluation. The displayed cup
examples overlap training data.

No practice-suite records, videos, per-episode predictions, model weights, or
clean-selection manifests are published in this repository. `data/`,
`videos/`, and `replay_results/` are Git-ignored. They must be populated only
by a user with separate authorization under the
[AIRoA dataset terms](https://huggingface.co/datasets/airoa-org/yubi-corl2026-umi-arena).

The [training and inference guide](../TRAINING_AND_INFERENCE.md) lists the
three model recipes, exact private A100/RTX 5090 locations, and authorized
operator steps. The separate **online Isaac Sim console** runs on port 8772 and connects over
SSH to the prepared `squirrel_5090` GPU workstation. It does not need local
practice data or a local Isaac Sim install. See the [model-debugging handoff](../MODEL_DEBUGGING.md)
for the server topology, private weight paths, use steps, and optimization
history; see [deployment notes](DEPLOYMENT.md) for file placement and adapter
contracts. The online page records current rendered observations and reports,
but no online run or model weight is included in Git.

```bash
python3 sim_console.py
# open http://127.0.0.1:8772/ on this machine
```

The browser page only binds to localhost. SSH access to `squirrel_5090` and
the existing private GPU deployment is required to run a model. LingBot 5k/10k
remain disabled until the corrected action timing and prompt are deployed and
causally retested. The published `squirrel_deployment/` scripts are references
for the existing private layout, not a one-click public model installation.

For source-only tests on a fresh Python environment, install
`python3 -m pip install -r requirements-tests.txt` first. The web control
servers themselves use the standard library; model adapters run in their
separate GPU environments.

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
restricted; do not commit them. The seven registered IDs in
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
