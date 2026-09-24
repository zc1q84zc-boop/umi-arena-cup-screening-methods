# SSH-backed review of the 552 quarantined cup episodes

This viewer uses the same **code in GitHub, data on A100, fetch on selection**
pattern as [YUBI_Visualization](https://github.com/Kostov0129/YUBI_Visualization).
It does not mirror the full dataset. Unlike a browser-only stream, it **does
persist** the selected episode's trajectory JSON and requested MP4 camera
clips under `.cache/review_viewer/` on the workstation, as requested. That
directory is Git-ignored and is never part of the repository. A catalog of
episode IDs and flags is read from A100 into server memory at startup.

## Run on the workstation

Prerequisites: authorized SSH access to the A100; `ssh openwam-a100` and
`scp` must work without an interactive password. On the A100, the configured
Python interpreter must have `pyarrow` and read access to the completed
`extracted_bad_episodes` directory.

```bash
python3 viewer/server.py --ssh-host openwam-a100
```

Open `http://127.0.0.1:8771/` on **the same workstation**. Use `--remote-root`,
`--remote-python`, `--cache-dir`, and `--port` to change defaults. The default
remote root matches the A100 workspace layout as of 2026-09-24. The server
binds only `127.0.0.1`; there is no public endpoint or GitHub Pages deployment.

The selector shows all 552 quarantined episodes and can filter human-marked
or machine-flagged ones. Selecting an episode pulls its Parquet motion columns
into a small local JSON cache. Opening a video pulls only that episode's
requested left, right, or center MP4 to the same cache. The browser gets data
from the localhost server with `Cache-Control: no-store`. Playback, time
scrubbing, event buttons, gripper curves, and action/observation motion curves
stay synchronized.

The local `.cache/` is **restricted derived data**. Do not commit it, upload
it, or share it beyond the registered competition team under the applicable
[AIRoA terms](https://huggingface.co/datasets/airoa-org/yubi-corl2026-umi-arena)
and confidentiality pledge. Checkpoint files are not used or transferred.

The flags are review candidates, not proof of three-robot infeasibility.
