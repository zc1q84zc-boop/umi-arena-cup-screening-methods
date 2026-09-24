#!/usr/bin/env python3
"""Build reversible cup-task clean/bad LeRobot selection views.

The raw packed Parquet and MP4 files are linked, not rewritten. A training
loader MUST use the emitted manifest's episode list; raw links alone still
contain other episodes. This keeps the source immutable and the output cheap.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


CUP_TASK = "Place the cup on the plate, then put it back to its original position"
EXPECTED_EPISODES = 3975
EXPECTED_FRAMES = 827323


def manifest_for(rows: list[dict], policy: str) -> dict:
    frames = sum(int(row["length"]) for row in rows)
    return {
        "summary": {
            "selection_policy": policy,
            "tasks": {CUP_TASK: {"episodes": len(rows), "frames": frames}},
            "total_episodes": len(rows),
            "total_frames": frames,
            "fps": 30,
            "total_hours": frames / 30 / 3600,
        },
        "episodes": rows,
    }


def write_view(path: Path, rows: list[dict], source: Path, policy: str) -> None:
    path.mkdir(parents=True, exist_ok=False)
    (path / "manifest.json").write_text(json.dumps(manifest_for(rows, policy), indent=2) + "\n")
    (path / "episode_indices.json").write_text(
        json.dumps([row["episode_index"] for row in rows]) + "\n")
    (path / "episode_prompts.json").write_text(
        json.dumps({str(row["episode_index"]): row["task"] for row in rows}, indent=2) + "\n")
    for name in ("data", "videos", "meta"):
        os.symlink(source / name, path / name, target_is_directory=True)
    (path / "README.md").write_text(
        "# Index-backed LeRobot v3 selection view\n\n"
        "Use `manifest.json` as the episode allowlist with the existing training "
        "loader. The linked source files are immutable and still physically "
        "contain other episodes; this view is not a standalone repack. "
        "Do not train by enumerating the linked raw files without the manifest.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scan-report", type=Path, required=True)
    parser.add_argument("--human-labels", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    manifest = json.loads(args.manifest.read_text())
    selected = {
        int(row["episode_index"]): row for row in manifest["episodes"]
        if row["task"] == CUP_TASK
    }
    if len(selected) != EXPECTED_EPISODES or sum(row["length"] for row in selected.values()) != EXPECTED_FRAMES:
        parser.error("source selection differs from 3,975 episodes / 827,323 frames")
    report = json.loads(args.scan_report.read_text())
    scanned = {int(row["episode_index"]): row for row in report["episodes"]}
    if len(scanned) != len(report["episodes"]) or set(scanned) != set(selected):
        parser.error("scan must contain each selected cup episode exactly once")
    human = {}
    for line in args.human_labels.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task") == CUP_TASK and row.get("review_status") == "confirmed_anomaly":
            human[int(row["episode_index"])] = row
    if not set(human).issubset(selected):
        parser.error("human label outside cup selection")

    # User requested a conservative clean training set: every machine-flagged
    # episode is quarantined even when a threshold flag is not proof of robot
    # infeasibility. The reason and evidence tier remain explicit/reversible.
    bad_ids = set(human) | {
        idx for idx, row in scanned.items() if row.get("flags_for_review_only")
    }
    clean_rows = [selected[idx] for idx in sorted(selected) if idx not in bad_ids]
    bad_rows = [selected[idx] for idx in sorted(bad_ids)]
    args.output.mkdir(parents=True, exist_ok=False)
    write_view(
        args.output / "clean_dataset", clean_rows, args.source_dataset.resolve(),
        "successful cup episodes MINUS human-confirmed action jumps and all machine anomaly candidates",
    )
    write_view(
        args.output / "quarantined_dataset", bad_rows, args.source_dataset.resolve(),
        "successful cup episodes with human-confirmed action jumps or machine anomaly flags",
    )
    labels_path = args.output / "episode_quality_labels.csv"
    with labels_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "episode_index", "frames", "decision", "human_confirmed_jump",
            "human_frames", "machine_flags", "candidate_frames",
            "evidence_tier", "source_location_corrected",
        ])
        writer.writeheader()
        for idx in sorted(selected):
            item = scanned[idx]
            human_row = human.get(idx, {})
            flags = item.get("flags_for_review_only", [])
            events = item.get("candidate_events", [])
            if item["frames"] != selected[idx]["length"] and idx not in bad_ids:
                parser.error(f"unaccounted frame loss for retained episode {idx}")
            writer.writerow({
                "episode_index": idx,
                "frames": selected[idx]["length"],
                "decision": "quarantine" if idx in bad_ids else "keep",
                "human_confirmed_jump": bool(human_row),
                "human_frames": ";".join(map(str, human_row.get("confirmed_frames", []))),
                "machine_flags": ";".join(flags),
                "candidate_frames": ";".join(map(str, sorted({e["frame_index"] for e in events}))),
                "evidence_tier": (
                    "human_confirmed" if human_row else
                    "machine_candidate_only" if flags else "no_rule_trigger"
                ),
                "source_location_corrected": bool(item.get("corrected_source_shards")),
            })
    with (args.output / "quarantined_episodes.jsonl").open("w") as stream:
        for idx in sorted(bad_ids):
            stream.write(json.dumps({
                "source_manifest_row": selected[idx],
                "human_review": human.get(idx),
                "machine_scan": scanned[idx],
            }) + "\n")
    summary = {
        "original_episode_count": len(selected),
        "original_frame_count": EXPECTED_FRAMES,
        "clean_episode_count": len(clean_rows),
        "clean_frame_count": sum(row["length"] for row in clean_rows),
        "quarantined_episode_count": len(bad_rows),
        "quarantined_frame_count": sum(row["length"] for row in bad_rows),
        "human_confirmed_episode_count": len(human),
        "machine_flagged_episode_count": sum(bool(row.get("flags_for_review_only")) for row in scanned.values()),
        "policy": "conservative episode-level quarantine; reversible; threshold flags are not robot infeasibility proof",
        "source_dataset": str(args.source_dataset.resolve()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "README.md").write_text(
        "# Cup task clean dataset v1\n\n"
        "`clean_dataset/` is a training-ready, manifest-filtered LeRobot v3 view. "
        "`quarantined_dataset/` selects every excluded episode. "
        "`episode_quality_labels.csv` records the decision for all 3,975 episodes, "
        "and `quarantined_episodes.jsonl` contains source metadata and frame-level "
        "signals for every excluded episode. Source data and prior training runs "
        "are untouched.\n\n"
        "Conservative policy: all machine rule triggers are quarantined, including "
        "uncertain threshold candidates. This is a quality curation decision, "
        "not a claim of proven robot infeasibility. The source links still point "
        "to packed raw files; only use these views with their manifests.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
