#!/usr/bin/env python3
"""Export a complete, non-destructive cup-episode review ledger.

Machine flags and prior human labels remain separate. This never edits the
source dataset, running training manifest, or any model checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CUP_TASK = "Place the cup on the plate, then put it back to its original position"
EXPECTED_EPISODES = 3975
EXPECTED_FRAMES = 827323


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scan-report", type=Path, required=True)
    parser.add_argument("--human-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    selected = {
        int(row["episode_index"]): row
        for row in manifest["episodes"] if row["task"] == CUP_TASK
    }
    if len(selected) != EXPECTED_EPISODES or sum(row["length"] for row in selected.values()) != EXPECTED_FRAMES:
        parser.error("cup selection does not match the 3,975 episodes / 827,323 frames baseline")
    report = json.loads(args.scan_report.read_text())
    scanned = {int(row["episode_index"]): row for row in report["episodes"]}
    if len(scanned) != len(report["episodes"]):
        parser.error("duplicate episode in scan report")
    if not set(scanned).issubset(selected):
        parser.error("scan report contains an episode outside the selected cup task")
    if not args.allow_partial and set(scanned) != set(selected):
        parser.error(f"scan incomplete: {len(scanned)} / {len(selected)} episodes")
    for episode_index, row in scanned.items():
        if row["frames"] != selected[episode_index]["length"]:
            parser.error(f"frame length mismatch in episode {episode_index}")

    human = {}
    for line in args.human_labels.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task") == CUP_TASK and int(row["episode_index"]) in selected:
            human[int(row["episode_index"])] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = args.output_dir / "cup_episode_annotations.csv"
    events_path = args.output_dir / "cup_candidate_frames.csv"
    summary_path = args.output_dir / "cup_review_summary.json"
    episode_columns = [
        "episode_index", "frames", "scanned", "machine_status", "machine_flags",
        "candidate_event_count", "human_status", "human_label", "human_frames",
        "robot_feasibility_status", "training_exclusion",
    ]
    event_columns = ["episode_index", "frame_index", "kind", "details_json"]
    machine_flagged = conditional_fr3 = events_total = quarantined = 0
    with episodes_path.open("w", newline="") as stream, events_path.open("w", newline="") as event_stream:
        writer = csv.DictWriter(stream, fieldnames=episode_columns)
        event_writer = csv.DictWriter(event_stream, fieldnames=event_columns)
        writer.writeheader()
        event_writer.writeheader()
        for episode_index in sorted(selected):
            row = scanned.get(episode_index)
            h = human.get(episode_index, {})
            flags = row.get("flags_for_review_only", []) if row else []
            events = row.get("candidate_events", []) if row else []
            machine_flagged += bool(flags)
            is_quarantined = bool(flags) or h.get("review_status") == "confirmed_anomaly"
            quarantined += is_quarantined
            has_fr3_flag = any("fr3_tcp_speed_infeasible" in flag for flag in flags)
            conditional_fr3 += has_fr3_flag
            events_total += len(events)
            writer.writerow({
                "episode_index": episode_index,
                "frames": selected[episode_index]["length"],
                "scanned": bool(row),
                "machine_status": "review_candidate" if flags else "no_rule_trigger" if row else "pending_scan",
                "machine_flags": ";".join(flags),
                "candidate_event_count": len(events),
                "human_status": h.get("review_status", "not_reviewed"),
                "human_label": h.get("episode_review_label", ""),
                "human_frames": ";".join(map(str, h.get("confirmed_frames", []))),
                "robot_feasibility_status": (
                    "fr3_30hz_speed_conflict_if_tcp_offset_le_0p5m" if has_fr3_flag
                    else "not_certified"
                ),
                "training_exclusion": (
                    "quarantined_in_new_conservative_view" if is_quarantined else "kept_in_new_view"
                ),
            })
            for event in events:
                event_writer.writerow({
                    "episode_index": episode_index,
                    "frame_index": event["frame_index"],
                    "kind": event["kind"],
                    "details_json": json.dumps(event, separators=(",", ":")),
                })
    summary = {
        "task": CUP_TASK,
        "selected_episodes": EXPECTED_EPISODES,
        "selected_frames": EXPECTED_FRAMES,
        "scanned_episodes": len(scanned),
        "scanned_frames": sum(row["frames"] for row in scanned.values()),
        "machine_review_candidates": machine_flagged,
        "conditional_fr3_30hz_conflicts": conditional_fr3,
        "prior_human_labels": len(human),
        "candidate_frame_events": events_total,
        "episodes_quarantined_in_new_view": quarantined,
        "original_manifest_modified": False,
        "note": "Machine flags and human labels are conservatively quarantined in a new view, not proven robot-infeasible. No-rule-trigger is not a feasibility certificate. Nominal FR3 speed conflicts assume a rigid <=0.5 m hand-root-to-TCP offset and exact recorded timing.",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
