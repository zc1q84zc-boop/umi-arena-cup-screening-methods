#!/usr/bin/env python3
"""Intersect a quality allowlist with reference IK labels; never rewrite raw data."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def by_episode(rows: list[dict]) -> dict[int, dict]:
    result = {int(row["episode_index"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate episode_index")
    return result


def read_manifest(path: Path) -> tuple[dict, dict[int, dict]]:
    manifest = json.loads(path.read_text())
    rows = by_episode(manifest["episodes"])
    assert len(rows) == manifest["summary"]["total_episodes"]
    assert sum(int(row["length"]) for row in rows.values()) == manifest["summary"]["total_frames"]
    return manifest, rows


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--quarantined-manifest", type=Path, required=True)
    parser.add_argument("--ik-dir", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite: {args.output}")

    quality_manifest, quality = read_manifest(args.quality_manifest)
    _, quarantine = read_manifest(args.quarantined_manifest)
    assert not (quality.keys() & quarantine.keys())
    known = quality | quarantine
    ik_episode_path = args.ik_dir / "episode_labels.jsonl"
    ik_record_path = args.ik_dir / "record_labels.jsonl"
    ik_ids_path = args.ik_dir / "record_ids.txt"
    ik = by_episode(read_jsonl(ik_episode_path))
    records = read_jsonl(ik_record_path)
    record_ids = ik_ids_path.read_text().splitlines()
    assert len({r["uuid"] for r in records}) == len(records)
    assert len(set(record_ids)) == len(record_ids) == len(records)
    assert set(record_ids) == {r["uuid"] for r in records}
    owner = {}
    required = ("three_reference_intersection_pass", "franka_fr3_left",
                "franka_fr3_right", "openarm_v2_left", "openarm_v2_right",
                "agibot_g2_reference_dual")
    for record in records:
        assert all(record.get(key) is True for key in required)
        indices = [int(i) for i in record["episode_indices"]]
        assert len(set(indices)) == len(indices)
        for idx in indices:
            assert idx not in owner
            owner[idx] = record["uuid"]
            assert ik[idx]["uuid"] == record["uuid"]
            assert set(ik[idx]["episode_indices"]) == set(indices)
    assert set(owner) == set(ik)
    assert all(all(row.get(key) is True for key in required) for row in ik.values())
    source = args.source_dataset.resolve()
    assert all((source / name).is_dir() for name in ("data", "videos", "meta"))
    fps = int(quality_manifest["summary"]["fps"])
    assert fps == json.loads((source / "meta/info.json").read_text())["fps"]

    selected_ids = set(quality) & set(ik)
    selected = [quality[idx] for idx in sorted(selected_ids)]
    frames = sum(int(row["length"]) for row in selected)
    tasks = {}
    for row in selected:
        item = tasks.setdefault(row["task"], {"episodes": 0, "frames": 0})
        item["episodes"] += 1
        item["frames"] += int(row["length"])
    policy = "episode_index intersection of quality-clean allowlist and three-reference IK pass labels"
    manifest = {"summary": {"selection_policy": policy, "tasks": tasks,
                "total_episodes": len(selected), "total_frames": frames,
                "fps": fps, "total_hours": frames / fps / 3600}, "episodes": selected}
    membership = []
    for record in records:
        original = sorted(int(i) for i in record["episode_indices"])
        retained = [i for i in original if i in selected_ids]
        membership.append({"uuid": record["uuid"], "ik_episode_indices": original,
                           "retained_episode_indices": retained,
                           "excluded_episode_indices": sorted(set(original) - set(retained)),
                           "retained_frames": sum(int(quality[i]["length"]) for i in retained),
                           "all_ik_episodes_retained": bool(retained) and retained == original})
    retained_records = [r for r in membership if r["retained_episode_indices"]]
    whole = sum(r["all_ik_episodes_retained"] for r in retained_records)
    summary = {"quality_clean_episodes": len(quality), "quality_clean_frames": sum(int(r["length"]) for r in quality.values()),
               "ik_episodes": len(ik), "ik_records": len(records),
               "ik_source_record_frames": sum(int(r["frames"]) for r in records),
               "intersection_episodes": len(selected), "intersection_frames": frames,
               "fps": fps, "intersection_hours": frames / fps / 3600,
               "intersection_records": len(retained_records),
               "ik_records_all_episodes_retained": whole,
               "ik_records_partially_retained": len(retained_records) - whole,
               "ik_records_fully_excluded": len(records) - len(retained_records),
               "retained_episodes_per_record": dict(sorted(Counter(len(r["retained_episode_indices"]) for r in retained_records).items())),
               "quality_clean_excluded_by_ik": len(set(quality) - set(ik)),
               "ik_excluded_by_quality": len(set(ik) - set(quality)),
               "ik_in_quality_quarantine": len(set(ik) & set(quarantine)),
               "ik_outside_quality_review_scope": len(set(ik) - set(known)),
               "episode_intersection_exact": sorted(selected_ids) == sorted(set(quality) & set(ik)),
               "raw_source_modified": False, "reference_only": True,
               "continuity_certified": False, "collision_certified": False}
    assert len(quality) == len(selected) + summary["quality_clean_excluded_by_ik"]
    assert len(ik) == len(selected) + summary["ik_excluded_by_quality"]
    assert frames == sum(r["retained_frames"] for r in retained_records)

    inputs = [args.quality_manifest, args.quarantined_manifest, ik_episode_path, ik_record_path, ik_ids_path]
    provenance = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "source_dataset": str(source), "selection_policy": policy,
                  "inputs": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "episode_indices.json", sorted(selected_ids))
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "provenance.json", provenance)
    (args.output / "record_ids.txt").write_text("".join(r["uuid"] + "\n" for r in retained_records))
    (args.output / "record_membership.jsonl").write_text("".join(json.dumps(r) + "\n" for r in membership))
    with (args.output / "episode_decisions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["episode_index", "uuid", "quality_clean", "ik_pass", "retain", "quality_quarantined", "frames"])
        for idx in sorted(set(known) | set(ik)):
            writer.writerow([idx, owner.get(idx, ""), idx in quality, idx in ik,
                             idx in selected_ids, idx in quarantine, known.get(idx, {}).get("length", "")])
    for name in ("data", "videos", "meta"):
        os.symlink(source / name, args.output / name, target_is_directory=True)
    (args.output / "README.md").write_text(
        "# Quality-clean and reference-IK episode intersection\n\n"
        "Use manifest.json or episode_indices.json as the mandatory training allowlist. "
        "data/, videos/, and meta/ link to immutable packed source files, which still contain excluded episodes.\n\n"
        "Selection is the exact episode_index intersection. Partial records remain selected; "
        "record_membership.jsonl identifies them. all_ik_episodes_retained means all episodes "
        "listed by the IK source remain, not that both task phases have been verified. "
        "Task text and all manifest rows are preserved from the quality manifest; "
        "this operation does not change subtask prompts or action labels.\n\n"
        "IK labels are reference checks only: calibrated real-robot execution, joint continuity, "
        "and collision clearance are not certified. Raw data, original selections, and prior training runs are untouched.\n")
    assert json.loads((args.output / "episode_indices.json").read_text()) == sorted(selected_ids)
    assert {int(r["episode_index"]) for r in json.loads((args.output / "manifest.json").read_text())["episodes"]} == selected_ids
    print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
