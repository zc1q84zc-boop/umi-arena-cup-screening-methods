#!/usr/bin/env python3
"""OpenWAM clean-cup checkpoint replay on recorded practice observations.

The current center image is the only clean image condition. The generated
video is not decoded or fed back as a ground-truth future observation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import traceback

import av
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation
import setuptools  # noqa: F401 -- DeepSpeed/Python 3.12 distutils shim


TASK = "Place the cup on the plate, then put it back to its original position"
ACTIVE_SLOTS = np.asarray([*range(10), *range(34, 44)], dtype=np.int64)


class CupUnifyNormalizer:
    """Exact inverse of CupDataset.normalize -> scatter, including raw rot6D."""

    _dst_index = ACTIVE_SLOTS
    _unify_dim = 80

    def __init__(self, normalization_json: Path):
        self.stats = json.loads(normalization_json.read_text())
        for kind in ("action", "state"):
            for key in ("min", "max"):
                value = np.asarray(self.stats[kind][key], dtype=np.float32)
                if value.shape != (20,) or not np.isfinite(value).all():
                    raise ValueError(f"invalid {kind} {key} normalization statistics")

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, dtype=np.float32)
        if values.shape[-1] != 20 or not np.isfinite(values).all():
            raise ValueError("deploy proprio must be finite raw20")
        low = np.asarray(self.stats["state"]["min"], dtype=np.float32)
        high = np.asarray(self.stats["state"]["max"], dtype=np.float32)
        normalized = np.clip(2 * (values - low) / np.maximum(high - low, 1e-6) - 1, -1, 1)
        normalized[..., 3:9] = values[..., 3:9]
        normalized[..., 13:19] = values[..., 13:19]
        mapped = np.zeros((*values.shape[:-1], 80), dtype=np.float32)
        mapped[..., ACTIVE_SLOTS] = normalized
        return mapped

    def unnormalize(self, unified: np.ndarray) -> np.ndarray:
        values = np.asarray(unified, dtype=np.float32)
        if values.shape[-1] != 80 or not np.isfinite(values).all():
            raise ValueError("model action must be finite unified80")
        raw = values[..., ACTIVE_SLOTS]
        low = np.asarray(self.stats["action"]["min"], dtype=np.float32)
        high = np.asarray(self.stats["action"]["max"], dtype=np.float32)
        physical = (raw + 1) * np.maximum(high - low, 1e-6) / 2 + low
        physical[..., 3:9] = raw[..., 3:9]
        physical[..., 13:19] = raw[..., 13:19]
        return physical.astype(np.float32)


def install_cup_deploy_normalizer(prepared: Path, checkpoint_dir: Path) -> None:
    """Use the audited reader's two directional stats without altering weights."""
    from openwam.deploy import model_loader

    saved = np.load(checkpoint_dir / "normalization_stats.npy", allow_pickle=True).item()
    expected = json.loads((prepared / "normalization.json").read_text())
    if set(saved) != {"action", "state"}:
        raise ValueError("checkpoint normalization blocks do not match the clean cup reader")
    for kind in ("action", "state"):
        for key in ("min", "max", "mean", "std", "q01", "q99"):
            if not np.array_equal(np.asarray(saved[kind][key]), np.asarray(expected[kind][key])):
                raise ValueError(f"checkpoint {kind} {key} statistics changed")
    adapter = CupUnifyNormalizer(prepared / "normalization.json")

    def build_normalizer(cfg, ckpt_dir):
        dl = cfg.dataloader
        if (Path(ckpt_dir).resolve() != checkpoint_dir.resolve()
                or dl.type != "umi_cup_clean" or dl.normalize_mode != "min-max"
                or not dl.unify_action or list(dl.unify_action_map) != ["0-9", "34-43"]):
            raise ValueError("deploy config is not the audited clean cup recipe")
        return adapter

    model_loader._build_normalizer = build_normalizer


def atomic_json(path: Path, data: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temp, path)


def center_frame(reference: dict, frame_index: int, fps: int):
    target = reference["video_start_s"] + frame_index / fps
    if target >= reference["video_end_s"]:
        raise ValueError("center image timestamp extends beyond episode")
    with av.open(reference["video"]) as container:
        stream = container.streams.video[0]
        container.seek(int(target / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time is None or frame.time < target - 0.5 / fps:
                continue
            if abs(frame.time - target) <= 0.5 / fps:
                return ImageOps.pad(frame.to_image(), (320, 384),
                                    method=Image.Resampling.BILINEAR, color=(0, 0, 0))
            break
    raise ValueError(f"missing center image at frame {frame_index}")


def raw_state20(episode, index: int) -> np.ndarray:
    pose = episode.hand_poses[index]
    grip = episode.joints[index]
    state = np.empty(20, dtype=np.float32)
    for hand in (0, 1):
        matrix = Rotation.from_quat(pose[hand, 3:7]).as_matrix()
        start = hand * 10
        state[start:start + 3] = pose[hand, :3]
        state[start + 3:start + 6] = matrix[:, 0]
        state[start + 6:start + 9] = matrix[:, 1]
        state[start + 9] = grip[hand]
    return state


def decode_actions20(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if raw.ndim != 2 or raw.shape[1] != 20 or not np.isfinite(raw).all():
        raise ValueError(f"OpenWAM output must be finite (N,20), got {raw.shape}")
    poses = np.empty((len(raw), 2, 7), dtype=np.float64)
    grippers = np.empty((len(raw), 2), dtype=np.float64)
    for hand in (0, 1):
        start = hand * 10
        a = raw[:, start + 3:start + 6].astype(np.float64)
        b = raw[:, start + 6:start + 9].astype(np.float64)
        an = np.linalg.norm(a, axis=1)
        if np.any(an < 1e-5):
            raise ValueError("OpenWAM rotation6D first basis is degenerate")
        a /= an[:, None]
        b -= (a * b).sum(axis=1)[:, None] * a
        bn = np.linalg.norm(b, axis=1)
        if np.any(bn < 1e-5):
            raise ValueError("OpenWAM rotation6D second basis is degenerate")
        b /= bn[:, None]
        c = np.cross(a, b)
        matrix = np.stack([a, b, c], axis=-1)
        poses[:, hand, :3] = raw[:, start:start + 3]
        poses[:, hand, 3:7] = Rotation.from_matrix(matrix).as_quat()
        grippers[:, hand] = raw[:, start + 9]
    return poses, grippers


def compare_absolute(predicted, predicted_grippers, anchors, references, grippers, settings):
    from umi_arena.replay_metrics import delta_between, pose_error

    count = len(grippers)
    predicted = predicted[:count]
    predicted_grippers = predicted_grippers[:count]
    if len(predicted) != count or not np.isfinite(predicted).all():
        raise ValueError("OpenWAM did not return every action in the comparison window")
    position, rotation = [], []
    for hand in (0, 1):
        p, r = pose_error(predicted[:, hand], references[:, hand])
        position.append(p)
        rotation.append(r)
    position, rotation = np.asarray(position).T, np.asarray(rotation).T
    grip_error = np.abs(predicted_grippers - grippers)
    pred_relative = delta_between(predicted[:, 0], predicted[:, 1], "body")
    ref_relative = delta_between(references[:, 0], references[:, 1], "body")
    rel_position, rel_rotation = pose_error(pred_relative, ref_relative)
    if not all(np.isfinite(x).all() for x in (position, rotation, grip_error, rel_position, rel_rotation)):
        raise ValueError("OpenWAM prediction produced unbounded errors")
    within = ((position <= settings.position_cm).all(axis=1)
              & (rotation <= settings.rotation_deg).all(axis=1)
              & (grip_error <= settings.gripper_rad).all(axis=1))
    previous = np.concatenate([anchors[None], references[:-1]])
    movement = np.linalg.norm(references[..., :3] - previous[..., :3], axis=-1)
    angular = np.stack([pose_error(previous[:, hand], references[:, hand])[1]
                        for hand in (0, 1)], axis=1)
    moving = ((movement * settings.action_hz > 0.005)
              | (angular * settings.action_hz > 2)).any(axis=1)
    return {
        "position_cm": position.tolist(), "rotation_deg": rotation.tolist(),
        "gripper_rad": grip_error.tolist(),
        "inter_hand_position_cm": rel_position.tolist(),
        "inter_hand_rotation_deg": rel_rotation.tolist(),
        "within_tolerance": within.tolist(), "moving": moving.tolist(),
        "predicted_poses": predicted.tolist(),
        "predicted_grippers": predicted_grippers.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--openwam-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-episodes", type=int, default=None,
                        help="process at most this many new episodes before exiting (for smoke validation)")
    args = parser.parse_args()
    if not args.checkpoint_id.startswith("openwam-cup-clean-"):
        parser.error("checkpoint ID must identify the clean-cup OpenWAM run")
    if not (args.checkpoint_dir / args.checkpoint_name).is_file():
        parser.error("OpenWAM checkpoint file is absent")
    prepared = json.loads((args.prepared / "manifest.json").read_text())
    if (prepared["total_episodes"], prepared["total_frames"], prepared["active_action_slots"]) != (
            3423, 746914, [*range(10), *range(34, 44)]):
        raise ValueError("prepared OpenWAM dataset is not the audited bimanual cup recipe")
    sys.path[:0] = [str(args.evaluation_root), str(args.openwam_root)]
    from umi_arena.replay_data import Dataset
    from umi_arena.replay_metrics import Settings, summarize
    from openwam.deploy.server import build_server_from_config
    from omegaconf import OmegaConf

    install_cup_deploy_normalizer(args.prepared, args.checkpoint_dir)
    deploy_cfg = OmegaConf.load(args.openwam_root / "configs/deploy.yaml")
    if int(deploy_cfg.inference.denoise_steps) != 10 or deploy_cfg.inference.denoise_mode != "sync":
        raise ValueError("official OpenWAM deploy denoising defaults changed")
    # Keep the official inference schedule but avoid JIT compilation during a
    # qualitative replay of just eight recorded examples.
    OmegaConf.update(deploy_cfg, "optimization.compile.enabled", False, merge=False)

    suite = json.loads(args.suite.read_text())
    clean = json.loads(args.train_manifest.read_text())
    clean_ids = {int(e["episode_index"]) for e in clean["episodes"]}
    if len(clean_ids) != 3423:
        raise ValueError("training selection differs from clean cup manifest")
    task = [x for x in suite["tasks"] if x["dataset_task"] == TASK]
    if len(task) != 1:
        raise ValueError("practice suite must contain one cup task")
    suite_ids = [int(i) for recording in task[0]["recordings"] for i in recording["episodes"]]
    selected = [i for i in suite_ids if i in clean_ids]
    excluded = [i for i in suite_ids if i not in clean_ids]
    settings = Settings(30, "world", "next")

    if args.output.exists():
        if not args.resume:
            parser.error(f"output already exists: {args.output}")
        report = json.loads((args.output / "report.json").read_text())
        if report.get("checkpoint_id") != args.checkpoint_id or report.get("requested_episodes") != selected:
            raise ValueError("existing report does not match checkpoint and selection")
        report["normalization_adapter"] = (
            "audited CupDataset action/state min-max with rot6D passthrough and 20-to-80 slot map")
        report["inference_schedule"] = "official OpenWAM synchronous 10-step deploy default"
        atomic_json(args.output / "report.json", report)
    else:
        args.output.mkdir(parents=True)
        report = {
            "version": 3, "status": "running", "baseline": None,
            "checkpoint_id": args.checkpoint_id,
            "checkpoint_path": str((args.checkpoint_dir / args.checkpoint_name).resolve()),
            "dataset_revision": suite["dataset_revision"],
            "requested_episodes": selected,
            "excluded_practice_episodes": excluded,
            "settings": asdict(settings),
            "scope": "Clean cup training examples for visualization; not held-out or official evaluation",
            "training_recipe": "OpenWAM-Alpha action/proprio fine-tune; frozen video backbone",
            "normalization_adapter": "audited CupDataset action/state min-max with rot6D passthrough and 20-to-80 slot map",
            "inference_schedule": "official OpenWAM synchronous 10-step deploy default",
            "prompt": TASK, "episodes": [],
        }
        atomic_json(args.output / "report.json", report)

    server = build_server_from_config(deploy_cfg, str(args.checkpoint_dir),
                                      ckpt_name=args.checkpoint_name, device="cuda")
    engine = server.engine
    dataset = Dataset(args.dataset)
    done = {int(e["episode_index"]) for e in report["episodes"] if e.get("status") == "complete"}
    processed = 0
    for index in selected:
        if index in done:
            continue
        if args.limit_episodes is not None and processed >= args.limit_episodes:
            break
        print(f"OPENWAM_REPLAY_START checkpoint={args.checkpoint_id} episode={index}", flush=True)
        entry = {"episode_index": index, "repeat": 0, "status": "running", "training_overlap": True}
        try:
            episode = dataset.episode(index)
            episode.audit()
            ref = prepared["episodes"][str(index)]
            if ref["frames"] != episode.length:
                raise ValueError("center video reference differs from episode length")
            windows_for_summary, windows_for_console = [], []
            for start, anchors, reference, grippers in episode.windows(settings):
                image = center_frame(ref, start, 30)
                state = raw_state20(episode, start)
                begin = time.monotonic()
                generated = engine.generate({
                    "first_frame_image": [image], "prompt": TASK, "proprio": state,
                    "num_frames": 33, "video_num_frames": 9,
                    "height": 384, "width": 320, "seed": 42,
                })
                latency_ms = (time.monotonic() - begin) * 1000
                actions = np.asarray(generated["actions"], dtype=np.float32)
                if actions.shape[0] < len(reference):
                    raise ValueError("OpenWAM generated fewer actions than comparison frames")
                predicted, pred_grippers = decode_actions20(actions)
                window = compare_absolute(predicted, pred_grippers, anchors,
                                          reference, grippers, settings)
                window.update(frame=start, latency_ms=latency_ms)
                windows_for_summary.append(window)
                windows_for_console.append({key: window[key] for key in (
                    "frame", "predicted_poses", "predicted_grippers",
                    "position_cm", "rotation_deg", "gripper_rad",
                )})
            entry.update(status="complete", frames=episode.length,
                         summary=summarize(windows_for_summary, settings), windows=windows_for_console)
            print(f"OPENWAM_REPLAY_COMPLETE checkpoint={args.checkpoint_id} episode={index} windows={len(windows_for_console)}", flush=True)
        except Exception as exc:
            entry.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            print(f"OPENWAM_REPLAY_FAILED checkpoint={args.checkpoint_id} episode={index}: {exc}", flush=True)
        report["episodes"] = [old for old in report["episodes"] if old["episode_index"] != index] + [entry]
        atomic_json(args.output / "report.json", report)
        processed += 1
        if entry["status"] == "failed" and args.limit_episodes is not None:
            break
    report["status"] = "complete" if all(
        any(e["episode_index"] == i and e["status"] == "complete" for e in report["episodes"])
        for i in selected
    ) else "partial"
    atomic_json(args.output / "report.json", report)
    print(f"OPENWAM_REPLAY_RESULT {report['status']} complete={sum(e['status'] == 'complete' for e in report['episodes'])}/{len(selected)}", flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
