#!/usr/bin/env python3
"""Publish only a completed, verified 259632/259633 simulator replay video."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import re
import sys
import uuid
from datetime import datetime, timezone

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


COMMIT = "29652dc4e93903514b292b24988fd1a893b10e21"
SETUP = "replay_259632_259633_tuned"
SCENE = "dual_franka_yubi_cup40k_cupfriction_trial"
POLICY = "umi_left_second_height_replay.py"
STEPS = 215


def verify(source: Path, require_wrists: bool = False) -> dict:
    report = json.loads((source / "report.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    episodes = report.get("episodes", [])
    if (report.get("status") != "completed" or report.get("episodes_requested") != 1
            or report.get("steps_requested") != 215 or len(episodes) != 1
            or report.get("setup_requested") != SETUP
            or not str(report.get("scene", "")).endswith(SCENE + ".usda")
            or not str(report.get("policy", "")).endswith(POLICY)):
        raise ValueError("run does not match the tuned paired replay recipe")
    episode = episodes[0]
    first = episode.get("first_success_policy_step")
    transitions = episode.get("transitions", [])
    if (episode.get("policy_steps") != STEPS or len(transitions) != STEPS
            or [row.get("policy_step") for row in transitions] != list(range(STEPS))):
        raise ValueError("run does not contain the full 215-step paired replay")
    if (not episode.get("ever_success") or not isinstance(first, int)
            or first < 0 or first >= len(transitions)
            or not any(row.get("policy_step") == first and row.get("is_success")
                       for row in transitions)
            or episode.get("policy_steps") != len(transitions)):
        raise ValueError("reported first success has no matching simulator transition")
    if (manifest.get("status") != "completed"
            or manifest.get("setup_requested") != SETUP or manifest.get("video_fps") != 30
            or manifest.get("policy_fps") != 10 or len(manifest.get("episodes", [])) != 1):
        raise ValueError("run manifest does not match 30 Hz replay capture")
    video = source / "video.mp4"
    if not video.is_file() or video.stat().st_size < 1024:
        raise ValueError("simulator video is absent or empty")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1",
         str(video)], capture_output=True, text=True, check=True,
    )
    frame_count = int(result.stdout.strip())
    if (frame_count != episode.get("video_frames")
            or frame_count != episode.get("joint_samples")
            or frame_count != 1 + 3 * STEPS):
        raise ValueError("video, joints and policy steps are not frame-aligned")
    wrists = ('left_wrist', 'right_wrist')
    recorded = manifest.get('recorded_views', [])
    if require_wrists or any(name in recorded for name in wrists):
        if recorded != ['overview', *wrists] or manifest.get('simulator_profile') != 'tuned_v1':
            raise ValueError('three-view recording has no verified simulator profile')
        for name in wrists:
            path = source / f'video_{name}.mp4'
            if not path.is_file() or path.stat().st_size < 1024:
                raise ValueError(f'{name} recording is absent')
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                 '-show_entries', 'stream=nb_read_frames', '-of', 'default=nokey=1:noprint_wrappers=1', str(path)],
                capture_output=True, text=True, check=True)
            if int(result.stdout.strip()) != frame_count or episode.get('wrist_video_frames', {}).get(name) != frame_count:
                raise ValueError(f'{name} video is not frame-aligned')
        from simulator_profiles.tuned_v1.yubi_isaac_sim_env.wrist_rig import camera_pose
        import numpy as np
        audit = [json.loads(line) for line in (source / 'wrist_camera_poses.jsonl').read_text().splitlines()]
        if [row.get('sample_index') for row in audit] != list(range(frame_count)):
            raise ValueError('wrist pose audit is incomplete')
        times = np.array([row['physics_time_s'] for row in audit], dtype=float)
        if not np.isfinite(times).all() or not np.allclose(np.diff(times), 1/30, atol=1e-5, rtol=0):
            raise ValueError('wrist pose audit is not at the recorded 30 Hz physics timestamps')
        for row in audit:
            for name in wrists:
                view = row['views'][name]
                spec = manifest['cameras'][name]
                clipping = np.asarray(view.get('clipping_range_m', []), dtype=float)
                if (spec.get('clipping_range_m') != [0.01, 100.0] or clipping.shape != (2,)
                        or not np.allclose(clipping, [0.01,100.0], atol=1e-6)):
                    raise ValueError('wrist near clipping plane hides the local grasp workspace')
                if view['robot_side'] != name.split('_')[0] or spec['robot_side'] != view['robot_side']:
                    raise ValueError('wrist camera assigned to the wrong arm')
                p, q = camera_pose(view['base_pose'], spec['rigid_mount'])
                actual_q = np.asarray(view['camera_quaternion_wxyz'])
                if (actual_q.shape != (4,) or not np.isfinite(actual_q).all()
                        or not np.allclose(p, view['camera_position_m'], atol=1e-6, rtol=0)
                        or min(np.linalg.norm(q-actual_q), np.linalg.norm(q+actual_q)) > 1e-5):
                    raise ValueError('wrist camera is not rigidly following its gripper base')
    return {
        "state": "verified",
        "branch_commit": COMMIT,
        "source_episodes": [259632, 259633],
        "source_frames": 601,
        "offline_waypoints": 215,
        "policy_steps": STEPS,
        "video_fps": 30,
        "policy_fps": 10,
        "video_available": True,
        "verified_first_success_policy_step": first,
        "final_success": bool(episode.get("success")),
        "video_frames": frame_count,
        "recorded_views": recorded or ['overview'],
        "simulator_profile": manifest.get('simulator_profile', 'upstream_tuned_branch'),
        "detail": (f"Isaac 在控制步 {first} 首次判定杯子放盘成功；完整双臂轨迹 "
                   f"{len(transitions)} 步、{frame_count} 帧已核对。"
                   f"{'末帧杯子仍在盘内。' if episode.get('success') else '末帧杯子已取回盘外，符合第二段目标（放盘判定为 false）。'}"
                   "两段采集轨迹以 10 Hz 抽样执行、30 fps 录制，额外含初始接近段。"
                   "这是已知采集轨迹的调优重放，不是在线模型独立抓取。"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Private local copy of completed remote run")
    parser.add_argument('--register-console', action='store_true', help='Import this verified run into console history without rerunning physics')
    args = parser.parse_args()
    status = verify(args.source, require_wrists=True)
    target = Path(__file__).resolve().parent
    generation = args.source.name
    if not re.fullmatch(r'[A-Za-z0-9_-]+', generation):
        raise ValueError('Unsafe publication generation')
    destination = target / 'tuned_published' / generation
    staged = destination.with_name(generation + '.staging')
    if destination.exists() or staged.exists():
        raise FileExistsError('Publication generation already exists')
    staged.mkdir(parents=True)
    files = {'overview': 'video.mp4', 'left_wrist': 'video_left_wrist.mp4', 'right_wrist': 'video_right_wrist.mp4'}
    for filename in files.values():
        shutil.copyfile(args.source / filename, staged / filename)
    os.replace(staged, destination)
    status['video_urls'] = {name: f'/dataset-replay/tuned/artifacts/{generation}/{filename}' for name,filename in files.items()}
    if args.register_console:
        run_id = uuid.uuid4().hex[:12]
        run_dir = target.parent / 'sim_runs' / run_id
        run_dir.mkdir(mode=0o700)
        for name in (*files.values(), 'report.json', 'manifest.json', 'joints.csv', 'wrist_camera_poses.jsonl'):
            shutil.copyfile(args.source / name, run_dir / name)
        stamp = datetime.now(timezone.utc).isoformat()
        metadata = {'id': run_id, 'status':'completed', 'created_at':stamp, 'finished_at':stamp,
                    'policy':'tuned-paired-replay-259632-259633',
                    'policy_label':'成功示范重放 · 259632／259633 · tuned_v1（非模型）',
                    'setup_index':0, 'seed':42, 'steps':215, 'run_until_success':False,
                    'task_objective':'plate', 'camera':'overview', 'gpu':0,
                    'canonical_hand_sides':True, 'anatomical_mounts':{'left':'LeftMount','right':'RightMount'},
                    'simulator_profile':'tuned_v1', 'recorded_views':['overview','left_wrist','right_wrist'],
                    'inference_backend':None, 'evaluation_class':'recorded_demonstration_replay_not_online_model',
                    'imported_verified_run':generation, 'tuned_replay_validation':status}
        (run_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
        status['console_run_url'] = f'/?run={run_id}'
    status_path = target / "tuned_branch_status.json"
    temporary = status_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, status_path)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
