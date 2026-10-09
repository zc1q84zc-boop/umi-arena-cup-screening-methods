#!/usr/bin/env python3
"""Launch the packaged dual-Franka YUBI scene in a visible GPU Isaac Sim window.

Run with ``python -m yubi_isaac_sim_env.run`` from the repository root.
``--record-run DIR`` captures RGB video and named joint states for each episode.
Custom policies export ``act(observation, step, episode) -> action``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yubi_isaac_sim_env import create_sim, resolve_scene
from yubi_isaac_sim_env.recording import JointStateWriter, VideoWriter

# Isolated reference experiment. Never alters asset files or production launcher.
if os.environ.get("UMI_REFERENCE_DIAGNOSTIC"):
    import yubi_isaac_sim_env as _package
    from yubi_isaac_sim_env.env import ROBOT_PATHS
    _reference = json.loads(Path(os.environ["UMI_REFERENCE_DIAGNOSTIC"]).read_text())
    os.environ["YUBI_DIAGNOSTIC_INITIAL_ARM_JOINTS"] = json.dumps(_reference["initial_arm_joint_rad"])
    if os.environ.get("UMI_INITIAL_STATE_FILE"):
        raise ValueError("reference diagnostic cannot use another initial-state profile")
    class ReferenceEnvironment(_package.DualFrankaYubiCupPlateEnv):
        def __init__(self, *args, **kwargs):
            import omni.usd
            from pxr import UsdGeom, Gf
            stage = omni.usd.get_context().get_stage()
            stage.SetEditTarget(stage.GetSessionLayer())
            for side, robot_path in ROBOT_PATHS.items():
                mount = stage.GetPrimAtPath(robot_path).GetParent()
                xf = UsdGeom.Xformable(mount)
                xf.ClearXformOpOrder()
                cfg = _reference["mounts"][side]
                xf.AddTranslateOp().Set(Gf.Vec3d(*cfg["position_m"]))
                q = cfg["quaternion_wxyz"]
                xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
                if os.environ.get("UMI_REFERENCE_FLANGE_PLUS90") == "1":
                    joint = stage.GetPrimAtPath(robot_path + "/joints/yubi_mount_joint")
                    if not joint.IsValid():
                        raise ValueError("reference mount joint missing")
                    joint.GetAttribute("physics:localRot0").Set(Gf.Quatf(2**-.5, 0, 0, 2**-.5))
            super().__init__(*args, **kwargs)
        def reset(self, *args, **kwargs):
            observation = super().reset(*args, **kwargs)
            for side, goal in _reference['targets'].items():
                actual = observation['robots'][side]['tool_pose']
                q=np.asarray(actual['quaternion_wxyz']); q=q/np.linalg.norm(q)
                distance=np.linalg.norm(np.asarray(actual['position_m'])-goal['position_m'])
                angle=np.degrees(2*np.arccos(np.clip(abs(q@goal['quaternion_wxyz']),0,1)))
                if distance > .02 or angle > 10:
                    raise RuntimeError(f'reference reset gate failed: {side} {distance}m {angle}deg')
            return observation
    _package.DualFrankaYubiCupPlateEnv = ReferenceEnvironment


PACKAGE_DIR = Path(__file__).resolve().parent
OVERVIEW_CAMERA_PATH = "/World/OverviewCamera"
OVERVIEW_CAMERA_EYE = (-1.15, 0.0, 1.85)
OVERVIEW_CAMERA_TARGET = (0.45, 0.0, 0.90)
HEAD_CALIBRATION_PATH = PACKAGE_DIR / "head_camera_calibration.json"
WRIST_CAMERA_MODEL_PATH = Path(os.environ.get("UMI_WRIST_CAMERA_PROFILE", str(PACKAGE_DIR / "wrist_camera_model.json")))
WRIST_MOUNT = json.loads(WRIST_CAMERA_MODEL_PATH.read_text(encoding="utf-8"))["nominal_mount_in_yubi_frame"]
EXTRA_VIDEO_STREAMS = []


def _quat_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray((aw*bw-ax*bx-ay*by-az*bz,
                       aw*bx+ax*bw+ay*bz-az*by,
                       aw*by-ax*bz+ay*bw+az*bx,
                       aw*bz+ax*by-ay*bx+az*bw), dtype=float)


def _rotate_by_quat(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    vector = q[1:]
    doubled_cross = 2.0 * np.cross(vector, v)
    return v + q[0] * doubled_cross + np.cross(vector, doubled_cross)


def _nominal_wrist_pose_from_gpu_base(state: dict, side: str) -> dict:
    """Audit pose from the articulation tensor, without moving the RTX sensor.

    Isaac's USD child Xform can lag its GPU-physics parent.  In particular,
    Camera.get_world_pose() is not proof of the actual renderer viewpoint.
    """
    base = state["robots"][side]["link_poses"]["base"]
    base_position = np.asarray(base["position_m"], dtype=float)
    base_quaternion = np.asarray(base["quaternion_wxyz"], dtype=float)
    base_quaternion /= np.linalg.norm(base_quaternion)
    local_position = np.asarray(WRIST_MOUNT["translation_m"], dtype=float)
    pitch = math.radians(WRIST_MOUNT["rotation_x_deg"]) / 2.0
    roll = math.radians(WRIST_MOUNT["roll_about_optical_axis_deg"]) / 2.0
    local_quaternion = _quat_product(
        np.asarray((math.cos(pitch), math.sin(pitch), 0.0, 0.0)),
        np.asarray((math.cos(roll), 0.0, 0.0, math.sin(roll))),
    )
    return {
        "position_m": (base_position + _rotate_by_quat(base_quaternion, local_position)).tolist(),
        "quaternion_wxyz": _quat_product(base_quaternion, local_quaternion).tolist(),
    }


def _camera_spec(name: str) -> dict:
    if name == "head":
        calibration = json.loads(HEAD_CALIBRATION_PATH.read_text(encoding="utf-8"))
        return {"name": "head", "prim_path": calibration["camera_path"],
                "resolution": calibration["resolution_px"],
                "eye_m": calibration["eye_world_m"],
                "quaternion_wxyz": calibration["quaternion_wxyz"],
                "lens_model": calibration["lens_model"],
                "intrinsics_px": calibration["intrinsics_px"],
                "focal_length_m": calibration["focal_length_m"],
                "horizontal_aperture_m": calibration["horizontal_aperture_m"],
                "horizontal_flip_for_dataset": False,
                "calibration_file": str(HEAD_CALIBRATION_PATH)}
    if name in ("left_wrist", "right_wrist"):
        canonical_sides = os.environ.get("YUBI_CANONICAL_HAND_SIDES") == "1"
        side = ("RightMount" if name == "left_wrist" else "LeftMount") if canonical_sides else \
               ("LeftMount" if name == "left_wrist" else "RightMount")
        model = json.loads(WRIST_CAMERA_MODEL_PATH.read_text(encoding="utf-8"))
        return {
            "name": name,
            "prim_path": f"/World/Robots/{side}/Panda/yubi_base/wrist_camera",
            "camera_model": model["model"],
            "mount_override": model.get("per_arm_mount", {}).get(name.split("_")[0]),
            "resolution": model["render_resolution_px"],
            "lens_model": model["projection"],
            "intrinsics_px": model["intrinsics_px"],
            "distortion": model["distortion_k1_k4"],
            "nominal_fov_deg": model["nominal_fov_deg"],
            "focal_length_m": model["usd_focal_length_m"],
            "horizontal_aperture_m": model["usd_horizontal_aperture_m"],
            "horizontal_flip_for_dataset": False,
            "gpu_parent_pose_audit": "nominal base-to-camera pose is computed from the GPU rigid body; "
                                     "USD Camera.get_world_pose may be stale",
            "calibration_status": model["intrinsics_status"],
            "calibration_file": str(WRIST_CAMERA_MODEL_PATH),
        }
    eye = OVERVIEW_CAMERA_EYE
    target = OVERVIEW_CAMERA_TARGET
    diagnostic_eye = os.environ.get("YUBI_DIAGNOSTIC_OVERVIEW_EYE_M")
    diagnostic_target = os.environ.get("YUBI_DIAGNOSTIC_OVERVIEW_TARGET_M")
    if diagnostic_eye is not None or diagnostic_target is not None:
        if diagnostic_eye is None or diagnostic_target is None:
            raise ValueError("diagnostic camera eye and target must both be set")
        def parse_position(raw: str) -> tuple[float, float, float]:
            values = tuple(float(value) for value in raw.split(","))
            if len(values) != 3 or not all(math.isfinite(value) and abs(value) <= 3.0
                                            for value in values):
                raise ValueError("diagnostic camera position must be three finite metres")
            return values
        eye = parse_position(diagnostic_eye)
        target = parse_position(diagnostic_target)
        if math.dist(eye, target) < 0.2:
            raise ValueError("diagnostic camera eye and target are too close")
    return {"name": "overview", "prim_path": OVERVIEW_CAMERA_PATH,
            "resolution": [960, 540], "eye_m": eye,
            "target_m": target, "focal_length_m": 0.015,
            "horizontal_aperture_m": 0.036,
            "horizontal_flip_for_dataset": False}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_path(name: str) -> Path:
    requested = Path(name).expanduser()
    candidates = [requested, PACKAGE_DIR / requested, PACKAGE_DIR / "policies" / requested]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Policy script not found: {name}")


def _setup_file(name: str | None) -> Path | None:
    if name is None:
        return None
    requested = Path(name).expanduser()
    if requested.is_file():
        return requested.resolve()
    bundled = PACKAGE_DIR / "setups" / f"{name.replace(':', '_')}.json"
    return bundled.resolve() if bundled.is_file() else None


def _load_callable(path: Path, name: str):
    spec = importlib.util.spec_from_file_location("cup_plate_user_policy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import policy script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    action_fn = getattr(module, name, None)
    if not callable(action_fn):
        raise RuntimeError(f"Policy script must define {name}(observation, step, episode): {path}")
    return action_fn


def _load_policy(path: Path):
    return _load_callable(path, "act")


def _builtin_action(observation: dict, step: int, mode: str) -> dict:
    if mode == "hold":
        return {}
    names = observation["robots"]["left"]["joint_names"]
    indices = [names.index(f"panda_joint{i}") for i in range(1, 8)]
    offset = 0.12 * math.sin(2 * math.pi * (step + 1) / 20)
    result = {}
    for side, sign in (("left", 1), ("right", -1)):
        q = observation["robots"][side]["joint_positions"]
        targets = [float(q[i]) for i in indices]
        targets[0] = sign * offset
        result[side] = {"arm_joint_targets_rad": targets, "gripper_open_fraction": 1.0}
    return result


def _look_at_quaternion(eye: tuple[float, ...], target: tuple[float, ...]) -> np.ndarray:
    from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats

    eye_vec = np.asarray(eye, dtype=float)
    forward = np.asarray(target, dtype=float) - eye_vec
    forward /= np.linalg.norm(forward)
    up = np.array((0.0, 0.0, 1.0))
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # USD camera coordinates: +X right, +Y up, and -Z forward.
    rotation = np.column_stack((right, up, -forward))
    return rot_matrices_to_quats(rotation)


def _make_camera(spec: dict, gui: bool, recording_video: bool):
    from isaacsim.sensors.camera import Camera
    import omni.usd

    # An inspection-only lighting override applies to the chosen camera.
    stage = omni.usd.get_context().get_stage()
    light = stage.GetPrimAtPath("/World/Lights/InspectionLight")
    if light.IsValid():
        intensity = float(os.environ.get("UMI_REFERENCE_LIGHT_INTENSITY", "220.0"))
        if not math.isfinite(intensity) or not 100 <= intensity <= 1500:
            raise ValueError("reference light intensity must be 100..1500")
        stage.SetEditTarget(stage.GetSessionLayer())
        light.GetAttribute("inputs:intensity").Set(intensity)
    dark_fingers = os.environ.get("UMI_REFERENCE_DARK_FINGERS")
    if dark_fingers is not None:
        if dark_fingers != "1":
            raise ValueError("reference dark-fingers probe must be 1 or unset")
        from pxr import Gf

        stage.SetEditTarget(stage.GetSessionLayer())
        for side in ("LeftMount", "RightMount"):
            shader_path = f"/World/Robots/{side}/Panda/YubiLooks/Jaws/PreviewSurface"
            shader = stage.GetPrimAtPath(shader_path)
            if not shader.IsValid():
                raise RuntimeError(f"reference jaw visual material missing: {shader_path}")
            shader.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(0.10, 0.10, 0.11))
    width, height = spec["resolution"]
    if spec.get("mount_override"):
        from pxr import UsdGeom, Gf
        m=spec["mount_override"]
        stage.SetEditTarget(stage.GetSessionLayer())
        xform=UsdGeom.Xformable(stage.GetPrimAtPath(spec["prim_path"]))
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*m["translation_m"]))
        q=m["quaternion_wxyz"]
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0],*q[1:]))
    camera = Camera(
        prim_path=spec["prim_path"],
        name=f"cup_plate_{spec['name']}",
        resolution=(width, height),
        annotator_device="cpu",
    )
    if spec["name"] == "overview":
        camera.set_world_pose(
            position=np.asarray(spec["eye_m"], dtype=float),
            orientation=_look_at_quaternion(spec["eye_m"], spec["target_m"]),
            camera_axes="usd",
        )
        # Explicit RTX projection: legacy authored lens schemas can otherwise
        # override focal-length/aperture setters and yield a zoomed-in view.
        focal_px = width * spec["focal_length_m"] / spec["horizontal_aperture_m"]
        camera.set_opencv_pinhole_properties(
            cx=width / 2, cy=height / 2, fx=focal_px, fy=focal_px,
            pinhole=[0.0] * 12,
        )
    # Keep the RTX sensor intrinsics consistent with the authored USD camera.
    camera.set_focal_length(spec["focal_length_m"])
    camera.set_horizontal_aperture(spec["horizontal_aperture_m"])
    if spec["name"] in ("left_wrist", "right_wrist"):
        intrinsics = spec["intrinsics_px"]
        camera.set_opencv_fisheye_properties(
            cx=intrinsics["cx"], cy=intrinsics["cy"],
            fx=intrinsics["fx"], fy=intrinsics["fy"],
            fisheye=spec["distortion"],
        )
    if gui:
        from isaacsim.core.utils.viewports import create_viewport_for_camera, set_active_viewport_camera
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is None:
            create_viewport_for_camera("Cup and plate", spec["prim_path"], width, height)
            viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("Isaac Sim GUI started without an active viewport")
        set_active_viewport_camera(spec["prim_path"])
        if str(viewport.camera_path) != spec["prim_path"]:
            raise RuntimeError(f"GUI viewport did not switch to {spec['prim_path']}: {viewport.camera_path}")
    if recording_video:
        camera.initialize()
    return camera


def _rendered_rgb(camera, world, spec: dict) -> np.ndarray:
    # RTX annotators can be several frames behind a just-reset state.
    for _ in range(5):
        world.render()
    height, width = spec["resolution"][1], spec["resolution"][0]
    for _ in range(24):
        world.render()
        rgb = camera.get_rgb(device="cpu")
        if rgb is None:
            continue
        frame = np.asarray(rgb)
        if frame.shape == (height, width, 3) and np.isfinite(frame).all():
            if float(frame.max()) - float(frame.min()) >= 2:
                # The source scene's 2D-to-world projection reverses handedness.
                # Flip the rendered image so plate/cup positions match center.mp4.
                return np.ascontiguousarray(frame[:, ::-1]) if spec["horizontal_flip_for_dataset"] else frame
    raise RuntimeError(f"{spec['name']} camera did not produce a nonblank RTX RGB image")


def _policy_rgb(frame: np.ndarray, spec: dict) -> np.ndarray:
    """Return an owned uint8 image in the model-facing H×W×RGB convention."""
    height, width = spec["resolution"][1], spec["resolution"][0]
    value = np.asarray(frame)
    if value.shape != (height, width, 3) or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{spec['name']} RGB shape/dtype mismatch: {value.shape}, {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{spec['name']} RGB contains non-finite pixels")
    if value.dtype != np.uint8:
        value = np.clip(value * 255 if float(value.max()) <= 1 else value, 0, 255).astype(np.uint8)
    result = np.array(value, dtype=np.uint8, copy=True, order="C")
    return result


def _policy_observation(state: dict, cameras: dict, specs: dict, world) -> dict:
    """Add ephemeral RGB input to a state observation without logging pixels."""
    images = {}
    metadata = {}
    for name in ("head", "left_wrist", "right_wrist"):
        camera = cameras[name]
        spec = specs[name]
        images[name] = _policy_rgb(_rendered_rgb(camera, world, spec), spec)
        position, orientation = camera.get_world_pose(camera_axes="usd")
        # Isaac Sim 5.1 may return CUDA torch tensors here when GPU physics is
        # active. Metadata is serialized for the policy, so transfer explicitly.
        if hasattr(position, "detach"):
            position = position.detach().cpu().numpy()
        if hasattr(orientation, "detach"):
            orientation = orientation.detach().cpu().numpy()
        frame = camera.get_current_frame()
        metadata[name] = {
            "resolution_px": list(spec["resolution"]),
            "lens_model": spec.get("lens_model", "pinhole"),
            "intrinsics_px": spec.get("intrinsics_px"),
            "distortion_k1_k4": spec.get("distortion"),
            "calibration_status": spec.get("calibration_status", "estimated"),
            "calibration_file": spec.get("calibration_file"),
            "calibration_sha256": (hashlib.sha256(Path(spec["calibration_file"]).read_bytes()).hexdigest()
                                   if spec.get("calibration_file") else None),
            "horizontal_flip_for_dataset": bool(spec.get("horizontal_flip_for_dataset", False)),
            "pose_world": {
                "position_m": np.asarray(position, dtype=float).tolist(),
                "quaternion_wxyz": np.asarray(orientation, dtype=float).tolist(),
            },
            "physics_time_s": float(state["physics_time_s"]),
            "rendering_time_s": float(frame["rendering_time"]) if frame and frame.get("rendering_time") is not None else None,
        }
        if name in ("left_wrist", "right_wrist"):
            side = name.split("_")[0]
            if "base" in state.get("robots", {}).get(side, {}).get("link_poses", {}):
                metadata[name]["nominal_pose_world_from_gpu_base"] = _nominal_wrist_pose_from_gpu_base(
                    state, side
                )
                metadata[name]["pose_world_status"] = (
                    "USD xform may lag GPU articulation; use nominal_pose_world_from_gpu_base "
                    "for a geometry audit, not as a measured real-camera extrinsic"
                )
            else:
                metadata[name]["pose_world_status"] = "GPU YUBI-base pose unavailable in this observation"
    return {**state, "images": images, "image_metadata": metadata}


def _sample(video: VideoWriter | None, joints: JointStateWriter | None, camera, world,
            spec: dict, episode: int, sample_index: int, phase: str, policy_step: int,
            observation: dict) -> None:
    if video is not None:
        rgb = _rendered_rgb(camera, world, spec)
        video.write(rgb)
        preview_path = os.environ.get("YUBI_LIVE_PREVIEW_PATH")
        if preview_path and sample_index % 3 == 0:
            # The MP4 is not playable until ffmpeg closes it. Publish a small,
            # atomic JPEG so a remote console can show the current render.
            from PIL import Image

            pixels = np.asarray(rgb)
            if pixels.dtype != np.uint8:
                pixels = np.clip(pixels * 255 if float(pixels.max()) <= 1 else pixels,
                                 0, 255).astype(np.uint8)
            path = Path(preview_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            Image.fromarray(pixels, "RGB").save(temporary, format="JPEG", quality=80)
            os.replace(temporary, path)
    if joints is not None:
        joints.write(episode, sample_index, phase, policy_step, observation)
    for writer, sensor, sensor_spec in EXTRA_VIDEO_STREAMS:
        writer.write(_rendered_rgb(sensor, world, sensor_spec))


def _episode_path(path: Path | None, episode: int, total: int) -> Path | None:
    if path is None:
        return None
    if total == 1:
        return path
    return path.with_name(f"{path.stem}_episode_{episode:03d}{path.suffix}")


def _paths(args, episode: int) -> tuple[Path | None, Path | None]:
    if args.record_run is not None:
        if args.episodes == 1:
            return args.record_run / "video.mp4", args.record_run / "joints.csv"
        return (
            args.record_run / f"episode_{episode:03d}.mp4",
            args.record_run / f"episode_{episode:03d}_joints.csv",
        )
    return (
        _episode_path(args.record_video, episode, args.episodes),
        _episode_path(args.record_joints, episode, args.episodes),
    )


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="dual_franka_yubi_random_000_seed_20260924", help="Packaged scene name or USD path")
    parser.add_argument("--setup", help="Named, JSON-file, or custom object setup for reset")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50, help="Maximum 10 Hz policy steps per episode")
    parser.add_argument("--policy-hz", type=int, choices=(10, 30),
                        help="Optional per-run control rate; default keeps packaged scene rate (10 Hz)")
    parser.add_argument("--online-chunk-30hz", action="store_true",
                        help="Call the model at 10 Hz and execute each of its three predicted actions at 30 Hz")
    parser.add_argument("--until-success", action="store_true",
                        help="Keep stepping until the selected task objective succeeds or a stop is requested")
    parser.add_argument("--stop-file", type=Path,
                        help="Stop gracefully when this control file appears")
    parser.add_argument("--task-objective", choices=("plate", "plate_return"), default="plate",
                        help="plate preserves the simulator's original one-stage success; plate_return "
                             "also requires stable release at the cup's original position")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--index-start", type=int, default=0)
    parser.add_argument("--fixed-index", type=int)
    parser.add_argument("--policy", choices=("hold", "oscillate"), default="hold")
    parser.add_argument("--camera", choices=("head", "overview", "left_wrist", "right_wrist"), default="head",
                        help="GUI and recorded view; head matches the center videos approximately")
    parser.add_argument("--policy-script", help="Python file exporting act(observation, step, episode)")
    parser.add_argument("--trajectory-policy-script",
                        help="Python file exporting predict(observation, step, episode) -> world-frame tool waypoint chunk")
    parser.add_argument("--policy-images", choices=("none", "all"), default="none",
                        help="Pass head and both wrist RGB arrays to a custom policy (default: state only)")
    parser.add_argument("--headless", action="store_true", help="Disable GUI window, retaining GPU physics and RTX recording")
    parser.add_argument("--linger-seconds", type=float, help="Keep the GUI visible this long after the run (default: 5)")
    parser.add_argument("--keep-open", action="store_true", help="Keep rendering the GUI until its window is closed")
    parser.add_argument("--record-run", type=Path, help="Shortcut: capture MP4, joint CSV, manifest, and report in DIR")
    parser.add_argument("--record-video", type=Path, help="Capture rendered camera MP4")
    parser.add_argument("--record-joints", type=Path, help="Capture named joint position and velocity CSV")
    parser.add_argument("--record-fps", type=int, default=30,
                        help="Camera/joint samples per simulated second (default: 30, matching source videos)")
    parser.add_argument("--report", type=Path, help="Write JSON episode report (automatic with --record-run)")
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.steps < 1 or args.index_start < 0 or (args.fixed_index is not None and args.fixed_index < 0):
        parser.error("episodes and steps must be positive; scenario indices must be nonnegative")
    if args.linger_seconds is not None and args.linger_seconds < 0:
        parser.error("--linger-seconds must be nonnegative")
    if args.keep_open and args.headless:
        parser.error("--keep-open requires the GUI")
    if args.policy_script and args.trajectory_policy_script:
        parser.error("Choose one of --policy-script and --trajectory-policy-script")
    if args.policy_images == "all" and not (args.policy_script or args.trajectory_policy_script):
        parser.error("--policy-images all requires a custom policy script")
    if args.record_run and (args.record_video or args.record_joints or args.report):
        parser.error("--record-run already names the video, joint CSV, and report outputs")
    if args.record_video and args.record_video.suffix.lower() != ".mp4":
        parser.error("--record-video must end in .mp4")
    if args.record_joints and args.record_joints.suffix.lower() != ".csv":
        parser.error("--record-joints must end in .csv")
    if args.record_run:
        args.record_run = args.record_run.expanduser().resolve()
        if args.record_run.exists() and any(args.record_run.iterdir()):
            parser.error(f"Recording directory is not empty: {args.record_run}")
    for key in ("record_video", "record_joints", "report"):
        value = getattr(args, key)
        if value is not None:
            value = value.expanduser().resolve()
            if value.exists():
                parser.error(f"Output already exists: {value}")
            setattr(args, key, value)
    return args


def _linger(app, world, seconds: float, keep_open: bool) -> None:
    end = time.monotonic() + seconds
    try:
        while app.is_running() and (keep_open or time.monotonic() < end):
            world.render()
    except KeyboardInterrupt:
        pass


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    scene_path = Path(resolve_scene(args.scene)).resolve()
    selected_policy_script = args.trajectory_policy_script or args.policy_script
    policy_path = _policy_path(selected_policy_script) if selected_policy_script else None
    policy = None
    trajectory_executor = None
    config = json.loads((PACKAGE_DIR / "config.json").read_text(encoding="utf-8"))
    if args.online_chunk_30hz:
        if not args.trajectory_policy_script or args.policy_images != 'all':
            raise ValueError('30 Hz online chunks require a trajectory policy with all live images')
        config['policy_hz'] = 30
        config['success']['dwell_policy_steps'] *= 3
    elif args.policy_hz is not None:
        config["policy_hz"] = args.policy_hz
    policy_fps = int(config["policy_hz"])
    physics_fps = int(config["physics_hz"])
    record_fps = int(args.record_fps)
    if record_fps <= 0 or physics_fps % record_fps or record_fps % policy_fps:
        raise ValueError(
            f"--record-fps must be a positive multiple of policy_hz={policy_fps} "
            f"and divide physics_hz={physics_fps}"
        )
    capture_interval = physics_fps // record_fps
    video_enabled = args.record_run is not None or args.record_video is not None
    camera_spec = _camera_spec(args.camera)
    camera_width, camera_height = camera_spec["resolution"]
    report_path = args.record_run / "report.json" if args.record_run else args.report
    manifest_path = args.record_run / "manifest.json" if args.record_run else None
    report: dict = {
        "status": "started",
        "scene": str(scene_path),
        "setup_requested": args.setup,
        "seed": args.seed,
        "episodes_requested": args.episodes,
        "steps_requested": None if args.until_success else args.steps,
        "run_until_success": args.until_success,
        "policy": str(policy_path) if policy_path else args.policy,
        "policy_input_views": ["head", "left_wrist", "right_wrist"] if args.policy_images == "all" else [],
        "policy_output": "world_tool_waypoint_chunks" if args.trajectory_policy_script else "joint_targets",
        "task_objective": args.task_objective,
        "backend": None,
        "camera": camera_spec,
        "record_fps": record_fps,
        "model_request_hz": 10 if args.online_chunk_30hz else policy_fps,
        "action_execution_hz": policy_fps,
        "episodes": [],
    }
    input_paths = [
        scene_path,
        PACKAGE_DIR / "assets" / "franka_yubi_panda.usdc",
        PACKAGE_DIR / "assets" / "yubi" / "source_manifest.json",
        PACKAGE_DIR / "config.json",
        PACKAGE_DIR / "scene_config.json",
    ]
    if args.task_objective == "plate_return":
        input_paths.append(PACKAGE_DIR / "two_stage_task.py")
    if args.camera == "head" or args.policy_images == "all":
        input_paths.append(HEAD_CALIBRATION_PATH)
    if args.camera in ("left_wrist", "right_wrist") or args.policy_images == "all":
        input_paths.append(WRIST_CAMERA_MODEL_PATH)
    if policy_path is not None:
        input_paths.append(policy_path)
    setup_file = _setup_file(args.setup)
    if setup_file is not None:
        input_paths.append(setup_file)
    manifest: dict = {
        "format_version": 1,
        "scene": str(scene_path),
        "setup_requested": args.setup,
        "seed": args.seed,
        "policy": str(policy_path) if policy_path else args.policy,
        "policy_input_views": ["head", "left_wrist", "right_wrist"] if args.policy_images == "all" else [],
        "policy_output": "world_tool_waypoint_chunks" if args.trajectory_policy_script else "joint_targets",
        "task_objective": args.task_objective,
        "sampling": f"reset state plus one sample every {capture_interval} physics frames",
        "frame_alignment": "video frame n corresponds to joint CSV sample_index n within each episode",
        "video_fps": record_fps,
        "policy_fps": policy_fps,
        "model_request_hz": 10 if args.online_chunk_30hz else policy_fps,
        "camera": camera_spec,
        "input_sha256": {str(path): _hash(path) for path in input_paths},
        "episodes": [],
    }
    app = None
    env = None
    exit_code = 1
    try:
        app, env = create_sim(
            scene=scene_path,
            gui=not args.headless,
            setup=args.setup,
            seed=args.seed,
            scenario_index=args.fixed_index if args.fixed_index is not None else args.index_start,
            task_config=config,
        )
        if args.until_success:
            env.task_config["max_policy_steps"] = math.inf
        elif args.task_objective == "plate_return":
            # The two-stage evaluation may legitimately need more than the
            # scene's original one-stage 150-step horizon.  This is per-run
            # only; it does not modify the published scene configuration.
            env.task_config["max_policy_steps"] = max(
                env.task_config["max_policy_steps"], args.steps * (3 if args.online_chunk_30hz else 1))
        report["backend"] = env.backend
        if env.control_decimation % capture_interval:
            raise RuntimeError("Recording interval does not divide a policy step")
        # Isaac/Omniverse modules imported by a custom policy need Kit ready.
        if policy_path is not None:
            policy = _load_callable(policy_path, "predict" if args.trajectory_policy_script else "act")
        if args.trajectory_policy_script:
            from yubi_isaac_sim_env.policy_adapter import TrajectoryChunkExecutor
            trajectory_executor = TrajectoryChunkExecutor(policy_hz=policy_fps)
        policy_specs = ({name: _camera_spec(name) for name in ("head", "left_wrist", "right_wrist")}
                        if args.policy_images == "all" else {})
        camera = _make_camera(camera_spec, gui=not args.headless,
                              recording_video=video_enabled or args.camera in policy_specs)
        policy_cameras = {}
        if policy_specs:
            for name, spec in policy_specs.items():
                policy_cameras[name] = camera if name == args.camera else _make_camera(
                    spec, gui=False, recording_video=True
                )
        env.world.render()
        for episode in range(args.episodes):
            index = args.fixed_index if args.fixed_index is not None else args.index_start + episode
            observation = env.reset(seed=args.seed, scenario_index=index, setup=args.setup)
            from yubi_isaac_sim_env.two_stage_task import CupPlateReturnEvaluator
            two_stage = CupPlateReturnEvaluator(observation)
            if trajectory_executor is not None:
                trajectory_executor.reset(episode)
            video_path, joint_path = _paths(args, episode)
            entry = {
                "episode": episode,
                "scenario_index": index,
                "scenario_id": observation["scenario"]["id"],
                "video": str(video_path) if video_path else None,
                "joints": str(joint_path) if joint_path else None,
                "policy_steps": 0,
                "video_frames": 0,
                "joint_samples": 0,
                "success": False,
            }
            manifest["episodes"].append(entry)
            initial = observation
            transitions = []
            video = None
            joints = None
            try:
                video = VideoWriter(video_path, camera_width, camera_height, record_fps) if video_path else None
                if os.environ.get('UMI_RECORD_THREE_VIEWS') == '1' and video_path:
                    if args.camera != 'head' or set(policy_cameras) != {'head','left_wrist','right_wrist'}:
                        raise ValueError('three-view capture requires head camera and all policy images')
                    for name in ('left_wrist','right_wrist'):
                        w,h = policy_specs[name]['resolution']
                        stream_path = video_path.with_name(f'video_{name}.mp4')
                        EXTRA_VIDEO_STREAMS.append((VideoWriter(stream_path,w,h,record_fps),
                                                    policy_cameras[name],policy_specs[name]))
                joints = JointStateWriter(joint_path) if joint_path else None
                _sample(video, joints, camera, env.world, camera_spec, episode, 0, "reset", -1, observation)
                sample_index = 0
                stop_requested = False
                for step in itertools.count() if args.until_success else range(args.steps):
                    if args.stop_file is not None and args.stop_file.exists():
                        stop_requested = True
                        break
                    policy_observation = (_policy_observation(observation, policy_cameras, policy_specs, env.world)
                                          if policy_specs else observation)
                    if trajectory_executor is not None:
                        if args.online_chunk_30hz:
                            if trajectory_executor.remaining_waypoints:
                                raise RuntimeError('previous 30 Hz prediction was not fully executed')
                            def predict_three(obs, inner_step, inner_episode):
                                chunk = policy(obs, step, inner_episode)
                                if (len(chunk.get('waypoints', ())) != 3
                                        or chunk.get('execute_steps') != 3):
                                    raise ValueError('online model must return exactly three 30 Hz actions')
                                return chunk
                            action = trajectory_executor.act(
                                policy_observation, step*3, episode, predict_three)
                        else:
                            action = trajectory_executor.act(policy_observation, step, episode, policy)
                    else:
                        action = policy(policy_observation, step, episode) if policy else _builtin_action(
                            observation, step, args.policy
                        )
                    def capture(physics_substep: int) -> None:
                        nonlocal sample_index
                        if physics_substep % capture_interval:
                            return
                        sample_index += 1
                        frame_observation = env.observe() if joints is not None else observation
                        _sample(video, joints, camera, env.world, camera_spec,
                                episode, sample_index, "step", step, frame_observation)

                    observation, reward, terminated, truncated, info = env.step(
                        action, on_physics_step=capture if (video is not None or joints is not None) else None
                    )
                    if args.online_chunk_30hz:
                        for subaction in (1, 2):
                            if terminated or truncated:
                                break
                            # The model is not called again. IK uses the current
                            # GPU robot state for each queued 30 Hz target.
                            action = trajectory_executor.act(
                                observation, step*3+subaction, episode, predict_three)
                            observation, reward, terminated, truncated, info = env.step(
                                action, on_physics_step=capture if (video is not None or joints is not None) else None
                            )
                    task_state = two_stage.update(observation, plate_success=bool(info["is_success"]))
                    objective_success = (task_state["full_task_success"] if args.task_objective == "plate_return"
                                         else bool(info["is_success"]))
                    transitions.append(
                        {
                            "policy_step": step,
                            "physics_time_s": observation["physics_time_s"],
                            "action": action,
                            "reward": reward,
                            "terminated": terminated,
                            "truncated": truncated,
                            "is_success": info["is_success"],
                            "task_stage": task_state["stage"],
                            "full_task_success": task_state["full_task_success"],
                            "cup_position_m": observation["objects"]["cup"]["position_m"],
                            "cup_quaternion_wxyz": observation["objects"]["cup"]["quaternion_wxyz"],
                            "gripper_open_fraction": {
                                side: robot["gripper_open_fraction"] for side, robot in observation["robots"].items()
                            },
                        }
                    )
                    if objective_success or truncated or not app.is_running():
                        break
            finally:
                entry['extra_video_frames'] = {}
                for writer, sensor, sensor_spec in EXTRA_VIDEO_STREAMS:
                    writer.close()
                    entry['extra_video_frames'][sensor_spec['name']] = writer.frame_count
                EXTRA_VIDEO_STREAMS.clear()
                if joints is not None:
                    joints.close()
                    entry["joint_samples"] = joints.sample_count
                if video is not None:
                    video.close()
                    entry["video_frames"] = video.frame_count
            entry["policy_steps"] = len(transitions)
            entry["stop_reason"] = ("success" if transitions and (
                transitions[-1]["full_task_success"] if args.task_objective == "plate_return"
                else transitions[-1]["is_success"]
            ) else "user_stop" if stop_requested else "incomplete")
            entry["success"] = bool(transitions and (
                transitions[-1]["full_task_success"] if args.task_objective == "plate_return"
                else transitions[-1]["is_success"]
            ))
            episode_report = {
                **entry,
                "task_objective": args.task_objective,
                "plate_placed": two_stage.plate_placed,
                "full_task_success": two_stage.full_success,
                "initial_observation": initial,
                "final_observation": observation,
                "object_displacement_m": {
                    name: math.dist(initial["objects"][name]["position_m"], observation["objects"][name]["position_m"])
                    for name in ("cup", "plate")
                },
                "transitions": transitions,
            }
            report["episodes"].append(episode_report)
            print(
                f"Episode {episode}: setup={entry['scenario_id']}, steps={entry['policy_steps']}, "
                f"success={entry['success']}, video={entry['video']}, joints={entry['joints']}",
                flush=True,
            )
        if any(episode["stop_reason"] == "user_stop" for episode in report["episodes"]):
            report["status"] = "stopped"
            exit_code = 0
        elif args.until_success and any(not episode["success"] for episode in report["episodes"]):
            report["status"] = "failed"
            report["error"] = "simulation ended before the selected task objective succeeded"
            exit_code = 1
        else:
            report["status"] = "completed"
            exit_code = 0
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        manifest["status"] = report["status"]
        if "error" in report:
            manifest["error"] = report["error"]
        if report_path:
            _write_json(report_path, report)
            print(f"Report: {report_path}", flush=True)
        if manifest_path:
            _write_json(manifest_path, manifest)
            print(f"Manifest: {manifest_path}", flush=True)
        if app is not None and env is not None and not args.headless:
            _linger(app, env.world, args.linger_seconds if args.linger_seconds is not None else 5.0, args.keep_open)
        sys.stdout.flush()
        sys.stderr.flush()
        if app is not None:
            # Isaac Sim 5.1 GPU teardown can hang or segfault on this host.
            # The video encoder, CSV, and JSON files were closed and flushed.
            os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
