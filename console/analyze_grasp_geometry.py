"""Inspect simulated YUBI jaw meshes against the nominal cup sidewall.

This offline diagnostic transforms the same STL vertices used by the USD
collision hull from each observed finger-link frame into simulator world
coordinates. It reports nearest mesh-vertex distances; it does not replace
PhysX contact-force measurement or certify real hardware calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MESHES = ROOT / "dual-franka-yubi-isaac-sim/yubi_isaac_sim_env/assets/yubi/meshes"
PIVOTS = {"left_finger": (0.015, 0.025, 0.045),
          "right_finger": (-0.015, 0.025, 0.045)}
STL_DTYPE = np.dtype([
    ("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attributes", "<u2")
])


def mesh_vertices(name: str) -> np.ndarray:
    path = MESHES / ("jaw_left.stl" if name == "left_finger" else "jaw_right.stl")
    blob = path.read_bytes()
    count = int.from_bytes(blob[80:84], "little")
    if len(blob) != 84 + 50 * count:
        raise ValueError(f"unexpected STL length: {path}")
    triangles = np.frombuffer(blob, dtype=STL_DTYPE, count=count, offset=84)
    return np.unique(triangles["vertices"].reshape(-1, 3), axis=0).astype(np.float64) - PIVOTS[name]


def rotate(vertices: np.ndarray, quaternion: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm([w, x, y, z])
    w, x, y, z = np.array([w, x, y, z]) / norm
    matrix = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return vertices @ matrix.T


def cup_sidewall_distance(vertices: np.ndarray, cup: np.ndarray) -> dict:
    height = 0.075
    relative = vertices - cup
    z = relative[:, 2]
    clamped_z = np.clip(z, 0.0, height)
    radius = 0.027 + (0.040 - 0.027) * clamped_z / height
    radial = np.linalg.norm(relative[:, :2], axis=1)
    surface_distance = np.hypot(radial - radius, z - clamped_z)
    in_height = (z >= 0.0) & (z <= height)
    nearest = int(np.argmin(surface_distance))
    close = in_height & (surface_distance <= 0.003)
    bands = {}
    for low, high in ((0.0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.075)):
        mask = (z >= low) & (z < high)
        if mask.any():
            bands[f"{low:.3f}-{high:.3f}"] = round(float((radial[mask] - radius[mask]).min()), 4)
    return {
        "nearest_sidewall_vertex_m": round(float(surface_distance[nearest]), 4),
        "nearest_vertex_world_m": [round(float(value), 5) for value in vertices[nearest]],
        "nearest_vertex_height_above_table_m": round(float(z[nearest]), 4),
        "nearest_vertex_xy_from_cup_m": [round(float(value), 4) for value in relative[nearest, :2]],
        "nearest_vertex_azimuth_deg": round(float(np.degrees(np.arctan2(relative[nearest, 1], relative[nearest, 0]))), 1),
        "near_contact_height_range_m": [round(float(z[close].min()), 4),
                                        round(float(z[close].max()), 4)] if close.any() else None,
        "vertices_at_cup_height": int(in_height.sum()),
        "min_radial_clearance_m": round(float((radial[in_height] - radius[in_height]).min()), 4)
        if in_height.any() else None,
        "min_radial_clearance_by_height_m": bands,
    }


def analyze(trace: Path, steps: list[int]) -> list[dict]:
    jaw_vertices = {name: mesh_vertices(name) for name in PIVOTS}
    selected = []
    for line in trace.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") != "step" or row["step"] not in steps:
            continue
        cup = np.asarray(row["cup_position_m"], dtype=np.float64)
        metrics = {}
        for name, local_vertices in jaw_vertices.items():
            pose = row["finger_link_poses"][name]
            world_vertices = rotate(local_vertices, pose["quaternion_wxyz"]) + pose["position_m"]
            metric = cup_sidewall_distance(world_vertices, cup)
            tool_quaternion = np.asarray(row["tool_orientation_wxyz"], dtype=np.float64)
            tool_quaternion[1:] *= -1
            tool_relative = rotate(
                np.asarray(metric["nearest_vertex_world_m"], dtype=np.float64).reshape(1, 3)
                - np.asarray(row["tool_position_m"], dtype=np.float64),
                tool_quaternion,
            )[0]
            metric["nearest_vertex_in_tool_frame_m"] = [round(float(value), 5) for value in tool_relative]
            metrics[name] = metric
        selected.append({"step": row["step"], "phase": row["phase"],
                         "cup_displacement_m": round(row["cup_displacement_m"], 4),
                         "jaw_angle_rad": round(row["driven_jaw_rad"], 4),
                         "meshes": metrics})
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.trace, args.steps), indent=2))
