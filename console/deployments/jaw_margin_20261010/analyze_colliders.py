"""Compare deformed cup nodes with the composed USD collision hulls.

Distances use the convex hull triangle surface; they are a contact-envelope
proxy, not measured force. PhysX cooking may simplify the source hull.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_gpu import rotation


def material_height(index):
    ring = (index - 1) // 48
    return .5 if ring <= 3 else .5 + (ring - 3) / 12 * 74


def surface_distance(points, vertices, triangles):
    """Exact nearest distance to the source hull triangle surface."""
    tri = vertices[triangles]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    u, v = b - a, c - a
    normals = np.cross(u, v)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    pa = points[:, None, :] - a[None, :, :]
    height = np.einsum('nmd,md->nm', pa, normals)
    projected = points[:, None, :] - height[:, :, None] * normals
    ap = projected - a
    uu, vv, uv = (u*u).sum(1), (v*v).sum(1), (u*v).sum(1)
    up, vp = np.einsum('nmd,md->nm', ap, u), np.einsum('nmd,md->nm', ap, v)
    denominator = uu*vv - uv*uv
    s = (vv*up - uv*vp)/denominator
    t = (uu*vp - uv*up)/denominator
    inside = (s >= -1e-9) & (t >= -1e-9) & (s+t <= 1+1e-9)
    candidates = [projected]
    distances = [np.where(inside, height*height, np.inf)]
    for x, y in ((a, b), (b, c), (c, a)):
        edge = y-x
        f = np.clip(np.einsum('nmd,md->nm', points[:, None, :]-x, edge)/(edge*edge).sum(1), 0, 1)
        closest = x + f[:, :, None]*edge
        candidates.append(closest)
        distances.append(((points[:, None, :]-closest)**2).sum(2))
    distance = np.stack(distances)
    choice = distance.argmin(0)
    per_triangle = distance.min(0)
    facets = per_triangle.argmin(1)
    rows = np.arange(len(points))
    near = np.stack(candidates)[choice[rows, facets], rows, facets]
    return np.sqrt(per_triangle[rows, facets]), near


def bounded_surface_distance(points, vertices, triangles, margin, band=.002):
    """Cull nodes that cannot be closest or within the requested band.

    Any vertex-node distance bounds the global closest distance from above;
    positive hull facet margin bounds exterior distance from below.
    """
    upper = np.linalg.norm(points[:, None, :]-vertices, axis=2).min()
    indices = np.flatnonzero(margin <= max(upper, band)+1e-12)
    distance = np.full(len(points), np.inf)
    nearest = np.full_like(points, np.nan)
    distance[indices], nearest[indices] = surface_distance(points[indices], vertices, triangles)
    return distance, nearest


def analyze(root, hull_path, robot='left'):
    hulls = json.loads(hull_path.read_text())
    result = json.loads((root / 'result.json').read_text())
    online = result['classification'] == 'live_pi05_left_phase_reset_diagnostic'
    data = np.load(root / 'cup_nodes.npz')
    nodes = {int(i): n for i, n in zip(data['model_steps'], data['nodes'])}
    rows = []
    with (root / 'states.jsonl').open() as stream:
        for line in stream:
            r = json.loads(line)
            i = r['model_step']
            if r['servo_step'] % 3 != 2 or i not in nodes:
                continue
            if not online and not 1400 <= i <= 1599:
                continue
            if online and np.linalg.norm(np.array(r[robot]['tool_pose']['position_m']) - r['cup']['position_m']) > .15:
                continue
            jaws = {}
            for name, meshes in hulls.items():
                pose = r[robot]['link_poses'][name]
                local = (nodes[i] - pose['position_m']) @ rotation(pose['quaternion_wxyz'])
                jaws[name] = {}
                for mesh in meshes:
                    equations = np.array(mesh['equations'])
                    margin = (local @ equations[:, :3].T + equations[:, 3]).max(axis=1)
                    gap, nearest = bounded_surface_distance(local, np.array(mesh['vertices']), np.array(mesh['triangles']), margin)
                    signed_gap = np.where(margin <= 0, -gap, gap)
                    closest = int(gap.argmin())
                    band = np.flatnonzero(gap <= .002)
                    jaws[name][mesh['name']] = dict(
                        closest_boundary_node=closest,
                        closest_boundary_material_height_mm=material_height(closest),
                        closest_boundary_margin_mm=float(margin[closest] * 1000),
                        closest_surface_signed_gap_mm=float(signed_gap[closest] * 1000),
                        nearest_hull_surface_world_m=(nearest[closest] @ rotation(pose['quaternion_wxyz']).T + pose['position_m']).tolist(),
                        closest_boundary_world_m=nodes[i][closest].tolist(),
                        minimum_margin_mm=float(margin.min() * 1000),
                        inside_node_count=int((margin <= 0).sum()),
                        nodes_in_2mm_surface_band=band.tolist(),
                        material_heights_in_2mm_surface_band_mm=[material_height(int(j)) for j in band],
                        source_contact_offset_m=mesh['contact_offset_m'],
                        source_rest_offset_m=mesh['rest_offset_m'])
            rows.append(dict(model_step=i, physics_time_s=r['physics_time_s'], jaws=jaws))
    output = dict(response_gain=result['response_gain'],robot=robot,
                  measurement='Exact Euclidean node distance to source convex hull triangle surface; +/-2mm is an analysis band, not a success threshold or measured force. Actual cooked hull may differ. Maximum facet margin retained only as containment diagnostic.',
                  node_material_height='Rest mesh height; world heights rotate with cup.', samples=rows)
    (root / ('collider_analysis.json' if robot=='left' else 'right_collider_analysis.json')).write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(samples=len(rows), response_gain=result['response_gain'])))
    if not online:
        for r in rows:
            if r['model_step'] in (1498, 1500, 1502, 1504, 1506):
                print(json.dumps(r))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--hulls', type=Path, required=True)
    p.add_argument('--robot', choices=('left','right'), default='left')
    a = p.parse_args()
    analyze(a.root, a.hulls, a.robot)
