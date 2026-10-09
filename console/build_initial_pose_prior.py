"""Select a real first-frame medoid; never average incompatible arm poses."""
import json
from pathlib import Path
import numpy as np
from online_calibration import MirroredReplayPrior

def build():
    root = Path(__file__).resolve().parent
    calibration = MirroredReplayPrior()
    samples = []
    for episode in [61164, 136238, 231149, 259632, 262232]:
        data = json.loads((root / f'dataset_replay/official_cup_5/episode-{episode}.json').read_text())
        obs = data['frames'][0]['observation']
        arms = {}
        for i, side in enumerate(['left', 'right']):
            p = obs['poses_xyzw'][i]
            position, quaternion = calibration.hand_to_world_tool(side, p[:3], [p[6], *p[3:6]])
            arms[side] = dict(position_m=position.tolist(), quaternion_wxyz=quaternion.tolist(),
                             gripper_open_fraction=calibration.sim_gripper(obs['grippers_rad'][i]))
        samples.append(dict(episode=episode, source_observation=obs, targets=arms))
    def distance(a, b):
        total = 0.
        for side in ['left', 'right']:
            x, y = a['targets'][side], b['targets'][side]
            total += np.linalg.norm(np.array(x['position_m'])-y['position_m']) / .1
            total += 2*np.arccos(np.clip(abs(np.dot(x['quaternion_wxyz'], y['quaternion_wxyz'])),0,1))
        return total
    costs = [sum(distance(a,b) for b in samples) for a in samples]
    chosen = samples[int(np.argmin(costs))]
    return dict(schema_version=1, id='official_five_first_frame_medoid_20260929', measured=False,
                purpose='approximate simulation initialization only; not a performance evaluation',
                calibration=calibration.audit(), selected=chosen, medoid_costs=costs, candidates=samples,
                safeguards=dict(max_steps=160, position_tolerance_m=.02, orientation_tolerance_deg=10,
                                minimum_tool_separation_m=.18, minimum_tool_height_m=.86))

if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
