"""Export only a converged diagnostic state, with explicit fit provenance."""
import json
from pathlib import Path
import numpy as np

def build():
    root=Path(__file__).resolve().parent
    prior=json.loads((root/'initial_pose_prior.json').read_text())
    rows=[json.loads(l) for l in (root/'sim_validation/initial_pose_fit_20260929/initial_pose_fit.jsonl').read_text().splitlines()]
    row=rows[-1]
    for side in ['left','right']:
        assert row['errors'][side]['position_m']<.01
        assert row['errors'][side]['orientation_deg']<2
    assert row['errors']['tool_separation_m']>.18 and row['errors']['minimum_tool_height_m']>.86
    joints={}
    for side,robot in row['robots'].items():
        q=[robot['joint_positions'][robot['joint_names'].index(f'panda_joint{i}')] for i in range(1,8)]
        limits=np.asarray(robot['arm_joint_limits_rad'])
        assert np.all(np.array(q)>=limits[:,0]+.06) and np.all(np.array(q)<=limits[:,1]-.06)
        joints[side]=q
    return dict(schema_version=1,id='cup_demo_medoid_initial_state_20260929',simulation_fit_verified=True,
                physical_calibration_measured=False,source_episode=prior['selected']['episode'],source_frame=0,
                source_calibration=prior['calibration'],arm_joint_positions_rad=joints,
                gripper_open_fraction={s:prior['selected']['targets'][s]['gripper_open_fraction'] for s in joints},
                world_tool_targets=prior['selected']['targets'],fit_errors=row['errors'],
                caveat='simulation-only approximate initialization; no full-mesh collision certification, human elbow reconstruction, or grasp success claim')

if __name__=='__main__': print(json.dumps(build(),indent=2))
