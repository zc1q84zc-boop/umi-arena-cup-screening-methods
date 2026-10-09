"""Current dual-wrist input; one future 100 ms action per observation."""
import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.request import Request,urlopen
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from pi05_isaac_online_adapter import (current_robot,encode_wrist,rotate,conj,mul,normalize,
    _source_from_fraction,_fraction_from_source,CALIBRATION,
    SIM_GRIPPER_CLOSED_RAD,SIM_GRIPPER_OPEN_RAD)
from intersection_prompts import cup_instruction,PROMPT_SOURCE,PROMPT_PROTOCOL

MODEL=os.environ['INTERSECTION_MODEL_ID']
URL=os.environ['INTERSECTION_ONLINE_URL']
AUDIT=Path(os.environ['SIM_ADAPTER_AUDIT_DIR'])


def waypoint10hz(current,actions):
    assert actions.shape==(1,16) and np.isfinite(actions).all()
    result={};diagnostics={}
    for side,offset,jaw in [('left',0,14),('right',7,15)]:
        position,quaternion,grip=current[side]
        delta=actions[0,offset:offset+3]
        if np.linalg.norm(delta)>.08:raise ValueError(f'{side} future 100ms translation exceeds 8 cm')
        xyzw=actions[0,offset+3:offset+7];dq=normalize([xyzw[3],*xyzw[:3]])
        angle=2*math.acos(min(1,abs(float(dq[0]))))
        if angle>1.05:raise ValueError(f'{side} future 100ms rotation exceeds 1.05 rad')
        target_p=position+rotate(quaternion,delta);target_q=normalize(mul(quaternion,dq))
        target_grip=_fraction_from_source(actions[0,jaw])
        target_grip=min(grip+.25,max(grip-.25,target_grip))
        before=position.copy()
        if CALIBRATION:
            target_p,target_q=CALIBRATION.hand_to_world_tool(side,target_p,target_q)
            before,_=CALIBRATION.hand_to_world_tool(side,position,quaternion)
        distance=float(np.linalg.norm(target_p-before))
        if distance>.08:raise ValueError(f'{side} calibrated 100ms tool displacement exceeds 8 cm')
        result[side]={'position_m':target_p.tolist(),'quaternion_wxyz':target_q.tolist(),
            'gripper_open_fraction':target_grip}
        diagnostics[side]={'world_position_delta_m':distance,'rotation_delta_rad':angle,
            'source_target_gripper_rad':float(actions[0,jaw]),'future_rows_consumed':1}
    return result,diagnostics


def predict(observation,step,episode):
    started=time.monotonic();prompt=cup_instruction(observation['task_stage'])
    current={side:current_robot(observation,side) for side in ('left','right')}
    lp,lq,lg=current['left'];rp,rq,rg=current['right']
    relative_p=rotate(conj(lq),rp-lp);relative_q=normalize(mul(conj(lq),rq))
    jpg={side:encode_wrist(observation['images'][side+'_wrist'],side) for side in ('left','right')}
    payload={'left_jpeg':base64.b64encode(jpg['left']).decode('ascii'),
        'right_jpeg':base64.b64encode(jpg['right']).decode('ascii'),
        'relative_pose_xyzw':[*relative_p.tolist(),*relative_q[1:].tolist(),float(relative_q[0])],
        'gripper_rad':[_source_from_fraction(lg),_source_from_fraction(rg)],
        'episode':int(episode),'step':int(step),'prompt':prompt}
    with urlopen(Request(URL,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=180) as response:
        result=json.load(response)
    assert result['model']==MODEL and result['episode']==episode and result['step']==step
    assert result['prompt']==prompt and result['prompt_source']==PROMPT_SOURCE and result['prompt_protocol']==PROMPT_PROTOCOL
    timing=result['action_timing']
    assert timing['action_hz']==10 and timing['pose_rows']==timing['gripper_rows']==[0]
    assert result['future_observation_used'] is False
    actions=np.asarray(result['actions'],dtype=np.float64)
    assert actions.shape==(1,16) and np.isfinite(actions).all()
    # Integration uses only row0 once. The following two servo ticks hold its
    # endpoint; neither additional future rows nor another 3x composition run.
    waypoint,diagnostics=waypoint10hz(current,actions)
    waypoints=[copy.deepcopy(waypoint) for _ in range(3)]
    AUDIT.mkdir(parents=True,exist_ok=True)
    audit={'episode':int(episode),'step':int(step),'model':MODEL,'prompt':prompt,
        'prompt_source':PROMPT_SOURCE,'prompt_protocol':PROMPT_PROTOCOL,
        'task_stage':observation['task_stage'],'model_input_views':['left_wrist','right_wrist'],
        'server_prompt':result['prompt'],'inference_mode':'future_aligned_10hz_row0',
        'observation_origin':'current_simulator_render_and_robot_state',
        'image_metadata':observation.get('image_metadata',{}),
        'image_shapes':{side:[480,640,3] for side in ('left','right')},
        'relative_pose_xyzw':payload['relative_pose_xyzw'],'gripper_rad':payload['gripper_rad'],
        'image_sha256':{s:hashlib.sha256(jpg[s]).hexdigest() for s in jpg},
        'actions':actions.tolist(),'waypoint':waypoint,'mapping':diagnostics,
        'substep_waypoints':waypoints,
        'gripper_calibration':{**(CALIBRATION.gripper if CALIBRATION else {}),
            'sim_closed_rad':SIM_GRIPPER_CLOSED_RAD,'sim_open_rad':SIM_GRIPPER_OPEN_RAD},
        'action_timing':timing,'model_request_hz':10,'target_update_hz':10,'execution_hz':30,
        'future_observation_used':False,'calibration':CALIBRATION.audit() if CALIBRATION else None,
        'model_latency_ms':result['latency_ms'],'total_adapter_latency_ms':(time.monotonic()-started)*1000}
    with (AUDIT/'online_adapter.jsonl').open('a') as stream:stream.write(json.dumps(audit)+'\n')
    if step<3:
        for side,raw in jpg.items():(AUDIT/f'input_{side}_{step:04d}.jpg').write_bytes(raw)
    return {'action_dt_s':1/30,'waypoints':waypoints,'execute_steps':3}
