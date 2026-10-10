"""Reset-only target-fit/contact tests, explicitly separate from policy success."""
import argparse
import copy
import json
import os
from pathlib import Path
import traceback
import numpy as np
from .catalog import ROOT
from .evaluation import world_points,in_frame,stable,lid_covers_box,rotation


def target_fixture(task):
    value=copy.deepcopy(task);objects={o['id']:o for o in value['objects']};key=task['task_id']
    if key=='pens':
        holder=objects['holder'];base=np.asarray(holder['position_m']);floor=holder['wall_m']
        pens=[o for o in value['objects'] if o['kind']=='pen']
        for i,obj in enumerate(pens):
            x=(i%2-.5)*.024;y=(i//2-1)*.015;r=obj['radius_m'];length=obj['size_m'][0]
            obj['position_m']=(base+[x-r,y,floor+length/2+.001]).tolist()
            obj['quaternion_wxyz']=[np.sqrt(.5),0,np.sqrt(.5),0]
    elif key=='sps':
        tray=objects['sorter'];counts={}
        for step in value['primitives']:
            n=step['cell'];j=counts.get(n,0);counts[n]=j+1
            cell=tray['cells'][str(n)];cx,cy,cz=cell['center_local_m']
            if n in (1,2,3,4,5):dx,dy=0,(-.0195,.0195)[j]
            elif n==7:dx,dy=[(-.025,-.018),(.025,-.018),(0,.020)][j]
            else:dx,dy=0,0
            objects[step['object']]['position_m']=(np.asarray(tray['position_m'])+[cx+dx,cy+dy,cz+.001]).tolist()
    elif key=='phone':
        box=objects['box'];p=np.asarray(box['position_m']);floor=box['wall_m']
        objects['divider']['position_m']=(p+[0,0,floor+.001]).tolist()
        objects['phone']['position_m']=(p+[0,0,floor+objects['divider']['size_m'][2]+.002]).tolist()
        objects['lid']['position_m']=(p+[0,0,.001]).tolist()
    else:raise ValueError('Cable dynamics/insertion needs controlled gripper validation, not a gravity-only fit test')
    return value


def inspect(task,state):
    objects={o['id']:o for o in task['objects']};key=task['task_id'];checks={}
    for obj in task['objects']:
        if obj['id'] in ('holder','sorter','box','lid'):continue
        ostate=state['objects'][obj['id']];pts=world_points(obj,ostate)
        target_name='holder' if key=='pens' else 'sorter' if key=='sps' else 'box'
        target=objects[target_name];local=in_frame(pts,state['objects'][target_name]);lo=local.min(axis=0);hi=local.max(axis=0)
        if key=='sps':
            primitive=next(p for p in task['primitives'] if p['object']==obj['id']);cell=target['cells'][str(primitive['cell'])]
            cx,cy,cz=cell['center_local_m'];sx,sy,sz=cell['size_m']
            contained=np.all(lo[:2]>np.array([cx-sx/2,cy-sy/2])-.001) and np.all(hi[:2]<np.array([cx+sx/2,cy+sy/2])+.001) and lo[2]>cz-.003 and hi[2]<cz+sz+.004
        else:
            sx,sy,sz=target['inner_size_m'];contained=np.all(lo[:2]>-np.array([sx,sy])/2-.001) and np.all(hi[:2]<np.array([sx,sy])/2+.001)
            if key=='phone':
                aligned=rotation(state['objects'][target_name]['quaternion_wxyz']).T@rotation(ostate['quaternion_wxyz'])
                contained=contained and lo[2]>target['wall_m']-.003 and hi[2]<target['size_m'][2]+.003 and aligned[2,2]>np.cos(np.radians(10))
            else:contained=contained and lo[2]>=target['wall_m']-.004
        checks[obj['id']]=dict(contained=bool(contained),stable=bool(stable(ostate)),local_bounds_m=[lo.tolist(),hi.tolist()])
    if key=='phone':
        checks['lid']=dict(contained=lid_covers_box(objects['lid'],state['objects']['lid'],objects['box'],state['objects']['box']),stable=stable(state['objects']['lid']))
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=('pens','sps','phone'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=True,active_gpu=int(os.environ.get('ISAAC_ACTIVE_GPU','0')),physics_gpu=0,multi_gpu=False))
    result=dict(status='starting',task_id=a.task,scope='reset-only physical goal-fit probe; no policy inference or scored trial')
    try:
        import omni.usd
        from .env import ArenaTaskEnv
        context=omni.usd.get_context();context.open_stage(str(ROOT/'scenes'/f'{a.task}.usda'))
        for _ in range(10):app.update()
        env=ArenaTaskEnv(a.task,render=False,randomize=False)
        env.task=target_fixture(env.task);state=env.reset()
        for _ in range(1200):env.world.step(render=False)
        state=env.observe();checks=inspect(env.task,state)
        passed=all(c['contained'] and c['stable'] for c in checks.values())
        result.update(status='ok' if passed else 'failed',checks=checks,gpu_backend=env.backend,
                      settled_physics_steps=1200,final_objects=state['objects'],model_requests=0)
    except Exception as exc:result.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    finally:
        try:(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        finally:app.close()
    print(a.task,result['status'],flush=True)
    return int(result['status']!='ok')

if __name__=='__main__':raise SystemExit(main())
