"""Ground-truth grading only. Nothing in this module produces robot actions."""
from __future__ import annotations
import copy
import math
import numpy as np


def rotation(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-9:
        raise ValueError('Invalid wxyz quaternion')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def corners(bounds):
    lo,hi=bounds
    return np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])


def world_points(meta,state):
    points=np.asarray(meta.get('local_envelope_points_m',corners(meta['local_bounds_m'])),dtype=float)
    return points@rotation(state['quaternion_wxyz']).T+state['position_m']


def in_frame(points,state):
    return (np.asarray(points)-state['position_m'])@rotation(state['quaternion_wxyz'])


def stable(obj):
    return bool(np.linalg.norm(obj['linear_velocity_m_s']) <= .04
                and np.linalg.norm(obj['angular_velocity_rad_s']) <= .35)


def lid_covers_box(lid_meta,lid_state,box_meta,box_state):
    """Pure shape fit; contact manipulation is checked separately by the grader."""
    boxpoints=in_frame(world_points(box_meta,box_state),lid_state)
    lidpoints=in_frame(world_points(lid_meta,lid_state),box_state)
    sx,sy,_=lid_meta['inner_size_m']
    aligned=rotation(box_state['quaternion_wxyz']).T@rotation(lid_state['quaternion_wxyz'])
    return bool(aligned[2,2]>math.cos(math.radians(8)) and abs(aligned[0,0])>math.cos(math.radians(8))
                and np.all(np.abs(boxpoints[:,:2])<np.array([sx,sy])/2+.0006)
                and -.006<=lidpoints[:,2].min()<=.010
                and abs(lidpoints[:,2].max()-(box_meta['size_m'][2]+lid_meta['wall_m']))<.012)


class PrimitiveEvaluator:
    def __init__(self, task, observation):
        self.task=task;self.meta={o['id']:o for o in task['objects']}
        self.index=0;self.streak=0;self.failed=False;self.records=[];self.grasped_sides=set()
        self.start=copy.deepcopy(observation['objects']);self.initial=copy.deepcopy(self.start)
        self.manipulated=False;self.step_started_s=observation['physics_time_s']

    @property
    def primitive(self):
        return self.task['primitives'][self.index] if self.index<len(self.task['primitives']) and not self.failed else None

    @property
    def prompt(self):
        return self.primitive['prompt'] if self.primitive else ''

    def _candidate(self, obs):
        p=self.primitive;name=p['object'];obj=obs['objects'][name];meta=self.meta[name]
        pts=world_points(meta,obj);hand=p['hand'];mode=p['mode']
        sides=set(obj.get('grasp_sides',[]))
        allowed=sides & ({hand} if hand!='either' else {'left','right'})
        # A contact-based grasp and real displacement are required. At reset,
        # an object merely in a goal region must never count as manipulation.
        moved=(np.linalg.norm(np.asarray(obj['position_m'])-self.start[name]['position_m'])>.012
               or np.linalg.norm(rotation(obj['quaternion_wxyz'])-rotation(self.start[name]['quaternion_wxyz']))>.3)
        self.grasped_sides |= allowed
        self.manipulated |= bool(allowed and moved)
        released=(not sides and any(obs['robots'][s]['gripper_open_fraction']>=.65 for s in self.grasped_sides))
        target=obs['objects'].get(p.get('target'));tm=self.meta.get(p.get('target'))
        local=in_frame(pts,target) if target else None
        lo=local.min(axis=0) if target else None;hi=local.max(axis=0) if target else None
        geometry=False;requires_release=True
        if mode=='pen_in':
            sx,sy,sz=tm['inner_size_m'];floor=tm['wall_m']
            upright=abs((rotation(target['quaternion_wxyz']).T@rotation(obj['quaternion_wxyz']))[2,0])>math.cos(math.radians(25))
            geometry=(upright and np.all(lo[:2]>=-np.array([sx,sy])/2+.001)
                      and np.all(hi[:2]<=np.array([sx,sy])/2-.001)
                      and floor-.004<=lo[2]<=floor+.020 and hi[2]>tm['size_m'][2]+.008)
        elif mode=='cell_in':
            cell=tm['cells'][str(p['cell'])];center=np.array(cell['center_local_m']);sx,sy,sz=cell['size_m']
            geometry=(np.all(lo[:2]>=center[:2]-[sx/2,sy/2]+.001)
                      and np.all(hi[:2]<=center[:2]+[sx/2,sy/2]-.001)
                      and lo[2]>=center[2]-.004 and hi[2]<=center[2]+sz+.004)
        elif mode=='box_in':
            sx,sy,sz=tm['inner_size_m'];floor=tm['wall_m']
            geometry=(np.all(lo[:2]>=-np.array([sx,sy])/2+.001)
                      and np.all(hi[:2]<=np.array([sx,sy])/2-.001)
                      and lo[2]>=floor-.004 and hi[2]<=tm['size_m'][2]+.002)
        elif mode=='table_out':
            # Any supported tabletop location outside the holder is valid.
            geometry=(abs(pts[:,2].min()-.75)<.008 and np.all(np.abs(pts[:,:2])<[.39,.62])
                      and (lo[0]>tm['size_m'][0]/2+.006 or hi[0]<-tm['size_m'][0]/2-.006
                           or lo[1]>tm['size_m'][1]/2+.006 or hi[1]<-tm['size_m'][1]/2-.006))
        elif mode=='upright':
            requires_release=False
            geometry=(abs(rotation(obj['quaternion_wxyz'])[2,0])>math.cos(math.radians(15))
                      and abs(pts[:,2].min()-.75)<.015 and hand in sides)
        elif mode=='lift':
            requires_release=False
            geometry=pts[:,2].min()>.75+p['lift_m'] and hand in sides
        elif mode=='lid_closed':
            # Box fits inside the lid's hollow sleeve, and the top panel covers it.
            geometry=lid_covers_box(meta,obj,tm,target)
        elif mode in ('lid_open','box_out'):
            requires_release=False
            separated=(lo[2]>tm['size_m'][2]+.015 or lo[0]>tm['size_m'][0]/2+.010
                       or hi[0]<-tm['size_m'][0]/2-.010 or lo[1]>tm['size_m'][1]/2+.010
                       or hi[1]<-tm['size_m'][1]/2-.010)
            geometry=bool(separated)
        elif mode=='usb_insert':
            requires_release=False;cfg=self.task['socket']
            tip=rotation(obj['quaternion_wxyz'])@np.array([.011,0,.004])+obj['position_m']
            mouth=np.asarray(tm['mouth_local_m']);rel=in_frame([tip],target)[0]-mouth
            alignment=rotation(target['quaternion_wxyz']).T@rotation(obj['quaternion_wxyz'])
            axis=alignment[:,0]
            projected_y=abs(alignment[1,1])*cfg['tongue_y_m']+abs(alignment[1,2])*cfg['tongue_z_m']
            projected_z=abs(alignment[2,1])*cfg['tongue_y_m']+abs(alignment[2,2])*cfg['tongue_z_m']
            # Position-only shortcuts reject rotated plugs and insufficient depth.
            geometry=(cfg['insertion_min_m']<=rel[0]<=cfg['depth_m']+.0002
                      and abs(rel[1])<=cfg['lateral_tolerance_y_m']
                      and abs(rel[2])<=cfg['lateral_tolerance_z_m']
                      and projected_y<=cfg['opening_y_m'] and projected_z<=cfg['opening_z_m']
                      and axis[0]>=math.cos(math.radians(cfg['angle_tolerance_deg'])))
        else:raise ValueError(mode)
        return bool(geometry and self.manipulated and stable(obj) and (released or not requires_release)),dict(
            geometry=bool(geometry),manipulated=self.manipulated,grasped_sides=sorted(self.grasped_sides),released=bool(released),stable=bool(stable(obj)))

    def update(self, observation):
        if not self.primitive:return self.summary()
        candidate,details=self._candidate(observation)
        self.streak=self.streak+1 if candidate else 0
        if self.streak>=self.task['dwell_steps']:
            self.records.append(dict(index=self.index,prompt=self.prompt,status='success',
                                     physics_time_s=observation['physics_time_s'],evidence=details))
            self.index+=1;self.streak=0;self.manipulated=False;self.grasped_sides.clear()
            self.start=copy.deepcopy(observation['objects']);self.step_started_s=observation['physics_time_s']
        return {**self.summary(),'candidate':candidate,'evidence':details}

    def fail(self, reason='operator_failure'):
        if self.primitive:
            self.records.append(dict(index=self.index,prompt=self.prompt,status='failed',reason=reason))
            self.failed=True
        return self.summary()

    def summary(self):
        total=len(self.task['primitives']);successes=sum(r['status']=='success' for r in self.records)
        statuses=['not_attempted']*total
        for rec in self.records:statuses[rec['index']]=rec['status']
        if self.primitive:statuses[self.index]='in_progress'
        return dict(task_id=self.task['task_id'],primitive_index=self.index,current_prompt=self.prompt,
                    primitive_statuses=statuses,successful_primitives=successes,planned_primitives=total,
                    score=successes/total,full_task_success=successes==total,
                    terminated=self.failed or successes==total,records=copy.deepcopy(self.records))
