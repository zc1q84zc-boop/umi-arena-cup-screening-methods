"""Reuse the established GPU robot/controller API with task rigid bodies."""
from __future__ import annotations
import copy
import json
import math
import numpy as np
from ..env import DualFrankaYubiCupPlateEnv, ROBOT_PATHS, ROOT as PACKAGE, as_list
from .catalog import ROOT
from .evaluation import PrimitiveEvaluator, rotation


def sample_layout(task,seed,index,randomize=True):
    result=copy.deepcopy(task['objects'])
    if not randomize:return result
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(index)]))
    limit=task['randomization']['translation_m'];yaw=math.radians(rng.uniform(-task['randomization']['yaw_deg'],task['randomization']['yaw_deg']))
    delta=rng.uniform(-limit,limit,size=2);c,s=math.cos(yaw),math.sin(yaw)
    transform=np.array([[c,-s],[s,c]])
    for obj in result:
        p=np.asarray(obj['position_m'],float);p[:2]=transform@p[:2]+delta;obj['position_m']=p.tolist()
        w,x,y,z=obj['quaternion_wxyz'];cy,sy=math.cos(yaw/2),math.sin(yaw/2)
        obj['quaternion_wxyz']=[cy*w-sy*z,cy*x-sy*y,cy*y+sy*x,cy*z+sy*w]
    return result


class ArenaTaskEnv(DualFrankaYubiCupPlateEnv):
    def __init__(self,task_id,*,render=False,max_steps=1200,randomize=True):
        self.task=json.loads((ROOT/'scenes'/f'{task_id}.json').read_text())
        self.task_id=task_id;self.randomize=randomize;self.evaluator=None
        task_config=json.loads((PACKAGE/'config.json').read_text())
        task_config.update(physics_hz=self.task['physics_hz'],policy_hz=self.task['policy_hz'],max_policy_steps=max_steps)
        grasp_objects={p['object'] for p in self.task['primitives']}
        filters=[path+'/'+finger for path in ROBOT_PATHS.values() for finger in ('yubi_leftfinger','yubi_rightfinger')]
        self.grasp_objects=grasp_objects
        super().__init__(task_config=task_config,render=render,
            object_paths={o['id']:o['prim_path'] for o in self.task['objects']},
            object_contact_filters={name:filters for name in grasp_objects})

    def _resolve_reset_setup(self,setup,*,seed,scenario_index,scene_config):
        if setup is not None:raise ValueError('Arena tasks use their own layout; choose seed/index instead of cup setup')
        # Use the current console's established initial arm posture rather
        # than the folded generic Panda home, preserving wrist viewpoints.
        shared_reset=json.loads((PACKAGE/'setups/online_aligned_v2.json').read_text())
        return dict(id=f'{self.task_id}:seed{seed}:index{scenario_index}',
                    objects=sample_layout(self.task,seed,scenario_index,self.randomize),
                    initial_arm_joint_rad=shared_reset['initial_arm_joint_rad'],initial_gripper_open_fraction=1.)

    def _reset_objects(self,scenario):
        torch=self.torch
        for obj in scenario['objects']:
            view=self.objects[obj['id']]
            view.set_world_poses(positions=torch.tensor([obj['position_m']],dtype=torch.float32,device=self.device),
                                orientations=torch.tensor([obj['quaternion_wxyz']],dtype=torch.float32,device=self.device))
            view.set_velocities(torch.zeros((1,6),dtype=torch.float32,device=self.device))

    def reset(self,*args,**kwargs):
        self.evaluator=None
        observation=super().reset(*args,**kwargs)
        self.evaluator=PrimitiveEvaluator(self.task,observation)
        return observation

    def observe(self):
        state=super().observe()
        for name in self.grasp_objects:
            # Pair forces are telemetry for evaluation, not policy inputs.
            view=self.objects[name]
            # Isaac's numpy backend cannot clone a GPU contact tensor. Copy to
            # the CPU explicitly after fetching the un-cloned PhysX buffer.
            forces=np.asarray(as_list(view.get_contact_force_matrix(clone=False,dt=1/self.task_config['physics_hz']))).reshape(-1,4,3)
            if forces.shape[0]!=1 or not np.isfinite(forces).all():raise RuntimeError('Invalid task contact telemetry')
            pair=np.linalg.norm(forces[0],axis=1)
            state['objects'][name]['finger_contact_forces_N']=pair.tolist()
            state['objects'][name]['grasp_sides']=[side for i,side in enumerate(('left','right'))
                if pair[2*i]>.01 and pair[2*i+1]>.01 and state['robots'][side]['gripper_open_fraction']<.9]
        return state

    def _success(self,observation):
        return False  # The cup evaluator is replaced by primitive grading below.

    def step(self,action,**kwargs):
        state,_,_,truncated,info=super().step(action,**kwargs)
        before=self.evaluator.index;grading=self.evaluator.update(state)
        if truncated and not grading['terminated']:grading=self.evaluator.fail('local_step_budget_exhausted')
        info.update(task=grading,is_success=grading['full_task_success'])
        return state,float(self.evaluator.index-before),grading['terminated'],truncated,info
