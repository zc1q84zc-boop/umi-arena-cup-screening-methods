import copy
import math
import unittest
import numpy as np
from yubi_isaac_sim_env.arena_tasks.catalog import load_task,task_catalog
from yubi_isaac_sim_env.arena_tasks.evaluation import PrimitiveEvaluator,world_points,lid_covers_box
from yubi_isaac_sim_env.arena_tasks.env import sample_layout
from yubi_isaac_sim_env.arena_tasks.interface import observation,action_chunk,KEYS
from yubi_isaac_sim_env.umi_gripper_mapping import open_fraction_to_source_angle


def fixture(key):
    task=load_task(key)
    for obj in task['objects']:
        x,y,z=obj['size_m'];obj['local_bounds_m']=[[-x/2,-y/2,0],[x/2,y/2,z]]
        if 'wall_m' in obj:obj['inner_size_m']=[x-2*obj['wall_m'],y-2*obj['wall_m'],z-obj['wall_m']]
        if obj['kind']=='sorter':
            t=obj['wall_m'];obj['cells']={}
            for n in range(1,10):
                r,c=divmod(n-1,3);obj['cells'][str(n)]=dict(center_local_m=[x/2-(r+.5)*x/3,y/2-(c+.5)*y/3,t],size_m=[x/3-2*t,y/3-2*t,z-t])
        if obj['kind']=='socket':obj['mouth_local_m']=[-x/2,0,.006]
    state=dict(physics_time_s=0.,objects={o['id']:dict(position_m=o['position_m'][:],quaternion_wxyz=o['quaternion_wxyz'][:],linear_velocity_m_s=[0,0,0],angular_velocity_rad_s=[0,0,0],grasp_sides=[]) for o in task['objects']},robots={s:dict(gripper_open_fraction=1.) for s in ('left','right')})
    return task,state


def move(state,name,position,hand='right',quaternion=None):
    obj=state['objects'][name];obj['position_m']=position;obj['grasp_sides']=[hand]
    state['robots'][hand]['gripper_open_fraction']=.3
    if quaternion is not None:obj['quaternion_wxyz']=quaternion


def dwell(ev,state):
    for i in range(5):state['physics_time_s']+=.1;ev.update(state)


class ArenaTests(unittest.TestCase):
    def test_actual_shape_envelope_avoids_empty_bbox_corners(self):
        meta=dict(local_bounds_m=[[0,0,0],[1,1,1]],local_envelope_points_m=[[0,0,0],[1,0,0],[0,1,1]])
        state=dict(position_m=[2,3,4],quaternion_wxyz=[1,0,0,0])
        np.testing.assert_equal(world_points(meta,state),[[2,3,4],[3,3,4],[2,4,5]])

    def test_detachable_lid_fit_accepts_half_turn_rejects_sideways(self):
        task,state=fixture('phone');objects={o['id']:o for o in task['objects']}
        state['objects']['lid']['position_m']=state['objects']['box']['position_m'][:]
        args=(objects['lid'],state['objects']['lid'],objects['box'],state['objects']['box'])
        self.assertTrue(lid_covers_box(*args))
        state['objects']['lid']['quaternion_wxyz']=[0,0,0,1]
        self.assertTrue(lid_covers_box(*args))
        state['objects']['lid']['quaternion_wxyz']=[math.sqrt(.5),0,0,math.sqrt(.5)]
        self.assertFalse(lid_covers_box(*args))

    def test_official_counts_and_order(self):
        self.assertEqual([t['primitive_count'] for t in task_catalog()],[12,15,1,8])
        t=load_task('phone');self.assertEqual([p['hand'] for p in t['primitives']],['right','left','right','right','right','right','right','right'])
        self.assertEqual([p['cell'] for p in load_task('sps')['primitives']],[1,1,2,2,3,3,4,4,5,5,6,7,7,7,8])

    def test_full_task_does_not_succeed_on_reset(self):
        for key in ('pens','sps','cable','phone'):
            t,s=fixture(key);ev=PrimitiveEvaluator(t,s);dwell(ev,s)
            self.assertEqual(ev.index,0);self.assertEqual(ev.summary()['score'],0)

    def test_requires_grasp_and_release(self):
        t,s=fixture('phone');ev=PrimitiveEvaluator(t,s);box=s['objects']['box']['position_m']
        position=[box[0],box[1],box[2]+.003]
        move(s,'divider',position);dwell(ev,s);self.assertEqual(ev.index,0)
        s['objects']['divider']['grasp_sides']=[];s['robots']['right']['gripper_open_fraction']=1.
        dwell(ev,s);self.assertEqual(ev.index,1)
        self.assertIn('left gripper',ev.prompt)

    def test_goal_teleport_without_contact_not_counted(self):
        t,s=fixture('phone');ev=PrimitiveEvaluator(t,s)
        s['objects']['divider']['position_m']=[.02,.16,.755]
        dwell(ev,s);self.assertEqual(ev.index,0)

    def test_wrong_hand_cannot_complete_phone_step(self):
        t,s=fixture('phone');ev=PrimitiveEvaluator(t,s);ev.index=1
        move(s,'phone',[-.14,-.055,.833],'right',[math.sqrt(.5),0,math.sqrt(.5),0]);dwell(ev,s)
        self.assertEqual(ev.index,1)
        s['objects']['phone']['grasp_sides']=['left'];s['robots']['left']['gripper_open_fraction']=.3
        dwell(ev,s);self.assertEqual(ev.index,2)

    def test_lying_pen_on_holder_rim_rejected(self):
        t,s=fixture('pens');ev=PrimitiveEvaluator(t,s)
        move(s,'pencil_1',[.17,0,.852]);ev.update(s)
        s['objects']['pencil_1']['grasp_sides']=[];s['robots']['right']['gripper_open_fraction']=1.
        dwell(ev,s);self.assertEqual(ev.index,0)

    def test_upright_pen_insert_and_return_use_same_object(self):
        t,s=fixture('pens');ev=PrimitiveEvaluator(t,s)
        move(s,'pencil_1',[.17,0,.8425],quaternion=[math.sqrt(.5),0,math.sqrt(.5),0]);ev.update(s)
        s['objects']['pencil_1']['grasp_sides']=[];s['robots']['right']['gripper_open_fraction']=1.
        dwell(ev,s);self.assertEqual(ev.index,1)
        ev.index=6;ev.start=copy.deepcopy(s['objects']);ev.manipulated=False;ev.grasped_sides.clear()
        move(s,'pencil_1',[-.13,-.16,.753],quaternion=[1,0,0,0]);ev.update(s)
        s['objects']['pencil_1']['grasp_sides']=[];s['robots']['right']['gripper_open_fraction']=1.
        dwell(ev,s);self.assertEqual(ev.index,7)

    def test_wrong_cell_and_overhanging_part_rejected(self):
        t,s=fixture('sps');ev=PrimitiveEvaluator(t,s);tray=s['objects']['sorter']['position_m']
        move(s,'finger_1',[tray[0],tray[1],tray[2]+.003]);ev.update(s)
        s['objects']['finger_1']['grasp_sides']=[];s['robots']['right']['gripper_open_fraction']=1.
        dwell(ev,s);self.assertEqual(ev.index,0)
        s['objects']['finger_1']['position_m']=[tray[0]+.12,tray[1]+.0867,tray[2]+.003]
        dwell(ev,s);self.assertEqual(ev.index,1)

    def test_socket_rejects_lateral_error_and_rotated_connector(self):
        t,s=fixture('cable');ev=PrimitiveEvaluator(t,s)
        move(s,'usb_plug',[.014,.024,.786]);dwell(ev,s);self.assertEqual(ev.index,0)
        s['objects']['usb_plug']['position_m']=[.014,.025,.786]
        s['objects']['usb_plug']['quaternion_wxyz']=[math.sqrt(.5),0,0,math.sqrt(.5)]
        dwell(ev,s);self.assertEqual(ev.index,0)
        s['objects']['usb_plug']['position_m']=[.014,.029,.790]
        s['objects']['usb_plug']['quaternion_wxyz']=[math.sqrt(.5),math.sqrt(.5),0,0]
        dwell(ev,s);self.assertEqual(ev.index,0)
        s['objects']['usb_plug']['position_m']=[.014,.025,.786]
        s['objects']['usb_plug']['quaternion_wxyz']=[1,0,0,0]
        dwell(ev,s);self.assertTrue(ev.summary()['full_task_success'])

    def test_failure_keeps_remaining_steps_in_denominator(self):
        t,s=fixture('sps');ev=PrimitiveEvaluator(t,s)
        ev.records=[dict(index=i,status='success') for i in range(8)];ev.index=8
        out=ev.fail('failed insertion');self.assertEqual(out['score'],8/15)
        self.assertEqual(out['primitive_statuses'][9:],['not_attempted']*6)
        self.assertTrue(out['terminated'])

    def test_reset_randomization_reproducible(self):
        t=load_task('cable')
        self.assertEqual(sample_layout(t,42,0),sample_layout(t,42,0))
        self.assertNotEqual(sample_layout(t,42,0),sample_layout(t,42,1))
        self.assertEqual(sample_layout(t,42,0,False),t['objects'])


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.state={'objects':{'secret':{'future_pose':1}},'scenario':{'private':1},'robots':{s:dict(tool_body_name='yubi_tool',tool_pose=dict(position_m=[0,.3*i,1],quaternion_wxyz=[1,0,0,0]),gripper_open_fraction=.6) for i,s in enumerate(('left','right'))}}
        self.images={s+'_wrist':np.zeros((480,640,3),np.uint8) for s in ('left','right')}

    def test_exact_official_inputs_no_oracle_or_center(self):
        self.images['head']=np.zeros((480,640,3),np.uint8)
        out=observation(self.state,self.images,'next primitive')
        self.assertEqual(set(out),KEYS);self.assertEqual(out['observation.joint_states'].shape,(2,))
        self.assertEqual(out['observation.pose.left_hand_root_to_right_hand_root.absolute'].shape,(7,))
        self.assertEqual(out['prompt'],'next primitive')
        out['observation.image.left'][0,0]=1;self.assertEqual(self.images['left_wrist'][0,0,0],0)

    def test_identity_action_roundtrip_and_single_integration(self):
        a=np.zeros((16,16));a[:,6]=a[:,13]=1;a[:,14:]=open_fraction_to_source_angle(.6)
        out=action_chunk(a,self.state,1)
        for side in ('left','right'):
            self.assertTrue(np.allclose(out['waypoints'][0][side]['position_m'],self.state['robots'][side]['tool_pose']['position_m']))
        a[0,0]=.01;out=action_chunk(a,self.state,1)
        self.assertAlmostEqual(np.linalg.norm(np.array(out['waypoints'][0]['left']['position_m'])-[0,0,1]),.01)
        self.assertEqual(len(out['waypoints']),1)

    def test_chunk_and_image_contract_errors(self):
        for a in (np.zeros((1,16)),np.zeros((16,15)),np.full((16,16),np.nan)):
            with self.assertRaises(ValueError):action_chunk(a,self.state)
        self.images['left_wrist']=np.zeros((240,320,3),np.uint8)
        with self.assertRaises(ValueError):observation(self.state,self.images,'test')

if __name__=='__main__':unittest.main()
