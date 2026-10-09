"""Offline rigid-cup grasp evidence; never changes model inputs or actions."""
import argparse
import json
import math
from pathlib import Path


def evaluate(rows, table_z=.75, bottom_radius=.027, top_radius=.04, height=.075):
    # Exactly one endpoint per independent 30Hz servo action, not every 60Hz
    # physics sample and not duplicate held 10Hz endpoint predictions.
    endpoints={}
    for row in rows:
        if row['request_step']>=0:endpoints[row['action_index']]=row
    samples=[endpoints[key] for key in sorted(endpoints)]
    result={side:{'longest_stable_actions':0,'current':0,'bilateral_actions':0,
                  'bilateral_actions_above_5mm':0,'bilateral_actions_above_50mm':0,
                  'peak_finger_force_N':{'left_finger':0.,'right_finger':0.,'base':0.},
                  'force_weighted_contact_height_above_cup_bottom_mm':{},
                  'stable_start_action':None} for side in ('left','right')}
    contact_heights={side:{part:[0.,0.] for part in ('left_finger','right_finger','base')}
                     for side in result}
    peak_clearance=-math.inf;peak_origin=-math.inf;last=None;peak_record=None
    for row in samples:
        p=row['cup_position_m'];w,x,y,z=row['cup_quaternion_wxyz']
        norm=math.sqrt(w*w+x*x+y*y+z*z)
        assert abs(norm-1)<.01
        cosine=1-2*(x*x+y*y)/(norm*norm)
        tilt=math.degrees(math.acos(max(-1.,min(1.,cosine))))
        sine=math.sqrt(max(0.,1-cosine*cosine))
        # Actual lowest outer point of the cup's two frustum rim disks.
        clearance=min(p[2]-bottom_radius*sine,p[2]+height*cosine-top_radius*sine)-table_z
        if clearance>peak_clearance:
            peak_clearance=clearance
            peak_record={'action_index':row['action_index'],'request_step':row['request_step'],
                         'time_s':row['physics_time_s'],'clearance_mm':clearance*1000,
                         'tilt_deg':tilt,'cup_position_m':p}
        peak_origin=max(peak_origin,p[2]-table_z)
        continuous=(last is not None and row['action_index']==last['action_index']+1
                    and abs(row['physics_time_s']-last['physics_time_s']-1/30)<1e-5)
        if continuous:
            dt=row['physics_time_s']-last['physics_time_s']
            speed=math.dist(p,last['cup_position_m'])/dt
            q0=last['cup_quaternion_wxyz'];q1=row['cup_quaternion_wxyz']
            dot=abs(sum(a*b for a,b in zip(q0,q1)))
            angular_speed=2*math.acos(min(1.,dot))/dt
        else:speed=angular_speed=math.inf
        for side in result:
            state=result[side];contacts=row['robots'][side]['cup_contacts']
            forces={part:contacts[part]['positive_normal_force_sum_N'] for part in state['peak_finger_force_N']}
            for part,force in forces.items():state['peak_finger_force_N'][part]=max(state['peak_finger_force_N'][part],force)
            both=all(forces[part]>.02 for part in ('left_finger','right_finger'))
            state['bilateral_actions']+=int(both)
            state['bilateral_actions_above_5mm']+=int(both and clearance>=.005)
            state['bilateral_actions_above_50mm']+=int(both and clearance>=.05)
            # Cup-local vertical axis, so a tilted rim is not mislabeled body.
            axis=[2*(x*z+w*y)/norm**2,2*(y*z-w*x)/norm**2,cosine]
            for part in contact_heights[side]:
                for contact in contacts[part].get('contacts',[]):
                    force=max(0.,contact['normal_force_N'])
                    local_z=sum(a*(v-origin) for a,v,origin in zip(axis,contact['point_world_m'],p))
                    contact_heights[side][part][0]+=local_z*force
                    contact_heights[side][part][1]+=force
            stable=(continuous and clearance>=.05 and tilt<=15 and speed<=.05
                    and angular_speed<=.3 and both and forces['base']<=.02
                    and not any(contacts[part]['buffer_truncated'] for part in contacts))
            state['current']=state['current']+1 if stable else 0
            if state['current']>state['longest_stable_actions']:
                state['longest_stable_actions']=state['current']
                state['stable_start_action']=row['action_index']-state['current']+1
        last=row
    for side,state in result.items():
        state.pop('current')
        state['force_weighted_contact_height_above_cup_bottom_mm']={
            part:weighted/total*1000 if total else None
            for part,(weighted,total) in contact_heights[side].items()}
    passed=any(state['longest_stable_actions']>=10 for state in result.values())
    return {'independent_30hz_actions':len(samples),'peak_lowest_cup_clearance_mm':peak_clearance*1000 if samples else None,
            'peak_cup_origin_height_above_table_mm':peak_origin*1000 if samples else None,
            'peak_clearance_sample':peak_record,
            'arms':result,'numeric_stable_grasp':passed,'visual_review_required':passed,
            'criteria':{'minimum_clearance_mm':50,'max_tilt_deg':15,'max_speed_m_s':.05,
                        'max_angular_speed_rad_s':.3,'min_independent_actions':10,
                        'both_finger_force_N':.02,'max_base_support_force_N':.02},
            'assumptions':{'rigid_cup_only':True,'table_top_z_m':table_z,'bottom_radius_m':bottom_radius,
                           'top_radius_m':top_radius,'height_m':height,'real_hardware_calibration':False},
            'task_or_placement_success_claimed':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=json.loads((args.run/'report.json').read_text())
    assert report['status'] in ('completed','stopped')
    assert 'cup_physics_profile' not in report,'Use the nodal PVC verifier for deformable cups'
    rows=[json.loads(line) for line in (args.run/'gripper_contact_audit.jsonl').read_text().splitlines()]
    result=evaluate(rows)
    result.update(run_id=args.run.name,report_status=report['status'],
                  full_task_success=report['episodes'][0]['full_task_success'],
                  plate_placed=report['episodes'][0]['plate_placed'])
    if args.output:
        with args.output.open('x') as out:out.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
