"""Create explicit, reviewable task fixtures from the official open-task order."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'simulator_profiles/arena_tasks_v1/yubi_isaac_sim_env/arena_tasks/configs'
T=.75

def body(name,kind,xy,size,mass=.02,color=(.15,.15,.15),**extra):
 return dict(id=name,kind=kind,position_m=[*xy,T+.002],quaternion_wxyz=[1,0,0,0],size_m=size,mass_kg=mass,color=list(color),**{'dynamic':True,**extra})
def step(prompt,obj,mode,target=None,hand='either',**extra):
 return dict(prompt=prompt,object=obj,mode=mode,target=target,hand=hand,**extra)
def task(n,name,title,objects,steps,**extra):
 return dict(schema_version=1,task_id=name,official_task_number=n,label=title,source_url='https://umi-arena.airoa.io/evaluation',reference_video=f'https://umi-arena.airoa.io/videos/task_{n}.mp4',reference_checked_date='2026-10-10',geometry_status='CAD where identified; other dimensions/layout are demo-based estimates, not measured calibration',physics_status='mass/friction and contact tolerances are provisional',physics_hz=120,policy_hz=10,dwell_steps=5,objects=objects,primitives=steps,body_numerics=dict(position_iterations=64,velocity_iterations=8,sleep_threshold_m2_s2=.0001,stabilization_threshold_m2_s2=.00001),randomization=dict(translation_m=.012,yaw_deg=5,scope='rigid layout perturbation per episode; cable chain and plug move together'),**extra)
objects=[body('holder','open_box',[.17,0],[.065,.065,.100],.06,[.012,.012,.012],wall_m=.003)]
steps=[]
for k,length,radius,color in [('pencil',.175,.0035,[.65,.42,.10]),('ballpoint',.145,.005,[.80,.80,.82]),('marker',.140,.007,[.55,.018,.025])]:
 for j in range(2):objects.append(body(f'{k}_{j+1}','pen',[-.13,{'pencil':-.14,'ballpoint':0,'marker':.14}[k]+(j-.5)*.035],[length,2*radius,2*radius],.007 if k=='pencil' else .014,color,radius_m=radius))
 for j in range(2):steps.append(step(f'Pick up a {k if k!="ballpoint" else "ballpoint pen"} on the table and put it into the pen holder',f'{k}_{j+1}','pen_in','holder'))
for k in ('pencil','ballpoint','marker'):
 for j in range(2):steps.append(step(f'Take a {k if k!="ballpoint" else "ballpoint pen"} out of the pencil holder and put it on the table',f'{k}_{j+1}','table_out','holder'))
a=task(2,'pens','任务 2 · 笔筒取放',objects,steps)
objects=[body('sorter','sorter',[0,-.23],[.36,.26,.065],.22,[.015,.32,.62],wall_m=.003,grid=[3,3])];steps=[]
groups=[('finger',2,1),('flap',2,2),('gear',2,3),('shaft',2,4),('gear_box',2,5),('quest_holder',1,6),('camera_bracket',3,7),('grip',1,8)]
idx=0
for kind,count,cell in groups:
 for j in range(count):
  name=f'{kind}_{j+1}'; xy=[-.27+(idx//3)*.135,.04+(idx%3)*.15];idx+=1
  color=[.32,.015,.035] if kind=='finger' else [.06,.055,.06] if kind not in ('gear','shaft') else [.62,.65,.66]
  asset=kind+'_left' if j==1 and kind in ('finger','flap') else kind
  objects.append(body(name,'cad' if kind!='gear' else 'gear',xy,[.06,.04,.02],.016,color,cad_asset=asset if kind!='gear' else None))
  ordinal=('1st','2nd','3rd')[j]
  part={'quest_holder':'QUEST_HOLDER','gear_box':'GEAR BOX','camera_bracket':'CAMERA_BRACKET'}.get(kind,kind.upper())
  prompt=f'Pick {ordinal+" " if count>1 else ""}{part} part and place into cell {cell}'
  steps.append(step(prompt,name,'cell_in','sorter',cell=cell))
b=task(3,'sps','任务 3 · YUBI 零件分拣',objects,steps)
objects=[body('yubi_fixture','fixture',[.03,-.09],[.11,.08,.08],.20,[.045,.04,.045],dynamic=False),
 body('usb_plug','usb_plug',[-.10,.07],[.022,.012,.008],.012,[.12,.12,.12]),
 body('socket','socket',[.03,-.025],[.018,.016,.012],.004,[.62,.64,.66],dynamic=False)]
# Socket mouth faces -X. Plugin tongue local +X: end at x=+11 mm.
objects[-1]['position_m'][2]=T+.034
c=task(4,'cable','任务 4 · USB-C 插接',objects,[step('Pick up the cables and insert the USB-C cable into microcontroller board of YUBI','usb_plug','usb_insert','socket')],cable=dict(length_m=.48,radius_m=.002,segments=32,mass_kg=.028,bend_limit_deg=38,swing_limit_deg=38,model='rigid capsule chain with limited spherical joints; not an elastic continuum'),socket=dict(opening_y_m=.0092,opening_z_m=.0036,tongue_y_m=.0084,tongue_z_m=.0026,depth_m=.006,insertion_min_m=.0035,lateral_tolerance_y_m=.00035,lateral_tolerance_z_m=.0004,angle_tolerance_deg=12))
objects=[body('box','open_box',[.02,.16],[.190,.104,.046],.045,[.86,.86,.83],wall_m=.0025),body('lid','lid',[.21,-.06],[.198,.112,.050],.035,[.82,.035,.045],wall_m=.0025),body('divider','divider',[.22,.17],[.175,.089,.017],.012,[.91,.91,.88],wall_m=.002),body('phone','phone',[-.14,-.055],[.166,.078,.009],.190,[.012,.016,.02])]
steps=[step('Pick up and insert the divider into the box with the right gripper','divider','box_in','box','right'),step('Stand the smartphone upright with the left gripper','phone','upright',hand='left'),step('Pick up and lift the smartphone with the right gripper','phone','lift',hand='right',lift_m=.05),step('Place the smartphone into the box with the right gripper','phone','box_in','box','right'),step('Close the lid with the right gripper','lid','lid_closed','box','right'),step('Open the lid with the right gripper','lid','lid_open','box','right'),step('Take the smartphone out of the box with the right gripper','phone','box_out','box','right'),step('Remove the divider with the right gripper','divider','box_out','box','right')]
d=task(5,'phone','任务 5 · 手机装盒',objects,steps,lid_model='detachable sleeve, no hinge, as seen in the official clip')
ROOT.mkdir(parents=True,exist_ok=True)
for x in (a,b,c,d):
 # The current calibrated head view maps world +Y to image left. Match the
 # official clips without reflecting either robot or its policy pose frame.
 for obj in x['objects']:obj['position_m'][1]*=-1
 x['layout_camera_convention']='world +Y projects to image left with the shared head calibration'
 p=ROOT/f'{x["task_id"]}.json';p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');print(p.name,len(x['primitives']))
