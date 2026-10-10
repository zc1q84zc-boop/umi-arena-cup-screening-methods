"""Static data figure from completed physics comparisons."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
for gain,color in ((4,'#b2182b'),(8,'#2166ac')):
    rows=json.loads((here/f'gain{gain}/plot_samples.json').read_text())
    rows=[r for r in rows if 140<=r['reference_time_s']<=160]
    t=np.array([r['reference_time_s'] for r in rows])
    axes[0].plot(t,[r['tool_error_mm'] for r in rows],label=f'Gain {gain}',color=color,linewidth=1)
    axes[1].plot(t,[r['cup_tilt_deg'] for r in rows],label=f'Gain {gain}',color=color,linewidth=1)
    axes[2].plot(t,[r['cup_clearance_mm'] for r in rows],label=f'Gain {gain}',color=color,linewidth=1)
axes[0].set_title('Frozen PI0.5 left phase: gain 4 vs 8 | matched reset, 240 Hz / 128 iterations')
axes[0].set_ylabel('Tool target error\n(mm)');axes[0].legend()
axes[1].set_ylabel('Cup tilt\n(deg)');axes[1].axhline(15,color='#777',linestyle=':',linewidth=1)
axes[2].set_ylabel('Lowest-node clearance\n(mm)');axes[2].set_xlabel('Source-aligned time (s); both diagnostics reset at source 87 s')
for ax in axes:ax.grid(alpha=.2)
fig.savefig(here/'gpu_comparison.png',dpi=150);plt.close(fig)
data={}
for gain in (4,8):
    data[gain]=json.loads((here/f'gain{gain}/analysis.json').read_text())
summary=dict(classification='frozen model Cartesian target physics comparison',gains=[4,8],
             windows={},full_task_success_claim=False)
for name in data[4]['windows']:
    a,b=data[4]['windows'][name],data[8]['windows'][name]
    summary['windows'][name]=dict(baseline=a,candidate=b,
        tool_error_mean_reduction_percent=100*(1-b['tool_error_mm']/a['tool_error_mm']),
        governor_joint_mae_reduction_percent=100*(1-b['arm_governor_mae_rad']/a['arm_governor_mae_rad']))
summary['outcomes']={gain:{k:data[gain][k] for k in ('longest_upright_clearance_above10mm_s','first_left_tilt_over30_time_s','peak_left_clearance_mm')} for gain in data}
summary['comparison_repetitions']=1
summary['grip_position_verification']=dict(
    basis='Composed USD convex colliders and actual 769 FEM nodes',
    source_model_step=1502, contact_force_measured=False,
    interpretation='Cup rim proximity persists despite faster response; opposite jaw contacts are geometrically asymmetric. Causality and force are not established.')
for gain in (4,8):
    geometry=json.loads((here/f'gain{gain}/collider_analysis.json').read_text())
    r=next(r for r in geometry['samples'] if r['model_step']==1502)
    summary['grip_position_verification'][f'gain{gain}']={
        f'{side}_{mesh}_closest_material_height_mm':r['jaws'][side][mesh]['closest_boundary_material_height_mm']
        for side,mesh in (('left_finger','TexturedContact'),('right_finger','TexturedContact'),('right_finger','ProximalCollision'))}
summary['physics_and_safety']=dict(physics_hz=240,solver_iterations=128,velocity_limit_rad_s=.8,
    acceleration_limit_rad_s2=1.5,camera_profiles_changed=False,model_targets_modified_in_frozen_trial=False)
(here/'gpu_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
