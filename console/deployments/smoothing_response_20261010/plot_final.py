"""Frozen-target response comparison, including deformation tradeoffs."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
profiles=[('gain4','Arm4 / jaw4','#555555'),('gain8','Arm8 / jaw8','#b2182b'),
          ('arm8_jaw4','Arm8 / jaw4','#2166ac')]
fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True,constrained_layout=True)
metrics=[('tool_error_mm','Tool target error (mm)'),('cup_tilt_deg','Cup tilt (deg)'),
         ('cup_clearance_mm','Lowest-node clearance (mm)'),('shape_change_mm','Peak nodal shape change (mm)')]
summary=dict(classification='Frozen model Cartesian targets, matched left-phase reset',
             physics_hz=240,solver_iterations=128,comparison_repetitions=1,full_task_success_claim=False,
             configurations={})
for folder,label,color in profiles:
    rows=json.loads((here/folder/'plot_samples.json').read_text())
    rows=[r for r in rows if 146<=r['reference_time_s']<=158]
    for ax,(key,ylabel) in zip(axes,metrics):
        ax.plot([r['reference_time_s'] for r in rows],[r[key] for r in rows],label=label,color=color,linewidth=1.1)
        ax.set_ylabel(ylabel);ax.grid(alpha=.2)
    a=json.loads((here/folder/'analysis.json').read_text())
    summary['configurations'][folder]={k:a[k] for k in ('windows','longest_upright_clearance_above10mm_s',
        'peak_left_clearance_mm','peak_shape_change_mm','peak_rim_compression_percent')}
axes[0].legend(ncol=3);axes[0].set_title('PI0.5 frozen left phase: tracking and grasp tradeoffs | 240 Hz / 128 iterations')
axes[1].axhline(15,color='#999',linestyle=':',linewidth=1)
axes[-1].set_xlabel('Source-aligned time (s); both arms execute frozen original model targets')
fig.savefig(here/'three_configuration_comparison.png',dpi=150);plt.close(fig)
(here/'three_configuration_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Saved three-profile comparison with deformation metrics')
