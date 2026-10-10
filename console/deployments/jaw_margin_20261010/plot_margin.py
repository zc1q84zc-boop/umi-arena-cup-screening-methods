"""Plot completed closure-margin comparisons without changing evaluation gates."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('trials', type=Path, nargs='+')
    args = p.parse_args()
    roots = [args.baseline, *args.trials]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    panels = [('cup_clearance_mm','Lowest cup node above table (mm)'),
              ('cup_tilt_deg','Cup tilt (degrees)'),
              ('shape_change_mm','Cup shape change (mm)'),
              ('jaw_actual_rad','Actual driven jaw angle (rad)')]
    for root in roots:
        result = json.loads((root / 'result.json').read_text())
        samples = json.loads((root / 'plot_samples.json').read_text())
        margin = result.get('left_jaw_margin_rad',0)
        selected = [r for r in samples if 140 <= r.get('reference_time_s',r['physics_time_s']) <= 160]
        x = [r.get('reference_time_s',r['physics_time_s']) for r in selected]
        for axis,(key,title) in zip(axes.flat,panels):
            axis.plot(x,[r[key] for r in selected],label=f'Closure margin {margin:.3f} rad',linewidth=1.3)
            axis.set_ylabel(title);axis.grid(alpha=.2)
    axes[0,0].axhline(10,color='black',linestyle=':',alpha=.5)
    axes[0,1].axhline(15,color='black',linestyle=':',alpha=.5)
    for axis in axes[1]:axis.set_xlabel('Source-aligned simulation time (s)')
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Same model pose targets; arm/jaw response 4/4; 240 Hz physics; 128 solver iterations')
    fig.tight_layout();fig.savefig(args.output,dpi=160);plt.close(fig)


if __name__ == '__main__': main()
