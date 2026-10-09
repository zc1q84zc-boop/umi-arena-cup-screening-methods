"""Scientific trace comparison, generated from independent 30Hz action samples."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('evidence',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();data=json.loads(args.evidence.read_text())
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for name,result in data['results'].items():
        rows=result['drop_window'];t=np.array([r['time_s'] for r in rows])
        axes[0].plot(t,[r['clearance_mm'] for r in rows],label=name)
        axes[1].plot(t,[sum(r['normal_force_N'].values()) for r in rows],label=name)
        axes[2].plot(t,[r['jaw_actual_rad'] for r in rows],label=name+' actual')
        axes[2].plot(t,[r['jaw_requested_rad'] for r in rows],linestyle='--',alpha=.65,label=name+' target')
    axes[0].axhline(50,color='gray',linestyle=':',label='50 mm clearance threshold')
    axes[0].set_ylabel('Lowest cup clearance (mm)');axes[1].set_ylabel('Two-finger normal force (N)')
    axes[2].set_ylabel('Jaw angle (rad; larger = open)');axes[2].set_xlabel('Simulation time (s)')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Frozen commands: jaw-only margin diagnostic (not online model success)')
    fig.tight_layout();fig.savefig(args.output,dpi=150)


if __name__=='__main__':main()
