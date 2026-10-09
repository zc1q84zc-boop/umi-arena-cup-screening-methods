"""Read-only contact telemetry. Never used by a policy or action converter."""
from __future__ import annotations

import math


def _list(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, 'tolist') else list(value)


def decode_pair(data, net_force):
    """Isaac 5.1: counts at index 4, starts at 5; caller supplies dt=1/Hz."""
    forces, points, normals, separations, counts, starts = map(_list, data)
    count, start = int(counts[0][0]), int(starts[0][0])
    if count < 0 or start < 0:
        raise ValueError('Invalid PhysX contact buffer indices')
    end = min(start + count, *(len(x) for x in (forces, points, normals, separations)))
    records = [dict(point_world_m=points[i], normal_world=normals[i],
                    normal_force_N=float(forces[i][0]), separation_m=float(separations[i][0]))
               for i in range(start, end)]
    vector = [float(x) for x in _list(net_force)[0][0]]
    if not all(math.isfinite(x) for x in vector) or not all(
            math.isfinite(r['normal_force_N']) for r in records):
        raise ValueError('Nonfinite PhysX contact force')
    return dict(net_force_world_N=vector, net_force_norm_N=math.sqrt(sum(x*x for x in vector)),
                positive_normal_force_sum_N=sum(max(0., r['normal_force_N']) for r in records),
                reported_contact_count=count, buffered_contact_count=len(records),
                buffer_truncated=len(records) != count, contacts=records)


class ContactSummary:
    """Summarize 60 Hz samples without mistaking cancelling forces for zero."""
    def __init__(self, dt_s, threshold_N=.02):
        self.dt_s, self.threshold_N = dt_s, threshold_N
        self.samples = 0
        self.cup_support_peak_N = 0.
        self.buffer_truncated_samples = 0
        self.arms = {s: dict(peak_normal_force_N={p: 0. for p in
                     ('left_finger', 'right_finger', 'base')}, bilateral_samples=0,
                     longest_bilateral_samples=0, current_streak=0) for s in ('left', 'right')}

    def update(self, row):
        self.samples += 1
        self.cup_support_peak_N = max(self.cup_support_peak_N,
                                      math.dist(row['cup_net_contact_force_world_N'], [0., 0., 0.]))
        for side, state in row['robots'].items():
            stats = self.arms[side]
            for part, contact in state['cup_contacts'].items():
                stats['peak_normal_force_N'][part] = max(stats['peak_normal_force_N'][part],
                                                        contact['positive_normal_force_sum_N'])
                self.buffer_truncated_samples += int(contact['buffer_truncated'])
            both = all(state['cup_contacts'][p]['positive_normal_force_sum_N'] > self.threshold_N
                       for p in ('left_finger', 'right_finger'))
            stats['bilateral_samples'] += int(both)
            stats['current_streak'] = stats['current_streak'] + 1 if both else 0
            stats['longest_bilateral_samples'] = max(stats['longest_bilateral_samples'],
                                                     stats['current_streak'])

    def result(self):
        return dict(samples=self.samples, sample_hz=1/self.dt_s, force_unit='N',
                    bilateral_threshold_per_finger_N=self.threshold_N,
                    cup_support_peak_N=self.cup_support_peak_N,
                    cup_contact_signal_observed=self.cup_support_peak_N > self.threshold_N,
                    buffer_truncated_samples=self.buffer_truncated_samples,
                    arms={s: {k: v for k, v in value.items() if k != 'current_streak'}
                          for s, value in self.arms.items()},
                    caution='Bilateral contact is not proof of a stable grasp; no actuator torque inferred.')
