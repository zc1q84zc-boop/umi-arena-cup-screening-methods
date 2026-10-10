"""Physics-rate simulation governor, preserving the online motion limits.

Bounds command velocity/acceleration, not the physical robot's actual motion.
The single driven-jaw reference is mirrored after governing, never twice.
"""
import numpy as np


class ContinuousTargets:
    def __init__(self, position, limits, dt, velocity=.8, acceleration=1.5, response_gain=4., jaw_response_gain=None):
        self.q = np.asarray(position, dtype=float).copy()
        self.v = np.zeros_like(self.q)
        self.limits = np.asarray(limits, dtype=float)
        self.dt, self.vmax, self.amax = float(dt), float(velocity), float(acceleration)
        self.response_gain = float(response_gain)
        self.jaw_response_gain = self.response_gain if jaw_response_gain is None else float(jaw_response_gain)
        self.gains = np.full_like(self.q, self.response_gain)
        self.gains[..., -1] = self.jaw_response_gain
        if (self.limits.shape != self.q.shape + (2,) or not np.isfinite(self.q).all()
                or not np.isfinite(self.limits).all()
                or not np.isfinite([dt, velocity, acceleration, self.response_gain, self.jaw_response_gain]).all()
                or min(dt, velocity, acceleration, self.response_gain, self.jaw_response_gain) <= 0
                or max(self.response_gain, self.jaw_response_gain) * self.dt > 1):
            raise ValueError('invalid command governor initialization')
        if np.any(self.q < self.limits[..., 0]) or np.any(self.q > self.limits[..., 1]):
            raise ValueError('initial command outside joint limits')

    def step(self, goal):
        goal = np.asarray(goal, dtype=float)
        if goal.shape != self.q.shape or not np.isfinite(goal).all():
            raise ValueError('invalid joint goal')
        goal = np.clip(goal, self.limits[..., 0], self.limits[..., 1])
        a, dt = self.amax, self.dt
        distance = np.maximum(0., np.stack((self.q-self.limits[..., 0], self.limits[..., 1]-self.q), -1))
        safe = np.sqrt((a*dt)**2 + 2*a*distance) - a*dt
        lo = np.maximum(np.maximum(-self.vmax, self.v-a*dt), -safe[..., 0])
        hi = np.minimum(np.minimum(self.vmax, self.v+a*dt), safe[..., 1])
        if np.any(lo > hi+1e-9):
            raise ValueError('no acceleration-safe command inside joint limits')
        self.v = np.clip(self.gains*(goal-self.q), lo, hi)
        self.q = self.q + self.v*dt
        return self.q.copy()
