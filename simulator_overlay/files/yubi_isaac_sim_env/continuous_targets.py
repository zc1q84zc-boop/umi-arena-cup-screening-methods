"""Simulation-time joint command governor; does not certify actual dynamics."""
import numpy as np


class ContinuousTargets:
    def __init__(self, position, limits, dt, velocity=.8, acceleration=1.5):
        self.q = np.asarray(position, dtype=float).copy()
        self.v = np.zeros_like(self.q)
        self.limits = np.asarray(limits, dtype=float)
        self.dt, self.vmax, self.amax = float(dt), float(velocity), float(acceleration)
        if not np.isfinite(self.q).all() or min(dt, velocity, acceleration) <= 0:
            raise ValueError("invalid command governor initialization")
        if np.any(self.q < self.limits[..., 0]) or np.any(self.q > self.limits[..., 1]):
            raise ValueError("initial command outside joint limits")

    def step(self, goal):
        goal = np.asarray(goal, dtype=float)
        if goal.shape != self.q.shape or not np.isfinite(goal).all():
            raise ValueError("invalid joint goal")
        goal = np.clip(goal, self.limits[..., 0], self.limits[..., 1])
        a, dt = self.amax, self.dt
        # Reserve one full substep plus braking distance before either limit.
        distance = np.maximum(0., np.stack((self.q-self.limits[..., 0], self.limits[..., 1]-self.q), -1))
        safe = np.sqrt((a*dt)**2 + 2*a*distance) - a*dt
        lo = np.maximum(np.maximum(-self.vmax, self.v-a*dt), -safe[..., 0])
        hi = np.minimum(np.minimum(self.vmax, self.v+a*dt), safe[..., 1])
        if np.any(lo > hi+1e-9):
            raise ValueError("no acceleration-safe command inside joint limits")
        self.v = np.clip(4.*(goal-self.q), lo, hi)
        self.q = self.q + self.v*dt
        return self.q.copy()
