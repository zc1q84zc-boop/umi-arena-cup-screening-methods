"""Observed cup-on-plate then released return; evaluation only, not control."""
import math


class CupPlateReturnEvaluator:
    def __init__(self, observation):
        self.origin = tuple(observation['objects']['cup']['position_m'])
        self.plate_placed = False
        self.return_streak = 0

    def update(self, observation, *, plate_success):
        # A cup held over the plate is not a completed placement. The next
        # primitive must not start before the right gripper has released it.
        self.plate_placed |= (bool(plate_success)
                              and observation['robots']['right']['gripper_open_fraction'] >= .65)
        cup = observation['objects']['cup']
        xyz = cup['position_m']
        _, x, y, _ = cup['quaternion_wxyz']
        released = all(r['gripper_open_fraction'] >= .65 for r in observation['robots'].values())
        candidate = (math.dist(xyz[:2], self.origin[:2]) <= .035
                     and cup.get('deformation', {}).get('max_nodal_shape_change_m', 0.) < .015
                     and abs(xyz[2] - self.origin[2]) <= .008
                     and 1 - 2*(x*x+y*y) >= math.cos(math.radians(15))
                     and math.sqrt(sum(v*v for v in cup['linear_velocity_m_s'])) <= .05
                     and math.sqrt(sum(v*v for v in cup['angular_velocity_rad_s'])) <= .3
                     and released)
        self.return_streak = self.return_streak + 1 if self.plate_placed and candidate else 0
        complete = self.return_streak >= 5
        return {'stage': 'complete' if complete else 'return_to_origin' if self.plate_placed else 'place_on_plate',
                'plate_placed': self.plate_placed, 'return_candidate': candidate,
                'return_streak': self.return_streak, 'released': released, 'full_task_success': complete}
