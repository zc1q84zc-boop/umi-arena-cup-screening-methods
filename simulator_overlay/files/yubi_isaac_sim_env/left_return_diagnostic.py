"""Isolated left-arm phase test; not a full two-arm task success claim."""
from copy import deepcopy
import math


def closure_target(model_fraction, extra_fraction):
    """Bounded stroke bias, NOT extra force or a measured aperture correction.

    Fade out at the existing .65 open/release boundary to avoid a target jump.
    Never use cup position/contact feedback or tighten an opening command.
    """
    if (type(extra_fraction) not in (int, float) or not math.isfinite(extra_fraction)
            or not 0 <= extra_fraction <= .05):
        raise ValueError('extra closure must be finite and between 0 and .05')
    if not math.isfinite(model_fraction) or not 0 <= model_fraction <= 1:
        raise ValueError('model gripper fraction must be finite and between 0 and 1')
    weight = min(1., max(0., (.65 - model_fraction) / .10))
    return max(0., model_fraction - extra_fraction * weight)


class LeftReturnDiagnostic:
    def __init__(self, observation, *, extra_closure_fraction=0.):
        closure_target(1., extra_closure_fraction)  # validate before simulation use
        self.extra_closure_fraction = extra_closure_fraction
        self.closure_audit = []
        scenario = observation['scenario']
        self.origin = tuple(scenario['diagnostic_return_origin_m'])
        cup = observation['objects']['cup']
        plate = observation['objects']['plate']
        self.initial_z = cup['position_m'][2]
        if (math.dist(cup['position_m'][:2], plate['position_m'][:2]) > .02
                or not .001 <= self.initial_z - plate['position_m'][2] <= .010
                or math.dist(cup['linear_velocity_m_s'], [0, 0, 0]) > .05):
            raise ValueError('left diagnostic requires a settled cup on the plate')
        if math.dist(self.origin[:2], cup['position_m'][:2]) < .10:
            raise ValueError('return target must not be the initial on-plate cup position')
        right = observation['robots']['right']
        self.right_pose = {**deepcopy(right['tool_pose']),
                           'gripper_open_fraction': right['gripper_open_fraction']}
        self.right_command = {
            'arm_joint_targets_rad': [right['joint_positions'][right['joint_names'].index(f'panda_joint{i}')]
                                     for i in range(1, 8)],
            'gripper_open_fraction': right['gripper_open_fraction'],
        }
        if len(self.right_command['arm_joint_targets_rad']) != 7:
            raise ValueError('right hold requires seven named Panda arm joints')
        self.plate_placed = True  # reset intervention, NOT accomplished by the model
        self.lift_streak = self.return_streak = 0
        self.lift_confirmed = False
        self.max_lift_m = 0.
        self.last = {}

    def filter_chunk(self, chunk):
        filtered = deepcopy(chunk)
        self.closure_audit = []
        for waypoint in filtered['waypoints']:
            waypoint['right'] = deepcopy(self.right_pose)
            if self.extra_closure_fraction:
                model = waypoint['left']['gripper_open_fraction']
                applied = closure_target(model, self.extra_closure_fraction)
                waypoint['left']['gripper_open_fraction'] = applied
                self.closure_audit.append({'model_open_fraction': model,
                    'command_open_fraction': applied,
                    'extra_closure_fraction_applied': model-applied,
                    'max_extra_closure_fraction': self.extra_closure_fraction,
                    'force_limit_changed': False, 'friction_changed': False,
                    'oracle_feedback': False})
        return filtered

    def filter_action(self, action):
        return {**action, 'right': deepcopy(self.right_command)}

    def update(self, observation, *, plate_success=False):
        cup = observation['objects']['cup']
        xyz = cup['position_m']
        w, x, y, z = cup['quaternion_wxyz']
        upright = 1 - 2*(x*x + y*y) >= math.cos(math.radians(15))
        lift = xyz[2] - self.initial_z
        self.max_lift_m = max(self.max_lift_m, lift)
        left = observation['robots']['left']
        lift_candidate = (lift >= .050 and upright and left['gripper_open_fraction'] < .65
                          and math.dist(xyz, left['tool_pose']['position_m']) <= .15)
        # Evaluated at every 30-Hz executed action, not just model request boundaries.
        self.lift_streak = self.lift_streak + 1 if lift_candidate else 0
        self.lift_confirmed |= self.lift_streak >= 10
        released = left['gripper_open_fraction'] >= .65
        candidate = (self.lift_confirmed and upright and released
                     and math.dist(xyz[:2], self.origin[:2]) <= .035
                     and abs(xyz[2] - self.origin[2]) <= .008
                     and math.dist(cup['linear_velocity_m_s'], [0, 0, 0]) <= .05
                     and math.dist(cup['angular_velocity_rad_s'], [0, 0, 0]) <= .3)
        self.return_streak = self.return_streak + 1 if candidate else 0
        complete = self.return_streak >= 15
        self.last = {
            'stage': 'diagnostic_complete' if complete else 'left_return' if self.lift_confirmed else 'left_pick_from_plate',
            'plate_placed': True, 'plate_placed_by_reset': True, 'released': released,
            'return_candidate': candidate, 'return_streak': self.return_streak,
            'stable_lift_confirmed': self.lift_confirmed, 'max_cup_lift_m': self.max_lift_m,
            'diagnostic_success': complete, 'full_task_success': False,
            'return_origin_m': list(self.origin), 'right_arm_held': True,
            'max_extra_closure_fraction': self.extra_closure_fraction,
            'closure_bias_is_diagnostic_assistance': bool(self.extra_closure_fraction),
        }
        return self.last
