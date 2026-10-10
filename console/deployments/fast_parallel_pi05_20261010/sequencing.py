"""Explicit right-place, right-clearance/home, then left-return orchestration.

Only the two manipulation phases use policy predictions. Homing and holding
are controller interventions, recorded separately from model actions.
"""
import copy
import math

ARM_NAMES = tuple(f'panda_joint{i}' for i in range(1, 8))


def arm_q(robot):
    return [float(robot['joint_positions'][robot['joint_names'].index(n)]) for n in ARM_NAMES]


def pose_error(current, goal):
    distance = math.dist(current['position_m'], goal['position_m'])
    a, b = current['quaternion_wxyz'], goal['quaternion_wxyz']
    dot = abs(sum(x*y for x, y in zip(a, b))) / math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    return distance, 2*math.acos(min(1., max(0., dot)))


class Sequencer:
    def __init__(self, observation):
        self.home = copy.deepcopy(observation['robots'])
        self.home_q = {s: arm_q(r) for s, r in self.home.items()}
        self.phase = 'right_place'
        self.last_policy_phase = None
        self.phase_started_s = float(observation['physics_time_s'])
        self.release_since = self.home_since = None
        self.plate_placed = False
        self.right_home_verified = False
        self.handoff_time_s = None
        self.first_left_policy_step = None
        self.events = []
        self.metrics = {}
        self.release_q = self.lift_pose = None

    @property
    def requested_task_stage(self):
        return 'place_on_plate' if self.phase == 'right_place' else 'return_to_origin' if self.phase == 'left_return' else self.phase

    def hold(self, side):
        grip = self.home['left']['gripper_open_fraction'] if side == 'left' else 1.
        return {'arm_joint_targets_rad': list(self.home_q[side]), 'gripper_open_fraction': grip}

    def set_phase(self, phase, observation):
        previous = self.phase
        self.phase = phase
        self.phase_started_s = float(observation['physics_time_s'])
        self.events.append({'from': previous, 'to': phase, 'physics_time_s': self.phase_started_s, 'metrics': dict(self.metrics)})

    def placed(self, observation):
        if self.phase != 'right_place':
            return
        self.plate_placed = True
        # Preserve the proven placement policy's inactive-hand posture. Only
        # after release is that standby hand held during the right retreat.
        self.home['left'] = copy.deepcopy(observation['robots']['left'])
        self.home_q['left'] = arm_q(self.home['left'])
        self.release_q = arm_q(observation['robots']['right'])
        self.lift_pose = copy.deepcopy(observation['robots']['right']['tool_pose'])
        self.lift_pose['position_m'][2] += .12
        self.set_phase('right_release', observation)

    def observe(self, observation):
        right = observation['robots']['right']
        t = float(observation['physics_time_s'])
        position_error, angle_error = pose_error(right['tool_pose'], self.home['right']['tool_pose'])
        q_error = max(abs(a-b) for a, b in zip(arm_q(right), self.home_q['right']))
        velocity = max(abs(float(right['joint_velocities'][right['joint_names'].index(n)])) for n in ARM_NAMES)
        self.metrics = {'right_home_position_error_m': position_error,
            'right_home_orientation_error_rad': angle_error,
            'right_home_max_joint_error_rad': q_error,
            'right_max_joint_velocity_rad_s': velocity,
            'right_gripper_open_fraction': right['gripper_open_fraction']}
        if self.phase == 'right_release':
            if right['gripper_open_fraction'] >= .95:
                if self.release_since is None:
                    self.release_since = t
                if t-self.release_since >= .2-1e-6:
                    self.set_phase('right_lift', observation)
            else:
                self.release_since = None
        elif self.phase == 'right_lift':
            lift_error, _ = pose_error(right['tool_pose'], self.lift_pose)
            self.metrics['right_clearance_pose_error_m'] = lift_error
            if lift_error <= .01 and right['gripper_open_fraction'] >= .95:
                self.set_phase('right_home', observation)
        elif self.phase == 'right_home':
            candidate = (position_error <= .005 and angle_error <= math.radians(2)
                and q_error <= .02 and velocity <= .05 and right['gripper_open_fraction'] >= .95)
            if candidate:
                if self.home_since is None:
                    self.home_since = t
                self.metrics['right_home_stable_s'] = t-self.home_since
                if t-self.home_since >= .5-1e-6:
                    self.right_home_verified = True
                    self.handoff_time_s = t
                    self.set_phase('left_return', observation)
            else:
                self.home_since = None
                self.metrics['right_home_stable_s'] = 0.

    def policy_waypoints(self, observation):
        target = {}
        for side in ('left', 'right'):
            target[side] = {**copy.deepcopy(observation['robots'][side]['tool_pose']), 'gripper_open_fraction': 1.}
        if self.phase == 'right_lift':
            target['right'] = {**copy.deepcopy(self.lift_pose), 'gripper_open_fraction': 1.}
        return {'action_dt_s': 1/30, 'waypoints': [copy.deepcopy(target) for _ in range(3)], 'execute_steps': 3}

    def filter_action(self, action, observation):
        output = copy.deepcopy(action)
        if self.phase not in ('right_place', 'left_return') or (self.phase == 'left_return' and self.last_policy_phase != 'left_return'):
            output['left'] = self.hold('left')
        if self.phase == 'right_release':
            output['right'] = {'arm_joint_targets_rad': list(self.release_q), 'gripper_open_fraction': 1.}
        elif self.phase == 'right_home':
            # Bound each joint target preview; the unchanged physics-rate
            # governor enforces 0.8 rad/s and 1.5 rad/s^2 on the actual targets.
            current = arm_q(observation['robots']['right'])
            goal = [q+max(-.08, min(.08, target-q)) for q, target in zip(current, self.home_q['right'])]
            output['right'] = {'arm_joint_targets_rad': goal, 'gripper_open_fraction': 1.}
        elif self.phase == 'left_return':
            output['right'] = self.hold('right')
        return output

    def audit(self):
        return {'phase': self.phase, 'policy_phase': self.last_policy_phase,
            'plate_placed': self.plate_placed, 'right_home_verified': self.right_home_verified,
            'handoff_time_s': self.handoff_time_s, 'first_left_policy_step': self.first_left_policy_step,
            **self.metrics}
