import copy

def predict(observation, step, episode):
    waypoint={side:{**copy.deepcopy(robot['tool_pose']),'gripper_open_fraction':robot['gripper_open_fraction']} for side,robot in observation['robots'].items()}
    return {'action_dt_s':1/30,'waypoints':[copy.deepcopy(waypoint) for _ in range(3)],'execute_steps':3}
