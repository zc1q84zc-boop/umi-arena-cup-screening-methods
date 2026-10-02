"""Diagnostic-only dual YUBI gripper range probe (not a trained model)."""

def predict(observation, step, episode):
    target = 1.0 if step < 10 else 0.0
    return {
        "action_dt_s": 0.1,
        "waypoints": [{
            "left": {"gripper_open_fraction": target},
            "right": {"gripper_open_fraction": target},
        }],
        "execute_steps": 1,
    }
