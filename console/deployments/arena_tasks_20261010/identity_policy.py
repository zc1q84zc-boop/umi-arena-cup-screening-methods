"""Runtime wire-contract probe only; does not attempt to solve any task."""
import numpy as np


class Policy:
    def __init__(self, checkpoint_dir):
        self.calls = 0

    def infer(self, observation):
        assert set(observation) == {
            'observation.image.left', 'observation.image.right',
            'observation.pose.left_hand_root_to_right_hand_root.absolute',
            'observation.joint_states', 'prompt'}
        for side in ('left', 'right'):
            image = observation['observation.image.' + side]
            assert image.shape == (480, 640, 3) and image.dtype == np.uint8
        pose = observation['observation.pose.left_hand_root_to_right_hand_root.absolute']
        joints = observation['observation.joint_states']
        assert pose.shape == (7,) and pose.dtype == np.float32
        assert joints.shape == (2,) and joints.dtype == np.float32
        assert np.isfinite(pose).all() and np.isfinite(joints).all()
        assert isinstance(observation['prompt'], str) and observation['prompt']
        self.calls += 1
        actions = np.zeros((16, 16), dtype=np.float32)
        actions[:, 6] = actions[:, 13] = 1.0
        actions[:, 14:] = joints
        return actions
