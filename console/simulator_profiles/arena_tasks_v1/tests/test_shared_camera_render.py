import unittest
from unittest.mock import patch

import numpy as np

from yubi_isaac_sim_env import run
from yubi_isaac_sim_env.shared_camera_render import install_shared_camera_render


class World:
    def __init__(self):
        self.current_time = 1.
        self.calls = 0
        self.cameras = {}

    def render(self):
        self.calls += 1
        for camera in self.cameras.values():
            camera.stamp = self.current_time - (.1 if camera.stale else 0)
            camera.rgb = np.arange(24, dtype=np.uint8).reshape(2,4,3) + self.calls


class Camera:
    def __init__(self):
        self.position = np.zeros(3)
        self.orientation = np.array([1.,0,0,0])
        self.stamp = 0.
        self.stale = False
        self.rgb = None
        self.frequency = 120

    def get_frequency(self):
        return self.frequency

    def set_frequency(self, value):
        self.frequency = value

    def get_rgb(self, device):
        return self.rgb

    def get_current_frame(self):
        return {'rendering_time':self.stamp}

    def get_world_pose(self, camera_axes):
        return self.position.copy(), self.orientation.copy()

    def get_clipping_range(self):
        return (.01,100.)


class Video:
    def __init__(self):
        self.frames = []

    def write(self, rgb):
        self.frames.append(np.array(rgb, copy=True))


class SharedRenderTests(unittest.TestCase):
    def setUp(self):
        self.names = ('head','left_wrist','right_wrist')
        self.world = World()
        self.cameras = {n:Camera() for n in self.names}
        self.world.cameras = self.cameras
        self.specs = {n:{'name':n,'resolution':[4,2]} for n in self.names}
        self.state = {'physics_time_s':1.,'robots':{}}
        self.shared = install_shared_camera_render(run)
        self.environment = patch.dict('os.environ', {'SIM_ADAPTER_AUDIT_DIR':''})
        self.environment.start()

    def tearDown(self):
        self.shared.uninstall()
        self.environment.stop()

    def record(self, phase='step'):
        videos = {n:Video() for n in self.names}
        run._sample(videos['head'],None,self.cameras['head'],self.world,self.specs['head'],
                    0,0,phase,0,self.state,{n:videos[n] for n in self.names[1:]},
                    self.cameras,self.specs)
        return videos

    def policy(self):
        return run._policy_observation(self.state,self.cameras,self.specs,self.world)

    def test_three_views_use_one_render_batch_and_policy_reuses_exact_recorded_pixels(self):
        videos = self.record()
        self.assertEqual(self.world.calls,6)
        obs = self.policy()
        self.assertEqual(self.world.calls,6)
        for name in self.names:
            np.testing.assert_array_equal(obs['images'][name],videos[name].frames[-1])
            self.assertEqual(obs['image_metadata'][name]['rendering_time_s'],1.)
            self.assertTrue(obs['image_metadata'][name]['shared_render_reused'])
        self.assertEqual(self.shared.stats['reused_policy_groups'],1)

    def test_rate_limited_sensor_is_changed_to_acquire_each_rendered_frame(self):
        original = self.world.render
        def rate_limited_render():
            original()
            for camera in self.cameras.values():
                if camera.frequency != -1:
                    camera.stamp -= .004
        self.world.render = rate_limited_render
        self.record()
        self.assertEqual(self.world.calls, 6)
        self.assertTrue(all(c.frequency == -1 for c in self.cameras.values()))

    def test_physics_time_change_invalidates_cache(self):
        self.record()
        self.world.current_time = self.state['physics_time_s'] = 1.1
        self.policy()
        self.assertEqual(self.world.calls,12)

    def test_camera_pose_change_at_same_time_invalidates_cache(self):
        self.record()
        self.cameras['right_wrist'].position[0] = .02
        obs = self.policy()
        self.assertEqual(self.world.calls,12)
        self.assertFalse(obs['image_metadata']['right_wrist']['shared_render_reused'])

    def test_reset_at_same_time_invalidates_cache(self):
        self.record()
        self.record('reset')
        self.assertEqual(self.world.calls,12)

    def test_one_stale_camera_rejects_whole_group(self):
        self.cameras['right_wrist'].stale = True
        with self.assertRaisesRegex(RuntimeError,'requested physics time'):
            self.record()
        self.assertIsNone(self.shared.last)

    def test_model_image_is_owned_and_cannot_corrupt_cached_video(self):
        videos = self.record()
        obs = self.policy()
        obs['images']['head'][:] = 0
        repeated = self.policy()
        np.testing.assert_array_equal(repeated['images']['head'],videos['head'].frames[-1])

    def test_wrong_state_time_is_rejected_before_rendering(self):
        self.state['physics_time_s'] = .9
        with self.assertRaisesRegex(RuntimeError,'current physics state'):
            self.policy()
        self.assertEqual(self.world.calls,0)

    def test_horizontal_flip_is_applied_once(self):
        self.specs['head']['horizontal_flip_for_dataset'] = True
        videos = self.record()
        obs = self.policy()
        np.testing.assert_array_equal(obs['images']['head'],self.cameras['head'].rgb[:,::-1])
        np.testing.assert_array_equal(obs['images']['head'],videos['head'].frames[-1])

    def test_render_cannot_advance_physics(self):
        original = self.world.render
        def invalid():
            original()
            self.world.current_time += .001
        self.world.render = invalid
        with self.assertRaisesRegex(RuntimeError,'advanced physics'):
            self.record()


if __name__ == '__main__':
    unittest.main()
