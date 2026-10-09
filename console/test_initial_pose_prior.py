import unittest
from pathlib import Path
import numpy as np
from build_initial_pose_prior import build

class InitialPosePriorTests(unittest.TestCase):
    @unittest.skipUnless(all((Path(__file__).parent / f'dataset_replay/official_cup_5/episode-{episode}.json').is_file()
                            for episode in (61164, 136238, 231149, 259632, 262232)),
                         'requires separately authorized private practice records')
    def test_real_medoid_and_no_action_leakage(self):
        p=build()
        self.assertEqual(len(p['candidates']),5)
        self.assertEqual(p['selected']['episode'],136238)
        self.assertNotIn('action',p['selected']['source_observation'])
        self.assertFalse(p['measured'])
        for side in ['left','right']:
            goal=p['selected']['targets'][side]
            self.assertAlmostEqual(np.linalg.norm(goal['quaternion_wxyz']),1.)
            self.assertGreater(goal['position_m'][2],.86)
            self.assertTrue(0<=goal['gripper_open_fraction']<=1)
        self.assertGreater(p['selected']['targets']['left']['position_m'][1],0)
        self.assertLess(p['selected']['targets']['right']['position_m'][1],0)

if __name__=='__main__': unittest.main()
