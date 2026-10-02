import json
from pathlib import Path
import tempfile
import unittest
import hashlib

import numpy as np
from online_calibration import Calibration, MirroredReplayPrior, matrix, mirror_sagittal_rotation


class CalibrationTests(unittest.TestCase):
    def test_provisional_roundtrip_both_arms(self):
        c = MirroredReplayPrior()
        rng = np.random.default_rng(23)
        for side in ("left", "right"):
            for _ in range(100):
                p, q = rng.normal(size=3), rng.normal(size=4)
                q /= np.linalg.norm(q)
                hp, hq = c.tool_to_hand(side, p, q)
                tp, tq = c.hand_to_world_tool(side, hp, hq)
                np.testing.assert_allclose(matrix(tp,tq), matrix(p,q), atol=1e-12)
        self.assertEqual(c.t[:2].tolist(), [0., 0.])
        self.assertAlmostEqual(c.source_gripper(1.), .78)
        self.assertAlmostEqual(c.sim_gripper(0.), 0.)
        self.assertGreater(c.sim_gripper(.74), c.sim_gripper(.46))
        for x in np.linspace(0,1,25):
            self.assertAlmostEqual(c.sim_gripper(c.source_gripper(x)),x)

    def test_sagittal_mirror_signs_and_involution(self):
        a,b=np.deg2rad([17,22])/2
        rx=matrix([0,0,0],[np.cos(a),np.sin(a),0,0])[:3,:3]
        ry=matrix([0,0,0],[np.cos(b),0,np.sin(b),0])[:3,:3]
        np.testing.assert_allclose(mirror_sagittal_rotation(rx),rx.T)
        np.testing.assert_allclose(mirror_sagittal_rotation(ry),ry)
        np.testing.assert_allclose(mirror_sagittal_rotation(mirror_sagittal_rotation(rx @ ry)),rx @ ry)
        self.assertAlmostEqual(np.linalg.det(mirror_sagittal_rotation(rx @ ry)),1)
        with self.assertRaises(ValueError): mirror_sagittal_rotation(np.diag([1,-1,1]))

    def fixture(self, root):
        camera = root / 'camera.json'
        camera.write_text('{}')
        identity = {'position_m': [0,0,0], 'quaternion_wxyz': [1,0,0,0]}
        profile = {'schema_version': 1, 'id': 'unit-test-only', 'verified': True,
                   'world_from_source': {'position_m':[0.1,-0.2,0.7], 'quaternion_wxyz':[0.70710678,0,0,-0.70710678]},
                   'hand_to_tool': {'left': identity, 'right': {'position_m':[0.08927,0,0], 'quaternion_wxyz':[0,0,0,1]}},
                   'gripper': {'source_open_rad':0, 'source_closed_rad':0.414,
                               'sim_open_fraction':1, 'sim_closed_fraction':1/3},
                   'camera_files_sha256': {str(camera):hashlib.sha256(camera.read_bytes()).hexdigest()}}
        path = root/'profile.json'; path.write_text(json.dumps(profile))
        return path, profile, camera

    def test_both_arms_round_trip_and_rotation_lever_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _, _ = self.fixture(Path(tmp)); c = Calibration(path)
            for side in ['left','right']:
                p,q = [0.3,-0.1,0.2],[0.2,0.3,0.4,0.5]
                wp,wq = c.hand_to_world_tool(side,p,q)
                hp,hq = c.tool_to_hand(side,wp,wq)
                np.testing.assert_allclose(matrix(hp,hq),matrix(p,q),atol=1e-12)
            p1,_=c.hand_to_world_tool('right',[0,0,0],[1,0,0,0])
            p2,_=c.hand_to_world_tool('right',[0,0,0],[0,0,0,1])
            self.assertAlmostEqual(np.linalg.norm(p2-p1),2*0.08927)

    def test_closure_direction_and_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _, _ = self.fixture(Path(tmp)); c=Calibration(path)
            self.assertGreater(c.sim_gripper(0),c.sim_gripper(0.414))
            for x in [0,0.1,0.3,0.414]:
                self.assertAlmostEqual(c.source_gripper(c.sim_gripper(x)),x)

    def test_reject_unverified_or_episode_specific_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,d,_=self.fixture(Path(tmp)); d['verified']=False; path.write_text(json.dumps(d))
            with self.assertRaises(ValueError): Calibration(path)
            d['verified']=True; d['source_contact_frame']=58; path.write_text(json.dumps(d))
            with self.assertRaises(ValueError): Calibration(path)

    def test_reject_changed_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_,camera=self.fixture(Path(tmp)); camera.write_text('{"roll":180}')
            with self.assertRaisesRegex(ValueError,'camera configuration changed'): Calibration(path)

if __name__ == '__main__': unittest.main()
