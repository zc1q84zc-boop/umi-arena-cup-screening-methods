import importlib.util
from pathlib import Path
import unittest
from collections import Counter
import numpy as np

path=Path(__file__).parent/'simulator_profiles/tuned_v1/yubi_isaac_sim_env/pvc_shell.py'
spec=importlib.util.spec_from_file_location('pvc_shell',path)
shell=importlib.util.module_from_spec(spec);spec.loader.exec_module(shell)


class ShellTests(unittest.TestCase):
    def test_manufactured_shell_rest_angles_and_official_bending_formula(self):
        p=shell.PROFILE
        self.assertEqual(p['rest_bend_angles_default'],'restShapeDefault')
        self.assertAlmostEqual(p['surface_bend_stiffness_Pa'],
                               p['youngs_modulus_Pa']/(12*(1-p['poissons_ratio']**2)))
        self.assertFalse(p['measured'])

    def test_manifold_open_cup_no_lid_and_no_degenerate_triangles(self):
        points,faces=shell.cup_mesh()
        self.assertEqual(points.shape,(769,3))
        cross=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
        self.assertTrue((np.linalg.norm(cross,axis=1)>1e-9).all())
        edges=Counter(tuple(sorted((int(a),int(b)))) for f in faces for a,b in zip(f,np.roll(f,1)))
        self.assertEqual(sum(v==1 for v in edges.values()),48)
        self.assertTrue(all(v in (1,2) for v in edges.values()))
        boundary={i for e,v in edges.items() if v==1 for i in e}
        self.assertEqual(boundary,set(range(len(points)-48,len(points))))
        self.assertTrue((cross[:48,2]<0).all())
        centers=points[faces].mean(1)
        self.assertTrue((np.sum(cross[336:,:2]*centers[336:,:2],axis=1)>0).all())

    def test_shape_metrics_ignore_rigid_motion(self):
        p,_=shell.cup_mesh();angle=.37
        r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
        q=p@r.T+[.1,.2,.75]
        translation,rot,residual=shell.rigid_fit(p,q)
        np.testing.assert_allclose(translation,[.1,.2,.75],atol=1e-9)
        np.testing.assert_allclose(rot,r,atol=1e-9)
        self.assertLess(np.max(abs(residual)),1e-9)
        metrics=shell.deformation_metrics(p,q)
        self.assertAlmostEqual(metrics['rim_compression_fraction'],0,places=6)
        self.assertLess(metrics['max_nodal_shape_change_m'],1e-9)

    def test_compression_is_not_rigid_translation(self):
        p,_=shell.cup_mesh();q=p.copy();q[:,0]*=.9
        m=shell.deformation_metrics(p,q)
        self.assertAlmostEqual(m['rim_compression_fraction'],.1,places=5)
        self.assertGreater(m['max_nodal_shape_change_m'],.003)
        with self.assertRaises(ValueError): shell.rigid_fit(p,np.full_like(p,np.nan))


if __name__=='__main__': unittest.main()
