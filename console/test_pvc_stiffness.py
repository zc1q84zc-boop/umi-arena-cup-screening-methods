import hashlib
import json
from pathlib import Path
import runpy
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import sim_console
from verify_pvc_stiffness import stable_grasp

PACKAGE = Path(__file__).parent/'simulator_profiles/tuned_v1/yubi_isaac_sim_env'
registry = runpy.run_path(str(PACKAGE/'pvc_stiffness.py'))
shell = runpy.run_path(str(PACKAGE/'pvc_shell.py'))
response = runpy.run_path(str(PACKAGE/'pvc_response_probe.py'))
numerics = runpy.run_path(str(PACKAGE/'pvc_numerics.py'))


class StiffnessTests(unittest.TestCase):
    def test_precision_changes_only_id_iterations_and_exact_physics_rate(self):
        base = registry['profile_for']('pvc_shell_e3000mpa_v1', shell['PROFILE'])
        precision = numerics['precision_profile'](base)
        for key in base.keys()-{'id', 'solver_position_iterations'}:
            self.assertEqual(base[key], precision[key])
        self.assertEqual(precision['solver_position_iterations'], 128)
        self.assertEqual(numerics['physics_hz_for'](precision['id']), 240)
        self.assertEqual(numerics['physics_hz_for'](base['id']), 120)
        with self.assertRaises(ValueError): numerics['precision_profile'](shell['PROFILE'])

    def test_precision_marker_requires_material_readback_and_exact_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root/'simulator_profiles/tuned_v1/yubi_isaac_sim_env'
            package.mkdir(parents=True)
            filenames = ('pvc_shell.py','pvc_stiffness.py','pvc_precision_probe.py','pvc_numerics.py','pvc_response_probe.py')
            for filename in filenames: shutil.copyfile(PACKAGE/filename, package/filename)
            key = numerics['PRECISION_ID']
            marker = root/'sim_validation/pvc_stiffness'/key/'verified_probe.json'
            marker.parent.mkdir(parents=True)
            profile = numerics['precision_profile'](registry['profile_for']('pvc_shell_e3000mpa_v1', shell['PROFILE']))
            readback = dict(verified_composed_usd=True, solver_position_iterations=128,
                values={'omniphysics:youngsModulus':3e9, 'omniphysics:surfaceThickness':.001})
            report = dict(status='completed', purpose='physical_platen_compression_not_model_grasp',
                prescribed_vertex_animation=False, profile=profile, physics_hz=240,
                unloaded_shape_preserved=True, physical_deformation_observed=True,
                recovered_after_release=True, platen_motion_verified=True, sample_count=1920, video_frames=240,
                material_before_reset=readback, material_after_reset=readback)
            for field, filename in zip(('shell_sha256','registry_sha256','source_sha256','numerics_sha256','readback_source_sha256'), filenames):
                report[field] = hashlib.sha256((package/filename).read_bytes()).hexdigest()
            marker.write_text(json.dumps(report))
            with patch('sim_console.ROOT', root):
                self.assertEqual(sim_console._require_verified_pvc_probe(key), report)
                report['material_after_reset']['solver_position_iterations'] = 32
                marker.write_text(json.dumps(report))
                with self.assertRaises(ValueError): sim_console._require_verified_pvc_probe(key)

    def test_response_diagnostic_keeps_base_unchanged_and_rejects_unbounded_parameters(self):
        base = dict(shell['PROFILE'])
        p = response['diagnostic_profile'](3., 128, 1., base)
        self.assertEqual(p['youngs_modulus_Pa'], 3e9)
        self.assertEqual(p['solver_position_iterations'], 128)
        self.assertEqual(p['mass_kg'], base['mass_kg'])
        self.assertEqual(p['thickness_m'], base['thickness_m'])
        self.assertEqual(base, shell['PROFILE'])
        thick = response['diagnostic_profile'](3., 128, 1.5, base)
        self.assertAlmostEqual(thick['mass_kg'], 1.5*base['mass_kg'])
        for bad in ((20,128,1), (3,256,1), (3,128,5)):
            with self.assertRaises(ValueError): response['diagnostic_profile'](*bad, base)

    def test_hard_to_soft_geometry_and_nonmaterial_parameters_are_identical(self):
        base = shell['PROFILE']
        previous = float('inf')
        points, faces = shell['cup_mesh'](base)
        for key, modulus in registry['STIFFNESS_SPECS']:
            p = registry['profile_for'](key, base)
            self.assertLess(modulus, previous)
            previous = modulus
            self.assertEqual(p['youngs_modulus_Pa'], modulus)
            self.assertAlmostEqual(p['surface_bend_stiffness_Pa'], modulus/(12*(1-.38**2)))
            for field in base.keys()-{'id','youngs_modulus_Pa','surface_bend_stiffness_Pa'}:
                self.assertEqual(p[field], base[field])
            actual, triangles = shell['cup_mesh'](p)
            np.testing.assert_array_equal(actual, points)
            np.testing.assert_array_equal(triangles, faces)
        with self.assertRaises(ValueError): registry['profile_for']('arbitrary', base)
        self.assertEqual(base['youngs_modulus_Pa'], 1e9)

    def test_physical_marker_binds_exact_modulus_and_current_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root/'simulator_profiles/tuned_v1/yubi_isaac_sim_env'
            package.mkdir(parents=True)
            for name in ('pvc_shell.py','pvc_stiffness.py','pvc_stiffness_probe.py'):
                shutil.copyfile(PACKAGE/name, package/name)
            key = registry['STIFFNESS_SPECS'][0][0]
            marker = root/'sim_validation/pvc_stiffness'/key/'verified_probe.json'
            marker.parent.mkdir(parents=True)
            report = dict(status='completed', purpose='physical_platen_compression_not_model_grasp',
                prescribed_vertex_animation=False, profile=registry['profile_for'](key,shell['PROFILE']),
                unloaded_shape_preserved=True,physical_deformation_observed=True,recovered_after_release=True,
                platen_motion_verified=True,sample_count=960,video_frames=240)
            for field, name in (('shell_sha256','pvc_shell.py'),('source_sha256','pvc_stiffness_probe.py'),('registry_sha256','pvc_stiffness.py')):
                report[field] = hashlib.sha256((package/name).read_bytes()).hexdigest()
            marker.write_text(json.dumps(report))
            with patch('sim_console.ROOT', root):
                self.assertEqual(sim_console._require_verified_pvc_probe(key), report)
                report['profile']['youngs_modulus_Pa'] /= 2
                marker.write_text(json.dumps(report))
                with self.assertRaises(ValueError): sim_console._require_verified_pvc_probe(key)
                report['profile'] = registry['profile_for'](key,shell['PROFILE'])
                marker.write_text(json.dumps(report))
                (package/'pvc_stiffness_probe.py').write_text('modified')
                with self.assertRaises(ValueError): sim_console._require_verified_pvc_probe(key)

    def row(self, action=0, **extra):
        return dict(action_index=action, min_node_world_z_m=.801,
                    cup_quaternion_wxyz=[1,0,0,0], cup_linear_velocity_m_s=[0,0,0],
                    cup_angular_velocity_rad_s=[0,0,0], max_nodal_shape_change_m=.002, **extra)

    def test_ten_unique_consecutive_upright_holds_required(self):
        self.assertTrue(stable_grasp([self.row(i) for i in range(10)])['stable_grasp_verified'])
        self.assertFalse(stable_grasp([self.row(i) for i in range(9)])['stable_grasp_verified'])
        self.assertFalse(stable_grasp([self.row(1)]*20)['stable_grasp_verified'])
        self.assertFalse(stable_grasp([self.row(i*2) for i in range(20)])['stable_grasp_verified'])

    def test_rim_deformation_tipping_root_fit_and_fast_motion_are_not_grasps(self):
        for change in ({'min_node_world_z_m':.7505,'cup_position_m':[0,0,.9]},
                       {'min_node_world_z_m':.80025},
                       {'cup_quaternion_wxyz':[.7071,.7071,0,0]},
                       {'max_nodal_shape_change_m':.02}, {'cup_linear_velocity_m_s':[.06,0,0]},
                       {'min_node_world_z_m':float('nan')}):
            rows = [{**self.row(i), **change} for i in range(20)]
            self.assertFalse(stable_grasp(rows)['stable_grasp_verified'])


if __name__ == '__main__': unittest.main()
