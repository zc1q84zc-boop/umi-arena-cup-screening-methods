"""Exercise output reservation without importing the Isaac runtime."""
import ast
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).parent / 'simulator_profiles/tuned_v1/yubi_isaac_sim_env/run.py'
TREE = ast.parse(SOURCE.read_text())
HELPER = next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
              and n.name == '_prepare_recording_directory')
NAMESPACE = {'Path': Path}
exec(compile(ast.Module(body=[HELPER], type_ignores=[]), str(SOURCE), 'exec'), NAMESPACE)
prepare = NAMESPACE['_prepare_recording_directory']


class RecordingStartupTests(unittest.TestCase):
    def test_reserves_directory_before_environment_can_emit_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'new' / 'run'
            prepare(path)
            self.assertTrue(path.is_dir())
            audit = path / 'cup_deformation.jsonl'
            with audit.open('a') as handle:
                handle.write('{"physics_time_s": 0}\n')
            self.assertTrue(audit.is_file())

    def test_empty_directory_allowed_but_recorded_run_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            prepare(path)
            report = path / 'report.json'
            report.write_text('preserve')
            with self.assertRaises(FileExistsError):
                prepare(path)
            self.assertEqual(report.read_text(), 'preserve')

    def test_reservation_precedes_sim_start_and_is_outside_report_handler(self):
        main = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        reservation = next(n for n in main.body if isinstance(n, ast.If)
                           and '_prepare_recording_directory' in ast.unparse(n))
        handler = next(n for n in main.body if isinstance(n, ast.Try))
        self.assertLess(reservation.lineno, handler.lineno)
        self.assertIn('create_sim(', ast.unparse(handler))


if __name__ == '__main__':
    unittest.main()
