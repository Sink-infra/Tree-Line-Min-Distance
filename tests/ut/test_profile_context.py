"""Shared Host catalogue, native launch identity, and incomplete rounds."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.common.cases import MOCK_CASES
from tests.common.profile_device import (
    collect_shared_samples, launch_attribution, prepare_context_catalog, prepare_context_inputs,
)
from tests.st.scripts.profile_mock_workloads import summarize


class ProfileContextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cases = [MOCK_CASES[0], MOCK_CASES[2], MOCK_CASES[12]]

    def log(self, text):
        path = self.root / 'process.log'
        path.write_text(text)
        return path

    def test_omitted_cases_have_only_host_shape_metadata(self):
        folder = self.root / 'catalog'
        prepare_context_catalog(folder, MOCK_CASES, self.cases)
        self.assertEqual((folder / 'order.txt').read_text().splitlines(), [c.name for c in MOCK_CASES])
        self.assertEqual({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()},
                         {'order.txt'} | {c.name + '/case.conf' for c in MOCK_CASES})
        self.assertEqual((folder / 'mock_02/case.conf').read_text(), 'b=1\nn=19\nm=7\ndtype=float32\n')

    def test_noncanonical_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ordered'):
            prepare_context_catalog(self.root / 'catalog', MOCK_CASES, self.cases[::-1])

    def test_unmeasured_prefix_contains_only_original_inputs(self):
        selected = [MOCK_CASES[0], MOCK_CASES[2]]
        prefix = prepare_context_inputs(self.root, MOCK_CASES, selected)
        self.assertEqual([c.name for c in prefix], ['mock_01', 'mock_02', 'mock_03'])
        self.assertEqual({p.name for p in (self.root / 'mock_02').iterdir()},
                         {'case.conf', 'tree_points.bin', 'line_segments.bin'})
        self.assertFalse((self.root / 'mock_04').exists())
        from tests.common.reference import materialize_case
        from tests.common.tensor_io import sha256
        import hashlib
        data = materialize_case(MOCK_CASES[1])
        for name in ('tree_points', 'line_segments'):
            self.assertEqual(sha256(self.root / 'mock_02' / (name + '.bin')),
                             hashlib.sha256(getattr(data, name).tobytes(order='C')).hexdigest())

    def test_launch_indices_are_per_kernel_image(self):
        log = self.log('<CaseReplay> mock_01 123\n'
                       '<ProfData> Start profiling on device 0, kernel: TreeLineMinDistance_float_0\n'
                       '<CaseReplay> mock_03 123\n'
                       '<ProfData> Start profiling on device 0, kernel: TreeLineMinDistance_half_0\n'
                       '<CaseReplay> mock_13 123\n'
                       '<ProfData> Start profiling on device 0, kernel: TreeLineMinDistance_float_0\n')
        mapping, pid = launch_attribution(log, self.cases)
        self.assertEqual(pid, '123')
        self.assertEqual(mapping, {('TreeLineMinDistance_float_0', '0'): 'mock_01',
                                   ('TreeLineMinDistance_half_0', '0'): 'mock_03',
                                   ('TreeLineMinDistance_float_0', '1'): 'mock_13'})

    def test_changed_process_cannot_be_attributed_as_shared(self):
        log = self.log('<CaseReplay> mock_01 123\n<CaseReplay> mock_03 456\n')
        with self.assertRaisesRegex(ValueError, 'one application'):
            launch_attribution(log, self.cases)

    def test_five_calls_share_a_case_marker(self):
        kernel = '<ProfData> Start profiling on device 0, kernel: TreeLineMinDistance_float_0\n'
        mapping, _ = launch_attribution(self.log('<CaseReplay> mock_01 123\n' + kernel * 5), self.cases)
        self.assertEqual(list(mapping.values()), ['mock_01'] * 5)

    def test_extra_kernel_launch_cannot_share_a_case_marker(self):
        log = self.log('<CaseReplay> mock_01 123\n'
                       + '<ProfData> Start profiling on device 0, kernel: TreeLineMinDistance_float_0\n' * 6)
        with self.assertRaisesRegex(ValueError, 'call count'):
            launch_attribution(log, self.cases)

    def test_runtime_failure_allows_only_a_case_prefix(self):
        self.assertEqual(launch_attribution(self.log('<CaseReplay> mock_01 123\n'), self.cases), ({}, '123'))
        with self.assertRaisesRegex(ValueError, 'requested order'):
            launch_attribution(self.log('<CaseReplay> mock_03 123\n'), self.cases)

    def test_complete_case_requires_five_attributed_tasks(self):
        cases = [self.cases[0], self.cases[2]]
        kernel = 'TreeLineMinDistance_float_0'
        line = '<ProfData> Start profiling on device 0, kernel: ' + kernel + '\n'
        log = self.log('<CaseReplay> mock_01 123\n' + line * 5
                       + '<CaseReplay> mock_13 123\n' + line * 5)
        outcomes = [{'name': c.name, 'execution': {'status': 'executed'}} for c in cases]
        samples = [{'op_name': kernel, 'launch_index': str(i), 'process_id': '123',
                    'duration_us': i + 1} for i in range(10)]
        with patch('tests.common.profile_device.collect_samples',
                   return_value={'samples': samples, 'csv_files': []}), \
                patch('tests.common.profile_device.kernel_dtypes', return_value={kernel: 'float32'}), \
                patch('tests.common.profile_device.verify_profiled_binary') as verify:
            profiles, _ = collect_shared_samples(self.root, log, cases, outcomes, self.root)
            self.assertEqual(profiles['mock_01']['statistics']['median_us'], 3)
            self.assertEqual(profiles['mock_13']['statistics']['median_us'], 8)
            self.assertEqual(verify.call_count, 10)
            samples.pop()
            with self.assertRaisesRegex(ValueError, 'incomplete native call sequence'):
                collect_shared_samples(self.root, log, cases, outcomes, self.root)

    def test_missing_round_cannot_produce_valid_timing(self):
        case = self.cases[0]
        runs = {case.name: [{'status': 'Pass', 'passed': True, 'error_percent': 0.0}] * 4}
        entry = summarize([case], runs, 5)[0]
        self.assertFalse(entry['passed'])
        self.assertFalse(entry['timing_valid'])
        self.assertEqual(entry['accuracy_observations'],
                         [{'status': 'Pass', 'error_percent': 0.0, 'count': 4}])

    def test_context_kernel_cannot_be_assigned_to_a_selected_case(self):
        kernel = 'TreeLineMinDistance_float_0'
        line = '<ProfData> Start profiling on device 0, kernel: ' + kernel + '\n'
        log = self.log('<CaseReplay> mock_01 123\n' + line * 5
                       + '<CaseReplay> mock_02 123\n' + line)
        samples = [{'op_name': kernel, 'launch_index': str(i), 'process_id': '123',
                    'duration_us': i + 1} for i in range(6)]
        outcome = [{'name': 'mock_01', 'execution': {'status': 'executed'}}]
        with patch('tests.common.profile_device.collect_samples',
                   return_value={'samples': samples, 'csv_files': []}), \
                patch('tests.common.profile_device.kernel_dtypes', return_value={kernel: 'float32'}), \
                patch('tests.common.profile_device.verify_profiled_binary'):
            with self.assertRaisesRegex(ValueError, 'unique Case'):
                collect_shared_samples(self.root, log, [MOCK_CASES[0]], outcome, self.root,
                                       launch_cases=MOCK_CASES[:2])


if __name__ == '__main__':
    unittest.main()
