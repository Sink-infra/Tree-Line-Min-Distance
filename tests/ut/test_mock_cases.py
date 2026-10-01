"""Check Mock declarations, generated tensors, and mathematical references."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.common.checks import check_declarations, check_generated_data
from tests.common.cases import MOCK_CASES as CASES
from tests.common.generate_data import generate_suite
from tests.common.mock_cases import load_mock_inputs
from tests.common.reference import materialize_case, tree_line_min_distance_reference
from tests.common.verify_result import verify_case_output


class MockTests(unittest.TestCase):
    def test_declarations(self):
        check_declarations(self, CASES, "mock")

    def test_generated_data(self):
        check_generated_data(self, CASES, "mock")

    def test_point_order_preserves_geometry_and_reference(self):
        for dtype in ("float16", "float32"):
            case = replace(CASES[1], dtype=dtype, point_order_seed=None)
            shuffled = replace(case, point_order_seed=17)
            before, after = materialize_case(case), materialize_case(shuffled)
            np.testing.assert_array_equal(before.line_segments, after.line_segments)
            self.assertEqual(before.tree_points.dtype, after.tree_points.dtype)
            self.assertTrue(after.tree_points.flags.c_contiguous)
            expected = tree_line_min_distance_reference(case, before)
            observed = tree_line_min_distance_reference(shuffled, after)
            by_point = {point.tobytes(): (distance, index)
                        for point, distance, index in zip(before.tree_points[0],
                            expected.min_distance[0], expected.nearest_segment_idx[0])}
            self.assertEqual(len(by_point), len(before.tree_points[0]))
            self.assertCountEqual(by_point, [point.tobytes() for point in after.tree_points[0]])
            for point, distance, index in zip(after.tree_points[0],
                    observed.min_distance[0], observed.nearest_segment_idx[0]):
                self.assertEqual((distance, index), by_point[point.tobytes()])

    def test_point_order_is_recorded_in_fixture_metadata(self):
        case = replace(CASES[1], point_order_seed=17)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "fixtures"
            generate_suite(output, [case], suite="mock")
            metadata = json.loads((output / case.name / "case.json").read_text())
            self.assertEqual(metadata["patterns"]["point_order_seed"], 17)
            points = np.fromfile(output / case.name / "tree_points.bin", dtype="<f4")
            np.testing.assert_array_equal(points.reshape(case.tree_points_shape),
                                          materialize_case(case).tree_points)

    def test_small_geometry_and_zero_distance_counts(self):
        expected = {
            "mock_13": ([np.sqrt(2)], [0]),
            "mock_14": ([np.sqrt(2)], [0]),
            "mock_15": ([0, 0, 5], [1, 0, 0]),
            "mock_16": ([5, 0, 5], [0, 0, 0]),
            "mock_17": ([np.sqrt(2)], [2]),
        }
        for case in CASES[12:]:
            with self.subTest(case=case.name):
                result = tree_line_min_distance_reference(case, materialize_case(case))
                distance, index = expected[case.name]
                np.testing.assert_allclose(result.min_distance.ravel(), distance, rtol=0, atol=1e-12)
                np.testing.assert_array_equal(result.nearest_segment_idx.ravel(), index)

    def test_dense_fixture_spans_independent_three_dimensional_inputs(self):
        points, segments = load_mock_inputs(CASES[10])
        for values in (points.reshape(-1, 3), segments.reshape(-1, 3)):
            self.assertTrue(np.all(np.abs(values) <= 100))
            self.assertTrue(np.all(np.abs(values.mean(axis=0)) < 2))
            self.assertTrue(np.all((values.std(axis=0) > 50) & (values.std(axis=0) < 65)))

    def test_degenerate_winner_is_before_the_ordinary_segment_start(self):
        points, segments = load_mock_inputs(CASES[14])
        first, last = segments[0, 0]
        self.assertLess(float(points[0, 0, 0]), float(first[0]))
        self.assertGreater(float(points[0, 1, 0]), float(first[0]))
        self.assertLess(float(points[0, 1, 0]), float(last[0]))
        self.assertGreater(float(points[0, 2, 0]), float(last[0]))
        np.testing.assert_array_equal(segments[0, 1, 0], points[0, 0])
        np.testing.assert_array_equal(segments[0, 1, 1], points[0, 0])

    def test_nonzero_tie_uses_a_shared_start_plane(self):
        case = CASES[16]
        data = materialize_case(case)
        np.testing.assert_array_equal(data.line_segments[..., 0, 2], 0)
        self.assertEqual(float(data.line_segments[0, 2, 1, 0] - data.line_segments[0, 2, 0, 0]), 1)
        self.assertEqual(float(data.line_segments[0, 5, 1, 0] - data.line_segments[0, 5, 0, 0]), 1)

    def test_fixed_spatial_populations(self):
        first_counts = (1, 4, 2, 2, 3, 6, 4, 15, 16384, 0, 128, 20)
        plane_counts = (1, 4, 2, 2, 3, 6, 4, 15, 16384, 0, 128, 52)
        last_counts = (1, 5, 1, 1, 7, 15, 2, 13, 16384, 0, 95, 0)
        for case, first_count, plane_count, last_count in zip(CASES[:12], first_counts, plane_counts, last_counts):
            with self.subTest(case=case.name):
                data = materialize_case(case)
                b, n, _ = case.tree_points_shape
                m = case.line_segments_shape[1]
                cell = 1600.0 / max(m, 2)
                cells = data.line_segments.astype(np.float64).mean(axis=2)[..., 0]
                if case.fixture_file or case.input_pattern == "indexed_random_overlapping" or case.input_pattern.startswith("dense_random"):
                    ids = tree_line_min_distance_reference(case, data).nearest_segment_idx
                else:
                    ids = np.rint(data.tree_points[..., 0].astype(np.float64) / cell + (m - 1) / 2).astype(int)
                    self.assertTrue(np.all((ids >= 0) & (ids < m)))
                    cell_to_segment = np.argsort(cells, axis=-1)
                    ids = cell_to_segment[np.arange(b)[:, None], ids]
                if m >= 4:
                    self.assertTrue(np.all(np.any(np.diff(cells, axis=-1) < 0, axis=-1)))
                self.assertEqual(int(np.count_nonzero(ids == 0)), first_count)
                self.assertEqual(int(np.count_nonzero(ids == m - 1)), last_count)
                starts = data.line_segments[..., 0, :]
                same_plane = starts[..., 2] == starts[:, 0:1, 2]
                self.assertEqual(int(np.count_nonzero(same_plane[np.arange(b)[:, None], ids])), plane_count)
                vectors = data.line_segments[..., 1, :].astype(np.float64) - starts.astype(np.float64)
                self.assertFalse(np.any(np.all(vectors[..., 1:] == 0, axis=-1)))
                if case.input_pattern.endswith("no_tail"):
                    self.assertFalse(np.any(ids == m - 1))

    def test_single_point_segment_list_retains_x_direction_after_quantization(self):
        case = CASES[9]
        data = materialize_case(case)
        self.assertTrue(np.all(data.line_segments[..., 0, 0] != data.line_segments[..., 1, 0]))
        result = tree_line_min_distance_reference(case, data)
        self.assertTrue(np.all(result.min_distance > 0.25))
        self.assertTrue(np.all((result.nearest_segment_idx > 0)
                               & (result.nearest_segment_idx < case.line_segments_shape[1] - 1)))

    def test_dense_intervals_have_unique_interior_winners(self):
        for case, dense_batches in ((CASES[10], 7), (CASES[11], 4)):
            data = materialize_case(case)
            selected = data.line_segments[:, np.arange(16) * 1023 // 15].astype(np.float64)
            centers = selected.mean(axis=2)
            widths = np.sum(np.abs(selected[..., 1, :] - selected[..., 0, :]), axis=1)
            dense = np.all(widths > 4 * np.ptp(centers, axis=1), axis=1)
            self.assertEqual(int(dense.sum()), dense_batches)
            expected = tree_line_min_distance_reference(case, data)
            segments = data.line_segments[np.arange(len(data.tree_points))[:, None],
                                         expected.nearest_segment_idx].astype(np.float64)
            vector = segments[..., 1, :] - segments[..., 0, :]
            offset = data.tree_points.astype(np.float64) - segments[..., 0, :]
            ratio = np.sum(offset * vector, axis=-1) / np.sum(vector * vector, axis=-1)
            if case.fixture_file:
                self.assertEqual(round(float(np.mean((ratio < 0) | (ratio > 1))) * 100, 2), 12.62)
                self.assertEqual(round(float(np.mean(ratio < 0)) * 100, 2), 6.15)
                for bound, fraction in ((2, 18.55), (4, 50.86), (8, 86.15)):
                    self.assertEqual(round(float(np.mean(expected.min_distance < bound)) * 100, 2), fraction)
            else:
                self.assertTrue(np.all((ratio > 0.2) & (ratio < 0.8)))
            self.assertTrue(np.all(expected.min_distance > (0.0 if case.name == "mock_11" else 0.5)))
            copy = materialize_case(case)
            copy.tree_points.fill(0)
            np.testing.assert_array_equal(materialize_case(case).tree_points, data.tree_points)

    def test_dense_geometry_seed_and_index_stream_are_independent(self):
        first = replace(CASES[10], input_pattern="dense_random", fixture_file=None, geometry_seed=174422,
                        tree_points_shape=(2, 64, 3),
                        line_segments_shape=(2, 16, 2, 3))
        second = replace(first, geometry_seed=first.geometry_seed + 1)
        data1, data2 = materialize_case(first), materialize_case(second)
        self.assertFalse(np.array_equal(data1.line_segments, data2.line_segments))
        reference1 = tree_line_min_distance_reference(first, data1)
        reference2 = tree_line_min_distance_reference(second, data2)
        np.testing.assert_array_equal(reference1.nearest_segment_idx,
                                      reference2.nearest_segment_idx)

    def test_dense_geometry_seed_metadata_and_validation(self):
        case = replace(CASES[10], input_pattern="dense_random", fixture_file=None, geometry_seed=174422,
                       tree_points_shape=(1, 3, 3),
                       line_segments_shape=(1, 8, 2, 3))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixtures"
            generate_suite(root, [case], suite="mock")
            metadata = json.loads((root / case.name / "case.json").read_text())
            self.assertEqual(metadata["patterns"]["geometry_seed"], case.geometry_seed)
        materialize_case(replace(case, geometry_seed=1))
        for invalid in (-1, True, 1.0, 1.5):
            with self.assertRaises(ValueError):
                materialize_case(replace(case, geometry_seed=invalid))

    def test_quantized_random_points_keep_positive_distance(self):
        for case in CASES[:12]:
            with self.subTest(case=case.name):
                data = materialize_case(case)
                self.assertTrue(np.all(np.any(data.line_segments[..., 0, :] != data.line_segments[..., 1, :], axis=-1)))
                b, n, _ = case.tree_points_shape
                m = case.line_segments_shape[1]
                step = 1600 / max(m, 2)
                axis = 1 if case.input_pattern.startswith("parallel_") else 0
                if case.fixture_file or case.input_pattern == "indexed_random_overlapping" or case.input_pattern.startswith("dense_random"):
                    ids = tree_line_min_distance_reference(case, data).nearest_segment_idx
                else:
                    ids = np.rint(data.tree_points[..., axis].astype(np.float64) / step + (m - 1) / 2).astype(int)
                    centers = data.line_segments.astype(np.float64).mean(axis=2)[..., axis]
                    cell_to_segment = np.argsort(centers, axis=-1)
                    ids = cell_to_segment[np.arange(b)[:, None], ids]
                segments = data.line_segments[np.arange(b)[:, None], ids].astype(np.float64)
                offset = data.tree_points.astype(np.float64) - segments[..., 0, :]
                vector = segments[..., 1, :] - segments[..., 0, :]
                length = np.sum(vector * vector, axis=-1)
                t = np.zeros((b, n))
                np.divide(np.sum(offset * vector, axis=-1), length, out=t, where=length != 0)
                residual = offset - np.clip(t, 0, 1)[..., None] * vector
                distances = np.linalg.norm(residual, axis=-1)
                if case.input_pattern in ("indexed_random_near_zero", "indexed_random_tail_mix_near_zero", "parallel_near_zero"):
                    self.assertEqual(int(np.count_nonzero(distances <= 0.001)), 1)
                    self.assertTrue(np.all(distances > 0))
                elif case.name == "mock_11":
                    self.assertTrue(np.all(distances > 0))
                    self.assertTrue(np.any(distances < 0.25))
                else:
                    self.assertTrue(np.all(distances > 0.25))

    def test_fixed_inputs_are_independent_and_match_the_declared_encoding(self):
        for case in CASES:
            if not case.fixture_file:
                continue
            points, segments = load_mock_inputs(case)
            self.assertEqual(points.shape, case.tree_points_shape)
            self.assertEqual(segments.shape, case.line_segments_shape)
            self.assertEqual(points.dtype, np.dtype(case.dtype))
            self.assertTrue(points.flags.c_contiguous and segments.flags.c_contiguous)
            expected = points.copy()
            points.fill(0)
            np.testing.assert_array_equal(load_mock_inputs(case)[0], expected)
        with self.assertRaises(ValueError):
            load_mock_inputs(replace(CASES[0], fixture_file="../outside.npz"))
        with self.assertRaises(ValueError):
            load_mock_inputs(replace(CASES[0], tree_points_shape=(1, 2, 3)))

    def test_errors_in_different_outputs_keep_separate_tensor_fractions(self):
        case = CASES[14]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_suite(root / "fixtures", [case], suite="mock")
            np.zeros(3, dtype="<f4").tofile(root / "distance.bin")
            np.zeros(3, dtype="<i4").tofile(root / "index.bin")
            result = verify_case_output(root / "fixtures" / case.name,
                                        root / "distance.bin", root / "index.bin")
            self.assertFalse(result.passed)
            self.assertEqual(result.distance_mismatch_count, 1)
            self.assertEqual(result.index_mismatch_count, 1)
            self.assertEqual(result.mismatch_count, 2)
            self.assertAlmostEqual(result.output_error_percent, 100 / 3)


if __name__ == "__main__":
    unittest.main()
