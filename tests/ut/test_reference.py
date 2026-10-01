"""Analytical geometry, scalar-oracle, and output-verifier self-checks."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.common.cases import STORAGE_DTYPES, TOLERANCES
from tests.common.checks import check_declarations, check_generated_data
from tests.common.generate_data import generate_suite
from tests.common.reference import CaseData, materialize_case, tree_line_min_distance_reference
from tests.common.verify_result import verify_case_output
from tests.common.cases import CORRECTNESS_CASES as CASES


def find_case(pattern, dtype="float32"):
    return next(c for c in CASES if c.input_pattern == pattern and c.dtype == dtype)


def scalar_reference(data):
    """Independent scalar formulation using AP - t AC, with strict tie order."""
    distances = np.empty(data.tree_points.shape[:2], dtype=np.float64)
    indices = np.empty(data.tree_points.shape[:2], dtype=np.int32)
    for batch, points in enumerate(data.tree_points):
        for i, point in enumerate(points):
            best, winner = math.inf, -1
            for j, segment in enumerate(data.line_segments[batch]):
                ac = [float(segment[1, k]) - float(segment[0, k]) for k in range(3)]
                ap = [float(point[k]) - float(segment[0, k]) for k in range(3)]
                length = sum(v * v for v in ac)
                t = 0.0 if length == 0 else min(1.0, max(0.0, sum(x * y for x, y in zip(ap, ac)) / length))
                squared = sum((x - t * y) ** 2 for x, y in zip(ap, ac))
                if squared < best:
                    best, winner = squared, j
            distances[batch, i], indices[batch, i] = math.sqrt(best), winner
    return distances, indices


class CorrectnessTests(unittest.TestCase):
    def test_declarations(self):
        check_declarations(self, CASES, "correctness")

    def test_generated_data(self):
        check_generated_data(self, CASES, "correctness")

    def test_analytical_geometry_for_both_dtypes(self):
        targets = {
            "documented_endpoints": ([5, 0, 5], [0, 0, 0]),
            "documented_tie": ([math.sqrt(2)], [0]),
            "documented_zero_length": ([5], [0]),
            "oblique_3d": ([0, 0, math.sqrt(18)], [0, 0, 0]),
            "negative_coordinates": ([3, 0, 5], [0, 0, 0]),
            "all_degenerate": ([0, 0, 0], [0, 3, 6]),
            "tie_groups": ([5, 5, 5], [19, 31, 32]),
            "large_coordinates": ([math.sqrt(2) * 1000, 1000, 0], [0, 0, 0]),
        }
        for dtype in ("float16", "float32"):
            for pattern, (distance, index) in targets.items():
                with self.subTest(dtype=dtype, pattern=pattern):
                    case = find_case(pattern, dtype)
                    result = tree_line_min_distance_reference(case, materialize_case(case))
                    np.testing.assert_allclose(result.min_distance, [distance], rtol=1e-14, atol=1e-14)
                    np.testing.assert_array_equal(result.nearest_segment_idx, [index])

    def test_zero_distance_and_positive_tail_cases(self):
        for pattern, distance, index in (("all_zero", 0, 0), ("zero_distance_tie", 0, 2),
                                         ("cross_tile_tie", 5, 31), ("last_segment", 1, 512),
                                         ("tail_positive", 5, 32)):
            for dtype in ("float16", "float32"):
                case = find_case(pattern, dtype)
                result = tree_line_min_distance_reference(case, materialize_case(case))
                np.testing.assert_array_equal(result.min_distance, np.full(result.min_distance.shape, distance))
                np.testing.assert_array_equal(result.nearest_segment_idx, np.full(result.min_distance.shape, index))

    def test_batches_are_independent(self):
        for dtype in ("float16", "float32"):
            case = find_case("batch_isolation", dtype)
            result = tree_line_min_distance_reference(case, materialize_case(case))
            np.testing.assert_array_equal(result.min_distance, np.repeat([[1], [2], [3]], 3, axis=1))
            np.testing.assert_array_equal(result.nearest_segment_idx, np.repeat([[0], [3], [6]], 3, axis=1))

    def test_zero_lengths_do_not_divide_and_tiny_lengths_are_preserved(self):
        for case in CASES:
            if "zero_length" in case.covers or "tiny_nonzero_length" in case.covers:
                with self.subTest(case=case.name), np.errstate(divide="raise", invalid="raise", over="raise"):
                    result = tree_line_min_distance_reference(case, materialize_case(case))
                    self.assertTrue(np.isfinite(result.min_distance).all())
                    if case.input_pattern == "tiny_segment":
                        self.assertEqual(result.nearest_segment_idx[0, 0], 1)
                        self.assertEqual(result.min_distance[0, 0], 0)
        case = find_case("quantized_zero_length", "float16")
        data = materialize_case(case)
        np.testing.assert_array_equal(data.line_segments[:, :, 0], data.line_segments[:, :, 1])
        result = tree_line_min_distance_reference(case, data)
        np.testing.assert_array_equal(result.min_distance, [[5, 0, 5]])

    def test_fp16_argmin_is_computed_before_output_conversion(self):
        case = find_case("output_rounding_tie", "float16")
        data = materialize_case(case)
        point_distances = np.linalg.norm(data.line_segments[0, :, 0].astype(np.float32), axis=-1)
        self.assertGreater(point_distances[0], point_distances[1])
        self.assertEqual(point_distances.astype(np.float16)[0], point_distances.astype(np.float16)[1])
        result = tree_line_min_distance_reference(case, data)
        self.assertEqual(result.nearest_segment_idx[0, 0], 1)

    def test_reference_matches_independent_scalar_formulation(self):
        for dtype in ("float16", "float32"):
            original = find_case("separated_random", dtype)
            case = replace(original, tree_points_shape=(2, 19, 3), line_segments_shape=(2, 7, 2, 3))
            data = materialize_case(case)
            expected, idx = scalar_reference(data)
            actual = tree_line_min_distance_reference(case, data, point_tile=3, segment_tile=2)
            np.testing.assert_allclose(actual.min_distance, expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(actual.nearest_segment_idx, idx)

    def test_tile_sizes_do_not_change_ties_or_values(self):
        for pattern in ("cross_tile_tie", "tie_groups", "separated_random"):
            case = find_case(pattern)
            data = materialize_case(case)
            expected = tree_line_min_distance_reference(case, data)
            for point_tile, segment_tile in ((1, 1), (7, 31), (19, 33), (1024, 1024)):
                actual = tree_line_min_distance_reference(case, data, point_tile=point_tile, segment_tile=segment_tile)
                np.testing.assert_array_equal(actual.min_distance, expected.min_distance)
                np.testing.assert_array_equal(actual.nearest_segment_idx, expected.nearest_segment_idx)
            for point_tile, segment_tile in ((0, 1), (1, 0), (-1, 2)):
                with self.assertRaises(ValueError):
                    tree_line_min_distance_reference(case, data, point_tile=point_tile, segment_tile=segment_tile)

    def test_random_cases_have_separated_nearest_distances_in_fp32(self):
        # Check the input construction after quantization, independently in FP32.
        for dtype in ("float16", "float32"):
            case = find_case("separated_random", dtype)
            data = materialize_case(case)
            a = data.line_segments[:, :, 0].astype(np.float32)
            ac = data.line_segments[:, :, 1].astype(np.float32) - a
            ap = data.tree_points.astype(np.float32)[:, :, None] - a[:, None]
            length = np.sum(ac * ac, axis=-1)
            t = np.sum(ap * ac[:, None], axis=-1) / length[:, None]
            q = a[:, None] + np.clip(t, 0, 1)[..., None] * ac[:, None]
            d2 = np.sum((data.tree_points.astype(np.float32)[:, :, None] - q) ** 2, axis=-1)
            order = np.sort(d2, axis=-1)
            self.assertTrue(np.all(order[..., 1] - order[..., 0] > 1e-3 * np.maximum(1, order[..., 1])))
            reference = tree_line_min_distance_reference(case, data)
            np.testing.assert_array_equal(np.argmin(d2, axis=-1), reference.nearest_segment_idx)

    def test_verifier_rejects_wrong_distance_or_index_independently(self):
        for dtype in ("float16", "float32"):
            case = find_case("documented_zero_length", dtype)
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                generate_suite(root / "data", (case,))
                folder = root / "data" / case.name
                distance_path, index_path = root / "distance.bin", root / "index.bin"
                np.array([5], dtype=STORAGE_DTYPES[dtype]).tofile(distance_path)
                np.array([0], dtype="<i4").tofile(index_path)
                self.assertTrue(verify_case_output(folder, distance_path, index_path).passed)
                for value in (25, 0, -1e-5, np.nan, np.inf, -np.inf):
                    np.array([value], dtype=STORAGE_DTYPES[dtype]).tofile(distance_path)
                    result = verify_case_output(folder, distance_path, index_path)
                    self.assertFalse(result.passed)
                    self.assertEqual(result.distance_mismatch_count, 1)
                    self.assertEqual(result.index_mismatch_count, 0)
                    self.assertEqual(result.first_distance_mismatch, [0, 0])
                    json.dumps(result.__dict__, allow_nan=False)
                np.array([5], dtype=STORAGE_DTYPES[dtype]).tofile(distance_path)
                for value in (-1, 1, 2, 2**31 - 1):
                    np.array([value], dtype="<i4").tofile(index_path)
                    result = verify_case_output(folder, distance_path, index_path)
                    self.assertFalse(result.passed)
                    self.assertEqual(result.index_mismatch_count, 1)
                    self.assertEqual(result.distance_mismatch_count, 0)

    def test_verifier_applies_each_element_tolerance(self):
        case = replace(find_case("tail_positive"), tree_points_shape=(4, 4097, 3))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_suite(root / "data", (case,))
            folder = root / "data" / case.name
            actual = np.full(case.tree_points_shape[:2], 5, dtype="<f4")
            actual[-1, -1] += 0.01
            atol, rtol = TOLERANCES[case.dtype]
            self.assertLess(np.mean(np.abs(actual - 5)), atol + rtol * 5)
            actual.tofile(root / "distance.bin")
            indices = folder / "reference_nearest_segment_idx_int32.bin"
            result = verify_case_output(folder, root / "distance.bin", indices)
            self.assertFalse(result.passed)
            self.assertEqual(result.distance_mismatch_count, 1)
            self.assertEqual(result.first_distance_mismatch, [3, 4096])

    def test_verifier_uses_contract_tolerance_without_rounding_reference(self):
        for dtype in ("float16", "float32"):
            case = find_case("documented_tie", dtype)
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                generate_suite(root / "data", (case,))
                folder = root / "data" / case.name
                actual_path = root / "distance.bin"
                index_path = folder / "reference_nearest_segment_idx_int32.bin"
                atol, rtol = TOLERANCES[dtype]
                target = math.sqrt(2)
                for fraction, passes in ((0.5, True), (2.0, False)):
                    np.array([target + fraction * (atol + rtol * target)], dtype=STORAGE_DTYPES[dtype]).tofile(actual_path)
                    self.assertEqual(verify_case_output(folder, actual_path, index_path).passed, passes)

    def test_verifier_rejects_incomplete_files_and_corrupted_references(self):
        case = find_case("documented_endpoints")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_suite(root / "data", (case,))
            folder = root / "data" / case.name
            distance_path, index_path = root / "distance.bin", root / "index.bin"
            np.array([5, 0, 5], dtype="<f4").tofile(distance_path)
            np.array([0, 0, 0], dtype="<i4").tofile(index_path)
            for path in (distance_path, index_path):
                original = path.read_bytes()
                for content in (original[:-1], original + b"x", b""):
                    path.write_bytes(content)
                    with self.assertRaisesRegex(ValueError, "bytes"):
                        verify_case_output(folder, distance_path, index_path)
                path.write_bytes(original)
            reference = folder / "reference_min_distance_fp64.bin"
            reference.write_bytes(b"\x00" * reference.stat().st_size)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_case_output(folder, distance_path, index_path)

    def test_verifier_rejects_changed_contract_or_malformed_metadata(self):
        case = find_case("documented_endpoints")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_suite(root / "data", (case,))
            folder = root / "data" / case.name
            metadata_path = folder / "case.json"
            original = metadata_path.read_text()
            for field in ("schema", "tolerance", "input_shape", "output_shape", "root"):
                metadata = json.loads(original)
                if field == "schema":
                    metadata["schema_version"] = 99
                elif field == "tolerance":
                    metadata["expected_output"]["min_distance"]["comparison"]["atol"] = 1.0
                elif field == "input_shape":
                    metadata["input"]["line_segments"]["shape"] = [1, 1, 3, 3]
                elif field == "output_shape":
                    metadata["expected_output"]["min_distance"]["shape"] = [1.0, 3]
                else:
                    metadata = []
                metadata_path.write_text(json.dumps(metadata))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    verify_case_output(folder, root / "unused_distance", root / "unused_index")

    def test_verifier_rejects_changed_missing_or_truncated_inputs(self):
        case = find_case("documented_endpoints")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_suite(root / "data", (case,))
            folder = root / "data" / case.name
            distance_path, index_path = root / "distance.bin", root / "index.bin"
            np.array([5, 0, 5], dtype="<f4").tofile(distance_path)
            np.array([0, 0, 0], dtype="<i4").tofile(index_path)
            for name in ("tree_points", "line_segments"):
                path = folder / (name + ".bin")
                original = path.read_bytes()
                changed = np.frombuffer(original, dtype="<f4").copy()
                changed[0] = -9
                changed.tofile(path)
                with self.subTest(input=name, failure="changed"), self.assertRaisesRegex(ValueError, "SHA256"):
                    verify_case_output(folder, distance_path, index_path)
                path.write_bytes(original[:-1])
                with self.subTest(input=name, failure="truncated"), self.assertRaisesRegex(ValueError, "bytes"):
                    verify_case_output(folder, distance_path, index_path)
                path.unlink()
                with self.subTest(input=name, failure="missing"), self.assertRaises(FileNotFoundError):
                    verify_case_output(folder, distance_path, index_path)
                path.write_bytes(original)
            self.assertTrue(verify_case_output(folder, distance_path, index_path).passed)


if __name__ == "__main__":
    unittest.main()
