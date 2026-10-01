"""Coverage contract, catalog, and data-generation self-checks."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.common.checks import check_declarations, check_generated_data
from tests.common.generate_data import _select_cases, generate_suite
from tests.common.reference import (
    CaseData, materialize_case, tree_line_min_distance_reference, validate_case, validate_data,
    ReferenceResult,
)
from tests.common.cases import COVERAGE_CASES as CASES
from tests.common.cases import CORRECTNESS_CASES
from tests.__main__ import summary


class CoverageTests(unittest.TestCase):
    def test_declarations(self):
        check_declarations(self, CASES, "coverage")

    def test_generated_data(self):
        check_generated_data(self, CASES, "coverage")

    def test_each_dtype_covers_dimension_extremes_and_documented_tails(self):
        for dtype in ("float16", "float32"):
            cases = [c for c in CASES if c.expectation == "success" and c.dtype == dtype]
            self.assertTrue(any(c.tree_points_shape == (8, 16384, 3)
                                and c.line_segments_shape == (8, 1024, 2, 3) for c in cases))
            self.assertTrue({1, 3, 19, 33, 127, 1003, 4097, 16384}.issubset(
                {c.tree_points_shape[1] for c in cases}))
            self.assertTrue({1, 7, 19, 31, 33, 129, 513, 1024}.issubset(
                {c.line_segments_shape[1] for c in cases}))

    def test_rejected_inputs_cannot_be_evaluated(self):
        for case in CASES:
            if case.expectation == "reject":
                with self.subTest(case=case.name), self.assertRaisesRegex(ValueError, "rejected case"):
                    tree_line_min_distance_reference(case, materialize_case(case))

    def test_coordinate_range_is_a_recommendation(self):
        case = CASES[0]
        data = CaseData(np.full(case.tree_points_shape, 1001, dtype="<f4"),
                        np.zeros(case.line_segments_shape, dtype="<f4"))
        self.assertEqual(validate_data(case, data).errors, ())

    def test_actual_tensor_contract_is_checked(self):
        case = CASES[1]
        data = materialize_case(case)
        with self.subTest("wrong_shape"):
            wrong = CaseData(data.tree_points[:, :1], data.line_segments)
            self.assertIn("TREE_POINTS_SHAPE", validate_data(case, wrong).errors)
        with self.subTest("wrong_dtype"):
            wrong = CaseData(data.tree_points.astype(np.float64), data.line_segments)
            self.assertIn("TREE_POINTS_STORAGE_DTYPE", validate_data(case, wrong).errors)
        with self.subTest("noncontiguous"):
            backing = np.empty((1, 3, 6), dtype=np.float32)
            backing[..., ::2] = data.tree_points
            wrong = CaseData(backing[..., ::2], data.line_segments)
            self.assertIn("TREE_POINTS_CONTIGUOUS", validate_data(case, wrong).errors)

    def test_generator_audits_before_creating_directory(self):
        case = CASES[0]
        invalid_collections = (
            (), (case, case), (case, CORRECTNESS_CASES[0]),
            (replace(case, name="coverage_../escape"),),
            (replace(case, category="other"),),
            (replace(case, expectation="reject", expected_error="N_RANGE"),),
            (replace(case, expectation="success", tree_points_shape=(1, 0, 3)),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            for cases in invalid_collections:
                with self.subTest(cases=cases), self.assertRaises(ValueError):
                    generate_suite(root, cases)
                self.assertFalse(root.exists())

    def test_selection_and_summary(self):
        self.assertEqual(_select_cases(CASES, "success", [CASES[0].name]), (CASES[0],))
        for expectation, names in (("all", ["missing"]), ("reject", [CASES[0].name])):
            with self.assertRaises(ValueError):
                _select_cases(CASES, expectation, names)
        result = summary()
        self.assertEqual(set(result), {"coverage", "correctness", "mock"})
        self.assertEqual(result["mock"]["count"], 17)
        self.assertEqual(result["coverage"]["count"], len(CASES))
        self.assertEqual(result["coverage"]["max_pairs"], 8 * 16384 * 1024)
        self.assertEqual(validate_case(CASES[0]).errors, ())

    def test_repeated_generation_has_identical_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = generate_suite(root / "first", (CASES[1],))
            second = generate_suite(root / "second", (CASES[1],))
            self.assertEqual(first, second)
            self.assertEqual((root / "first" / CASES[1].name / "case.json").read_bytes(),
                             (root / "second" / CASES[1].name / "case.json").read_bytes())

    def test_large_distance_corruption_is_detected(self):
        for dtype in ("float16", "float32"):
            case = next(c for c in CASES if c.name == f"coverage_{dtype}_n1003_m129")
            for location in (0, -1):
                def wrong_reference(case, data, **kwargs):
                    result = tree_line_min_distance_reference(case, data, **kwargs)
                    distance = result.min_distance.copy()
                    distance.ravel()[location] += 1.0
                    return ReferenceResult(distance, result.nearest_segment_idx)
                with self.subTest(dtype=dtype, location=location):
                    with patch("tests.common.generate_data.tree_line_min_distance_reference", wrong_reference):
                        with self.assertRaises(AssertionError):
                            # A standalone TestCase lets this helper's assertion propagate.
                            check_generated_data(unittest.TestCase(), (case,), "coverage")

    def test_rejection_dtypes_have_portable_storage_and_normal_errors(self):
        for dtype in ("bool", "int8", "int16", "int64", "uint64"):
            for field in ("dtype", "line_segments_dtype"):
                error = "TREE_POINTS_DTYPE" if field == "dtype" else "LINE_SEGMENTS_DTYPE"
                case = replace(CASES[0], name=f"coverage_reject_{field}_{dtype}",
                               expectation="reject", expected_error=error, **{field: dtype})
                with self.subTest(dtype=dtype, field=field):
                    self.assertIn(error, validate_data(case, materialize_case(case)).errors)
                    with tempfile.TemporaryDirectory() as temporary:
                        generate_suite(Path(temporary) / "data", (case,))
        unknown = replace(CASES[0], dtype="unsupported_encoding")
        arrays = CaseData(np.zeros((1, 1, 3), dtype=np.float32), np.zeros((1, 1, 2, 3), dtype=np.float32))
        self.assertIn("TREE_POINTS_DTYPE", validate_data(unknown, arrays).errors)
        with self.assertRaisesRegex(ValueError, "portable fixture storage"):
            materialize_case(unknown)

    def test_noninteger_or_negative_shape_dimensions_are_rejected(self):
        for dimension in (1.0, True, "1", -1):
            case = replace(CASES[0], tree_points_shape=(dimension, 1, 3))
            with self.subTest(dimension=dimension):
                self.assertIn("TREE_POINTS_DIMENSIONS", validate_case(case).errors)
                with self.assertRaisesRegex(ValueError, "integer dimensions"):
                    materialize_case(case)


if __name__ == "__main__":
    unittest.main()
