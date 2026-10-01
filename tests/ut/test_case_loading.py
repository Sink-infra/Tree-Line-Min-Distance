"""Case catalogue schemas, selection boundaries, and ST fixture generation."""

from dataclasses import asdict
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.common.cases import (
    CASE_GROUPS, COVERAGE_CASES, CORRECTNESS_CASES, MOCK_CASES,
    load_cases, select_cases,
)
from tests.common.verify_result import verify_case_output
from tests.st.scripts.gen_data import main


class CatalogueTests(unittest.TestCase):
    def test_categories_and_named_selections(self):
        self.assertEqual({name: len(cases) for name, cases in CASE_GROUPS.items()},
                         {"coverage": 48, "correctness": 35, "mock": 17})
        full = select_cases("full")
        self.assertEqual(len(full), 83)
        self.assertEqual(len(select_cases("all")), 100)
        self.assertFalse({c.name for c in full} & {c.name for c in MOCK_CASES})
        with self.assertRaises(ValueError):
            select_cases("coverage", names=(MOCK_CASES[0].name,))
        custom = select_cases("custom", names=(COVERAGE_CASES[0].name, MOCK_CASES[0].name))
        self.assertEqual({c.name for c in custom}, {COVERAGE_CASES[0].name, MOCK_CASES[0].name})
        with self.assertRaises(ValueError):
            select_cases("custom")

    def test_tags_and_expectations(self):
        selected = select_cases("coverage", tags=COVERAGE_CASES[0].covers, expectation="success")
        self.assertIn(COVERAGE_CASES[0], selected)
        self.assertTrue(all(set(COVERAGE_CASES[0].covers).issubset(c.covers) for c in selected))
        rejected = select_cases("full", expectation="reject")
        self.assertEqual(len(rejected), 28)
        self.assertTrue(all(c.category == "coverage" and c.expected_error for c in rejected))
        with self.assertRaises(ValueError):
            select_cases("mock", tags=("absent_tag",))
        with self.assertRaises(ValueError):
            select_cases("mock", expectation="reject")

    def test_catalogue_rejects_duplicate_and_invalid_entries(self):
        row = asdict(COVERAGE_CASES[0])
        row.pop("category")
        row = json.loads(json.dumps(row))
        invalid_rows = (
            [row, row],
            [{**row, "name": "coverage_../escape"}],
            [{**row, "tree_points_shape": [True, 1, 3]}],
            [{**row, "unexpected_option": 1}],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for rows in invalid_rows:
                with (root / "coverage.json").open("w") as stream:
                    json.dump({"schema_version": 1, "operator": "TreeLineMinDistance",
                               "category": "coverage", "cases": rows}, stream)
                with self.assertRaises(ValueError):
                    load_cases("coverage", case_root=root)

    def test_rejection_specifications_load_without_operator_acceptance(self):
        loaded = load_cases("coverage")
        rejected = [c for c in loaded if c.expectation == "reject"]
        self.assertTrue(any(0 in c.tree_points_shape for c in rejected))
        self.assertTrue(any(c.dtype == "int32" for c in rejected))
        self.assertTrue(any(c.tree_points_format != "ND" for c in rejected))
        self.assertTrue(all(c.expected_error for c in rejected))

    def test_listing_has_no_fixture_side_effects(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "data"
            output = io.StringIO()
            with redirect_stdout(output):
                result = main(["--suite", "mock", "--case", MOCK_CASES[0].name,
                               "--list", "--output-dir", str(destination)])
            self.assertEqual(result, 0)
            self.assertIn(MOCK_CASES[0].name, output.getvalue())
            self.assertNotIn(MOCK_CASES[1].name, output.getvalue())
            self.assertFalse(destination.exists())

    def test_full_generation_preserves_case_categories_and_goldens(self):
        chosen = (COVERAGE_CASES[0], next(c for c in CORRECTNESS_CASES if c.dtype == "float16"))
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "full"
            args = ["--suite", "full", "--output-dir", str(destination)]
            for case in chosen:
                args += ["--case", case.name]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(manifest["category"], "full")
            self.assertEqual(manifest["counts"], {"total": 2, "success": 2, "reject": 0})
            for case in chosen:
                folder = destination / case.name
                metadata = json.loads((folder / "case.json").read_text())
                self.assertEqual(metadata["category"], case.category)
                distance = np.fromfile(folder / "reference_min_distance_fp64.bin", dtype="<f8")
                distance.astype("<f2" if case.dtype == "float16" else "<f4").tofile(folder / "distance.bin")
                self.assertTrue(verify_case_output(
                    folder, folder / "distance.bin", folder / "reference_nearest_segment_idx_int32.bin").passed)


if __name__ == "__main__":
    unittest.main()
