"""Suite declaration and generated-asset self-checks."""

import json
from pathlib import Path
import tempfile

import numpy as np

from tests.common.cases import STORAGE_DTYPES, TOLERANCES
from tests.common.generate_data import generate_suite, sha256
from tests.common.reference import materialize_case, tree_line_min_distance_reference, validate_data
from tests.common.verify_result import verify_case_output


def _known_nearest_distances(data, ids):
    """Independently evaluate AP - t AC for the construction's known winners."""
    batch = np.arange(ids.shape[0])[:, None]
    nearest = data.line_segments[batch, ids].astype(np.float64)
    ap = data.tree_points.astype(np.float64) - nearest[..., 0, :]
    ac = nearest[..., 1, :] - nearest[..., 0, :]
    length = np.sum(ac * ac, axis=-1)
    projection = np.zeros(ids.shape, dtype=np.float64)
    np.divide(np.sum(ap * ac, axis=-1), length, out=projection, where=length != 0)
    residual = ap - np.clip(projection, 0, 1)[..., None] * ac
    return np.sqrt(np.sum(residual * residual, axis=-1))


def check_declarations(test, cases, category):
    test.assertEqual(len(cases), len({case.name for case in cases}))
    for case in cases:
        with test.subTest(case=case.name):
            test.assertEqual(case.category, category)
            test.assertTrue(case.name.startswith(category + "_"))
            validation = validate_data(case, materialize_case(case))
            if case.expectation == "reject":
                test.assertIn(case.expected_error, validation.errors)
            else:
                test.assertEqual(case.expectation, "success")
                test.assertEqual(validation.errors, ())
                test.assertTrue(case.covers)


def check_generated_data(test, cases, category):
    """Generate every case, including maximum shapes, and verify stored outputs."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / category
        manifest = generate_suite(root, cases)
        test.assertEqual(manifest["category"], category)
        test.assertEqual(manifest["counts"]["total"], len(cases))
        test.assertEqual(manifest["catalog_audit_failures"], [])
        test.assertEqual(manifest, json.loads((root / "manifest.json").read_text()))
        for case, entry in zip(cases, manifest["cases"]):
            with test.subTest(case=case.name):
                folder = root / case.name
                test.assertEqual(entry["case_metadata_sha256"], sha256(folder / "case.json"))
                metadata = json.loads((folder / "case.json").read_text())
                test.assertEqual(metadata["category"], category)
                test.assertEqual(metadata["attributes"], {})
                data = materialize_case(case)
                for name, values in (("tree_points", data.tree_points), ("line_segments", data.line_segments)):
                    descriptor = metadata["files"][name]
                    path = folder / descriptor["path"]
                    test.assertEqual(path.stat().st_size, values.nbytes)
                    test.assertEqual(sha256(path), descriptor["sha256"])
                    np.testing.assert_array_equal(
                        np.fromfile(path, dtype=descriptor["storage_dtype"]), values.ravel())
                if case.expectation == "reject":
                    test.assertIsNone(metadata["expected_output"])
                    test.assertEqual(set(metadata["files"]), {"tree_points", "line_segments"})
                    continue
                policy = metadata["expected_output"]["min_distance"]["comparison"]
                test.assertEqual((policy["atol"], policy["rtol"]), TOLERANCES[case.dtype])
                distance_path = folder / metadata["files"]["reference_min_distance"]["path"]
                index_path = folder / metadata["files"]["reference_nearest_segment_idx"]["path"]
                stored = np.fromfile(distance_path, dtype="<f8")
                stored_idx = np.fromfile(index_path, dtype="<i4")
                test.assertEqual(stored.size, case.tree_points_shape[0] * case.tree_points_shape[1])
                if case.pair_count <= 100000:
                    reference = tree_line_min_distance_reference(case, data, point_tile=31, segment_tile=19)
                    np.testing.assert_array_equal(stored, reference.min_distance.ravel())
                    np.testing.assert_array_equal(stored_idx, reference.nearest_segment_idx.ravel())
                elif case.input_pattern == "separated_random":
                    b, n = case.tree_points_shape[:2]
                    ids = (np.arange(n)[None, :] * 17 + np.arange(b)[:, None] * 13) % case.line_segments_shape[1]
                    np.testing.assert_array_equal(stored_idx, ids.ravel())
                    # O(B*N) validation checks every distance even at maximum shapes.
                    expected = _known_nearest_distances(data, ids)
                    np.testing.assert_allclose(stored, expected.ravel(), rtol=1e-11, atol=1e-10)
                else:
                    reference = tree_line_min_distance_reference(case, data, point_tile=31, segment_tile=19)
                    np.testing.assert_array_equal(stored, reference.min_distance.ravel())
                    np.testing.assert_array_equal(stored_idx, reference.nearest_segment_idx.ravel())
                actual_distance = root / "actual_distance.bin"
                stored.astype(STORAGE_DTYPES[case.dtype]).tofile(actual_distance)
                test.assertTrue(verify_case_output(folder, actual_distance, index_path).passed)
        with test.assertRaises(FileExistsError):
            generate_suite(root, cases)
