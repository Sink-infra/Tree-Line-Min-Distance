"""Shared output assertions for generated TreeLineMinDistance test cases."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from tests.common.cases import STORAGE_DTYPES, TOLERANCES
from tests.common.tensor_io import verified_tensor_path, verify_input_files


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    element_count: int
    distance_mismatch_count: int
    index_mismatch_count: int
    mismatch_count: int
    max_abs_error: Optional[float]
    first_distance_mismatch: Optional[list]
    first_index_mismatch: Optional[list]
    message: str


def _load_values(path: Path, dtype: str, element_count: int) -> np.ndarray:
    storage_dtype = np.dtype(STORAGE_DTYPES[dtype])
    expected_bytes = element_count * storage_dtype.itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{path} contains {path.stat().st_size} bytes, expected {expected_bytes}")
    return np.fromfile(path, dtype=storage_dtype)


def _first_mismatch(mask: np.ndarray, shape: tuple):
    locations = np.flatnonzero(mask)
    return [int(x) for x in np.unravel_index(int(locations[0]), shape)] if locations.size else None


def verify_case_output(case_dir: Path, actual_distance_path: Path, actual_index_path: Path) -> VerificationResult:
    """Use the contract tolerances; indices must match exactly before distance casting."""
    case_dir = Path(case_dir)
    metadata = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("case metadata must be a JSON object")
    if metadata.get("schema_version") != 1 or metadata.get("operator") != "TreeLineMinDistance":
        raise ValueError("unsupported case metadata; regenerate the case")
    if metadata["expectation"] != "success" or metadata["expected_output"] is None:
        raise ValueError("rejection cases do not have outputs to verify")
    distance_spec = metadata["expected_output"]["min_distance"]
    index_spec = metadata["expected_output"]["nearest_segment_idx"]
    raw_shape = distance_spec["shape"]
    if not isinstance(raw_shape, list) or any(type(dim) is not int for dim in raw_shape):
        raise ValueError("output shape must contain integer dimensions")
    shape = tuple(raw_shape)
    dtype = distance_spec["dtype"]
    if len(shape) != 2 or not 1 <= shape[0] <= 8 or not 1 <= shape[1] <= 16384 or dtype not in TOLERANCES:
        raise ValueError("invalid output shape or dtype")
    points = metadata["input"]["tree_points"]
    segments = metadata["input"]["line_segments"]
    if (points["shape"] != [*shape, 3] or points["dtype"] != dtype
            or segments["dtype"] != dtype or len(segments["shape"]) != 4
            or segments["shape"][0] != shape[0] or segments["shape"][2:] != [2, 3]):
        raise ValueError("input and output contracts do not match")
    for tensor in (points, segments):
        if tensor["format"] != "ND" or tensor["contiguous"] is not True:
            raise ValueError("successful cases require contiguous ND inputs")
    m = segments["shape"][1]
    if type(m) is not int:
        raise ValueError("segment count must be an integer")
    if (index_spec["shape"] != list(shape) or index_spec["dtype"] != "int32"
            or not 1 <= m <= 1024 or index_spec["range"] != [0, m - 1]):
        raise ValueError("invalid index output contract")
    atol, rtol = TOLERANCES[dtype]
    if (distance_spec["comparison"]["atol"], distance_spec["comparison"]["rtol"]) != (atol, rtol):
        raise ValueError("case tolerances do not match TreeLineMinDistance.md")
    count = math.prod(shape)
    verify_input_files(case_dir, metadata)
    files = metadata["files"]
    reference_distance = verified_tensor_path(case_dir, files["reference_min_distance"], "float64", count)
    reference_index = verified_tensor_path(case_dir, files["reference_nearest_segment_idx"], "int32", count)
    expected = _load_values(reference_distance, "float64", count)
    expected_idx = _load_values(reference_index, "int32", count)
    if not np.isfinite(expected).all() or np.any(expected < 0) or np.any((expected_idx < 0) | (expected_idx >= m)):
        raise ValueError("invalid reference values")
    actual = _load_values(Path(actual_distance_path), dtype, count).astype(np.float64)
    actual_idx = _load_values(Path(actual_index_path), "int32", count)
    finite = np.isfinite(actual)
    diff = np.full(count, np.inf)
    diff[finite] = np.abs(actual[finite] - expected[finite])
    distance_bad = (~finite) | (actual < 0) | (diff > atol + rtol * np.abs(expected))
    index_bad = (actual_idx != expected_idx) | (actual_idx < 0) | (actual_idx >= m)
    distance_count = int(np.count_nonzero(distance_bad))
    index_count = int(np.count_nonzero(index_bad))
    # Null denotes an unbounded error in JSON; all finite errors have a numeric maximum.
    max_error = float(np.max(diff)) if finite.all() else None
    passed = distance_count == 0 and index_count == 0
    return VerificationResult(
        passed, count, distance_count, index_count, int(np.count_nonzero(distance_bad | index_bad)),
        max_error, _first_mismatch(distance_bad, shape), _first_mismatch(index_bad, shape),
        "pass" if passed else f"failed: distance mismatches={distance_count}, index mismatches={index_count}",
    )
