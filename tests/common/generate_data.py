"""Generate TreeLineMinDistance inputs, unrounded references, and manifests."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import platform
import re
from typing import Sequence

import numpy as np

from tests.common.cases import CaseSpec, TOLERANCES, filter_cases as _select_cases
from tests.common.tensor_io import sha256, save_tensor
from tests.common.reference import (
    infer_output_shape, materialize_case,
    tree_line_min_distance_reference, validate_data,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _audit_case(case: CaseSpec) -> None:
    if not re.fullmatch(r"[a-z0-9_]+", case.name) or not case.name.startswith(case.category + "_"):
        raise ValueError(f"invalid case name: {case.name}")
    errors = validate_data(case, materialize_case(case)).errors
    if case.expectation == "success":
        if errors or case.expected_error is not None:
            raise ValueError(f"success case has an invalid contract: {case.name}: {errors}")
    elif case.expectation == "reject":
        if not case.expected_error or case.expected_error not in errors:
            raise ValueError(f"rejection label is missing from validation: {case.name}: {errors}")
    else:
        raise ValueError(f"unknown expectation: {case.expectation}")


def _case_metadata(case: CaseSpec, folder: Path) -> dict:
    data = materialize_case(case)
    validation = validate_data(case, data)
    files = {
        "tree_points": save_tensor(folder, "tree_points.bin", data.tree_points, case.dtype),
        "line_segments": save_tensor(folder, "line_segments.bin", data.line_segments,
                                       case.resolved_line_segments_dtype),
    }
    expected = None
    if case.expectation == "success":
        reference = tree_line_min_distance_reference(case, data)
        shape = list(infer_output_shape(case))
        atol, rtol = TOLERANCES[case.dtype]
        files["reference_min_distance"] = save_tensor(
            folder, "reference_min_distance_fp64.bin", reference.min_distance.astype("<f8"), "float64")
        files["reference_nearest_segment_idx"] = save_tensor(
            folder, "reference_nearest_segment_idx_int32.bin", reference.nearest_segment_idx.astype("<i4"), "int32")
        expected = {
            "min_distance": {"shape": shape, "dtype": case.dtype,
                             "comparison": {"atol": atol, "rtol": rtol, "scope": "every element",
                                            "reference": "unrounded FP64 distance from quantized inputs",
                                            "finite": True, "nonnegative": True}},
            "nearest_segment_idx": {"shape": shape, "dtype": "int32",
                                    "range": [0, case.line_segments_shape[1] - 1],
                                    "comparison": "exact; smallest index wins equal distances"},
        }
    inputs = {}
    for name, tensor, dtype, tensor_format in (
        ("tree_points", data.tree_points, case.dtype, case.tree_points_format),
        ("line_segments", data.line_segments, case.resolved_line_segments_dtype, case.line_segments_format),
    ):
        inputs[name] = {"shape": list(tensor.shape), "dtype": dtype, "format": tensor_format,
                        "contiguous": True,
                        "strides_in_elements": [s // tensor.dtype.itemsize for s in tensor.strides]}
    return {
        "schema_version": 1, "operator": "TreeLineMinDistance", "name": case.name,
        "category": case.category, "expectation": case.expectation,
        "expected_error": case.expected_error, "validation_errors": list(validation.errors),
        "validation_error_domain": "input-contract validator",
        "input": inputs, "attributes": {}, "expected_output": expected,
        "patterns": {"input": case.input_pattern, "seed": case.seed},
        "covers": list(case.covers), "note": case.note, "files": files,
    }


def generate_suite(output_dir: Path, cases: Sequence[CaseSpec], *, suite: str = None) -> dict:
    """Create one audited suite in a new directory; existing data is never replaced."""
    output_dir = Path(output_dir)
    categories = {case.category for case in cases}
    supported = {"coverage", "correctness", "mock"}
    if not categories or not categories.issubset(supported):
        raise ValueError("generate a nonempty named suite")
    if suite is None:
        if len(categories) != 1:
            raise ValueError("mixed categories require an explicit suite")
        suite = next(iter(categories))
    allowed = {name: {name} for name in supported}
    allowed.update(full={"coverage", "correctness"}, all=supported, custom=supported)
    if suite not in allowed or not categories.issubset(allowed[suite]):
        raise ValueError("case categories do not match the selected suite")
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("case names must be unique within a suite")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    for case in cases:
        _audit_case(case)
    output_dir.mkdir(parents=True, exist_ok=False)
    entries = []
    for case in cases:
        folder = output_dir / case.name
        folder.mkdir()
        metadata_path = folder / "case.json"
        _write_json(metadata_path, _case_metadata(case, folder))
        entries.append({"name": case.name, "expectation": case.expectation,
                        "metadata": f"{case.name}/case.json",
                        "case_metadata_sha256": sha256(metadata_path)})
    counts = Counter(case.expectation for case in cases)
    manifest = {
        "schema_version": 1, "operator": "TreeLineMinDistance", "category": suite,
        "schema_order": ["tree_points", "line_segments", "min_distance", "nearest_segment_idx"],
        "generation": {"numpy_version": np.__version__, "python_version": platform.python_version(),
                       "deterministic_for_recorded_environment": True, "endianness": "little",
                       "overwrite_policy": "refuse", "reference_compute_dtype": "float64",
                       "point_tile": 256, "segment_tile": 128},
        "catalog_audit_failures": [],
        "counts": {"total": len(cases), "success": counts["success"], "reject": counts["reject"]},
        "cases": entries,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest
