"""Mock catalogue and immutable, quantized fixture inputs."""

from pathlib import Path

import numpy as np


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "st" / "fixtures"


def load_mock_cases(case_root):
    from tests.common.cases import load_cases

    return load_cases("mock", case_root=case_root)


def load_mock_inputs(case):
    if case.category != "mock" or not case.fixture_file:
        raise ValueError("fixed inputs require a Mock fixture declaration")
    path = (FIXTURE_ROOT / case.fixture_file).resolve()
    if path.parent != FIXTURE_ROOT.resolve() or path.suffix != ".npz":
        raise ValueError("Mock fixture must be an NPZ file in st/fixtures")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"tree_points", "line_segments"}:
            raise ValueError("Mock fixture requires exactly the two operator inputs")
        points = np.array(archive["tree_points"], copy=True, order="C")
        segments = np.array(archive["line_segments"], copy=True, order="C")
    for values, shape, dtype in (
        (points, case.tree_points_shape, case.dtype),
        (segments, case.line_segments_shape, case.resolved_line_segments_dtype),
    ):
        if values.shape != shape or values.dtype != np.dtype(dtype):
            raise ValueError("Mock fixture shape or dtype differs from its declaration")
        if not np.isfinite(values).all():
            raise ValueError("Mock fixture coordinates must be finite")
    return points, segments
