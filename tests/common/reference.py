"""Dtype-quantized inputs and a tiled FP64 geometric reference."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Tuple

import numpy as np

from tests.common.cases import CaseSpec, STORAGE_DTYPES, SUPPORTED_DTYPES, fixture_storage_dtype


@dataclass(frozen=True)
class ValidationResult:
    errors: Tuple[str, ...]


@dataclass(frozen=True)
class CaseData:
    tree_points: np.ndarray
    line_segments: np.ndarray


@dataclass(frozen=True)
class ReferenceResult:
    # Distances remain unrounded; indices are selected before output conversion.
    min_distance: np.ndarray
    nearest_segment_idx: np.ndarray


def validate_case(case: CaseSpec) -> ValidationResult:
    """Check declared shapes, types, and formats against the input contract."""
    errors = []
    points, segments = case.tree_points_shape, case.line_segments_shape
    points_dims_valid = all(type(dim) is int and dim >= 0 for dim in points)
    segments_dims_valid = all(type(dim) is int and dim >= 0 for dim in segments)
    if not points_dims_valid:
        errors.append("TREE_POINTS_DIMENSIONS")
    if not segments_dims_valid:
        errors.append("LINE_SEGMENTS_DIMENSIONS")
    if len(points) != 3:
        errors.append("TREE_POINTS_RANK")
    elif points_dims_valid:
        if not 1 <= points[0] <= 8:
            errors.append("B_RANGE")
        if not 1 <= points[1] <= 16384:
            errors.append("N_RANGE")
        if points[2] != 3:
            errors.append("TREE_POINTS_XYZ")
    if len(segments) != 4:
        errors.append("LINE_SEGMENTS_RANK")
    elif segments_dims_valid:
        if not 1 <= segments[0] <= 8:
            errors.append("B_RANGE")
        if not 1 <= segments[1] <= 1024:
            errors.append("M_RANGE")
        if segments[2] != 2:
            errors.append("LINE_SEGMENTS_ENDPOINTS")
        if segments[3] != 3:
            errors.append("LINE_SEGMENTS_XYZ")
    if points and segments and points[0] != segments[0]:
        errors.append("BATCH_MISMATCH")
    if case.dtype not in SUPPORTED_DTYPES:
        errors.append("TREE_POINTS_DTYPE")
    if case.resolved_line_segments_dtype not in SUPPORTED_DTYPES:
        errors.append("LINE_SEGMENTS_DTYPE")
    if case.dtype != case.resolved_line_segments_dtype:
        errors.append("DTYPE_MISMATCH")
    if case.tree_points_format != "ND":
        errors.append("TREE_POINTS_FORMAT")
    if case.line_segments_format != "ND":
        errors.append("LINE_SEGMENTS_FORMAT")
    return ValidationResult(tuple(dict.fromkeys(errors)))


def validate_data(case: CaseSpec, data: CaseData) -> ValidationResult:
    """Check actual, already quantized tensors without imposing a coordinate bound."""
    errors = list(validate_case(case).errors)
    for name, values, shape, dtype in (
        ("TREE_POINTS", data.tree_points, case.tree_points_shape, case.dtype),
        ("LINE_SEGMENTS", data.line_segments, case.line_segments_shape, case.resolved_line_segments_dtype),
    ):
        if values.shape != shape:
            errors.append(name + "_SHAPE")
        storage = STORAGE_DTYPES.get(dtype)
        if storage is not None and values.dtype != np.dtype(storage):
            errors.append(name + "_STORAGE_DTYPE")
        if not values.flags.c_contiguous:
            errors.append(name + "_CONTIGUOUS")
        if not np.isfinite(values).all():
            errors.append(name + "_NONFINITE")
    return ValidationResult(tuple(dict.fromkeys(errors)))


def infer_output_shape(case: CaseSpec) -> Tuple[int, int]:
    validation = validate_case(case)
    if validation.errors:
        raise ValueError(f"invalid case {case.name}: {validation.errors}")
    return case.tree_points_shape[:2]


def _separated_random(case: CaseSpec, rng: np.random.Generator) -> CaseData:
    """Oblique segments use indexed cells with sparse or overlapping x intervals."""
    b, n, _ = case.tree_points_shape
    m = case.line_segments_shape[1]
    step = 1600.0 / max(m, 2)
    centers = rng.uniform(-100.0, 100.0, size=(b, m, 3))
    centers[..., 0] = (np.arange(m) - (m - 1) / 2.0) * step
    overlapping = case.input_pattern == "indexed_random_overlapping"
    if overlapping:
        side = int(np.ceil(np.sqrt(m)))
        centers[..., 1] = (np.arange(m) % side - (side - 1) / 2) * 4.0
        centers[..., 2] = (np.arange(m) // side - (side - 1) / 2) * 4.0
        centers[..., 2] += rng.uniform(-0.03, 0.03, size=(b, m))
    directions = rng.normal(size=(b, m, 3))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    vectors = directions * (step * 0.3)
    if overlapping:
        vectors[..., 0] = np.where(directions[..., 0] < 0, -1.0, 1.0) * step * 128
    elif case.input_pattern == "indexed_random_distinct_x":
        vectors[..., 0] = np.where(directions[..., 0] < 0, -1.0, 1.0) * step * 0.5
    segments = np.stack((centers - vectors / 2, centers + vectors / 2), axis=2)
    if case.input_pattern.startswith("indexed_random"):
        if case.input_pattern == "indexed_random_tail_mix_near_zero":
            midpoint = (segments[:, -1, 0] + segments[:, -1, 1]) / 2
            direction = segments[:, -1, 1] - segments[:, -1, 0]
            segments[:, -1, 0] = midpoint - 0.75 * direction
            segments[:, -1, 1] = midpoint + 0.75 * direction
        if case.input_pattern == "indexed_random_planes_no_tail":
            if m < 6:
                raise ValueError("shared start planes require at least six segments")
            height = float(np.asarray(segments[0, 0, 0, 2], dtype=fixture_storage_dtype(case.dtype)))
            segments[0, 4:6, :, 2] += height - segments[0, 4:6, 0:1, 2]
        quantized = segments.astype(fixture_storage_dtype(case.dtype))
        collapsed = np.all(quantized[..., 0, 1:] == quantized[..., 1, 1:], axis=-1)
        segments[..., 1, 2][collapsed] += 0.125
        if case.input_pattern in ("indexed_random_near_zero", "indexed_random_tail_mix_near_zero"):
            segments[0, 0, :, 1:] = [(-0.125, 0.25), (0.125, 0.5)]
        vectors = segments[..., 1, :] - segments[..., 0, :]
    indexed = case.input_pattern.startswith("indexed_random")
    if indexed:
        choices = m - 1 if case.input_pattern in ("indexed_random_no_tail", "indexed_random_planes_no_tail") else m
        if choices < 1:
            raise ValueError("tail exclusion requires at least two segments")
        ids = np.random.default_rng(case.seed).integers(0, choices, size=(b, n))
    else:
        ids = (np.arange(n)[None, :] * 17 + np.arange(b)[:, None] * 13) % m
    batch = np.arange(b)[:, None]
    limits = (0.25, 0.75) if overlapping or case.input_pattern == "indexed_random_interior" else (-0.25, 1.25)
    t = rng.uniform(*limits, size=(b, n, 1))
    if case.input_pattern == "indexed_random_tail_mix_near_zero":
        locations = np.argwhere(ids == m - 1)
        if len(locations) < 4:
            raise ValueError("mixed tail projections require four final-cell points")
        t[ids == m - 1] = -0.25
        for batch_index, point_index in locations[:4]:
            t[batch_index, point_index, 0] = 0.5
    points = segments[batch, ids, 0] + t * vectors[batch, ids]
    points += rng.uniform(-0.015, 0.015, size=(b, n, 3)) * step
    if indexed:
        # A perpendicular offset in the yz plane remains positive after FP16 quantization.
        perpendicular = np.stack((np.zeros((b, m)), directions[..., 2], -directions[..., 1]), axis=-1)
        lengths = np.linalg.norm(perpendicular, axis=-1, keepdims=True)
        perpendicular = np.divide(perpendicular, lengths, out=np.zeros_like(perpendicular), where=lengths != 0)
        perpendicular[lengths[..., 0] == 0] = (0, 1, 0)
        points += perpendicular[batch, ids] * max(0.5, step * 0.04)
        if case.input_pattern in ("indexed_random_near_zero", "indexed_random_tail_mix_near_zero"):
            locations = np.flatnonzero(ids[0] == 0)
            if not locations.size:
                raise ValueError("near-zero construction requires a point in the first segment cell")
            start = segments[0, 0, 0].astype(fixture_storage_dtype(case.dtype)).astype(np.float64)
            points[0, locations[0]] = start + (0, 2.0 ** -12, 0)
    if indexed and m >= 4:
        order = np.arange(m)
        order[1:3] = (2, 1)
        segments = segments[:, order]
    return CaseData(points, segments)


@lru_cache(maxsize=4)
def _dense_random(case: CaseSpec) -> CaseData:
    """Independent finite endpoints and points with unique interior winners."""
    b, n, _ = case.tree_points_shape
    m = case.line_segments_shape[1]
    if case.geometry_seed is not None and (type(case.geometry_seed) is not int or case.geometry_seed < 0):
        raise ValueError("geometry seed must be a nonnegative integer")
    rng = np.random.default_rng(case.seed if case.geometry_seed is None else case.geometry_seed)
    storage = fixture_storage_dtype(case.dtype)
    segments = rng.uniform(-800.0, 800.0, size=case.line_segments_shape)
    segments = segments.astype(storage).astype(np.float64)
    if case.input_pattern.endswith("planes_no_tail"):
        plane_case = replace(case, input_pattern="indexed_random_planes_no_tail", geometry_seed=None)
        plane_data = _separated_random(plane_case, np.random.default_rng(case.seed))
        plane_starts = plane_data.line_segments[..., 0, 2].astype(storage)
        shared = plane_starts == plane_starts[:, 0:1]
        starts = segments[..., 0, 2]
        coincident = (starts == starts[:, 0:1]) & ~shared
        starts[coincident] += 1.0
        starts[shared] = np.broadcast_to(starts[:, 0:1], starts.shape)[shared]
    choices = m - 1 if case.input_pattern.endswith("no_tail") else m
    ids = np.random.default_rng(case.seed).integers(0, choices, size=(b, n))
    if m >= 4:
        mapped = ids.copy()
        mapped[ids == 1] = 2
        mapped[ids == 2] = 1
        ids = mapped
    batch = np.arange(b)[:, None]
    starts = segments[batch, ids, 0]
    vectors = segments[batch, ids, 1] - starts
    direction = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
    points = np.empty(case.tree_points_shape)
    pending = np.ones((b, n), dtype=bool)
    for attempt in range(12):
        cross = np.cross(direction, rng.normal(size=case.tree_points_shape))
        cross /= np.linalg.norm(cross, axis=-1, keepdims=True)
        values = starts + rng.uniform(0.25, 0.75, size=(b, n, 1)) * vectors + 4.0 * cross
        points[pending] = values[pending]
        data = CaseData(points.astype(storage), segments.astype(storage))
        expected = tree_line_min_distance_reference(case, data)
        pending = (expected.nearest_segment_idx != ids) | (expected.min_distance < 0.5)
        if not np.any(pending):
            data.tree_points.setflags(False)
            data.line_segments.setflags(False)
            return data
    raise ValueError("unable to separate dense segment projections")


def _parallel_random(case: CaseSpec, rng: np.random.Generator) -> CaseData:
    """Parallel segments share their x interval; y cells identify unique winners."""
    b, n, _ = case.tree_points_shape
    m = case.line_segments_shape[1]
    segments = np.zeros(case.line_segments_shape, dtype=np.float64)
    segments[..., 0, 0] = -1000
    segments[..., 1, 0] = 1000
    step = 1600.0 / max(m, 2)
    segments[..., 0, 1] = (np.arange(m) - (m - 1) / 2.0) * step
    segments[..., 1, 1] = segments[..., 0, 1]
    choices = m - 1 if case.input_pattern == "parallel_no_tail" else m
    ids = np.random.default_rng(case.seed).integers(0, choices, size=(b, n))
    points = np.empty(case.tree_points_shape, dtype=np.float64)
    points[..., 0] = rng.uniform(-500, 500, size=(b, n))
    points[..., 1] = segments[np.arange(b)[:, None], ids, 0, 1]
    points[..., 2] = rng.uniform(90, 110, size=(b, n))
    if case.input_pattern == "parallel_near_zero":
        locations = np.flatnonzero(ids[0] == 0)
        if not locations.size:
            raise ValueError("near-zero construction requires a first-cell point")
        points[0, locations[0]] = (-1000, segments[0, 0, 0, 1], 2.0 ** -12)
    return CaseData(points, segments)


def _geometry(pattern: str, shape: Tuple[int, ...], segment_shape: Tuple[int, ...]) -> CaseData:
    b, n, _ = shape
    m = segment_shape[1]
    points = np.zeros(shape, dtype=np.float64)
    segments = np.full(segment_shape, 500.0, dtype=np.float64)
    if pattern == "documented_endpoints":
        points[:] = [(-3, 4, 0), (5, 0, 0), (13, -4, 0)]
        segments[:] = [(0, 0, 0), (10, 0, 0)]
    elif pattern == "documented_tie":
        segments[0] = [[(1, 1, 0), (2, 1, 0)], [(1, -1, 0), (2, -1, 0)]]
    elif pattern == "documented_zero_length":
        segments[0] = [[(3, 4, 0), (3, 4, 0)], [(0, 6, 0), (0, 7, 0)]]
    elif pattern == "mixed_degenerate_points":
        points[:] = [(0, 0, 0), (0.5, 0, 0), (10, 4, 0)]
        segments[:] = [[(0, 0, 0), (1, 0, 0)], [(10, 2, 0), (10, 2, 0)]]
    elif pattern == "oblique_3d":
        points[:] = [(0, 0, 0), (2, 4, 4), (1, 5, -1)]
        segments[:] = [(0, 0, 0), (2, 4, 4)]
    elif pattern == "negative_coordinates":
        points[:] = [(-13, -4, -2), (-5, -4, -2), (3, -8, -2)]
        segments[:] = [(-10, -4, -2), (0, -4, -2)]
    elif pattern == "all_zero":
        segments.fill(0)
    elif pattern == "all_degenerate":
        points[:] = [(0, 0, 0), (3, 0, 0), (6, 0, 0)]
        for j in range(m):
            segments[:, j] = (j, 0, 0)
    elif pattern == "zero_distance_tie":
        points[..., 0] = np.arange(n) % 3
        segments[:, 2] = [(0, 0, 0), (2, 0, 0)]
        segments[:, 5] = [(2, 0, 0), (0, 0, 0)]
    elif pattern == "nonzero_tie":
        segments[..., 0, 2] = 0
        segments[..., 1, 2] = 1
        segments[:, 2] = [(3, 4, 0), (4, 4, 0)]
        segments[:, 5] = [(-4, 4, 0), (-3, 4, 0)]
    elif pattern == "cross_tile_tie":
        segments[:, 31] = [(3, 4, 0), (3, 4, 0)]
        segments[:, 32] = [(3, -4, 0), (3, -4, 0)]
        segments[:, m - 1] = [(-3, 4, 0), (-3, 4, 0)]
    elif pattern == "tie_groups":
        points[:] = [(0, 0, 0), (0, 100, 0), (0, -100, 0)]
        for i, winners in enumerate(((19, 33), (31, 129), (32, 512))):
            y = points[0, i, 1]
            segments[:, winners[0]] = (3, y + 4, 0)
            segments[:, winners[1]] = (-3, y - 4, 0)
    elif pattern in ("last_segment", "tail_positive"):
        points[..., 2] = (np.arange(n) % 17) * 0.25
        if pattern == "last_segment":
            segments[:, -1] = [(1, 0, -10), (1, 0, 10)]
        else:
            segments[:, -1] = [(3, 4, -10), (3, 4, 10)]
    elif pattern == "batch_isolation":
        for batch in range(b):
            points[batch, :, 1] = np.arange(n)
            segments[batch, batch * 3] = [(batch + 1, -10, 0), (batch + 1, 10, 0)]
    elif pattern == "large_coordinates":
        points[:] = [(0, 1000, 1000), (1000, 1000, 0), (-1000, 0, 0)]
        segments[:] = [(-1000, 0, 0), (1000, 0, 0)]
    elif pattern == "tiny_segment":
        tiny = 2.0 ** -24
        points[..., 0] = tiny
        segments[:, 0] = [(0, 0, 0), (0, 0, 0)]
        segments[:, 1] = [(0, 0, 0), (tiny, 0, 0)]
    elif pattern == "output_rounding_tie":
        segments[:, 0] = [(1, 1 / 64, 0), (1, 1 / 64, 0)]
        segments[:, 1] = [(1, 0, 0), (1, 0, 0)]
    elif pattern == "quantized_zero_length":
        points[:] = [(997, 4, 0), (1000, 0, 0), (1003, -4, 0)]
        segments[:] = [(1000, 0, 0), (1000.125, 0, 0)]
    else:
        raise ValueError(f"unknown input pattern: {pattern}")
    return CaseData(points, segments)


def materialize_case(case: CaseSpec) -> CaseData:
    """Generate repeatable inputs and quantize before any reference calculation."""
    points_dtype = fixture_storage_dtype(case.dtype)
    segments_dtype = fixture_storage_dtype(case.resolved_line_segments_dtype)
    validation = validate_case(case)
    if any(error.endswith("_DIMENSIONS") for error in validation.errors):
        raise ValueError(f"fixture shapes require nonnegative integer dimensions: {validation.errors}")
    rng = np.random.default_rng(case.seed)
    if validate_case(case).errors:
        # Malformed shapes and types are preserved in rejection fixtures.
        def rejection_values(shape, storage):
            kind = np.dtype(storage).kind
            if kind in ("u", "b"):
                return rng.integers(0, 2 if kind == "b" else 16, size=shape)
            return rng.uniform(-10, 10, size=shape)
        data = CaseData(rejection_values(case.tree_points_shape, points_dtype),
                        rejection_values(case.line_segments_shape, segments_dtype))
    elif case.input_pattern == "fixed_fixture":
        from tests.common.mock_cases import load_mock_inputs
        data = CaseData(*load_mock_inputs(case))
    elif case.input_pattern in ("separated_random", "indexed_random", "indexed_random_no_tail", "indexed_random_near_zero", "indexed_random_planes_no_tail", "indexed_random_interior", "indexed_random_tail_mix_near_zero", "indexed_random_overlapping", "indexed_random_distinct_x"):
        data = _separated_random(case, rng)
    elif case.input_pattern in ("dense_random", "dense_random_planes_no_tail"):
        if case.geometry_seed is not None and (type(case.geometry_seed) is not int or case.geometry_seed < 0):
            raise ValueError("geometry seed must be a nonnegative integer")
        data = _dense_random(case)
    elif case.input_pattern in ("parallel_random", "parallel_no_tail", "parallel_near_zero"):
        data = _parallel_random(case, rng)
    elif case.input_pattern.startswith("nonfinite_"):
        data = _separated_random(case, rng)
        _, tensor, kind = case.input_pattern.split("_")
        values = data.tree_points if tensor == "points" else data.line_segments
        values.flat[0] = {"nan": np.nan, "posinf": np.inf, "neginf": -np.inf}[kind]
    else:
        data = _geometry(case.input_pattern, case.tree_points_shape, case.line_segments_shape)
    if case.point_order_seed is not None:
        if type(case.point_order_seed) is not int or case.point_order_seed < 0:
            raise ValueError("point order seed must be a nonnegative integer")
        if data.tree_points.ndim != 3:
            raise ValueError("point ordering requires rank-three tree points")
        order = np.random.default_rng(case.point_order_seed).permutation(data.tree_points.shape[1])
        data = CaseData(data.tree_points[:, order], data.line_segments)
    return CaseData(
        np.array(data.tree_points, dtype=points_dtype, order="C", copy=True),
        np.array(data.line_segments, dtype=segments_dtype, order="C", copy=True),
    )


def tree_line_min_distance_reference(
    case: CaseSpec, data: CaseData, *, point_tile: int = 256, segment_tile: int = 128,
) -> ReferenceResult:
    """Evaluate finite segments in FP64 with bounded point/segment work tiles.

    The input arrays already contain the actual FP16/FP32 coordinates. Division
    only occurs where the quantized endpoints differ. Segment tiles are visited
    in ascending order, and an equal distance never replaces an earlier index.
    """
    validation = validate_data(case, data)
    if validation.errors:
        raise ValueError(f"cannot evaluate rejected case {case.name}: {validation.errors}")
    if point_tile <= 0 or segment_tile <= 0:
        raise ValueError("tile sizes must be positive")
    b, n = infer_output_shape(case)
    m = case.line_segments_shape[1]
    output = np.empty((b, n), dtype=np.float64)
    indices = np.empty((b, n), dtype=np.int32)
    for batch in range(b):
        segments = data.line_segments[batch].astype(np.float64)
        starts = segments[:, 0]
        vectors = segments[:, 1] - starts
        lengths = np.sum(vectors * vectors, axis=-1)
        for first in range(0, n, point_tile):
            points = data.tree_points[batch, first:first + point_tile].astype(np.float64)
            best = np.full(len(points), np.inf, dtype=np.float64)
            best_idx = np.zeros(len(points), dtype=np.int32)
            for offset in range(0, m, segment_tile):
                a = starts[offset:offset + segment_tile]
                ac = vectors[offset:offset + segment_tile]
                length = lengths[offset:offset + segment_tile]
                ap = points[:, None, :] - a[None, :, :]
                t = np.zeros(ap.shape[:2], dtype=np.float64)
                np.divide(np.sum(ap * ac[None, :, :], axis=-1), length[None, :],
                          out=t, where=length[None, :] != 0)
                np.clip(t, 0.0, 1.0, out=t)
                residual = points[:, None, :] - (a[None, :, :] + t[..., None] * ac[None, :, :])
                squared = np.sum(residual * residual, axis=-1)
                local_idx = np.argmin(squared, axis=1)
                local_best = squared[np.arange(len(points)), local_idx]
                improve = local_best < best

                best[improve] = local_best[improve]
                best_idx[improve] = (offset + local_idx[improve]).astype(np.int32)
            output[batch, first:first + len(points)] = np.sqrt(best)
            indices[batch, first:first + len(points)] = best_idx
    return ReferenceResult(output, indices)


def storage_description(dtype: str) -> dict:
    return {"logical_dtype": dtype, "storage_dtype": fixture_storage_dtype(dtype), "endianness": "little"}
