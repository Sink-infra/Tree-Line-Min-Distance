"""Case schema, JSON catalogues, and suite selection."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable, Optional, Sequence, Tuple
import json
from pathlib import Path
import re


SUPPORTED_DTYPES = ("float16", "float32")
TOLERANCES = {"float32": (1e-4, 1e-4), "float16": (1e-3, 1e-3)}
# Portable fixture encodings include rejection inputs; operator support stays FP16/FP32.
STORAGE_DTYPES = {
    "float16": "<f2", "float32": "<f4", "float64": "<f8", "bool": "|b1",
    "int8": "|i1", "int16": "<i2", "int32": "<i4", "int64": "<i8",
    "uint8": "|u1", "uint16": "<u2", "uint32": "<u4", "uint64": "<u8",
}


def fixture_storage_dtype(dtype: str) -> str:
    """Resolve a fixture encoding, reporting unknown encodings as a normal error."""
    if dtype not in STORAGE_DTYPES:
        raise ValueError(f"no portable fixture storage for dtype: {dtype}")
    return STORAGE_DTYPES[dtype]


@dataclass(frozen=True)
class CaseSpec:
    """One operator call; rejection labels describe the local contract validator."""

    name: str
    category: str
    expectation: str
    tree_points_shape: Tuple[int, ...]
    line_segments_shape: Tuple[int, ...]
    dtype: str = "float32"
    line_segments_dtype: Optional[str] = None
    input_pattern: str = "separated_random"
    seed: int = 20261001
    tree_points_format: str = "ND"
    line_segments_format: str = "ND"
    expected_error: Optional[str] = None
    covers: Tuple[str, ...] = ()
    note: str = ""

    @property
    def resolved_line_segments_dtype(self) -> str:
        return self.dtype if self.line_segments_dtype is None else self.line_segments_dtype

    @property
    def pair_count(self) -> int:
        if len(self.tree_points_shape) != 3 or len(self.line_segments_shape) != 4:
            return 0
        return self.tree_points_shape[0] * self.tree_points_shape[1] * self.line_segments_shape[1]


CATEGORIES = ("coverage", "correctness", "mock")
CASE_ROOT = Path(__file__).resolve().parents[1] / "st" / "cases"


def load_cases(category: str, *, case_root: Path = CASE_ROOT) -> tuple:
    """Load one declared category while preserving rejection input specifications."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown category: {category}")
    path = Path(case_root) / (category + ".json")
    document = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(document, dict) or document.get("schema_version") != 1
            or document.get("operator") != "TreeLineMinDistance"
            or document.get("category") != category
            or not isinstance(document.get("cases"), list) or not document["cases"]):
        raise ValueError(f"invalid case catalogue: {path}")
    allowed = {field.name for field in fields(CaseSpec)} - {"category"}
    required = {"name", "expectation", "tree_points_shape", "line_segments_shape"}
    result, names = [], set()
    for row in document["cases"]:
        if not isinstance(row, dict) or set(row) - allowed or required - set(row):
            raise ValueError(f"invalid case fields: {path}")
        values = dict(row)
        name = values["name"]
        if (not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name)
                or not name.startswith(category + "_") or name in names):
            raise ValueError(f"invalid or duplicate case name: {name}")
        names.add(name)
        if values["expectation"] not in ("success", "reject"):
            raise ValueError(f"invalid expectation: {name}")
        for key in ("tree_points_shape", "line_segments_shape"):
            shape = values[key]
            if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
                raise ValueError(f"invalid shape encoding: {name}")
            values[key] = tuple(shape)
        covers = values.get("covers", [])
        if not isinstance(covers, list) or any(not isinstance(tag, str) for tag in covers):
            raise ValueError(f"invalid coverage tags: {name}")
        values["covers"] = tuple(covers)
        if type(values.get("seed", 20261001)) is not int:
            raise ValueError(f"invalid seed: {name}")
        result.append(CaseSpec(category=category, **values))
    return tuple(result)


COVERAGE_CASES = load_cases("coverage")
CORRECTNESS_CASES = load_cases("correctness")
MOCK_CASES = load_cases("mock")
CASE_GROUPS = {
    "coverage": COVERAGE_CASES,
    "correctness": CORRECTNESS_CASES,
    "mock": MOCK_CASES,
}
FULL_CASES = COVERAGE_CASES + CORRECTNESS_CASES
CASES = FULL_CASES + MOCK_CASES
_BY_NAME = {case.name: case for case in CASES}
if len(_BY_NAME) != len(CASES):
    raise ValueError("case names must be unique across categories")


def cases_for_suite(suite: str = "full") -> tuple:
    if suite in CASE_GROUPS:
        return CASE_GROUPS[suite]
    if suite == "full":
        return FULL_CASES
    if suite == "all":
        return CASES
    raise ValueError(f"unknown suite: {suite}")


def filter_cases(cases: Sequence[CaseSpec], expectation: str = "all",
                 names: Iterable[str] = (), tags: Iterable[str] = ()) -> tuple:
    if expectation not in ("all", "success", "reject"):
        raise ValueError(f"unknown expectation: {expectation}")
    requested, required_tags = set(names), set(tags)
    missing = requested - {case.name for case in cases}
    if missing:
        raise ValueError(f"case names outside the selected suite: {sorted(missing)}")
    selected = tuple(case for case in cases
                     if (expectation == "all" or case.expectation == expectation)
                     and (not requested or case.name in requested)
                     and required_tags.issubset(case.covers))
    if not selected:
        raise ValueError("selection contains no cases")
    return selected


def select_cases(suite: str = "full", *, names: Iterable[str] = (),
                 tags: Iterable[str] = (), expectation: str = "all") -> tuple:
    requested = tuple(names)
    if suite == "custom":
        if not requested:
            raise ValueError("custom suite requires at least one case name")
        collection = CASES
    else:
        collection = cases_for_suite(suite)
    return filter_cases(collection, expectation, requested, tags)
