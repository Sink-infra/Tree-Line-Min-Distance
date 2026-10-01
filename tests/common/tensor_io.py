"""Binary tensor storage, file sizes, and content fingerprints."""

import hashlib
import math
from pathlib import Path

import numpy as np

from tests.common.cases import fixture_storage_dtype


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_tensor_path(case_dir: Path, descriptor: dict, dtype: str, count: int) -> Path:
    relative = Path(descriptor["path"])
    if relative.is_absolute() or len(relative.parts) != 1:
        raise ValueError("tensor file must be in the case directory")
    path = Path(case_dir) / relative
    storage = fixture_storage_dtype(dtype)
    expected_bytes = count * np.dtype(storage).itemsize
    if (descriptor["logical_dtype"] != dtype or descriptor["bytes"] != expected_bytes
            or descriptor["storage_dtype"] != storage or descriptor["endianness"] != "little"):
        raise ValueError(f"invalid tensor storage description: {relative}")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{path} contains {path.stat().st_size} bytes, expected {expected_bytes}")
    if sha256(path) != descriptor["sha256"]:
        raise ValueError(f"tensor SHA256 mismatch: {path}")
    return path


def verify_input_files(case_dir: Path, metadata: dict) -> dict:
    """Check both input files before execution or comparison, returning their paths."""
    paths = {}
    for name in ("tree_points", "line_segments"):
        tensor = metadata["input"][name]
        shape = tensor["shape"]
        if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError(f"invalid input shape: {name}")
        paths[name] = verified_tensor_path(case_dir, metadata["files"][name], tensor["dtype"], math.prod(shape))
    return paths


def save_tensor(folder: Path, filename: str, values: np.ndarray, dtype: str) -> dict:
    path = Path(folder) / filename
    values.ravel(order="C").tofile(path)
    return {"path": filename, "sha256": sha256(path), "bytes": path.stat().st_size,
            "logical_dtype": dtype, "storage_dtype": fixture_storage_dtype(dtype),
            "endianness": "little"}
