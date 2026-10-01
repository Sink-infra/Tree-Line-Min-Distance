"""Build the current TreeLineMinDistance operator package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess

from tests.common.tensor_io import sha256

ROOT = Path(__file__).resolve().parents[3]
TOOLKIT = Path("/usr/local/Ascend/cann-9.0.0")
RECEIPT = "suite-build.json"


def device_guard():
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("NPU build and execution require the configured Ascend Linux target")


def save(path, data):
    with Path(path).open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def source_hashes(source):
    return {str(path.relative_to(source)): sha256(path) for path in sorted(source.rglob("*"))
            if path.is_file() and "build" not in path.relative_to(source).parts}


def package_hashes(build):
    paths = [build / "libcust_opapi.so"]
    paths.extend(p for p in sorted((build / "tmp/vendors/custom").rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts)
    return {str(p.relative_to(build)): sha256(p) for p in paths}


def runtime_environment(build, toolkit):
    environment = dict(os.environ)
    paths = [str(build), str(toolkit / "lib64"), str(toolkit / "lib64/plugin/opskernel"),
             str(toolkit / "lib64/plugin/nnengine"), "/usr/local/Ascend/driver/lib64",
             "/usr/local/Ascend/driver/lib64/common", "/usr/local/Ascend/driver/lib64/driver"]
    environment["LD_LIBRARY_PATH"] = ":".join(paths + [environment.get("LD_LIBRARY_PATH", "")])
    environment["ASCEND_CUSTOM_OPP_PATH"] = str(build / "tmp/vendors/custom")
    environment["ASCEND_HOME_PATH"] = str(toolkit)
    environment["ASCEND_OPP_PATH"] = str(toolkit / "opp")
    environment["PATH"] = ":".join([str(toolkit / "bin"), str(toolkit / "tools/profiler/bin"),
                                    environment.get("PATH", "")])
    return environment


def source_command(argv, toolkit, folder, log, timeout=300):
    setup = toolkit / "set_env.sh"
    if not setup.is_file():
        raise ValueError("CANN environment setup script is missing")
    command = ["bash", "-c", "source " + shlex.quote(str(setup)) + ' && exec "$@"', "cann", *map(str, argv)]
    with Path(log).open("w") as stream:
        result = subprocess.run(command, cwd=folder, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed; see {log}")


def build_operator(source, build, toolkit=TOOLKIT, jobs=4):
    device_guard()
    source, build, toolkit = Path(source).resolve(), Path(build).resolve(), Path(toolkit).resolve()
    if (jobs < 1 or source == build or source in build.parents
            or build == ROOT or (ROOT in build.parents and ROOT / "build" not in build.parents)):
        raise ValueError("invalid build directory or job count")
    before = source_hashes(source)
    if not before or not (source / "CMakeLists.txt").is_file():
        raise ValueError("operator source directory is incomplete")
    build.mkdir(parents=True, exist_ok=True)
    receipt = build / RECEIPT
    if receipt.exists():
        receipt.unlink()
    source_command(["cmake", "-S", source, "-B", build,
                    "-DASCEND_COMPUTE_UNIT=ascend910b", "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_PREFIX_PATH=" + str(toolkit / "lib64/cmake")],
                   toolkit, ROOT, build / "configure.log")
    source_command(["cmake", "--build", build, "-j" + str(jobs)],
                   toolkit, ROOT, build / "build.log")
    original = build / "op_host/libcustom_ascendc_cust_optiling.so"
    tiling = build / "tmp/vendors/custom/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"
    tiling.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, tiling)
    after = source_hashes(source)
    if before != after:
        raise RuntimeError("operator sources changed during the build")
    data = {"source_dir": str(source), "build_dir": str(build), "toolkit": str(toolkit),
            "source_sha256": before, "binary_sha256": package_hashes(build),
            "header_sha256": sha256(build / "autogen/aclnn_tree_line_min_distance.h")}
    save(receipt, data)
    return data


def verify_build(build, toolkit):
    build, toolkit = Path(build).resolve(), Path(toolkit).resolve()
    data = json.loads((build / RECEIPT).read_text())
    if Path(data["build_dir"]) != build or Path(data["toolkit"]) != toolkit:
        raise ValueError("build directory or toolkit differs from the build receipt")
    if source_hashes(Path(data["source_dir"])) != data["source_sha256"]:
        raise ValueError("operator sources differ from the built package")
    for name, digest in data["binary_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or sha256(build / relative) != digest:
            raise ValueError("operator package differs from the build receipt")
    if package_hashes(build) != data["binary_sha256"]:
        raise ValueError("operator package file set differs from the build receipt")
    if sha256(build / "autogen/aclnn_tree_line_min_distance.h") != data["header_sha256"]:
        raise ValueError("generated operator header changed")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "code")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/current")
    parser.add_argument("--toolkit", type=Path, default=TOOLKIT)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    try:
        result = build_operator(args.source_dir, args.build_dir, args.toolkit, args.jobs)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.error(str(error))
    print("build=" + result["build_dir"])


if __name__ == "__main__":
    main()
