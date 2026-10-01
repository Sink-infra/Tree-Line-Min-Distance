"""Build the current TreeLineMinDistance operator package."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess

from tests.common.tensor_io import sha256

ROOT = Path(__file__).resolve().parents[3]
TOOLKIT = Path("/usr/local/Ascend/cann-9.0.0")
RECEIPT = "suite-build.json"


HOST_PROFILE = {"compiler": "GNU", "version": "11.4.0", "standard": "c++11", "abi": 0}
SDK_PROFILE = {"cann": "9.0.0", "asc_devkit": "1.0.0"}
DEVICE_PROFILE = {"standard": "c++17", "optimization": "source/SDK settings"}


def toolchain_metadata(toolkit):
    version = subprocess.check_output(["g++", "-dumpfullversion"], text=True).strip()
    cann = dict(line.split("=", 1) for line in (toolkit / "compiler/version.info").read_text().splitlines()
                if "=" in line).get("Version")
    header = toolkit / "include/ascendc/asc_devkit_version.h"
    found = re.search(r'ASC_DEVKIT_VERSION_STR\s+"([^"]+)"', header.read_text())
    sdk = {"cann": cann, "asc_devkit": found.group(1) if found else None}
    if version != HOST_PROFILE["version"] or sdk != SDK_PROFILE:
        raise ValueError("CANN 9.0.0, ASC DevKit 1.0.0, and GNU 11.4.0 are required")
    return {"host": dict(HOST_PROFILE), "sdk": sdk, "device": dict(DEVICE_PROFILE)}


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


def verify_kernel_options(build):
    module = Path(build) / "tmp/vendors/custom/op_impl/ai_core/tbe/custom_impl/dynamic/tree_line_min_distance.py"
    found = re.search(r"^\s*custom_compile_options\s*=\s*(\{[^\n]*\}),?\s*$",
                      module.read_text(), re.MULTILINE)
    options = ast.literal_eval(found.group(1)) if found else None
    if not isinstance(options, dict):
        raise ValueError("device compiler options are missing or invalid")
    return options


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


def prepare_build_directory(source, build):
    source, build = Path(source).resolve(), Path(build).resolve()
    if source == build or source in build.parents or build in source.parents:
        raise ValueError("source and build directories must be separate")
    if build.exists() and any(build.iterdir()):
        cache = build / "CMakeCache.txt"
        found = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$", cache.read_text(),
                          re.MULTILINE) if cache.is_file() else None
        if not found or Path(found.group(1)).resolve() != source:
            raise ValueError("existing build directory does not belong to these sources")
        shutil.rmtree(build)
    build.mkdir(parents=True, exist_ok=True)


def build_operator(source, build, toolkit=TOOLKIT, jobs=28):
    device_guard()
    source, build, toolkit = Path(source).resolve(), Path(build).resolve(), Path(toolkit).resolve()
    if (jobs < 1 or source == build or source in build.parents or build in source.parents
            or build == ROOT or (ROOT in build.parents and ROOT / "build" not in build.parents)):
        raise ValueError("invalid build directory or job count")
    toolchain = toolchain_metadata(toolkit)
    before = source_hashes(source)
    if not before or not (source / "CMakeLists.txt").is_file():
        raise ValueError("operator source directory is incomplete")
    prepare_build_directory(source, build)
    receipt = build / RECEIPT
    if receipt.exists():
        receipt.unlink()
    source_command(["cmake", "-S", source, "-B", build,
                    "-DASCEND_COMPUTE_UNIT=ascend910b", "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_PREFIX_PATH=" + str(toolkit / "lib64/cmake"),
                    "-DCMAKE_CXX_STANDARD=11", "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
                    "-DCMAKE_CXX_EXTENSIONS=OFF", "-DCMAKE_CXX_COMPILER=g++"],
                   toolkit, ROOT, build / "configure.log")
    flags = sorted((build / "op_host/CMakeFiles").glob("*.dir/flags.make"))
    if not flags or any("-std=c++11" not in path.read_text()
                        or "-D_GLIBCXX_USE_CXX11_ABI=0" not in path.read_text() for path in flags):
        raise ValueError("Host compiler flags differ from C++11 and ABI=0")
    source_command(["cmake", "--build", build, "-j" + str(jobs)],
                   toolkit, ROOT, build / "build.log", timeout=900)
    original = build / "op_host/libcustom_ascendc_cust_optiling.so"
    tiling = build / "tmp/vendors/custom/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"
    tiling.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, tiling)
    kernel_options = verify_kernel_options(build)
    after = source_hashes(source)
    if before != after:
        raise RuntimeError("operator sources changed during the build")
    data = {"toolchain": toolchain, "kernel_compile_options": kernel_options,
            "source_dir": str(source), "build_dir": str(build), "toolkit": str(toolkit),
            "source_sha256": before, "binary_sha256": package_hashes(build),
            "header_sha256": sha256(build / "autogen/aclnn_tree_line_min_distance.h")}
    save(receipt, data)
    return data


def verify_build(build, toolkit):
    build, toolkit = Path(build).resolve(), Path(toolkit).resolve()
    data = json.loads((build / RECEIPT).read_text())
    if data.get("toolchain") != {"host": HOST_PROFILE, "sdk": SDK_PROFILE, "device": DEVICE_PROFILE}:
        raise ValueError("build toolchain differs from the required configuration")
    if verify_kernel_options(build) != data.get("kernel_compile_options"):
        raise ValueError("device compiler options differ from the build receipt")
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
    parser.add_argument("--jobs", type=int, default=28)
    args = parser.parse_args()
    try:
        result = build_operator(args.source_dir, args.build_dir, args.toolkit, args.jobs)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.error(str(error))
    print("build=" + result["build_dir"])


if __name__ == "__main__":
    main()
