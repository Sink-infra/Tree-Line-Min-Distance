"""Current-build NPU execution, fixture integrity, and output checks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import os
import math
import shlex
import signal
from pathlib import Path
import subprocess
import tempfile

from tests.common.generate_data import generate_suite
from tests.common.tensor_io import sha256, verify_input_files
from tests.common.verify_result import verify_case_output
from tests.st.scripts.build import (
    ROOT, TOOLKIT, device_guard, runtime_environment, save, source_command, verify_build,
)

RUNNER_SOURCE = ROOT / "tests/st/runner/tree_line_min_distance_runner.cpp"


def output_directory(path):
    path = Path(path).resolve()
    if path == ROOT or (ROOT in path.parents and ROOT / "build" not in path.parents):
        raise ValueError("runtime artifacts require an ignored build directory or an external directory")
    path.mkdir(parents=True, exist_ok=False)
    return path


def compile_runner(build, output, toolkit=TOOLKIT):
    build, output, toolkit = map(lambda p: Path(p).resolve(), (build, output, toolkit))
    receipt = verify_build(build, toolkit)
    before = sha256(RUNNER_SOURCE)
    runner = output / "tree_line_min_distance_runner"
    source_command(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", RUNNER_SOURCE,
                    "-I" + str(toolkit / "include"), "-I" + str(build / "autogen"),
                    "-L" + str(build), "-L" + str(toolkit / "lib64"),
                    "-Wl,-rpath," + str(build), "-Wl,-rpath," + str(toolkit / "lib64"),
                    "-lcust_opapi", "-lascendcl", "-lnnopbase", "-o", runner],
                   toolkit, ROOT, output / "runner-build.log")
    if sha256(RUNNER_SOURCE) != before:
        raise RuntimeError("runner source changed during compilation")
    verify_build(build, toolkit)
    return runner, {"operator": receipt, "runner_source_sha256": before, "runner_sha256": sha256(runner)}


def prepare_fixtures(output, cases, suite):
    fixtures = output / "fixtures"
    generate_suite(fixtures, cases, suite=suite)
    for case in cases:
        if case.expectation != "success":
            continue
        folder = fixtures / case.name
        metadata = json.loads((folder / "case.json").read_text())
        verify_input_files(folder, metadata)
        b, n, _ = case.tree_points_shape
        m = case.line_segments_shape[1]
        (folder / "case.conf").write_text(f"b={b}\nn={n}\nm={m}\ndtype={case.dtype}\n")
    return fixtures


def fixture_hashes(folder):
    return {p.name: sha256(p) for p in sorted(Path(folder).iterdir()) if p.is_file()}


def require_fixture(folder, expected):
    if fixture_hashes(folder) != expected:
        raise ValueError("fixture metadata, input, or reference changed during execution")


@contextmanager
def device_lock(device):
    # The lock is shared by suite runners on this host.
    path = Path(tempfile.gettempdir()) / f"tree-line-min-distance-device-{os.getuid()}-{device}.lock"
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def launch(command, build, toolkit, output, timeout):
    environment = runtime_environment(build, toolkit)
    setup = toolkit / "set_env.sh"
    shell = ["bash", "-c", "source " + shlex.quote(str(setup)) + ' && exec "$@"', "cann", *map(str, command)]
    save(output / "command.json", {"argv": list(map(str, command)), "timeout_seconds": timeout})
    with (output / "process.log").open("w") as stream:
        process = subprocess.Popen(shell, cwd=ROOT, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return {"exit_code": process.wait(timeout=timeout), "timed_out": False}
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return {"exit_code": process.returncode, "timed_out": True}


def execution_marker(output, filename="execution.txt"):
    fields = {}
    for line in (Path(output) / filename).read_text().splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in fields:
            raise ValueError("invalid or duplicate execution marker field")
        fields[key] = value
    return fields


def assess_execution(fixture, output, process, library, repeat, first_required=True):
    result = {"status": "Runtime Error", "passed": False, **process, "first": None, "final": None}
    reasons = []
    marker = {}
    try:
        marker = execution_marker(output)
        result["execution"] = marker
        snapshots = (("first_", "first"), ("", "final")) if first_required else (("", "final"),)
        for prefix, key in snapshots:
            checked = verify_case_output(fixture, output / (prefix + "min_distance.bin"),
                                         output / (prefix + "nearest_segment_idx.bin"))
            result[key] = {**asdict(checked), "error_percent": checked.mismatch_count * 100 / checked.element_count}
    except (OSError, ValueError, KeyError, TypeError) as error:
        reasons.append(str(error))
    if process["timed_out"]:
        result["status"] = "Timeout"
    elif process["exit_code"] != 0 or marker.get("status") != "executed" or marker.get("api_code") != "0":
        reasons.append(marker.get("detail", "runner did not complete successfully"))
    elif (first_required and result["first"] is None) or result["final"] is None:
        result["status"] = "Data Error"
    elif (marker.get("library") != str(Path(library).resolve()) or marker.get("repeat") != str(repeat)
          or (not first_required and marker.get("mode") != "performance")):
        result["status"] = "Execution Error"
        reasons.append("loaded library or launch count differs from the requested build")
    elif marker.get("guards_ok") != "1" or marker.get("inputs_unchanged") != "1":
        result["status"] = "Memory Error"
        reasons.append("tensor guards or device inputs changed")
    elif (first_required and not result["first"]["passed"]) or not result["final"]["passed"]:
        result["status"] = "Wrong Answer"
    else:
        result["status"], result["passed"] = "Pass", True
    result["diagnostics"] = reasons
    return result


def assess_batch(fixtures, output, cases, process, library, repeat, performance=False):
    results, stopped = [], False
    try:
        batch = execution_marker(output, "batch.txt")
        valid_batch = batch["status"] == "completed" and int(batch["case_count"]) == len(cases)
        valid_batch = valid_batch and process["exit_code"] in (0, 1) and not process["timed_out"]
    except (OSError, ValueError, KeyError):
        batch, valid_batch = {}, False
    for case in cases:
        folder = output / case.name
        try:
            marker = execution_marker(folder)
            code = int((folder / "case-exit.txt").read_text())
            if marker["status"] == "Skipped":
                if not stopped or code != 3:
                    raise ValueError("Skipped without a preceding runtime failure")
                result = {"status": "Skipped", "passed": None, "reason": marker.get("reason"),
                          "execution": marker, "first": None, "final": None}
            else:
                if stopped:
                    raise ValueError("case executed after preceding runtime failure")
                result = assess_execution(fixtures / case.name, folder,
                                          {"exit_code": code, "timed_out": process["timed_out"]},
                                          library, repeat, first_required=not performance)
                stopped = stopped or result["status"] in ("Runtime Error", "Timeout")
        except (OSError, ValueError, KeyError) as error:
            result = {"status": "Timeout" if process["timed_out"] else "Execution Error",
                      "passed": False, "diagnostics": [str(error)], "first": None, "final": None}
        results.append({"name": case.name, **result})
    completed = sum(r.get("execution", {}).get("status") == "executed" for r in results)
    if valid_batch:
        try:
            valid_batch = (int(batch.get("completed_cases", "-1")) == completed
                           and batch.get("stopped") == str(int(stopped))
                           and (process["exit_code"] == 0 or stopped))
        except ValueError:
            valid_batch = False
    # Initialization errors remain concrete API failures; incomplete successful output cannot pass.
    if not valid_batch:
        for result in results:
            if result["passed"] is True:
                result["passed"], result["status"] = False, "Execution Error"
                result.setdefault("diagnostics", []).append("invalid batch completion or process exit")
    return {"process": process, "completion": batch, "completed": valid_batch, "cases": results}


def run_batch(runner, build, toolkit, fixtures, cases, output, device, warmup, repeat,
              timeout, performance=False):
    output.mkdir(parents=True, exist_ok=False)
    before = {case.name: fixture_hashes(fixtures / case.name) for case in cases}
    order = output / "case-order.txt"
    order.write_text("".join(case.name + "\n" for case in cases))
    order_hash = sha256(order)
    command = [str(runner), "--fixture", str(fixtures), "--output", str(output), "--case-list", str(order),
               "--device", str(device), "--warmup", str(0 if performance else warmup), "--repeat", str(repeat),
               "--mode", "performance" if performance else "correctness"]
    if performance:
        count = len(cases) * repeat
        if count > 5000:
            raise ValueError("batch exceeds the profiler launch limit; reduce repeat or selection")
        command = ["msprof", "op", "--application=" + shlex.join(command),
                   "--output=" + str(output / "profile"), "--aic-metrics=BasicInfo",
                   "--launch-count=" + str(count), "--warm-up=" + str(warmup)]
    verify_build(build, toolkit)
    runner_hash = sha256(runner)
    process = launch(command, build, toolkit, output, timeout)
    if sha256(order) != order_hash:
        raise ValueError("batch case order changed during execution")
    for case in cases:
        require_fixture(fixtures / case.name, before[case.name])
    verify_build(build, toolkit)
    if sha256(runner) != runner_hash:
        raise ValueError("runner binary changed during execution")
    return assess_batch(fixtures, output, cases, process, build / "libcust_opapi.so", repeat, performance)


def run_case(runner, build, toolkit, fixture, output, device, warmup, repeat, timeout, profile=False):
    if profile:
        raise ValueError("performance uses run_batch with a separate cold correctness phase")
    output.mkdir(parents=True, exist_ok=False)
    before = fixture_hashes(fixture)
    verify_build(build, toolkit)
    command = [str(runner), "--fixture", str(fixture), "--output", str(output),
               "--device", str(device), "--warmup", str(warmup), "--repeat", str(repeat), "--mode", "correctness"]
    with device_lock(device):
        process = launch(command, build, toolkit, output, timeout)
    require_fixture(fixture, before)
    verify_build(build, toolkit)
    return assess_execution(fixture, output, process, build / "libcust_opapi.so", repeat)


def arguments(parser, default_suite):
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--toolkit", type=Path, default=TOOLKIT)
    parser.add_argument("--suite", choices=("coverage", "correctness", "mock", "full", "all", "custom"),
                        default=default_suite)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300)


def validate_arguments(args):
    device_guard()
    if args.device < 0 or not 0 <= args.warmup <= 500 or not 1 <= args.repeat <= 5000 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("invalid device, warmup, repeat, or timeout")
    args.build_dir, args.toolkit = args.build_dir.resolve(), args.toolkit.resolve()
    verify_build(args.build_dir, args.toolkit)
