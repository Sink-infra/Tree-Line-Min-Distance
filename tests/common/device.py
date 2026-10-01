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
import shutil
from pathlib import Path
import subprocess

from tests.common.generate_data import generate_suite
from tests.common.tensor_io import sha256, verify_input_files
from tests.common.verify_result import verify_case_output, first_error_percent
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


def compile_runner(build, output, toolkit=TOOLKIT, runner_dir=None):
    build, output, toolkit = map(lambda p: Path(p).resolve(), (build, output, toolkit))
    receipt = verify_build(build, toolkit)
    before = sha256(RUNNER_SOURCE)
    executable_dir = output if runner_dir is None else Path(runner_dir).resolve()
    if (executable_dir == ROOT or (ROOT in executable_dir.parents
            and ROOT / "build" not in executable_dir.parents)):
        raise ValueError("runner executable requires an ignored build directory or an external directory")
    executable_dir.mkdir(parents=True, exist_ok=True)
    runner = executable_dir / "tree_line_min_distance_runner"
    source_command(["g++", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-O2", "-Wall", "-Wextra", RUNNER_SOURCE,
                    "-I" + str(toolkit / "include"), "-I" + str(toolkit / "include/aclnn"), "-I" + str(build / "autogen"),
                    "-L" + str(build), "-L" + str(toolkit / "lib64"),
                    "-Wl,-rpath," + str(build), "-Wl,-rpath," + str(toolkit / "lib64"),
                    "-lcust_opapi", "-lascendcl", "-lnnopbase", "-lplatform", "-lstdc++fs", "-ldl", "-o", runner],
                   toolkit, ROOT, output / "runner-build.log")
    if sha256(RUNNER_SOURCE) != before:
        raise RuntimeError("runner source changed during compilation")
    verify_build(build, toolkit)
    fingerprint = sha256(runner)
    snapshot = output / "tree_line_min_distance_runner"
    if runner != snapshot:
        shutil.copyfile(runner, snapshot)
        shutil.copymode(runner, snapshot)
        if sha256(snapshot) != fingerprint:
            raise RuntimeError("runner snapshot differs from the executable")
    return runner, {"operator": receipt, "runner_source_sha256": before, "runner_sha256": fingerprint}


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
    path = Path("/tmp") / f"tree-line-min-distance-device-{os.getuid()}-{device}.lock"
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


CACHE_PROFILES = {"native": None, "Ascend910B4": 96 * 1024 * 1024}


def cache_profile_matches(marker, expected):
    if expected is None:
        return True
    if expected not in CACHE_PROFILES or marker.get("cache_profile") != expected or not marker.get("native_soc"):
        return False
    try:
        native = int(marker["native_l2_cache_bytes"])
        applied = int(marker["l2_cache_budget_bytes"])
    except (KeyError, TypeError, ValueError):
        return False
    return 0 < applied <= native and applied == (native if expected == "native" else CACHE_PROFILES[expected])


def assess_execution(fixture, output, process, library, repeat, first_required=True, output_init=None, vector_cores=None, cache_profile=None, compare_outputs=True):
    result = {"status": "Runtime Error", "passed": False, **process, "first": None, "final": None}
    reasons = []
    marker = {}
    try:
        marker = execution_marker(output)
        result["execution"] = marker
        snapshots = (("first_", "first"), ("", "final")) if first_required else (("", "final"),)
        for prefix, key in snapshots if compare_outputs else ():
            checked = verify_case_output(fixture, output / (prefix + "min_distance.bin"),
                                         output / (prefix + "nearest_segment_idx.bin"))
            result[key] = {
                **asdict(checked), "error_percent": checked.output_error_percent,
                "union_error_percent": checked.mismatch_count * 100 / checked.element_count,
            }
    except (OSError, ValueError, KeyError, TypeError) as error:
        reasons.append(str(error))
    if process["timed_out"]:
        result["status"] = "Timeout"
    elif process["exit_code"] != 0 or marker.get("status") != "executed" or marker.get("api_code") != "0":
        reasons.append(marker.get("detail", "runner did not complete successfully"))
    elif compare_outputs and ((first_required and result["first"] is None) or result["final"] is None):
        result["status"] = "Data Error"
    elif (marker.get("library") != str(Path(library).resolve()) or marker.get("repeat") != str(repeat)
          or (not first_required and marker.get("mode") != "performance")
          or (output_init is not None and marker.get("output_init") != output_init)
          or (vector_cores is not None and marker.get("vector_cores") != str(vector_cores))
          or not cache_profile_matches(marker, cache_profile)):
        result["status"] = "Execution Error"
        reasons.append("execution protocol differs from the requested build or resource limit")
    elif marker.get("guards_ok") != "1" or marker.get("inputs_unchanged") != "1":
        result["status"] = "Memory Error"
        reasons.append("tensor guards or device inputs changed")
    elif not compare_outputs:
        result["status"], result["passed"] = "Unmeasured Context", None
    elif (first_required and not result["first"]["passed"]) or not result["final"]["passed"]:
        result["status"] = "Wrong Answer"
    else:
        result["status"], result["passed"] = "Pass", True
    result["diagnostics"] = reasons
    result["error_percent"] = first_error_percent(result["first"], result["final"]) if result["status"] in ("Pass", "Wrong Answer") else None
    return result


def assess_batch(fixtures, output, cases, process, library, repeat, performance=False, output_init=None, vector_cores=None, cache_profile=None, sampled_names=None, context_repeat=None):
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
                sampled = sampled_names is None or case.name in sampled_names
                result = assess_execution(fixtures / case.name, folder,
                                          {"exit_code": code, "timed_out": process["timed_out"]},
                                          library, repeat if sampled else context_repeat,
                                          first_required=not performance, output_init=output_init,
                                          vector_cores=vector_cores, cache_profile=cache_profile,
                                          compare_outputs=sampled)
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
              timeout, performance=False, trace=False, output_init="zero", vector_cores=0, cache_profile="native"):
    if cache_profile not in CACHE_PROFILES:
        raise ValueError("unsupported cache profile")
    if output_init not in ("zero", "nan"):
        raise ValueError("unsupported output initialization")
    output.mkdir(parents=True, exist_ok=False)
    before = {case.name: fixture_hashes(fixtures / case.name) for case in cases}
    order = output / "case-order.txt"
    order.write_text("".join(case.name + "\n" for case in cases))
    order_hash = sha256(order)
    command = [str(runner), "--fixture", str(fixtures), "--output", str(output), "--case-list", str(order),
               "--device", str(device), "--warmup", str(0 if performance else warmup), "--repeat", str(repeat),
               "--mode", "performance" if performance else "correctness", "--output-init", output_init, "--vector-cores", str(vector_cores), "--cache-profile", cache_profile]
    if performance and trace:
        raise ValueError("cold task tracing and kernel measurement are separate phases")
    if trace:
        command = ["msprof", "--application=" + shlex.join(command),
                   "--output=" + str(output / "trace"), "--task-time=l2", "--runtime-api=on",
                   "--model-execution=on", "--ge-api=l1"]
    if performance:
        if len(cases) != 1 or repeat != 1 or warmup != 5:
            raise ValueError("kernel profiling requires one launch and five warm-up replays")
        command = ["msprof", "op", "--application=" + shlex.join(command),
                   "--output=" + str(output / "profile"), "--replay-mode=kernel",
                   "--warm-up=" + str(warmup)]
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
    return assess_batch(fixtures, output, cases, process, build / "libcust_opapi.so", repeat, performance, output_init=output_init, vector_cores=vector_cores, cache_profile=cache_profile)


def run_case(runner, build, toolkit, fixture, output, device, warmup, repeat, timeout, profile=False, output_init="zero", vector_cores=0, cache_profile="native"):
    if cache_profile not in CACHE_PROFILES:
        raise ValueError("unsupported cache profile")
    if profile:
        raise ValueError("performance uses the native kernel profiling entry point")
    output.mkdir(parents=True, exist_ok=False)
    before = fixture_hashes(fixture)
    verify_build(build, toolkit)
    command = [str(runner), "--fixture", str(fixture), "--output", str(output),
               "--device", str(device), "--warmup", str(warmup), "--repeat", str(repeat), "--mode", "correctness",
               "--output-init", output_init, "--vector-cores", str(vector_cores), "--cache-profile", cache_profile]
    with device_lock(device):
        process = launch(command, build, toolkit, output, timeout)
    require_fixture(fixture, before)
    verify_build(build, toolkit)
    return assess_execution(fixture, output, process, build / "libcust_opapi.so", repeat, output_init=output_init, vector_cores=vector_cores, cache_profile=cache_profile)


def arguments(parser, default_suite):
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--toolkit", type=Path, default=TOOLKIT)
    parser.add_argument("--runner-dir", type=Path)
    parser.add_argument("--suite", choices=("coverage", "correctness", "mock", "full", "all", "custom"),
                        default=default_suite)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-init", choices=("zero", "nan"), default="zero")
    parser.add_argument("--vector-cores", type=int, default=40 if default_suite == "mock" else 0)
    parser.add_argument("--cache-profile", choices=tuple(CACHE_PROFILES), default="Ascend910B4")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10 if default_suite == "mock" else 5)
    parser.add_argument("--timeout", type=float, default=300)


def validate_arguments(args):
    device_guard()
    if not 0 <= args.vector_cores <= 40:
        raise ValueError("vector cores must be between 0 and 40; zero uses native resources")
    if args.device < 0 or not 0 <= args.warmup <= 500 or not 1 <= args.repeat <= 5000 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("invalid device, warmup, repeat, or timeout")
    args.build_dir, args.toolkit = args.build_dir.resolve(), args.toolkit.resolve()
    verify_build(args.build_dir, args.toolkit)
