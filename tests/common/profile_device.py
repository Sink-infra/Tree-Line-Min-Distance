"""Shared-process native kernel replay and Case-to-image attribution."""

from pathlib import Path
import re
import shlex

from tests.common.device import assess_batch, fixture_hashes, launch, require_fixture
from tests.common.profiler import collect_samples, kernel_dtypes, statistics_us, verify_profiled_binary
from tests.common.reference import materialize_case
from tests.common.tensor_io import sha256, save_tensor
from tests.st.scripts.build import verify_build, save

APPLICATION_CALLS = 5
CONTEXT_CALLS = APPLICATION_CALLS * 7


def execution_prefix(catalog, selected):
    names = [case.name for case in catalog]
    chosen = [case.name for case in selected]
    if not chosen or len(names) != len(set(names)) or len(chosen) != len(set(chosen)):
        raise ValueError("empty or duplicate profiling Case selection")
    if [name for name in names if name in chosen] != chosen:
        raise ValueError("selected cases must be an ordered catalogue subset")
    return list(catalog[:names.index(chosen[-1]) + 1])


def prepare_context_catalog(folder, catalog, selected):
    """Preserve the complete Host shape and dtype catalogue."""
    execution_prefix(catalog, selected)
    names = [case.name for case in catalog]
    folder.mkdir(parents=True, exist_ok=False)
    for case in catalog:
        path = folder / case.name
        path.mkdir()
        b, n, _ = case.tree_points_shape
        m = case.line_segments_shape[1]
        (path / "case.conf").write_text(f"b={b}\nn={n}\nm={m}\ndtype={case.dtype}\n")
    (folder / "order.txt").write_text("".join(name + "\n" for name in names))


def prepare_context_inputs(fixtures, catalog, selected):
    """Materialize preceding inputs without references or numerical assessment."""
    prefix = execution_prefix(catalog, selected)
    chosen, descriptors = {case.name for case in selected}, {}
    for case in prefix:
        if case.name in chosen:
            continue
        folder = fixtures / case.name
        folder.mkdir()
        data = materialize_case(case)
        descriptors[case.name] = {
            "tree_points": save_tensor(folder, "tree_points.bin", data.tree_points, case.dtype),
            "line_segments": save_tensor(folder, "line_segments.bin", data.line_segments,
                                          case.resolved_line_segments_dtype),
        }
        b, n, _ = case.tree_points_shape
        m = case.line_segments_shape[1]
        (folder / "case.conf").write_text(f"b={b}\nn={n}\nm={m}\ndtype={case.dtype}\n")
    save(fixtures / "context-inputs.json", descriptors)
    return prefix


def launch_attribution(log, cases):
    names = [case.name for case in cases]
    markers, observed, indices = [], {}, {}
    current, pid, counts = None, None, {}
    for line in Path(log).read_text().splitlines():
        if line.startswith("<CaseReplay> "):
            fields = line.split()
            if (len(fields) != 3 or fields[1] not in names or fields[1] in markers
                    or not fields[2].isdigit() or int(fields[2]) <= 0):
                raise ValueError("invalid or duplicate Case launch marker")
            current = fields[1]
            if pid is not None and pid != fields[2]:
                raise ValueError("profiling Cases did not share one application process")
            pid = fields[2]
            markers.append(current)
        match = re.search(r"<ProfData> Start profiling on device \d+, kernel: (\S+)", line)
        if not match or not match[1].startswith("TreeLineMinDistance"):
            continue
        if current is None or counts.get(current, 0) >= APPLICATION_CALLS:
            raise ValueError("kernel launch exceeds the Case call count")
        kernel = match[1]
        index = indices.get(kernel, 0)
        indices[kernel] = index + 1
        observed[kernel, str(index)] = current
        counts[current] = counts.get(current, 0) + 1
    if markers != names[:len(markers)]:
        raise ValueError("Case launch markers differ from the requested order")
    return observed, pid


def collect_shared_samples(folder, log, cases, outcomes, build, launch_cases=None):
    if len(cases) != len(outcomes) or any(c.name != r.get("name") for c, r in zip(cases, outcomes)):
        raise ValueError("profiling outcomes differ from the Case order")
    observed, pid = launch_attribution(log, cases if launch_cases is None else launch_cases)
    executed = {case.name: case for case, result in zip(cases, outcomes)
                if result.get("execution", {}).get("status") == "executed"}
    if not executed:
        if list(Path(folder).rglob("OpBasicInfo*.csv")):
            raise ValueError("unexpected samples without executed Cases")
        return {}, observed
    types = kernel_dtypes(build)
    data = collect_samples(folder, len(executed) * APPLICATION_CALLS, require_device_profile=True)
    profiles = {}
    for sample in data["samples"]:
        name = observed.get((sample["op_name"], sample["launch_index"]))
        case = executed.get(name)
        dtype = [value for stem, value in types.items()
                 if sample["op_name"] == stem or sample["op_name"].startswith(stem + "_")]
        if case is None or dtype != [case.dtype]:
            raise ValueError("native launch has no unique Case and dtype")
        if sample["process_id"] and sample["process_id"] != pid:
            raise ValueError("sample PID differs from the shared application")
        verify_profiled_binary(folder, sample, build)
        profiles.setdefault(name, {"samples": [], "csv_files": data["csv_files"]})["samples"].append(sample)
    if set(profiles) != set(executed):
        raise ValueError("missing profiled Case")
    for profile in profiles.values():
        if len(profile["samples"]) != APPLICATION_CALLS:
            raise ValueError("Case has an incomplete native call sequence")
        profile["statistics"] = statistics_us(s["duration_us"] for s in profile["samples"])
    return profiles, observed


def run_profile_round(runner, build, toolkit, fixtures, cases, output, device,
                      timeout, output_init, vector_cores, cache_profile, context_catalog, execution_cases=None):
    execution_cases = cases if execution_cases is None else execution_cases
    output.mkdir(parents=True, exist_ok=False)
    before = {case.name: fixture_hashes(fixtures / case.name) for case in execution_cases}
    context_files = [path for path in Path(context_catalog).rglob("*") if path.is_file()]
    order = output / "case-order.txt"
    order.write_text("".join(case.name + "\n" for case in execution_cases))
    selection = output / "profile-cases.txt"
    selection.write_text("".join(case.name + "\n" for case in cases))
    protected = {path: sha256(path) for path in [Path(runner), order, selection, *context_files]}
    application = [str(runner), "--fixture", str(fixtures), "--output", str(output),
                   "--case-list", str(order), "--context-catalog", str(context_catalog),
                   "--profile-list", str(selection),
                   "--device", str(device), "--warmup", "0", "--repeat", str(APPLICATION_CALLS), "--mode", "performance",
                   "--output-init", output_init, "--vector-cores", str(vector_cores),
                   "--cache-profile", cache_profile]
    command = ["msprof", "op", "--application=" + shlex.join(application),
               "--output=" + str(output / "profile"), "--replay-mode=kernel", "--warm-up=5",
               "--aic-metrics=BasicInfo", "--mstx=on", "--mstx-include=selected",
               "--launch-count=" + str(len(cases) * APPLICATION_CALLS)]
    verify_build(build, toolkit)
    process = launch(command, build, toolkit, output, timeout)
    verify_build(build, toolkit)
    for path, digest in protected.items():
        if sha256(path) != digest:
            raise ValueError("profiling application or context catalogue changed during execution")
    if set(context_files) != {p for p in Path(context_catalog).rglob("*") if p.is_file()}:
        raise ValueError("profiling context file list changed during execution")
    for case in execution_cases:
        require_fixture(fixtures / case.name, before[case.name])
    chosen = {case.name for case in cases}
    batch = assess_batch(fixtures, output, execution_cases, process, build / "libcust_opapi.so", APPLICATION_CALLS,
                         performance=True, output_init=output_init,
                         vector_cores=vector_cores, cache_profile=cache_profile,
                         sampled_names=chosen, context_repeat=CONTEXT_CALLS)
    batch["contexts"] = [result for result in batch["cases"] if result["name"] not in chosen]
    batch["cases"] = [result for result in batch["cases"] if result["name"] in chosen]
    profiles, observed, error = {}, {}, None
    try:
        profiles, observed = collect_shared_samples(output / "profile", output / "process.log",
                                                    cases, batch["cases"], build, launch_cases=execution_cases)
    except (OSError, ValueError, KeyError) as issue:
        error = str(issue)
    for result in batch["cases"]:
        if result["name"] in profiles:
            result["profiling"] = profiles[result["name"]]
        elif result["status"] in ("Pass", "Wrong Answer"):
            result["profiling_error"], result["passed"] = error or "missing Case sample", False
            if result["status"] == "Pass":
                result["status"] = "Profiler Error"
    batch["launches"] = [{"kernel": kernel, "index": index, "case": name}
                         for (kernel, index), name in observed.items()]
    batch["profiling_error"] = error
    batch["completed"] = batch["completed"] and process["exit_code"] == 0 and not process["timed_out"]
    return batch
