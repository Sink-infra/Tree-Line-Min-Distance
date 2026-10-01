"""Profiler kernel samples and independent-process timing statistics."""

from __future__ import annotations

import csv
import math
import json
from pathlib import Path
import statistics
from decimal import Decimal, InvalidOperation

from tests.common.tensor_io import sha256
from tests.common.verify_result import first_error_percent


def statistics_us(samples):
    values = [float(value) for value in samples]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("kernel durations must be finite, positive, and nonempty")
    median = statistics.median(values)
    return {"count": len(values), "mean_us": statistics.mean(values),
            "median_us": median, "min_us": min(values), "max_us": max(values),
            "mad_us": statistics.median(abs(value - median) for value in values)}


def collect_samples(folder, expected_count, require_device_profile=False):
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("expected sample count must be positive")
    folder = Path(folder)
    files = sorted(folder.rglob("OpBasicInfo*.csv"))
    if not files:
        raise ValueError("Profiler did not produce OpBasicInfo CSV files")
    samples, sources, identities = [], [], set()
    for path in files:
        relative = str(path.relative_to(folder))
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"OpName", "TaskDuration(us)"}
            columns = {name.replace(" ", "") for name in reader.fieldnames or ()}
            if not required.issubset(columns):
                raise ValueError(f"missing profiler columns: {relative}")
            for row_number, row in enumerate(reader, 2):
                row = {key.replace(" ", ""): value for key, value in row.items() if key}
                name = row["OpName"]
                if not name.startswith("TreeLineMinDistance"):
                    continue
                value = float(row["TaskDuration(us)"])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("invalid profiler kernel duration")
                launch = row.get("LaunchIndex", "")
                if not launch and path.parent.name.isdigit():
                    launch = path.parent.name
                identity = (row.get("Pid", ""), name, launch) if launch else (relative, row_number)
                if identity in identities:
                    raise ValueError("duplicate profiler kernel launch")
                identities.add(identity)
                samples.append({"duration_us": value, "op_name": name, "launch_index": launch,
                                "source": relative, "row": row_number,
                                "block_dim": row.get("BlockDim", ""),
                                "current_frequency_mhz": row.get("CurrentFreq", row.get("aic_frequency(MHz)", "")),
                                "rated_frequency_mhz": row.get("RatedFreq", ""),
                                "op_type": row.get("OpType", ""), "device_id": row.get("DeviceId", ""),
                                "process_id": row.get("Pid", "")})
        sources.append({"path": relative, "sha256": sha256(path)})
    if expected_count == 1 and len(samples) == 1 and not samples[0]["launch_index"]:
        samples[0]["launch_index"] = "0"
    if require_device_profile:
        for sample in samples:
            if (sample["current_frequency_mhz"].strip() != "1650"
                    or sample["rated_frequency_mhz"].strip() != "1650"):
                raise ValueError("profiling requires Current/Rated Freq 1650/1650")
            raw = sorted((folder / sample["source"]).parent.glob("dump/DeviceProf*.bin"))
            if not raw or any(path.stat().st_size == 0 for path in raw):
                raise ValueError("missing or empty DeviceProf kernel data")
            sample["measurement_source"] = "DeviceProf"
            sample["metric"] = "Task Duration(us)"
            sample["device_profile_files"] = [
                {"path": str(path.relative_to(folder)), "sha256": sha256(path)} for path in raw]
    if len(samples) != expected_count:
        raise ValueError(f"profiler sample count {len(samples)} differs from requested {expected_count}")
    return {"samples": samples, "statistics": statistics_us(s["duration_us"] for s in samples),
            "csv_files": sources}


def aggregate_processes(processes):
    if not processes:
        raise ValueError("no profiler processes")
    means = [p["statistics"]["mean_us"] for p in processes]
    medians = [p["statistics"]["median_us"] for p in processes]
    pooled = [s["duration_us"] for p in processes for s in p["samples"]]
    all_samples = statistics_us(pooled)
    ordered = sorted(pooled)
    cut = len(ordered) // 5
    central = ordered[cut:len(ordered) - cut] if cut else ordered
    trimmed = statistics_us(central)
    return {"process_count": len(processes), "elapsed_us": all_samples["mean_us"],
            "statistic": "arithmetic mean of all full Task samples",
            "process_means_us": means, "median_of_process_medians_us": statistics.median(medians),
            "process_medians_us": medians, "all_samples": all_samples,
            "trim_fraction_per_tail": 0.2, "trimmed_samples": trimmed}


def kernel_dtypes(build):
    config = Path(build) / "tmp/vendors/custom/op_impl/ai_core/tbe/kernel/config/ascend910b/tree_line_min_distance.json"
    mapping = {}
    for entry in json.loads(config.read_text())["binList"]:
        inputs = entry["inputs"]
        dtype = inputs[0]["dtype"]
        if dtype not in ("float32", "float16") or inputs[1]["dtype"] != dtype:
            raise ValueError("unsupported kernel input types")
        stem = Path(entry["binInfo"]["jsonFilePath"]).stem
        if stem in mapping and mapping[stem] != dtype:
            raise ValueError("ambiguous kernel input types")
        mapping[stem] = dtype
    if not mapping:
        raise ValueError("operator package contains no kernel type information")
    return mapping



def collect_batch_trace(folder, cases, outcomes, build):
    """Match cold tasks to case order, shapes, types, and concrete kernel variants."""
    executed = [case for case, result in zip(cases, outcomes)
                if result.get("execution", {}).get("status") == "executed"]
    if len(cases) != len(outcomes) or any(r.get("name") != c.name for c, r in zip(cases, outcomes)):
        raise ValueError("case outcomes differ from the declared batch order")
    folder = Path(folder)
    types = kernel_dtypes(build)
    tasks, summaries, sources = [], {}, []
    for pattern, kind in (("task_time*.csv", "task"), ("op_summary*.csv", "summary")):
        for path in sorted(folder.rglob(pattern)):
            relative = str(path.relative_to(folder))
            sources.append({"path": relative, "sha256": sha256(path)})
            with path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                for row_number, row in enumerate(reader, 2):
                    if kind == "task":
                        name = row.get("kernel_name", "")
                        if not name.startswith("TreeLineMinDistance"):
                            continue
                        try:
                            start = Decimal(row["task_start(us)"].strip())
                            identity = (row["Device_id"], row["stream_id"], row["task_id"])
                        except (KeyError, InvalidOperation) as error:
                            raise ValueError("invalid task trace identity or timestamp") from error
                        if not start.is_finite() or start < 0:
                            raise ValueError("invalid task trace timestamp")
                        matches = [dtype for stem, dtype in types.items()
                                   if name == stem or name.startswith(stem + "_")]
                        if len(matches) != 1:
                            raise ValueError("task trace kernel type is ambiguous")
                        tasks.append({"kernel_name": name, "dtype": matches[0], "start": start,
                                      "identity": identity, "source": relative, "row": row_number})
                    elif row.get("OP Type", row.get("Op Type")) == "TreeLineMinDistance":
                        identity = (row["Device_id"], row["Stream ID"], row["Task ID"])
                        timestamp = row.get("Task Start Time(us)")
                        try:
                            timestamp = Decimal(timestamp.strip()) if timestamp is not None else None
                        except InvalidOperation as error:
                            raise ValueError("invalid operator trace timestamp") from error
                        if timestamp is not None and (not timestamp.is_finite() or timestamp < 0):
                            raise ValueError("invalid operator trace timestamp")
                        entries = summaries.setdefault(identity, [])
                        if any(previous == timestamp or previous is None or timestamp is None
                               for previous, _ in entries):
                            raise ValueError("duplicate operator trace identity")
                        entries.append((timestamp, row))
    if len(tasks) != len(executed) or sum(map(len, summaries.values())) != len(executed):
        raise ValueError("cold task or operator trace count differs from executed cases")
    tasks.sort(key=lambda task: task["start"])
    if (len({(task["identity"], task["start"]) for task in tasks}) != len(tasks)
            or len({task["start"] for task in tasks}) != len(tasks)):
        raise ValueError("duplicate or ambiguous cold task ordering")
    results = {}
    for case, task in zip(executed, tasks):
        entries = summaries.get(task["identity"], [])
        matched = [(timestamp, row) for timestamp, row in entries
                   if timestamp is None or timestamp == task["start"]]
        if len(matched) != 1:
            raise ValueError("cold task has no corresponding operator trace")
        timestamp, row = matched[0]
        entries.remove(matched[0])
        shapes = ";".join(",".join(map(str, shape)) for shape in
                          (case.tree_points_shape, case.line_segments_shape))
        dtype = "FLOAT" if case.dtype == "float32" else "FLOAT16"
        if (task["dtype"] != case.dtype or row.get("Input Shapes", "").strip('"') != shapes
                or row.get("Input Data Types") != dtype + ";" + dtype
                or row.get("Input Formats") != "ND;ND"):
            raise ValueError("cold task trace input contract differs from the case")
        results[case.name] = {"kernel_name": task["kernel_name"], "dtype": task["dtype"],
                             "task_start_us": str(task["start"]), "identity": list(task["identity"]),
                             "source": task["source"], "row": task["row"]}
    return {"cases": results, "csv_files": sources}


def collect_batch_samples(folder, cases, outcomes, repeat, build, trace=None):
    if len(cases) != len(outcomes) or any(r.get("name") != c.name for c, r in zip(cases, outcomes)):
        raise ValueError("case outcomes differ from the declared batch order")
    executed = [case for case, result in zip(cases, outcomes)
                if result.get("execution", {}).get("status") == "executed"]
    if not executed:
        if list(Path(folder).rglob("OpBasicInfo*.csv")):
            raise ValueError("unexpected samples without executed cases")
        return {}
    collected = collect_samples(folder, len(executed) * repeat, require_device_profile=True)
    types = kernel_dtypes(build)
    groups = {}
    for sample in collected["samples"]:
        matches = [dtype for stem, dtype in types.items()
                   if sample["op_name"] == stem or sample["op_name"].startswith(stem + "_")]
        if len(matches) != 1 or not sample["launch_index"].isdigit():
            raise ValueError("kernel type or launch ordering is ambiguous")
        sample = {**sample, "dtype": matches[0]}
        key = sample["op_name"] if trace is not None else matches[0]
        groups.setdefault(key, []).append(sample)
    for values in groups.values():
        if len({s["op_name"] for s in values}) != 1:
            raise ValueError("multiple kernel variants require a cold task trace")
        values.sort(key=lambda row: int(row["launch_index"]))
        if [int(s["launch_index"]) for s in values] != list(range(len(values))):
            raise ValueError("batch launch sequence has missing or unexpected indices")
    offsets, results = {}, {}
    for case in executed:
        if trace is None:
            key = case.dtype
        else:
            observed = trace["cases"].get(case.name)
            if observed is None or observed["dtype"] != case.dtype:
                raise ValueError("case has no matching cold kernel trace")
            key = observed["kernel_name"]
        offset = offsets.get(key, 0)
        selected = groups.get(key, [])[offset:offset + repeat]
        offsets[key] = offset + repeat
        if len(selected) != repeat:
            raise ValueError("case has incomplete profiler samples")
        paths = {s["source"] for s in selected}
        results[case.name] = {
            "samples": selected, "statistics": statistics_us(s["duration_us"] for s in selected),
            "csv_files": [p for p in collected["csv_files"] if p["path"] in paths],
        }
        if trace is not None:
            results[case.name]["cold_trace"] = trace["cases"][case.name]
    if any(offsets.get(key, 0) != len(values) for key, values in groups.items()):
        raise ValueError("unassigned batch samples")
    return results


def collect_process_samples(folder, cases, outcomes, children, build):
    if not (len(cases) == len(outcomes) == len(children)):
        raise ValueError("Case processes and outcomes have different lengths")
    executed = {}
    for case, outcome, child in zip(cases, outcomes, children):
        if case.name != outcome["name"] or case.name != child["name"]:
            raise ValueError("Case process order differs from outcomes")
        if outcome.get("execution", {}).get("status") == "executed":
            pid = str(child["pid"])
            if pid in executed:
                raise ValueError("duplicate Case process ID")
            executed[pid] = case
    if not executed:
        if list(Path(folder).rglob("OpBasicInfo*.csv")):
            raise ValueError("unexpected profiler samples without executed Cases")
        return {}
    collected = collect_samples(folder, len(executed), require_device_profile=True)
    types, results = kernel_dtypes(build), {}
    for sample in collected["samples"]:
        case = executed.get(sample["process_id"])
        matches = [dtype for stem, dtype in types.items()
                   if sample["op_name"] == stem or sample["op_name"].startswith(stem + "_")]
        if case is None or matches != [case.dtype] or case.name in results:
            raise ValueError("profiler PID or kernel dtype has no unique Case")
        verify_profiled_binary(folder, sample, build)
        results[case.name] = {
            "samples": [sample], "statistics": statistics_us([sample["duration_us"]]),
            "csv_files": [row for row in collected["csv_files"] if row["path"] == sample["source"]],
        }
    return results


def verify_profiled_binary(folder, sample, build):
    """Require the executed kernel image to match the selected package."""
    package = (Path(build) / "tmp/vendors/custom/op_impl/ai_core/tbe/kernel").resolve()
    config = package / "config/ascend910b/tree_line_min_distance.json"
    candidates = []
    for entry in json.loads(config.read_text())["binList"]:
        relative = Path(entry["binInfo"]["jsonFilePath"]).with_suffix(".o")
        stem = relative.stem
        if sample["op_name"] == stem or sample["op_name"].startswith(stem + "_"):
            binary = (package / relative).resolve()
            if package not in binary.parents:
                raise ValueError("kernel image escapes the selected package")
            candidates.append(binary)
    if len(candidates) != 1:
        raise ValueError("profiled kernel has no unique package image")
    actual = (Path(folder) / sample["source"]).parent / "dump/aicore_binary.o"
    expected = candidates[0]
    if not actual.is_file() or not actual.stat().st_size:
        raise ValueError("missing or empty profiled kernel image")
    digest = sha256(actual)
    if digest != sha256(expected):
        raise ValueError("profiled kernel differs from the selected package")
    sample["executed_binary"] = {
        "profile_path": str(actual.relative_to(folder)),
        "package_path": str(expected.relative_to(package)), "sha256": digest,
    }


def combine_phases(correctness, measurement):
    if correctness["status"] not in ("Pass", "Wrong Answer", "Skipped"):
        result = dict(correctness)
    else:
        result = dict(measurement)
        if (correctness["status"] == "Skipped") != (measurement["status"] == "Skipped"):
            result["status"], result["passed"] = "Execution Error", False
            result.setdefault("diagnostics", []).append("correctness and measurement batch stopping differ")
    result["correctness"] = correctness
    result["measurement"] = measurement
    result["accuracy_source"] = "profiled process final outputs"
    result["first"] = correctness.get("first")
    result["final"] = measurement.get("final")
    result["error_percent"] = first_error_percent(measurement.get("final")) if result["status"] in ("Pass", "Wrong Answer") else None
    return result
