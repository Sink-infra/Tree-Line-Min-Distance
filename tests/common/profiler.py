"""Profiler kernel samples and independent-process timing statistics."""

from __future__ import annotations

import csv
import math
import json
from pathlib import Path
import statistics

from tests.common.tensor_io import sha256


def statistics_us(samples):
    values = [float(value) for value in samples]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("kernel durations must be finite, positive, and nonempty")
    return {"count": len(values), "mean_us": statistics.mean(values),
            "median_us": statistics.median(values), "min_us": min(values), "max_us": max(values)}


def collect_samples(folder, expected_count):
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
                identity = (name, launch) if launch else (relative, row_number)
                if identity in identities:
                    raise ValueError("duplicate profiler kernel launch")
                identities.add(identity)
                samples.append({"duration_us": value, "op_name": name, "launch_index": launch,
                                "source": relative, "row": row_number,
                                "block_dim": row.get("BlockDim", ""),
                                "current_frequency_mhz": row.get("CurrentFreq", row.get("aic_frequency(MHz)", "")),
                                "op_type": row.get("OpType", ""), "device_id": row.get("DeviceId", "")})
        sources.append({"path": relative, "sha256": sha256(path)})
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
    return {"process_count": len(processes), "elapsed_us": statistics.median(means),
            "process_means_us": means, "median_of_process_medians_us": statistics.median(medians),
            "process_medians_us": medians, "all_samples": statistics_us(pooled)}


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


def collect_batch_samples(folder, cases, outcomes, repeat, build):
    if len(cases) != len(outcomes) or any(r.get("name") != c.name for c, r in zip(cases, outcomes)):
        raise ValueError("case outcomes differ from the declared batch order")
    executed = [case for case, result in zip(cases, outcomes)
                if result.get("execution", {}).get("status") == "executed"]
    if not executed:
        if list(Path(folder).rglob("OpBasicInfo*.csv")):
            raise ValueError("unexpected samples without executed cases")
        return {}
    collected = collect_samples(folder, len(executed) * repeat)
    types = kernel_dtypes(build)
    groups = {}
    for sample in collected["samples"]:
        matches = [dtype for stem, dtype in types.items()
                   if sample["op_name"] == stem or sample["op_name"].startswith(stem + "_")]
        if len(matches) != 1 or not sample["launch_index"].isdigit():
            raise ValueError("kernel type or launch ordering is ambiguous")
        sample = {**sample, "dtype": matches[0]}
        groups.setdefault(matches[0], []).append(sample)
    for values in groups.values():
        if len({s["op_name"] for s in values}) != 1:
            raise ValueError("multiple kernel variants prevent unambiguous batch attribution")
        values.sort(key=lambda row: int(row["launch_index"]))
        if [int(s["launch_index"]) for s in values] != list(range(len(values))):
            raise ValueError("batch launch sequence has missing or unexpected indices")
    offsets, results = {}, {}
    for case in executed:
        offset = offsets.get(case.dtype, 0)
        selected = groups.get(case.dtype, [])[offset:offset + repeat]
        offsets[case.dtype] = offset + repeat
        if len(selected) != repeat:
            raise ValueError("case has incomplete profiler samples")
        paths = {s["source"] for s in selected}
        results[case.name] = {
            "samples": selected, "statistics": statistics_us(s["duration_us"] for s in selected),
            "csv_files": [p for p in collected["csv_files"] if p["path"] in paths],
        }
    if any(offsets.get(dtype, 0) != len(values) for dtype, values in groups.items()):
        raise ValueError("unassigned batch samples")
    return results


def combine_phases(correctness, measurement):
    if correctness["status"] not in ("Pass", "Wrong Answer", "Skipped"):
        result = dict(correctness)
    else:
        result = dict(measurement)
        if correctness["status"] == "Wrong Answer" and measurement["status"] == "Pass":
            result["status"], result["passed"] = "Wrong Answer", False
        if (correctness["status"] == "Skipped") != (measurement["status"] == "Skipped"):
            result["status"], result["passed"] = "Execution Error", False
            result.setdefault("diagnostics", []).append("correctness and measurement batch stopping differ")
    result["correctness"] = correctness
    result["measurement"] = measurement
    result["first"] = correctness.get("first")
    result["final"] = measurement.get("final")
    return result
