"""Measure selected Mock Cases in a shared application each profiling round."""

from __future__ import annotations

import argparse
from collections import Counter
import subprocess

from tests.common.cases import MOCK_CASES, select_cases
from tests.common.device import (
    arguments, compile_runner, device_lock, output_directory, prepare_fixtures, validate_arguments,
)
from tests.common.profiler import aggregate_processes
from tests.common.profile_device import (
    APPLICATION_CALLS, CONTEXT_CALLS, prepare_context_catalog, prepare_context_inputs, run_profile_round,
)
from tests.st.scripts.build import save


def summarize(cases, runs, processes):
    entries = []
    for case in cases:
        selected = runs[case.name]
        profiles = [r["profiling"] for r in selected if "profiling" in r]
        observations = Counter((r["status"], r.get("error_percent")) for r in selected)
        complete = len(selected) == processes
        entry = {
            "name": case.name, "runs": selected, "collected_processes": len(selected),
            "passed": complete and all(r["passed"] is True for r in selected),
            "timing_valid": complete and len(profiles) == processes,
            "accuracy_observations": [
                {"status": status, "error_percent": error, "count": count}
                for (status, error), count in observations.items()],
        }
        if entry["timing_valid"]:
            entry["timing"] = aggregate_processes(profiles)
        entries.append(entry)
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser, "mock")
    parser.set_defaults(repeat=APPLICATION_CALLS)
    parser.add_argument("--processes", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        if args.warmup != 5:
            raise ValueError("profiling requires a warm-up setting of exactly five")
        if args.repeat != APPLICATION_CALLS:
            raise ValueError("profiling requires five application calls per case")
        validate_arguments(args)
        if args.processes < 1:
            raise ValueError("profiling requires at least one independent process with five Task samples per case")
        cases = select_cases(args.suite, names=args.case, tags=args.tag)
        if any(case.category != "mock" or case.expectation != "success" for case in cases):
            raise ValueError("profiling requires executable Mock Cases")
        output = output_directory(args.output_dir)
        runner, receipt = compile_runner(args.build_dir, output, args.toolkit, runner_dir=args.runner_dir)
        fixtures = prepare_fixtures(output, cases, args.suite)
        execution_cases = prepare_context_inputs(fixtures, MOCK_CASES, cases)
        context_catalog = output / "context-catalog"
        prepare_context_catalog(context_catalog, MOCK_CASES, cases)
        report = {
            "schema_version": 3, "build": receipt, "suite": args.suite, "device": args.device,
            "warmup": 5, "repeat": APPLICATION_CALLS, "processes": args.processes,
            "application_scope": "one shared Host/ACL process and one profiler invocation per round",
            "protocol": "cann9-kernel-shared-mstx-calls5-w5-basicinfo-deviceprof",
            "measurement_source": "DeviceProf", "metric": "Task Duration(us)",
            "replay_mode": "kernel", "metrics_group": "BasicInfo", "launch_limit": len(cases) * APPLICATION_CALLS,
            "kernel_cases": [case.name for case in cases],
            "unmeasured_contexts": [case.name for case in execution_cases if case not in cases],
            "context_calls_per_case": CONTEXT_CALLS,
            "host_context_order": [case.name for case in MOCK_CASES],
            "application_warmup_count": 0, "profiler_warmup_count": 5,
            "frequency_mhz": {"current": 1650, "rated": 1650},
            "accuracy_source": "profiled process final outputs", "cache_profile": args.cache_profile,
            "output_init": args.output_init, "vector_cores": args.vector_cores,
            "statistic": "arithmetic mean of all full DeviceProf Task samples; median and MAD retained",
            "batches": [], "cases": [],
        }
        runs = {case.name: [] for case in cases}
        for number in range(args.processes):
            round_folder = output / f"process-{number + 1:02d}"
            round_folder.mkdir()
            with device_lock(args.device):
                hot = run_profile_round(runner, args.build_dir, args.toolkit, fixtures, cases,
                                        round_folder / "measurement", args.device, args.timeout,
                                        args.output_init, args.vector_cores, args.cache_profile, context_catalog,
                                        execution_cases=execution_cases)
            report["batches"].append({"process": number + 1, "measurement": hot})
            for case, result in zip(cases, hot["cases"]):
                runs[case.name].append(result)
                print(f"{case.name} process {number + 1}: {result['status']}", flush=True)
                report["cases"] = summarize(cases, runs, args.processes)
                save(output / "performance.json", report)
        report["passed"] = (all(r["passed"] and r["timing_valid"] for r in report["cases"])
                            and all(b["measurement"]["completed"]
                                    for b in report["batches"]))
        save(output / "performance.json", report)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
        parser.error(str(error))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
