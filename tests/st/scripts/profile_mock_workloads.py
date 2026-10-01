"""Measure selected workloads with shared-session device batches."""

from __future__ import annotations

import argparse
import subprocess

from tests.common.cases import select_cases
from tests.common.device import (
    arguments, compile_runner, device_lock, output_directory, prepare_fixtures, run_batch, validate_arguments,
)
from tests.common.profiler import aggregate_processes, collect_batch_samples, combine_phases
from tests.st.scripts.build import save


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser, "mock")
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        validate_arguments(args)
        if args.processes < 1:
            raise ValueError("processes must be positive")
        cases = select_cases(args.suite, names=args.case, tags=args.tag)
        if any(case.expectation != "success" for case in cases):
            raise ValueError("profiling requires successful input contracts; select executable cases")
        if len(cases) * args.repeat > 5000:
            raise ValueError("batch exceeds the profiler launch limit")
        output = output_directory(args.output_dir)
        runner, receipt = compile_runner(args.build_dir, output, args.toolkit)
        fixtures = prepare_fixtures(output, cases, args.suite)
        report = {"schema_version": 2, "build": receipt, "suite": args.suite, "device": args.device,
                  "warmup": args.warmup, "repeat": args.repeat, "processes": args.processes,
                  "correctness_repeat": 1, "statistic": "median of process means", "batches": [], "cases": []}
        runs = {case.name: [] for case in cases}
        for number in range(args.processes):
            folder = output / f"process-{number + 1:02d}"
            folder.mkdir()
            with device_lock(args.device):
                cold = run_batch(runner, args.build_dir, args.toolkit, fixtures, cases, folder / "correctness",
                                 args.device, 0, 1, args.timeout)
                hot = run_batch(runner, args.build_dir, args.toolkit, fixtures, cases, folder / "measurement",
                                args.device, args.warmup, args.repeat, args.timeout, performance=True)
            profiling_error = None
            try:
                profiles = collect_batch_samples(folder / "measurement/profile", cases, hot["cases"],
                                                 args.repeat, args.build_dir)
            except (OSError, ValueError, KeyError) as error:
                profiles, profiling_error = {}, str(error)
            report["batches"].append({"correctness": cold, "measurement": hot,
                                     "profiling_error": profiling_error})
            for case, correct, measured in zip(cases, cold["cases"], hot["cases"]):
                result = combine_phases(correct, measured)
                if case.name in profiles:
                    result["profiling"] = profiles[case.name]
                elif result["status"] in ("Pass", "Wrong Answer"):
                    result["profiling_error"] = profiling_error or "missing case samples"
                    result["passed"] = False
                    if result["status"] == "Pass":
                        result["status"] = "Profiler Error"
                runs[case.name].append(result)
                print(f"{case.name} process {number + 1}: {result['status']}", flush=True)
            report["cases"] = []
            for case in cases:
                selected = runs[case.name]
                profiles_for_case = [r["profiling"] for r in selected if "profiling" in r]
                entry = {"name": case.name, "runs": selected,
                         "passed": all(r["passed"] is True for r in selected),
                         "timing_valid": len(profiles_for_case) == len(selected)}
                if entry["timing_valid"]:
                    entry["timing"] = aggregate_processes(profiles_for_case)
                report["cases"].append(entry)
            save(output / "performance.json", report)
        report["passed"] = (all(r["passed"] and r["timing_valid"] for r in report["cases"])
                            and all(b["correctness"]["completed"] and b["measurement"]["completed"]
                                    for b in report["batches"]))
        save(output / "performance.json", report)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
        parser.error(str(error))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
