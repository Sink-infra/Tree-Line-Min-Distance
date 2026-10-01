"""Execute selected cases in a shared Ascend session."""

from __future__ import annotations

import argparse
import subprocess

from tests.common.cases import select_cases
from tests.common.device import (
    arguments, compile_runner, device_lock, output_directory, prepare_fixtures, run_batch, validate_arguments,
)
from tests.st.scripts.build import save


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser, "full")
    args = parser.parse_args(argv)
    try:
        validate_arguments(args)
        cases = select_cases(args.suite, names=args.case, tags=args.tag)
        executable = [c for c in cases if c.expectation == "success"]
        if not executable:
            raise ValueError("selection contains no executable cases; input rejection contracts are covered by UT")
        output = output_directory(args.output_dir)
        runner, receipt = compile_runner(args.build_dir, output, args.toolkit, runner_dir=args.runner_dir)
        fixtures = prepare_fixtures(output, cases, args.suite)
        with device_lock(args.device):
            batch = run_batch(runner, args.build_dir, args.toolkit, fixtures, executable, output / "batch",
                              args.device, args.warmup, args.repeat, args.timeout, output_init=args.output_init, vector_cores=args.vector_cores, cache_profile=args.cache_profile)
        measured = {r["name"]: r for r in batch["cases"]}
        rows = [measured[c.name] if c.expectation == "success" else
                {"name": c.name, "status": "Skipped", "passed": None,
                 "reason": "input rejection contract; tested by UT"} for c in cases]
        report = {"schema_version": 2, "cache_profile": args.cache_profile, "output_init": args.output_init, "vector_cores": args.vector_cores, "build": receipt, "suite": args.suite, "device": args.device,
                  "warmup": args.warmup, "repeat": args.repeat, "batch": batch, "cases": rows,
                  "passed": batch["completed"] and all(r["passed"] is not False for r in rows)}
        save(output / "results.json", report)
        for row in rows:
            print(row["name"] + ": " + row["status"])
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
        parser.error(str(error))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
