"""Generate ST inputs, mathematical references, and metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.common.cases import select_cases
from tests.common.generate_data import generate_suite


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("coverage", "correctness", "mock", "full", "all", "custom"),
                        default="full")
    parser.add_argument("--case", action="append", default=[], help="case name; repeatable")
    parser.add_argument("--tag", action="append", default=[], help="required coverage tag; repeatable")
    parser.add_argument("--expectation", choices=("all", "success", "reject"), default="all")
    parser.add_argument("--list", action="store_true", help="list the selected cases")
    parser.add_argument("--output-dir", "--output", type=Path, help="new fixture directory")
    args = parser.parse_args(argv)
    try:
        selected = select_cases(args.suite, names=args.case, tags=args.tag, expectation=args.expectation)
        if args.list:
            for case in selected:
                print(f"{case.name:58} {case.expectation:7} {case.dtype:7} "
                      f"points={case.tree_points_shape} segments={case.line_segments_shape}")
            print(f"total={len(selected)}")
            return 0
        if args.output_dir is None:
            parser.error("--output-dir is required to generate fixtures")
        manifest = generate_suite(args.output_dir.resolve(), selected, suite=args.suite)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest["counts"], sort_keys=True))
    print(f"manifest: {args.output_dir.resolve() / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
