"""Report case counts and input sizes without generating test data."""

import argparse
from tests.common.cases import CASE_GROUPS
import json


def summary():
    result = {}
    for category, cases in CASE_GROUPS.items():
        valid = [case for case in cases if case.expectation == "success"]
        result[category] = {
            "count": len(cases), "runnable": len(valid), "reject": len(cases) - len(valid),
            "max_b": max(case.tree_points_shape[0] for case in valid),
            "max_n": max(case.tree_points_shape[1] for case in valid),
            "max_m": max(case.line_segments_shape[1] for case in valid),
            "max_pairs": max(case.pair_count for case in valid),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    result = summary()
    print(json.dumps(result if args.details else {k: v["count"] for k, v in result.items()},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
