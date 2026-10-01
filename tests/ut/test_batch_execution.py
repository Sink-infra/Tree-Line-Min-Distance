"""Batch stopping, cold output checks, and profiler attribution."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.common.cases import MOCK_CASES
from tests.common.device import assess_batch, assess_execution, prepare_fixtures
from tests.common.profiler import aggregate_processes, collect_batch_samples, combine_phases


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = [MOCK_CASES[0], MOCK_CASES[12]]
        self.fixtures = prepare_fixtures(self.root, self.cases, "mock")
        self.output = self.root / "output"
        self.output.mkdir()
        self.library = self.root / "libcust_opapi.so"
        for case in self.cases:
            folder = self.output / case.name
            folder.mkdir()
            self.success(case)
        self.batch()

    def batch(self, stopped=0, completed=2, status="completed"):
        (self.output / "batch.txt").write_text(
            f"status={status}\ncase_count=2\ncompleted_cases={completed}\nstopped={stopped}\n")

    def success(self, case, mode="correctness"):
        folder = self.output / case.name
        (folder / "execution.txt").write_text(
            "status=executed\napi_code=0\nguards_ok=1\ninputs_unchanged=1\nrepeat=2\n"
            f"library={self.library}\nmode={mode}\n")
        (folder / "case-exit.txt").write_text("0\n")
        fixture = self.fixtures / case.name
        distance = np.fromfile(fixture / "reference_min_distance_fp64.bin", dtype="<f8").astype("<f4")
        index = np.fromfile(fixture / "reference_nearest_segment_idx_int32.bin", dtype="<i4")
        for prefix in ("first_", ""):
            distance.tofile(folder / (prefix + "min_distance.bin"))
            index.tofile(folder / (prefix + "nearest_segment_idx.bin"))

    def runtime(self, case):
        folder = self.output / case.name
        (folder / "execution.txt").write_text("status=Runtime Error\napi_code=561002\nstage=workspace query\n")
        (folder / "case-exit.txt").write_text("1\n")

    def skipped(self, case):
        folder = self.output / case.name
        (folder / "execution.txt").write_text("status=Skipped\nreason=preceding runtime failure\n")
        (folder / "case-exit.txt").write_text("3\n")

    def assess(self, exit_code=0, performance=False):
        return assess_batch(self.fixtures, self.output, self.cases,
                            {"exit_code": exit_code, "timed_out": False}, self.library, 2, performance)

    def test_completed_batch(self):
        result = self.assess()
        self.assertTrue(result["completed"])
        self.assertTrue(all(c["passed"] for c in result["cases"]))

    def test_runtime_failure_stops_following_case(self):
        self.runtime(self.cases[0])
        self.skipped(self.cases[1])
        self.batch(stopped=1, completed=0)
        result = self.assess(exit_code=1)
        self.assertTrue(result["completed"])
        self.assertEqual([c["status"] for c in result["cases"]], ["Runtime Error", "Skipped"])
        self.assertEqual(result["cases"][0]["execution"]["api_code"], "561002")

    def test_unexpected_skip_cannot_pass(self):
        self.skipped(self.cases[1])
        result = self.assess()
        self.assertFalse(result["completed"])
        self.assertEqual(result["cases"][1]["status"], "Execution Error")

    def test_execution_after_runtime_failure_is_rejected(self):
        self.runtime(self.cases[0])
        self.batch(stopped=1, completed=1)
        result = self.assess(exit_code=1)
        self.assertEqual(result["cases"][1]["status"], "Execution Error")

    def test_signal_exit_rejects_completed_outputs(self):
        result = self.assess(exit_code=-11)
        self.assertFalse(result["completed"])
        self.assertFalse(any(c["passed"] for c in result["cases"]))

    def test_missing_completion_rejects_success(self):
        (self.output / "batch.txt").unlink()
        self.assertFalse(any(c["passed"] for c in self.assess()["cases"]))

    def test_invalid_completion_count_is_not_success(self):
        self.batch(completed="invalid")
        self.assertFalse(self.assess()["completed"])

    def test_wrong_completion_count(self):
        self.batch(completed=1)
        self.assertFalse(self.assess()["completed"])

    def test_performance_requires_final_output_and_mode(self):
        case = self.cases[0]
        folder = self.output / case.name
        self.success(case, mode="performance")
        for name in ("first_min_distance.bin", "first_nearest_segment_idx.bin"):
            (folder / name).unlink()
        process = {"exit_code": 0, "timed_out": False}
        result = assess_execution(self.fixtures / case.name, folder, process, self.library, 2, first_required=False)
        self.assertTrue(result["passed"])
        self.assertIsNone(result["first"])
        self.success(case, mode="correctness")
        result = assess_execution(self.fixtures / case.name, folder, process, self.library, 2, first_required=False)
        self.assertFalse(result["passed"])

    def test_cold_wrong_answer_blocks_correct_measurement(self):
        cold = {"status": "Wrong Answer", "passed": False, "first": {"passed": False}}
        hot = {"status": "Pass", "passed": True, "final": {"passed": True}}
        result = combine_phases(cold, hot)
        self.assertEqual(result["status"], "Wrong Answer")
        self.assertFalse(result["passed"])
        self.assertFalse(result["first"]["passed"])

    def test_cold_api_failure_is_preserved(self):
        cold = {"status": "Runtime Error", "passed": False, "execution": {"api_code": "561002"}}
        hot = {"status": "Pass", "passed": True}
        result = combine_phases(cold, hot)
        self.assertEqual(result["status"], "Runtime Error")
        self.assertEqual(result["execution"]["api_code"], "561002")

    def test_phase_stopping_must_agree(self):
        cold = {"status": "Skipped", "passed": None}
        hot = {"status": "Pass", "passed": True}
        self.assertEqual(combine_phases(cold, hot)["status"], "Execution Error")
        self.assertEqual(combine_phases(hot, cold)["status"], "Execution Error")


class BatchProfilerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile = self.root / "profile"
        self.profile.mkdir()
        self.build = self.root / "build"
        config = self.build / "tmp/vendors/custom/op_impl/ai_core/tbe/kernel/config/ascend910b"
        config.mkdir(parents=True)
        (config / "tree_line_min_distance.json").write_text(json.dumps({"binList": [
            {"inputs": [{"dtype": dtype}, {"dtype": dtype}],
             "binInfo": {"jsonFilePath": "kernels/" + stem + ".json"}}
            for dtype, stem in (("float32", "TreeLineMinDistance_single"), ("float16", "TreeLineMinDistance_half"))
        ]}))
        self.cases = [MOCK_CASES[0], MOCK_CASES[2], MOCK_CASES[1], MOCK_CASES[3]]
        self.outcomes = [{"name": c.name, "execution": {"status": "executed"}} for c in self.cases]

    def samples(self):
        for stem, count, base in (("TreeLineMinDistance_single_0", 15, 1), ("TreeLineMinDistance_half_1", 5, 10)):
            for launch in range(count):
                folder = self.profile / stem / str(launch)
                folder.mkdir(parents=True)
                (folder / "OpBasicInfo.csv").write_text(
                    "Op Name,Task Duration(us),Block Dim,Current Freq\n"
                    f"{stem},{base+launch/10},1,1650\n")

    def test_mixed_types_and_numeric_launch_order(self):
        self.samples()
        result = collect_batch_samples(self.profile, self.cases, self.outcomes, 5, self.build)
        self.assertEqual([s["launch_index"] for s in result["mock_04"]["samples"]], [str(x) for x in range(10, 15)])
        self.assertAlmostEqual(result["mock_01"]["statistics"]["mean_us"], 1.2)
        self.assertAlmostEqual(result["mock_03"]["statistics"]["mean_us"], 10.2)
        self.assertAlmostEqual(result["mock_02"]["statistics"]["mean_us"], 1.7)

    def test_missing_sample_fails(self):
        self.samples()
        (self.profile / "TreeLineMinDistance_single_0/14/OpBasicInfo.csv").unlink()
        with self.assertRaises(ValueError):
            collect_batch_samples(self.profile, self.cases, self.outcomes, 5, self.build)

    def test_unknown_kernel_fails(self):
        self.samples()
        p = self.profile / "TreeLineMinDistance_half_1/0/OpBasicInfo.csv"
        p.write_text(p.read_text().replace("TreeLineMinDistance_half_1", "TreeLineMinDistance_unknown_1"))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            collect_batch_samples(self.profile, self.cases, self.outcomes, 5, self.build)

    def test_outcome_order_cannot_change_attribution(self):
        self.samples()
        with self.assertRaisesRegex(ValueError, "batch order"):
            collect_batch_samples(self.profile, self.cases, list(reversed(self.outcomes)), 5, self.build)

    def test_unexecuted_batch_has_no_measurement(self):
        for outcome in self.outcomes:
            outcome["execution"]["status"] = "Skipped"
        self.assertEqual(collect_batch_samples(self.profile, self.cases, self.outcomes, 5, self.build), {})

    def test_unexecuted_batch_rejects_samples(self):
        self.samples()
        for outcome in self.outcomes:
            outcome["execution"]["status"] = "Skipped"
        with self.assertRaisesRegex(ValueError, "unexpected samples"):
            collect_batch_samples(self.profile, self.cases, self.outcomes, 5, self.build)

    def test_elapsed_time_uses_process_means(self):
        processes = [{"statistics": {"mean_us": mean, "median_us": median},
                      "samples": [{"duration_us": mean}]} for mean, median in ((4, 3), (6, 2), (100, 1))]
        result = aggregate_processes(processes)
        self.assertEqual(result["elapsed_us"], 6)
        self.assertEqual(result["process_means_us"], [4, 6, 100])


if __name__ == "__main__":
    unittest.main()
