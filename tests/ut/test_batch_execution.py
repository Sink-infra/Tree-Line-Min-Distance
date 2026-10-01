"""Batch stopping, cold output checks, and profiler attribution."""

import json
import argparse
import shlex
from unittest.mock import patch
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.common.cases import MOCK_CASES
from tests.common.device import assess_batch, assess_execution, prepare_fixtures, arguments, validate_arguments, run_batch
from tests.common.profiler import aggregate_processes, collect_batch_samples, collect_batch_trace, collect_process_samples, combine_phases, statistics_us


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
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

    def test_kernel_collection_uses_shared_batch_configuration(self):
        runner = self.root / "runner"
        runner.touch()
        sentinel = object()
        target = self.root / "measurement"
        with patch("tests.common.device.verify_build"), \
                patch("tests.common.device.launch", return_value={"exit_code": 0}) as launch, \
                patch("tests.common.device.assess_batch", return_value=sentinel):
            result = run_batch(runner, self.root / "package", self.root / "toolkit",
                               self.fixtures, self.cases[:1], target, 0, 5, 1, 60,
                               performance=True, vector_cores=40, cache_profile="Ascend910B4")
        self.assertIs(result, sentinel)
        command = launch.call_args.args[0]
        self.assertEqual(command[:2], ["msprof", "op"])
        for option in ("--replay-mode=kernel", "--warm-up=5"):
            self.assertIn(option, command)
        self.assertFalse(any(option.startswith("--aic-metrics=") for option in command))
        self.assertFalse(any(option.startswith("--launch-count=") for option in command))
        application = shlex.split(next(x.split("=", 1)[1] for x in command if x.startswith("--application=")))
        self.assertEqual(application[application.index("--warmup") + 1], "0")
        self.assertEqual(application[application.index("--repeat") + 1], "1")
        self.assertEqual(application[application.index("--cache-profile") + 1], "Ascend910B4")
        self.assertEqual((target / "case-order.txt").read_text().splitlines(), [self.cases[0].name])

    def cache_fields(self, profile="Ascend910B4", budget=100663296):
        for case in self.cases:
            path = self.output / case.name / "execution.txt"
            path.write_text(path.read_text() + f"cache_profile={profile}\nnative_soc=Ascend910B4-1\n"
                            f"native_l2_cache_bytes=176160768\nl2_cache_budget_bytes={budget}\n")

    def assess_cache(self, profile):
        return assess_batch(self.fixtures, self.output, self.cases,
                            {"exit_code": 0, "timed_out": False}, self.library, 2, cache_profile=profile)

    def test_target_cache_profile_is_part_of_execution_provenance(self):
        self.cache_fields()
        result = self.assess_cache("Ascend910B4")
        self.assertTrue(result["completed"])
        self.assertTrue(all(c["passed"] for c in result["cases"]))

    def test_native_cache_profile_preserves_capacity(self):
        self.cache_fields("native", 176160768)
        self.assertTrue(all(c["passed"] for c in self.assess_cache("native")["cases"]))

    def test_missing_cache_profile_cannot_pass_requested_configuration(self):
        self.assertTrue(all(c["status"] == "Execution Error" for c in self.assess_cache("Ascend910B4")["cases"]))

    def test_changed_cache_budget_cannot_pass(self):
        self.cache_fields(budget=100663295)
        self.assertTrue(all(c["status"] == "Execution Error" for c in self.assess_cache("Ascend910B4")["cases"]))

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

    def test_output_initialization_is_part_of_execution_provenance(self):
        for case in self.cases:
            path = self.output / case.name / "execution.txt"
            path.write_text(path.read_text() + "output_init=zero\n")
        result = assess_batch(self.fixtures, self.output, self.cases,
                              {"exit_code": 0, "timed_out": False}, self.library, 2, output_init="zero")
        self.assertTrue(all(row["passed"] for row in result["cases"]))
        result = assess_batch(self.fixtures, self.output, self.cases,
                              {"exit_code": 0, "timed_out": False}, self.library, 2, output_init="nan")
        self.assertEqual([row["status"] for row in result["cases"]], ["Execution Error", "Execution Error"])

    def test_vector_resource_limit_is_part_of_execution_provenance(self):
        for case in self.cases:
            path = self.output / case.name / "execution.txt"
            path.write_text(path.read_text() + "vector_cores=40\n")
        process = {"exit_code": 0, "timed_out": False}
        valid = assess_batch(self.fixtures, self.output, self.cases, process, self.library, 2, vector_cores=40)
        self.assertTrue(all(row["passed"] for row in valid["cases"]))
        wrong = assess_batch(self.fixtures, self.output, self.cases, process, self.library, 2, vector_cores=32)
        self.assertEqual([row["status"] for row in wrong["cases"]], ["Execution Error", "Execution Error"])

    def test_missing_vector_resource_record_cannot_pass(self):
        process = {"exit_code": 0, "timed_out": False}
        result = assess_batch(self.fixtures, self.output, self.cases, process, self.library, 2, vector_cores=40)
        self.assertFalse(any(row["passed"] for row in result["cases"]))

    def test_profiled_outputs_define_measurement_accuracy(self):
        cold = {"status": "Wrong Answer", "passed": False, "first": {"passed": False}}
        hot = {"status": "Pass", "passed": True, "final": {"passed": True, "error_percent": 0.0}}
        result = combine_phases(cold, hot)
        self.assertEqual(result["status"], "Pass")
        self.assertTrue(result["passed"])
        self.assertFalse(result["first"]["passed"])
        self.assertEqual(result["error_percent"], 0.0)

    def test_profiled_numerical_failure_sets_reported_error_percent(self):
        for cold_bad, hot_bad, expected in ((True, False, 0.0), (True, True, 20.0), (False, True, 20.0)):
            with self.subTest(cold=cold_bad, hot=hot_bad):
                cold = {"status": "Wrong Answer" if cold_bad else "Pass", "passed": not cold_bad, "first": {"passed": not cold_bad, "error_percent": 10.0 if cold_bad else 0.0}}
                hot = {"status": "Wrong Answer" if hot_bad else "Pass", "passed": not hot_bad, "final": {"passed": not hot_bad, "error_percent": 20.0 if hot_bad else 0.0}}
                result = combine_phases(cold, hot)
                self.assertEqual(result["status"], "Wrong Answer" if hot_bad else "Pass")
                self.assertEqual(result["error_percent"], expected)
                self.assertEqual(result["first"], cold["first"])
                self.assertEqual(result["final"], hot["final"])

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
        self.root = Path(self.temp.name).resolve()
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
                    "Op Name,Task Duration(us),Block Dim,Current Freq,Rated Freq\n"
                    f"{stem},{base+launch/10},1,1650,1650\n")

                (folder / "dump").mkdir()
                (folder / "dump/DeviceProf1.bin").write_bytes(b"kernel profiling sample")

    def test_mixed_types_and_numeric_launch_order(self):
        self.samples()
        result = collect_batch_samples(self.profile, self.cases, self.outcomes, 5, self.build)
        self.assertEqual([s["launch_index"] for s in result["mock_04"]["samples"]], [str(x) for x in range(10, 15)])
        self.assertAlmostEqual(result["mock_01"]["statistics"]["mean_us"], 1.2)
        self.assertAlmostEqual(result["mock_03"]["statistics"]["mean_us"], 10.2)
        self.assertAlmostEqual(result["mock_02"]["statistics"]["mean_us"], 1.7)

    def test_independent_processes_use_pid_for_case_attribution(self):
        cases = [self.cases[0], self.cases[2]]
        outcomes = [{"name": c.name, "execution": {"status": "executed"}} for c in cases]
        children = [{"name": c.name, "pid": pid} for c, pid in zip(cases, (123, 456))]
        binary = self.build / "tmp/vendors/custom/op_impl/ai_core/tbe/kernel/kernels/TreeLineMinDistance_single.o"
        binary.parent.mkdir()
        binary.write_bytes(b"selected kernel image")
        for pid, duration in ((456, 7), (123, 3)):
            folder = self.profile / str(pid) / "0"
            folder.mkdir(parents=True)
            (folder / "OpBasicInfo.csv").write_text(
                "Op Name,Task Duration(us),Pid,Current Freq,Rated Freq\n"
                f"TreeLineMinDistance_single,{duration},{pid},1650,1650\n")
            (folder / "dump").mkdir()
            (folder / "dump/DeviceProf1.bin").write_bytes(b"kernel profiling sample")
            (folder / "dump/aicore_binary.o").write_bytes(binary.read_bytes())
        result = collect_process_samples(self.profile, cases, outcomes, children, self.build)
        self.assertEqual(result[cases[0].name]["statistics"]["median_us"], 3)
        self.assertEqual(result[cases[1].name]["statistics"]["median_us"], 7)
        actual = self.profile / "123/0/dump/aicore_binary.o"
        actual.write_bytes(b"kernel from a different build")
        with self.assertRaisesRegex(ValueError, "kernel differs"):
            collect_process_samples(self.profile, cases, outcomes, children, self.build)
        actual.write_bytes(binary.read_bytes())
        children[1]["pid"] = 789
        with self.assertRaisesRegex(ValueError, "PID"):
            collect_process_samples(self.profile, cases, outcomes, children, self.build)

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

    def test_elapsed_mean_keeps_every_task_sample(self):
        values = [1] + [2] * 8 + [100]
        processes = [{"statistics": statistics_us(values),
                      "samples": [{"duration_us": x} for x in values]} for _ in range(3)]
        result = aggregate_processes(processes)
        self.assertEqual(result["elapsed_us"], 11.7)
        self.assertEqual(result["all_samples"]["count"], 30)
        self.assertEqual(result["trimmed_samples"]["count"], 18)
        self.assertEqual(result["process_means_us"], [11.7] * 3)
        self.assertEqual([s["duration_us"] for s in processes[0]["samples"]], values)


    def trace_files(self, names=None):
        names = names or ["TreeLineMinDistance_single_0", "TreeLineMinDistance_half_1",
                          "TreeLineMinDistance_single_768", "TreeLineMinDistance_single_0"]
        trace = self.root / "trace"
        trace.mkdir()
        with (trace / "task_time.csv").open("w", newline="") as stream:
            rows = csv.writer(stream)
            rows.writerow(["Device_id", "kernel_name", "stream_id", "task_id", "task_start(us)"])
            for i, name in reversed(list(enumerate(names))):
                rows.writerow(["0", name, "45", str(i), "1791681774000000." + str(i).zfill(3)])
        with (trace / "op_summary.csv").open("w", newline="") as stream:
            rows = csv.writer(stream)
            rows.writerow(["Device_id", "Op Name", "OP Type", "Stream ID", "Task ID",
                           "Input Shapes", "Input Data Types", "Input Formats"])
            for i, case in enumerate(self.cases):
                shapes = ";".join(",".join(map(str, s)) for s in
                                  (case.tree_points_shape, case.line_segments_shape))
                dtype = "FLOAT" if case.dtype == "float32" else "FLOAT16"
                rows.writerow(["0", "TreeLineMinDistance", "TreeLineMinDistance", "45", str(i),
                               '"' + shapes + '"', dtype + ";" + dtype, "ND;ND"])
        return trace

    def test_trace_orders_variants_and_checks_input_contracts(self):
        trace = collect_batch_trace(self.trace_files(), self.cases, self.outcomes, self.build)
        self.assertEqual(trace["cases"]["mock_02"]["kernel_name"], "TreeLineMinDistance_single_768")
        self.assertEqual(trace["cases"]["mock_04"]["identity"], ["0", "45", "3"])
        self.assertEqual(len(trace["csv_files"]), 2)

    def test_timestamps_distinguish_reused_stream_and_task_ids(self):
        folder = self.trace_files()
        for filename, id_key in (("task_time.csv", "task_id"), ("op_summary.csv", "Task ID")):
            path = folder / filename
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields, rows = list(reader.fieldnames), list(reader)
            if filename == "op_summary.csv":
                fields.append("Task Start Time(us)")
            for row in rows:
                original = row[id_key]
                row[id_key] = "0"
                if filename == "op_summary.csv":
                    row["Task Start Time(us)"] = "1791681774000000." + original.zfill(3)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        trace = collect_batch_trace(folder, self.cases, self.outcomes, self.build)
        self.assertEqual(len(trace["cases"]), len(self.cases))
        self.assertEqual(trace["cases"]["mock_04"]["identity"], ["0", "45", "0"])

    def test_operator_timestamp_must_match_the_task(self):
        folder = self.trace_files()
        path = folder / "op_summary.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = list(reader.fieldnames), list(reader)
        fields.append("Task Start Time(us)")
        for row in rows:
            row["Task Start Time(us)"] = "1791681774000001"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "corresponding operator trace"):
            collect_batch_trace(folder, self.cases, self.outcomes, self.build)

    def test_multiple_variants_use_cold_trace_and_per_variant_launch_indices(self):
        trace = collect_batch_trace(self.trace_files(), self.cases, self.outcomes, self.build)
        for stem, count, base in (("TreeLineMinDistance_single_0", 10, 1),
                                  ("TreeLineMinDistance_single_768", 5, 20),
                                  ("TreeLineMinDistance_half_1", 5, 10)):
            for launch in range(count):
                folder = self.profile / stem / str(launch)
                folder.mkdir(parents=True)
                (folder / "OpBasicInfo.csv").write_text(
                    "Op Name,Task Duration(us),Current Freq,Rated Freq\n"
                    + f"{stem},{base + launch / 10},1650,1650\n")
                (folder / "dump").mkdir()
                (folder / "dump/DeviceProf1.bin").write_bytes(b"kernel profiling sample")
        result = collect_batch_samples(self.profile, self.cases, self.outcomes, 5, self.build, trace=trace)
        self.assertAlmostEqual(result["mock_02"]["statistics"]["mean_us"], 20.2)
        self.assertAlmostEqual(result["mock_04"]["statistics"]["mean_us"], 1.7)
        with self.assertRaisesRegex(ValueError, "multiple kernel variants"):
            collect_batch_samples(self.profile, self.cases, self.outcomes, 5, self.build)

    def test_trace_rejects_wrong_shape_or_type(self):
        folder = self.trace_files()
        path = folder / "op_summary.csv"
        path.write_text(path.read_text().replace("FLOAT16;FLOAT16", "FLOAT;FLOAT"))
        with self.assertRaisesRegex(ValueError, "input contract"):
            collect_batch_trace(folder, self.cases, self.outcomes, self.build)

    def test_trace_rejects_duplicate_or_missing_tasks(self):
        folder = self.trace_files()
        path = folder / "task_time.csv"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines + [lines[1]]) + "\n")
        with self.assertRaisesRegex(ValueError, "count"):
            collect_batch_trace(folder, self.cases, self.outcomes, self.build)
        path.write_text("\n".join(lines[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "count"):
            collect_batch_trace(folder, self.cases, self.outcomes, self.build)

    def test_trace_rejects_wrong_variant_type(self):
        names = ["TreeLineMinDistance_half_1", "TreeLineMinDistance_half_1",
                 "TreeLineMinDistance_single_768", "TreeLineMinDistance_single_0"]
        with self.assertRaisesRegex(ValueError, "input contract"):
            collect_batch_trace(self.trace_files(names), self.cases, self.outcomes, self.build)

    def test_trace_rejects_nonfinite_or_ambiguous_order(self):
        folder = self.trace_files()
        path = folder / "task_time.csv"
        text = path.read_text()
        path.write_text(text.replace("1791681774000000.003", "NaN"))
        with self.assertRaisesRegex(ValueError, "timestamp"):
            collect_batch_trace(folder, self.cases, self.outcomes, self.build)
        path.write_text(text.replace("1791681774000000.003", "1791681774000000.002"))
        with self.assertRaisesRegex(ValueError, "ordering"):
            collect_batch_trace(folder, self.cases, self.outcomes, self.build)




class ResourceArgumentTests(unittest.TestCase):
    def parse(self, suite, *extra):
        parser = argparse.ArgumentParser()
        arguments(parser, suite)
        return parser.parse_args(["--build-dir", "/tmp/package", "--output-dir", "/tmp/results", *extra])

    def test_mock_resource_defaults_and_native_override(self):
        self.assertEqual(self.parse("mock").vector_cores, 40)
        self.assertEqual(self.parse("mock").cache_profile, "Ascend910B4")
        self.assertEqual(self.parse("mock", "--cache-profile", "native").cache_profile, "native")
        self.assertEqual((self.parse("mock").warmup, self.parse("mock").repeat), (5, 10))
        self.assertEqual(self.parse("full").repeat, 5)
        self.assertIsNone(self.parse("mock").runner_dir)
        self.assertEqual(self.parse("mock", "--runner-dir", "/tmp/runner-bin").runner_dir, Path("/tmp/runner-bin"))
        self.assertEqual(self.parse("full").vector_cores, 0)
        self.assertEqual(self.parse("mock", "--vector-cores", "0").vector_cores, 0)

    def test_device_lock_is_shared_across_temporary_directories(self):
        import os
        import select
        import subprocess
        import sys
        from tests.common.device import device_lock

        lock_id = 1000000 + os.getpid()
        script = ("from tests.common.device import device_lock; import sys; "
                  "print('ready', flush=True); "
                  f"scope=device_lock({lock_id}); scope.__enter__(); "
                  "print('acquired', flush=True); scope.__exit__(None,None,None)")
        child = None
        with tempfile.TemporaryDirectory() as temporary:
            try:
                with device_lock(lock_id):
                    environment = dict(os.environ, TMPDIR=temporary,
                                       PYTHONPATH=str(Path(__file__).resolve().parents[2]))
                    child = subprocess.Popen([sys.executable, "-c", script],
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                             text=True, env=environment)
                    self.assertEqual(child.stdout.readline().strip(), "ready")
                    self.assertEqual(select.select([child.stdout], [], [], 0.2)[0], [])
                    self.assertIsNone(child.poll())
                output, errors = child.communicate(timeout=10)
                self.assertEqual((child.returncode, output.strip()), (0, "acquired"), errors)
            finally:
                if child is not None and child.poll() is None:
                    child.kill()
                    child.communicate()

    def test_invalid_vector_core_bounds(self):
        with patch("tests.common.device.device_guard"), patch("tests.common.device.verify_build"):
            for value in ("-1", "41"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    validate_arguments(self.parse("mock", "--vector-cores", value))


if __name__ == "__main__":
    unittest.main()
