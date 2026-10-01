"""Assertions for current-build execution and profiler data integrity."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.common.cases import MOCK_CASES
from tests.common.device import (
    assess_execution, execution_marker, fixture_hashes, output_directory, prepare_fixtures, require_fixture,
)
from tests.common.profiler import aggregate_processes, collect_samples, statistics_us
from tests.common.tensor_io import sha256
from tests.st.scripts.build import ROOT, RECEIPT, HOST_PROFILE, SDK_PROFILE, DEVICE_PROFILE, save, source_hashes, verify_build, prepare_build_directory


class ProfilerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def csv(self, rows, name="OpBasicInfo.csv", header="OpName,TaskDuration(us),LaunchIndex"):
        path = self.root / name
        path.write_text(header + "\n" + "\n".join(rows) + "\n")
        return path

    def test_profiling_rejects_other_warmup_counts_before_build_access(self):
        import contextlib
        import io
        from unittest import mock
        from tests.st.scripts import profile_mock_workloads as profile

        diagnostic = io.StringIO()
        with mock.patch.object(profile, "validate_arguments") as validate:
            with contextlib.redirect_stderr(diagnostic), self.assertRaises(SystemExit) as rejected:
                profile.main(["--build-dir", str(self.root / "build"),
                              "--output-dir", str(self.root / "profile"), "--warmup", "20"])
        self.assertEqual(rejected.exception.code, 2)
        validate.assert_not_called()
        self.assertIn("exactly five", diagnostic.getvalue())

    def test_kernel_filter_and_statistics(self):
        self.csv(["OtherOp,99,0", "TreeLineMinDistance_abc,3,0", "TreeLineMinDistance_abc,5,1"])
        result = collect_samples(self.root, 2)
        self.assertEqual([s["duration_us"] for s in result["samples"]], [3, 5])
        self.assertEqual(result["statistics"]["median_us"], 4)
        self.assertEqual(result["csv_files"][0]["sha256"], sha256(self.root / "OpBasicInfo.csv"))

    def test_operator_profiler_schema(self):
        for launch, duration in enumerate((3.1, 4.2)):
            folder = self.root / "OPPROF_example/TreeLineMinDistance_abc" / str(launch)
            folder.mkdir(parents=True)
            (folder / "OpBasicInfo.csv").write_text(
                "Op Name,Task Duration(us),Block Dim,Current Freq,Op Type,Device Id,\n"
                f"TreeLineMinDistance_abc,{duration},40,1650,vector,0,\n")
        result = collect_samples(self.root, 2)
        self.assertEqual([s["launch_index"] for s in result["samples"]], ["0", "1"])
        self.assertEqual(result["samples"][0]["current_frequency_mhz"], "1650")

    def test_device_profile_requires_raw_samples_and_fixed_frequencies(self):
        folder = self.root / "TreeLineMinDistance_abc/0"
        folder.mkdir(parents=True)
        path = folder / "OpBasicInfo.csv"
        path.write_text("Op Name,Task Duration(us),Current Freq,Rated Freq\n"
                        "TreeLineMinDistance_abc,3.25,1650,1650\n")
        with self.assertRaisesRegex(ValueError, "DeviceProf"):
            collect_samples(self.root, 1, require_device_profile=True)
        (folder / "dump").mkdir()
        raw = folder / "dump/DeviceProf1.bin"
        raw.write_bytes(b"kernel sample")
        result = collect_samples(self.root, 1, require_device_profile=True)
        sample = result["samples"][0]
        self.assertEqual(sample["launch_index"], "0")
        self.assertEqual(sample["measurement_source"], "DeviceProf")
        self.assertEqual(sample["metric"], "Task Duration(us)")
        self.assertEqual(sample["device_profile_files"][0]["sha256"], sha256(raw))
        for current, rated in ((1500, 1650), (1650, 1500)):
            path.write_text("Op Name,Task Duration(us),Current Freq,Rated Freq\n"
                            f"TreeLineMinDistance_abc,3.25,{current},{rated}\n")
            with self.assertRaisesRegex(ValueError, "Current/Rated"):
                collect_samples(self.root, 1, require_device_profile=True)
        path.write_text("Op Name,Task Duration(us),Current Freq,Rated Freq\n"
                        "TreeLineMinDistance_abc,3.25,1650,1650\n")
        raw.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "empty DeviceProf"):
            collect_samples(self.root, 1, require_device_profile=True)

    def test_missing_csv(self):
        with self.assertRaises(ValueError):
            collect_samples(self.root, 1)

    def test_missing_columns(self):
        self.csv(["TreeLineMinDistance,3"], header="OpName,Other")
        with self.assertRaises(ValueError):
            collect_samples(self.root, 1)

    def test_incomplete_samples(self):
        self.csv(["TreeLineMinDistance,3,0"])
        with self.assertRaisesRegex(ValueError, "sample count"):
            collect_samples(self.root, 2)

    def test_extra_samples(self):
        self.csv(["TreeLineMinDistance,3,0", "TreeLineMinDistance,4,1"])
        with self.assertRaisesRegex(ValueError, "sample count"):
            collect_samples(self.root, 1)

    def test_duplicate_launch_across_csv(self):
        self.csv(["TreeLineMinDistance,3,0"])
        self.csv(["TreeLineMinDistance,3,0"], name="OpBasicInfo_duplicate.csv")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            collect_samples(self.root, 2)

    def test_invalid_duration(self):
        for value in ("nan", "inf", "-1", "0", ""):
            with self.subTest(value=value):
                self.csv([f"TreeLineMinDistance,{value},0"])
                with self.assertRaises(ValueError):
                    collect_samples(self.root, 1)

    def test_no_target_kernel(self):
        self.csv(["OtherOp,3,0"])
        with self.assertRaises(ValueError):
            collect_samples(self.root, 1)

    def test_pooled_summary_retains_process_statistics(self):
        processes = [{"statistics": statistics_us(values),
                      "samples": [{"duration_us": x} for x in values]} for values in ([1, 2, 30], [4, 6, 8])]
        result = aggregate_processes(processes)
        self.assertEqual(result["elapsed_us"], 8.5)
        self.assertEqual(result["median_of_process_medians_us"], 4)
        self.assertEqual(result["process_medians_us"], [2, 6])
        self.assertEqual(result["all_samples"]["count"], 6)
        self.assertEqual(result["trimmed_samples"]["count"], 4)
        self.assertEqual(result["trim_fraction_per_tail"], 0.2)


    def test_invalid_sample_cannot_be_trimmed(self):
        processes = [{"statistics": {"mean_us": 1, "median_us": 1},
                      "samples": [{"duration_us": x} for x in [0, 1, 1, 1, 1, 2]]}]
        with self.assertRaisesRegex(ValueError, "positive"):
            aggregate_processes(processes)


class ExecutionTests(unittest.TestCase):
    def test_context_does_not_read_or_compare_outputs(self):
        self.marker(mode="performance", repeat="35")
        from unittest.mock import patch
        with patch('tests.common.device.verify_case_output', side_effect=AssertionError('context assessed')):
            result = assess_execution(self.fixture, self.output, {'exit_code': 0, 'timed_out': False},
                                      self.library, 35, first_required=False, compare_outputs=False)
        self.assertEqual(result['status'], 'Unmeasured Context')
        self.assertIsNone(result['passed'])
        self.assertIsNone(result['error_percent'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.fixture = prepare_fixtures(self.root, [MOCK_CASES[0]], "mock") / MOCK_CASES[0].name
        self.output = self.root / "actual"
        self.output.mkdir()
        self.library = self.root / "libcust_opapi.so"
        self.marker()
        self.good_outputs()

    def marker(self, **changes):
        fields = {"status": "executed", "api_code": "0", "guards_ok": "1", "inputs_unchanged": "1",
                  "repeat": "2", "library": str(self.library), "alignment_bytes": "512"}
        fields.update(changes)
        (self.output / "execution.txt").write_text("".join(f"{k}={v}\n" for k, v in fields.items()))

    def good_outputs(self):
        distance = np.fromfile(self.fixture / "reference_min_distance_fp64.bin", dtype="<f8").astype("<f4")
        indices = np.fromfile(self.fixture / "reference_nearest_segment_idx_int32.bin", dtype="<i4")
        for prefix in ("first_", ""):
            distance.tofile(self.output / (prefix + "min_distance.bin"))
            indices.tofile(self.output / (prefix + "nearest_segment_idx.bin"))

    def assess(self, exit_code=0, timed_out=False):
        return assess_execution(self.fixture, self.output, {"exit_code": exit_code, "timed_out": timed_out},
                                self.library, 2)

    def test_pass_requires_both_outputs(self):
        self.assertTrue(self.assess()["passed"])
        self.assertEqual(self.assess()["first"]["error_percent"], 0)

    def test_first_output_wrong_last_correct(self):
        np.array([-1], dtype="<i4").tofile(self.output / "first_nearest_segment_idx.bin")
        result = self.assess()
        self.assertEqual(result["status"], "Wrong Answer")
        self.assertEqual(result["first"]["error_percent"], 100)
        self.assertTrue(result["final"]["passed"])

    def test_last_output_wrong_first_correct(self):
        np.array([np.nan], dtype="<f4").tofile(self.output / "min_distance.bin")
        result = self.assess()
        self.assertEqual(result["status"], "Wrong Answer")
        self.assertIsNone(result["final"]["max_abs_error"])

    def test_missing_final_output(self):
        (self.output / "min_distance.bin").unlink()
        self.assertEqual(self.assess()["status"], "Data Error")
        self.assertFalse(self.assess()["passed"])

    def test_truncated_output(self):
        (self.output / "nearest_segment_idx.bin").write_bytes(b"")
        self.assertEqual(self.assess()["status"], "Data Error")

    def test_nonzero_exit_with_good_outputs(self):
        self.assertEqual(self.assess(exit_code=-11)["status"], "Runtime Error")
        self.assertFalse(self.assess(exit_code=-11)["passed"])

    def test_api_failure(self):
        self.marker(status="Runtime Error", api_code="507011")
        result = self.assess(exit_code=1)
        self.assertEqual(result["execution"]["api_code"], "507011")
        self.assertFalse(result["passed"])

    def test_timeout(self):
        self.assertEqual(self.assess(exit_code=-9, timed_out=True)["status"], "Timeout")

    def test_guard_and_input_changes(self):
        for field in ("guards_ok", "inputs_unchanged"):
            with self.subTest(field=field):
                self.marker(**{field: "0"})
                self.assertEqual(self.assess()["status"], "Memory Error")

    def test_library_and_repeat_provenance(self):
        for changes in ({"library": str(self.root / "different.so")}, {"repeat": "3"}):
            self.marker(**changes)
            self.assertEqual(self.assess()["status"], "Execution Error")

    def test_missing_marker(self):
        (self.output / "execution.txt").unlink()
        self.assertFalse(self.assess()["passed"])

    def test_duplicate_marker_field(self):
        (self.output / "execution.txt").write_text("status=executed\nstatus=executed\n")
        with self.assertRaises(ValueError):
            execution_marker(self.output)
        self.assertFalse(self.assess()["passed"])

    def test_fixture_snapshot_detects_metadata_change(self):
        before = fixture_hashes(self.fixture)
        require_fixture(self.fixture, before)
        (self.fixture / "case.conf").write_text("b=2\n")
        with self.assertRaisesRegex(ValueError, "fixture"):
            require_fixture(self.fixture, before)

    def test_runtime_artifacts_exclude_source_tree(self):
        with self.assertRaises(ValueError):
            output_directory(ROOT / "tests/current-run")
        target = self.root / "new-output"
        self.assertEqual(output_directory(target), target)
        with self.assertRaises(FileExistsError):
            output_directory(target)


class BuildReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source, self.build, self.toolkit = [self.root / x for x in ("source", "build", "toolkit")]
        self.source.mkdir()
        self.build.mkdir()
        self.toolkit.mkdir()
        (self.source / "CMakeLists.txt").write_text("project(operator)\n")
        (self.source / "kernel.cpp").write_text("void kernel() {}\n")
        (self.build / "libcust_opapi.so").write_bytes(b"binary")
        (self.build / "autogen").mkdir()
        (self.build / "autogen/aclnn_tree_line_min_distance.h").write_bytes(b"header")
        module = self.build / "tmp/vendors/custom/op_impl/ai_core/tbe/custom_impl/dynamic/tree_line_min_distance.py"
        module.parent.mkdir(parents=True)
        module.write_text("custom_compile_options = {},\n")
        self.module = module
        self.receipt = {"toolchain": {"host": HOST_PROFILE, "sdk": SDK_PROFILE, "device": DEVICE_PROFILE}, "kernel_compile_options": {}, "source_dir": str(self.source), "build_dir": str(self.build),
                        "toolkit": str(self.toolkit), "source_sha256": source_hashes(self.source),
                        "binary_sha256": {"libcust_opapi.so": sha256(self.build / "libcust_opapi.so"),
                                          str(module.relative_to(self.build)): sha256(module)},
                        "header_sha256": sha256(self.build / "autogen/aclnn_tree_line_min_distance.h")}
        save(self.build / RECEIPT, self.receipt)

    def test_matching_package(self):
        self.assertEqual(verify_build(self.build, self.toolkit), self.receipt)

    def test_source_change(self):
        (self.source / "kernel.cpp").write_text("void changed() {}\n")
        with self.assertRaisesRegex(ValueError, "sources"):
            verify_build(self.build, self.toolkit)

    def test_binary_change(self):
        (self.build / "libcust_opapi.so").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "package"):
            verify_build(self.build, self.toolkit)

    def test_package_file_set_change(self):
        package = self.build / "tmp/vendors/custom/kernel"
        package.mkdir(parents=True)
        (package / "kernel.o").write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "package file set"):
            verify_build(self.build, self.toolkit)

    def test_header_change(self):
        (self.build / "autogen/aclnn_tree_line_min_distance.h").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "header"):
            verify_build(self.build, self.toolkit)

    def test_host_standard_and_abi_are_part_of_build_provenance(self):
        self.receipt["toolchain"]["host"] = dict(HOST_PROFILE, standard="c++17")
        save(self.build / RECEIPT, self.receipt)
        with self.assertRaisesRegex(ValueError, "toolchain"):
            verify_build(self.build, self.toolkit)

    def test_device_configuration_is_part_of_build_provenance(self):
        self.receipt["toolchain"]["device"] = dict(DEVICE_PROFILE, optimization="O3")
        save(self.build / RECEIPT, self.receipt)
        with self.assertRaisesRegex(ValueError, "toolchain"):
            verify_build(self.build, self.toolkit)

    def test_package_compiler_options_match_the_configuration(self):
        self.module.write_text("custom_compile_options = {'__ALLSOC__': ['', '-O3']},\n")
        self.receipt["binary_sha256"][str(self.module.relative_to(self.build))] = sha256(self.module)
        save(self.build / RECEIPT, self.receipt)
        with self.assertRaisesRegex(ValueError, "device compiler options"):
            verify_build(self.build, self.toolkit)

    def test_wrong_toolkit(self):
        with self.assertRaisesRegex(ValueError, "toolkit"):
            verify_build(self.build, self.root / "another-toolkit")

    def test_unsafe_binary_path(self):
        self.receipt["binary_sha256"] = {"../outside.so": "bad"}
        save(self.build / RECEIPT, self.receipt)
        with self.assertRaises(ValueError):
            verify_build(self.build, self.toolkit)




class BuildDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "kernel.cpp").write_text("void kernel() {}")
        self.build = self.root / "build"
        self.build.mkdir()

    def test_generated_state_is_independent_of_previous_options(self):
        (self.build / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=" + str(self.source) + "\n")
        stale = self.build / "autogen"
        stale.mkdir()
        (stale / "options.ini").write_text("O3")
        prepare_build_directory(self.source, self.build)
        self.assertEqual(list(self.build.iterdir()), [])
        self.assertTrue((self.source / "kernel.cpp").is_file())

    def test_unrelated_directory_is_preserved(self):
        marker = self.build / "notes.txt"
        marker.write_text("unrelated")
        with self.assertRaisesRegex(ValueError, "does not belong"):
            prepare_build_directory(self.source, self.build)
        self.assertEqual(marker.read_text(), "unrelated")

    def test_source_directory_is_preserved(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            prepare_build_directory(self.source, self.source)
        self.assertTrue((self.source / "kernel.cpp").is_file())


if __name__ == "__main__":
    unittest.main()
