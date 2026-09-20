"""Controller tests use explicit fixtures and never fit or simulate a model."""
import contextlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from qualified_inference import release
from qualified_inference import reporting

ROOT = release.ROOT

class ReleaseTests(unittest.TestCase):
    def test_package_lock(self):
        release.verify_package()

    def test_inventory_covers_all_nine_routes(self):
        routes = release.load(ROOT / "WORKFLOW.json")["routes"]
        self.assertEqual(len(routes), 9)
        self.assertEqual(len({r["name"] for r in routes}), 9)
        declared = [p for r in routes for p in r["outputs"]]
        self.assertEqual(len(declared), len(set(declared)))
        self.assertEqual(len(declared), 48)
        self.assertEqual(sum(r["name"] == "finite_window_weight_learning" for r in routes), 1)
        for route in routes:
            if route["name"] == "finite_window_weight_learning":
                self.assertIn("evaluate", route["args"])
                self.assertNotIn("all", route["args"])

    def test_existing_reference_contracts(self):
        for route in release.load(ROOT / "WORKFLOW.json")["routes"]:
            for name, ref in route["references"].items():
                release.check_frame(ROOT / ref["path"], route["csv_contracts"][name])

    def test_environment_mismatch_blocks_execution(self):
        with self.assertRaisesRegex(RuntimeError, "Environment differs"):
            release.require_environment({"python": "a"}, {"python": "b"})

    def test_missing_dependency_rejected(self):
        with patch("importlib.metadata.version", return_value="0.0"):
            with self.assertRaisesRegex(RuntimeError, "Dependency mismatch"):
                release.environment()

    def test_reproduction_pins_cover_required_dependencies(self):
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        pins = {}
        for line in (ROOT / "requirements-reproduction.txt").read_text().splitlines():
            name, version = line.split("==")
            pins[canonicalize_name(name)] = version
        for name, version in pins.items():
            self.assertEqual(importlib.metadata.version(name), version)
            for declaration in importlib.metadata.requires(name) or []:
                requirement = Requirement(declaration)
                if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                    continue
                dependency = canonicalize_name(requirement.name)
                self.assertIn(dependency, pins, f"{name} requires an unpinned dependency")
                self.assertTrue(requirement.specifier.contains(pins[dependency], prereleases=True),
                                f"{name}: incompatible dependency {requirement}")

    def test_output_cannot_overlap_package(self):
        for destination in (ROOT, ROOT / "output", ROOT.parent):
            with self.assertRaises(ValueError):
                release.output_destination(destination)

    def test_archive_mismatch_permits_reproduction(self):
        self.assertTrue(release.reproduction_permitted(
            {"execution_integrity": "pass", "archive_compatibility": "mismatch"}))
        self.assertFalse(release.reproduction_permitted(
            {"execution_integrity": "failed", "archive_compatibility": "pass"}))

    def test_acceptance_requires_both_reproduction_and_archives(self):
        good = {"same_environment_reproduction": "pass",
                "archive_compatibility": {"primary": "pass", "clean_clone": "pass"}}
        self.assertTrue(release.release_acceptable(good))
        for label in ("primary", "clean_clone"):
            altered = json.loads(json.dumps(good))
            altered["archive_compatibility"][label] = "mismatch"
            self.assertFalse(release.release_acceptable(altered))
        self.assertFalse(release.release_acceptable({**good, "same_environment_reproduction": "mismatch"}))
        self.assertFalse(release.release_acceptable({"same_environment_reproduction": "pass"}))

    def fixture_run(self, destination, *, resume=False, fail_execution=False, fail_integrity=False,
                    archive_error=False):
        calls = []
        routes = [{"name": "first", "outputs": ["first.csv"]},
                  {"name": "second", "outputs": ["second.csv"]}]
        def commands(output, root):
            return [(r, [r["name"]]) for r in routes]
        def launch(command, **kwargs):
            calls.append(command[0])
            if fail_execution:
                return SimpleNamespace(returncode=1)
            (destination / (command[0] + ".csv")).write_text("id,value\n1,2\n")
            return SimpleNamespace(returncode=0)
        def integrity(output, route, root):
            if fail_integrity:
                raise RuntimeError("fixture integrity failure")
            return {n: release.sha(output / n) for n in route["outputs"]}
        def archive(*args):
            if archive_error:
                raise RuntimeError("fixture archive comparison error")
            return {"status": "mismatch", "comparisons": []}
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(release, "verify_package", return_value="fixed"))
            stack.enter_context(patch.object(release, "environment", return_value={"fixture": True}))
            stack.enter_context(patch.object(release, "commands", commands))
            stack.enter_context(patch.object(release, "stage_inputs"))
            result = release.execute(destination, {"fixture": True}, resume=resume, launcher=launch,
                                     integrity=integrity,
                                     archive=archive)
        return result, calls

    def test_archived_failure_does_not_cancel_later_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.fixture_run(Path(tmp) / "run")
            self.assertEqual(calls, ["first", "second"])
            self.assertEqual(result["execution_integrity"], "pass")
            self.assertEqual(result["archive_compatibility"], "mismatch")

    def test_archive_check_error_does_not_discard_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.fixture_run(Path(tmp) / "run", archive_error=True)
            self.assertEqual(calls, ["first", "second"])
            self.assertEqual(result["execution_integrity"], "pass")
            self.assertEqual(result["archive_compatibility"], "mismatch")
            self.assertTrue(all(s["archive_compatibility"]["status"] == "error" for s in result["stages"]))

    def test_completed_stages_are_reused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            self.fixture_run(output)
            with self.assertRaises(FileExistsError):
                self.fixture_run(output)
            result, calls = self.fixture_run(output, resume=True)
            self.assertEqual(calls, [])
            self.assertEqual(result["execution_integrity"], "pass")

    def test_changed_completed_output_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            self.fixture_run(output)
            (output / "first.csv").write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "Completed outputs changed"):
                self.fixture_run(output, resume=True)

    def test_execution_failure_preserves_partial_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            with self.assertRaisesRegex(RuntimeError, "Execution failed"):
                self.fixture_run(output, fail_execution=True)
            self.assertTrue((output / "control/first.started.json").exists())
            self.assertFalse((output / "control/second.started.json").exists())
            with self.assertRaisesRegex(RuntimeError, "Incomplete stage preserved"):
                self.fixture_run(output, resume=True)

    def test_structural_failure_stops_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            with self.assertRaisesRegex(RuntimeError, "fixture integrity"):
                self.fixture_run(output, fail_integrity=True)
            self.assertFalse((output / "control/second.started.json").exists())

    def test_missing_or_duplicate_output_keys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            path.write_text("rep,value\n1,2\n1,3\n")
            with self.assertRaisesRegex(RuntimeError, "duplicate keys"):
                release.check_frame(path, {"rows": 2, "columns": ["rep", "value"], "keys": ["rep"]})

    def target_fixture(self):
        keys = ["rep", "config", "T0", "T1", "branch", "estimator", "method"]
        rows = []
        for mode in ("fixed_template", "reanchored_template"):
            for estimator in ("TSLS", "SIV"):
                for method in ("ARIMA-Z", "HAC", "AR"):
                    rows.append(dict(anchor_mode=mode, target_role="midpoint", rep=0, config="finite_length",
                                     T0=5, T1=10, branch="Basic", estimator=estimator, method=method,
                                     tau=1., se=1., covered=1, weight_mean=0.,
                                     weight_exposure_constraint=1., weight_max_abs=1.))
        raw = pd.DataFrame(rows)
        expected = raw[raw.anchor_mode.eq("fixed_template")].copy()
        expected["tau"] = 2.
        def compare(left, right, columns, **kwargs):
            for column in columns:
                if not np.array_equal(left[column], right[column]):
                    raise AssertionError("fixture numerical mismatch")
            return 0., 0.
        module = SimpleNamespace(target_map=lambda p: {"midpoint": 1.},
                                 midpoint_role=lambda p: "midpoint",
                                 ANCHOR_MODES=["fixed_template", "reanchored_template"],
                                 BRANCHES=["Basic"], METHODS=["ARIMA-Z", "HAC", "AR"],
                                 KEY_COLUMNS=keys, REFERENCE_NUMERIC_COLUMNS=["tau", "se"],
                                 numeric_difference_diagnostics=compare)
        protocol = {"production_reference": {"output_directory": "reference_outputs/simulation",
                                            "configuration": "finite_length"},
                    "numerical_equivalence": {"absolute_tolerance": 1e-10, "relative_tolerance": 5e-15}}
        reporting.install_target_checks(module)
        return module, protocol, raw, expected

    def test_target_archive_mismatch_is_reported_not_hidden(self):
        module, protocol, raw, reference = self.target_fixture()
        with patch.object(pd, "read_csv", return_value=reference):
            result = module.structural_checks(protocol, raw, 1,
                {"maximum_common_random_component_difference": 0.})
        self.assertTrue(all(r["status"] == "mismatch" for r in result["legacy_reference"].values()))

    def test_target_later_structural_failure_still_blocks(self):
        module, protocol, raw, reference = self.target_fixture()
        with patch.object(pd, "read_csv", return_value=reference):
            with self.assertRaisesRegex(AssertionError, "Common random components differ"):
                module.structural_checks(protocol, raw, 1,
                    {"maximum_common_random_component_difference": 1.})

    def test_target_source_remains_byte_identical(self):
        reporting.verify_sources()

    def test_ci_does_not_launch_simulations(self):
        workflow = (ROOT / ".github/workflows/reproducibility.yml").read_text()
        self.assertIn("--structural-only", workflow)
        self.assertNotIn("--execute", workflow)
        self.assertNotIn("--quick", workflow)

    def test_execution_requires_destination_and_environment(self):
        p = subprocess.run([sys.executable, "-B", "reproduce.py", "--execute"],
                           cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("requires --output", p.stderr)

if __name__ == "__main__":
    unittest.main()
