"""Structural tests: no fitting, stochastic sampling, or simulation execution."""
import ast
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qualified_inference import reporting as q
from qualified_inference.workflow import stage_objective
import validate_results as validation

BASE = json.loads((ROOT / "qualified_inference/structural_baseline.json").read_text())

def functions(path):
    return {n.name: n for n in ast.parse(path.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

def fingerprint(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()

class CandidateTests(unittest.TestCase):
    def setUp(self):
        # An unmocked fit or random draw is an error throughout this suite.
        for target in ("numpy.random.default_rng", "numpy.random.normal",
                       "statsmodels.tsa.arima.model.ARIMA.fit",
                       "scipy.optimize.fmin_l_bfgs_b"):
            guard = patch(target, side_effect=AssertionError("Numerical execution is outside this test"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_sources_are_exact(self):
        q.verify_sources()
        self.assertFalse(q.PROVENANCE["public_candidate_numerically_verified"])

    def test_design_inputs_are_exact(self):
        for name, expected in BASE["exact_files"].items():
            with self.subTest(file=name):
                self.assertEqual(q.digest(ROOT / name), expected)

    def test_other_functions_are_unchanged(self):
        for name, expected in BASE["functions"].items():
            actual = functions(ROOT / name)
            allowed = BASE["allowed_function_changes"].get(name, [])
            for func, digest in expected.items():
                if func not in allowed:
                    with self.subTest(file=name, function=func):
                        self.assertEqual(fingerprint(actual[func]), digest)

    def test_all_python_parses(self):
        for source in ROOT.rglob("*.py"):
            if any(part in (".git", ".venv", "__pycache__") for part in source.relative_to(ROOT).parts):
                continue
            ast.parse(source.read_text(), filename=str(source))

    def test_principal_binding(self):
        import run_simulation
        self.assertIs(run_simulation.raw_z_arp_design_based_se, q.principal_arima_result)

    def test_sensitivity_bindings(self):
        for name in ("finite_window_weight_learning", "target_anchor_sensitivity",
                     "state_panel_and_confounding_sensitivity"):
            module = importlib.import_module("analysis." + name)
            self.assertIs(module.raw_z_arp_design_based_se, q.reporting_arima_result)

    def test_orientation_reporting_and_calibration_are_distinct(self):
        import analysis.shock_orientation_sensitivity as orientation
        import sim.inference as legacy
        self.assertIs(orientation.raw_z_arp_design_based_se, legacy.raw_z_arp_design_based_se)
        self.assertIs(orientation.orientation_arima_result, q.orientation_arima_result)
        tree = functions(ROOT / "analysis/shock_orientation_sensitivity.py")
        calls = [n.func.id for n in ast.walk(tree["_inference_records"])
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertIn("orientation_arima_result", calls)
        self.assertNotIn("raw_z_arp_design_based_se", calls)

    def test_empirical_supplies_original_points(self):
        tree = functions(ROOT / "empirical.py")["run_sample"]
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                 isinstance(n.func, ast.Name) and n.func.id == "interval_for_points"]
        self.assertEqual(len(calls), 2)
        for call, suffix in zip(calls, ("t", "r")):
            self.assertEqual(ast.unparse(call.args[-2]), f"est_{suffix}['tau']")
            self.assertEqual(ast.unparse(call.args[-1]), f"est_{suffix}['pi']")

    def test_observed_point_values_are_forwarded(self):
        interval = SimpleNamespace(status="ok", se=1., critical=2., lower=-1.,
                                   upper=3., reason="", detail="")
        fit = SimpleNamespace(cache_key="structural_fixture")
        with patch.object(q.common, "fit_instrument", return_value=fit) as fit_call:
            with patch.object(q.common, "interval_from_fit", return_value=interval) as call:
                result = q.interval_for_points([1.], [2.], [3.], [4.], 5., 6.)
        self.assertEqual((result.tau, result.pi), (5., 6.))
        self.assertEqual(call.call_args.kwargs["tau_hat"], 5.)
        self.assertEqual(call.call_args.kwargs["pi_hat"], 6.)
        self.assertEqual(fit_call.call_args.args[0].tolist(), [4.])

    def test_principal_preserves_centered_arithmetic(self):
        point = SimpleNamespace(tau=5., pi=6.)
        with patch.object(q, "centered_slope_ratio_moments", return_value=point) as moments:
            with patch.object(q, "interval_for_points", return_value="fixture") as call:
                self.assertEqual(q.principal_arima_result([1.], [2.], [3.], z_process=[4.]), "fixture")
        moments.assert_called_once()
        self.assertEqual(call.call_args.args[-2:], (5., 6.))

    def test_wrong_specification_rejected_before_fit(self):
        for kwargs in ({"ar_order": 1}, {"varz_mode": "other"}):
            with self.assertRaises(q.ReportingInferenceUnavailable):
                q.principal_arima_result([], [], [], z_process=[], **kwargs)

    def test_failed_inference_cannot_be_silently_skipped_as_value_error(self):
        error = q.common.InferenceFailure("nonconvergence", "structural fixture")
        with patch.object(q.common, "fit_instrument", side_effect=error):
            with self.assertRaises(q.ReportingInferenceUnavailable):
                q.interval_for_points([], [], [], [], 1., 1.)
        self.assertFalse(issubclass(q.ReportingInferenceUnavailable, ValueError))

    def test_retry_only_for_nonconvergence(self):
        fitter = type(q._orientation.fitter)()
        method_globals = fitter.fit.__globals__
        error = q.common.InferenceFailure("invalid_input", "structural fixture")
        with patch.object(q.common, "fit_key", return_value="fixture"):
            with patch.object(q.common, "fit_instrument", side_effect=error):
                with patch.dict(method_globals, corrected_fit=lambda *a: self.fail("Retry on invalid input")):
                    with self.assertRaises(q.common.InferenceFailure):
                        fitter.fit([1.])

    def test_retry_keeps_original_success(self):
        fitter = type(q._orientation.fitter)()
        sentinel = object()
        with patch.object(q.common, "fit_key", return_value="fixture"):
            with patch.object(q.common, "fit_instrument", return_value=sentinel):
                value, info = fitter.fit([1.])
        self.assertIs(value, sentinel)
        self.assertFalse(info["retry_used"])

    def test_objective_bank_checksum_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluation"
            command = stage_objective(path)
            self.assertEqual(command[1:3], ["--stage", "evaluate"])
            self.assertEqual(q.digest(path / "population_objective.npz"), q.PROVENANCE["objective_bank_sha256"])
            with self.assertRaises(FileExistsError):
                stage_objective(path)

    def test_objective_cannot_write_inside_package(self):
        with self.assertRaises(ValueError):
            stage_objective(ROOT / "outputs/forbidden")

    def test_archive_hashes(self):
        validation.validate_reference_hashes()

    def test_archived_display_values_and_sensitivity_keys(self):
        reference = ROOT / "reference_outputs"
        validation.validate_empirical(reference / "empirical/results_summary.csv")
        validation.validate_simulation(reference / "simulation/summary.csv",
                                       reference / "simulation/paired_differences.csv")
        validation.validate_sensitivity_outputs(reference / "sensitivity")

    def test_reference_schema_is_not_silently_weakened(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, actual = Path(tmp) / "reference.csv", Path(tmp) / "actual.csv"
            pd.DataFrame({"id": [1], "value": [2]}).to_csv(reference, index=False)
            pd.DataFrame({"id": [1]}).to_csv(actual, index=False)
            with self.assertRaisesRegex(AssertionError, "Missing reference columns"):
                validation.compare_reference(actual, reference, keys=["id"])
            pd.DataFrame({"id": [1, 1], "value": [2, 2]}).to_csv(actual, index=False)
            with self.assertRaisesRegex(AssertionError, "Duplicate comparison keys"):
                validation.compare_reference(actual, reference, keys=["id"])

    def test_baseline_compatibility_archive_is_separate(self):
        import analysis.target_anchor_sensitivity as target
        self.assertNotEqual(q.digest(ROOT / "reference_outputs/simulation/raw.csv"),
                            q.digest(ROOT / "validation_data/reference_outputs/simulation/raw.csv"))
        self.assertEqual(target.structural_checks.__closure__[0].cell_contents["protocol"],
                         "target_anchor_cross_version_compatibility_v1")

    def test_tolerances_are_unchanged(self):
        self.assertEqual(validation.REFERENCE_RTOL, 1e-8)
        self.assertEqual(validation.REFERENCE_ATOL, 1e-10)
        compat = json.loads((q.SOURCES / "target_anchor_compat_protocol.json").read_text())
        self.assertEqual(compat["cross_version_numeric_equivalence"]["absolute_tolerance"], 1e-7)
        self.assertEqual(compat["cross_version_numeric_equivalence"]["relative_tolerance"], 1e-10)

    def test_runtime_versions(self):
        self.assertEqual(f"{sys.version_info.major}.{sys.version_info.minor}", BASE["python"])
        for name, version in BASE["environment"].items():
            self.assertEqual(importlib.metadata.version(name), version)

    def test_no_default_numerical_execution(self):
        result = subprocess.run([sys.executable, "-B", "reproduce.py"], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["mode"], "preflight_only")

    def test_no_personal_absolute_paths(self):
        for path in ROOT.rglob("*"):
            if any(part in (".git", ".venv", "__pycache__") for part in path.relative_to(ROOT).parts):
                continue
            if path.suffix in (".py", ".json", ".md", ".txt", ".yml", ".cff"):
                value = path.read_text()
                # Construct the search string so this test does not match itself.
                self.assertNotIn("/" + "Users/", value, str(path))

if __name__ == "__main__":
    unittest.main()
