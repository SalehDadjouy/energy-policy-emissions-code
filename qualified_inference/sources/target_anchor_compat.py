"""Run the frozen target-anchor design with version-2 reporting inference.

This isolated harness preserves the target-anchor DGP and every structural
check. It separates legacy NumPy 2.4.0 numeric-reference comparisons from
the approved ARIMA reporting change and runs under the pinned NumPy 2.4.4
environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd

from adapter import HERE, PUBLIC, verify_lock
from run_adapted import load_target, patch_reporting


COMPAT_PROTOCOL = HERE / "target_anchor_compat_protocol.json"
COMPAT_LOCK = HERE / "target_anchor_compat_lock.json"
ARIMA_METHOD = "ARIMA-Z"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_compat_protocol() -> dict[str, Any]:
    return json.loads(COMPAT_PROTOCOL.read_text(encoding="utf-8"))


def verify_compat_lock() -> dict[str, Any]:
    base = verify_lock()
    lock = json.loads(COMPAT_LOCK.read_text(encoding="utf-8"))
    expected = lock["source_sha256"]
    for filename in expected:
        path = Path(filename)
        observed = digest(path)
        if expected.get(str(path)) != observed:
            raise RuntimeError(f"Compatibility source is not frozen: {path}")
    if expected.get(str(PUBLIC / "analysis" / "target_anchor_sensitivity.py")) != digest(
        PUBLIC / "analysis" / "target_anchor_sensitivity.py"
    ):
        raise RuntimeError("Target-anchor source differs from the compatibility lock.")
    import scipy
    import statsmodels
    environment = {"python": platform.python_version(), "numpy": np.__version__,
                   "pandas": pd.__version__, "scipy": scipy.__version__,
                   "statsmodels": statsmodels.__version__}
    if environment != base["environment"]:
        raise RuntimeError(f"Environment differs from frozen environment: {environment}")
    return lock


def _numeric_difference(
    module: Any,
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: list[str],
    protocol: dict[str, Any],
) -> tuple[float, float]:
    equivalence = protocol.get("cross_version_numeric_equivalence", protocol.get("numerical_equivalence"))
    for column in columns:
        x = left[column].to_numpy(dtype=float)
        y = right[column].to_numpy(dtype=float)
        for predicate in (np.isnan, np.isposinf, np.isneginf):
            if not np.array_equal(predicate(x), predicate(y)):
                raise AssertionError(f"Nonfinite pattern mismatch for {column}.")
    return module.numeric_difference_diagnostics(
        left,
        right,
        columns,
        absolute_tolerance=float(equivalence["absolute_tolerance"]),
        relative_tolerance=float(equivalence["relative_tolerance"]),
    )


def compatibility_checks(
    module: Any,
    compat: dict[str, Any],
    protocol: dict[str, Any],
    raw: pd.DataFrame,
    *,
    reps: int,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Retain all target-anchor checks while classifying approved ARIMA changes."""
    targets = module.target_map(protocol)
    midpoint = module.midpoint_role(protocol)
    expected_rows = int(reps) * len(module.ANCHOR_MODES) * len(targets) * len(module.BRANCHES) * 2 * len(module.METHODS)
    if len(raw) != expected_rows:
        raise AssertionError(f"Raw row count {len(raw)} != {expected_rows}.")
    duplicate_keys = ["anchor_mode", "target_role", *module.KEY_COLUMNS]
    if raw.duplicated(duplicate_keys).any():
        raise AssertionError("Duplicate raw keys detected.")
    if set(raw["anchor_mode"]) != set(module.ANCHOR_MODES):
        raise AssertionError("Anchor-mode set mismatch.")
    if set(raw["target_role"]) != set(targets):
        raise AssertionError("Target-role set mismatch.")
    if set(raw["branch"]) != set(module.BRANCHES):
        raise AssertionError("Branch set mismatch.")
    if set(raw["method"]) != set(module.METHODS):
        raise AssertionError("Inference-method set mismatch.")
    if not raw["covered"].isin([0, 1]).all():
        raise AssertionError("Coverage indicator is not binary.")
    for column in ("weight_mean", "weight_exposure_constraint", "weight_max_abs"):
        if not np.isfinite(raw[column].to_numpy(dtype=float)).all():
            raise AssertionError(f"Nonfinite weight diagnostic: {column}.")
    if float(np.max(np.abs(raw["weight_mean"].to_numpy(dtype=float)))) > 1e-8:
        raise AssertionError("Weight centering constraint failed.")
    if float(np.max(np.abs(raw["weight_exposure_constraint"].to_numpy(dtype=float) - 1.0))) > 1e-8:
        raise AssertionError("Weight exposure constraint failed.")

    reference = protocol["production_reference"]
    keys = module.KEY_COLUMNS
    baseline = raw[
        (raw["anchor_mode"] == "fixed_template") & (raw["target_role"] == midpoint)
    ].sort_values(keys).reset_index(drop=True)
    reference_rows = pd.read_csv(PUBLIC / str(reference["output_directory"]) / "raw.csv")
    reference_rows = reference_rows[
        (reference_rows["config"] == reference["configuration"])
        & (reference_rows["rep"] < int(reps))
    ].sort_values(keys).reset_index(drop=True)
    if not baseline[keys].equals(reference_rows[keys]):
        raise AssertionError("Baseline simulation keys do not match the principal reference.")

    legacy_diagnostics: dict[str, dict[str, float | int]] = {}
    for method in module.METHODS:
        observed = baseline[baseline["method"] == method].reset_index(drop=True)
        expected = reference_rows[reference_rows["method"] == method].reset_index(drop=True)
        if not observed[keys].equals(expected[keys]):
            raise AssertionError(f"Legacy baseline keys differ for {method}.")
        columns = list(module.REFERENCE_NUMERIC_COLUMNS)
        if method == ARIMA_METHOD:
            columns.remove("se")
        maximum, relative = _numeric_difference(module, observed, expected, columns, compat)
        coverage_changes = int(np.count_nonzero(observed["covered"].to_numpy() != expected["covered"].to_numpy()))
        if method != ARIMA_METHOD and coverage_changes:
            raise AssertionError(f"Legacy containment decisions differ for {method}.")
        legacy_diagnostics[method] = {
            "max_abs_difference": maximum,
            "max_relative_difference": relative,
            "legacy_coverage_changes": coverage_changes,
            "legacy_arima_se_excluded": int(method == ARIMA_METHOD),
        }

    fixed_midpoint = baseline
    reanchored_midpoint = raw[
        (raw["anchor_mode"] == "reanchored_template") & (raw["target_role"] == midpoint)
    ].sort_values(keys).reset_index(drop=True)
    if not fixed_midpoint[keys].equals(reanchored_midpoint[keys]):
        raise AssertionError("Anchor-mode keys differ at the midpoint.")
    midpoint_maximum, midpoint_relative = _numeric_difference(
        module, fixed_midpoint, reanchored_midpoint, list(module.REFERENCE_NUMERIC_COLUMNS), protocol
    )
    if not np.array_equal(fixed_midpoint["covered"].to_numpy(), reanchored_midpoint["covered"].to_numpy()):
        raise AssertionError("Anchor-mode coverage differs at the midpoint.")

    tsls = raw[(raw["anchor_mode"] == "fixed_template") & (raw["estimator"] == "TSLS")]
    invariance_columns = [
        "bias", "se", "pi", "weight_mean", "weight_exposure_constraint", "weight_max_abs", "shock",
        "iv_moment_tau", "iv_moment_abs_tau", "iv_cov_zw", "iv_moment_bias_ratio",
        "bias_component_total", "bias_component_gfe", "bias_component_shock", "bias_component_eps",
    ]
    tsls_maximum = 0.0
    tsls_relative = 0.0
    tsls_coverage_equal = True
    for endpoint in [role for role in targets if role != midpoint]:
        midpoint_rows = tsls[tsls["target_role"] == midpoint].sort_values(keys).reset_index(drop=True)
        endpoint_rows = tsls[tsls["target_role"] == endpoint].sort_values(keys).reset_index(drop=True)
        if not midpoint_rows[keys].equals(endpoint_rows[keys]):
            raise AssertionError("Target comparison keys differ.")
        maximum, relative = _numeric_difference(module, midpoint_rows, endpoint_rows, invariance_columns, protocol)
        tsls_maximum = max(tsls_maximum, maximum)
        tsls_relative = max(tsls_relative, relative)
        tsls_coverage_equal = tsls_coverage_equal and np.array_equal(
            midpoint_rows["covered"].to_numpy(), endpoint_rows["covered"].to_numpy()
        )
    if not tsls_coverage_equal:
        raise AssertionError("Fixed-template TSLS coverage is not invariant across targets.")
    tolerance = float(protocol["numerical_equivalence"]["absolute_tolerance"])
    common_difference = float(runtime["maximum_common_random_component_difference"])
    if not np.isfinite(common_difference) or common_difference > tolerance:
        raise AssertionError("Common random components differ across target cells.")

    return {
        "status": "pass",
        "reps_checked": int(reps),
        "raw_rows": len(raw),
        "expected_raw_rows": expected_rows,
        "legacy_reference": legacy_diagnostics,
        "midpoint_anchor_mode_max_abs_difference": midpoint_maximum,
        "midpoint_anchor_mode_max_relative_difference": midpoint_relative,
        "fixed_template_tsls_max_abs_invariance_difference": tsls_maximum,
        "fixed_template_tsls_max_relative_invariance_difference": tsls_relative,
        "fixed_template_tsls_coverage_invariant": tsls_coverage_equal,
        "maximum_common_random_component_difference": float(runtime["maximum_common_random_component_difference"]),
        "cross_version_numeric_equivalence": compat["cross_version_numeric_equivalence"],
        "legacy_arima_reporting_exclusion": {
            "excluded_field": "se",
            "excluded_decision": "covered",
            "reason": "The version-2 ARIMA reporting calculation is the approved change under test.",
        },
        "production_reference_unchanged": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--execute-full", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compat = load_compat_protocol()
    verify_compat_lock()
    module = load_target("target_anchor")
    patch_reporting(module, "target_anchor")
    protocol = module.load_protocol()
    full_reps = int(protocol["paired_execution"]["reps"])
    if args.execute_full:
        if args.reps not in (None, full_reps):
            parser.error(f"--execute-full requires --reps {full_reps} or no --reps argument.")
        reps = full_reps
    else:
        if args.reps is None or args.reps <= 0 or args.reps >= full_reps:
            parser.error("Preflight requires 1 through full_reps - 1; full execution requires --execute-full.")
        reps = int(args.reps)
    output = args.out.expanduser().resolve()
    permitted = (HERE / "candidate_outputs" / "target_anchor").resolve()
    try:
        output.relative_to(permitted)
    except ValueError as exc:
        raise RuntimeError(f"Output must be below {permitted}.") from exc
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite an existing output: {output}")

    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "reps": reps, "protocol": compat["protocol"]}
    (output / "execution_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    raw, weight_contrasts, runtime = module.run_diagnostic(protocol, reps=reps)
    raw.to_csv(output / "raw_candidate.csv", index=False)
    weight_contrasts.to_csv(output / "weight_contrasts_candidate.csv", index=False)
    (output / "runtime_candidate.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    verify_compat_lock()
    original_check = module.structural_checks
    module.structural_checks = lambda p, r, reps, runtime: compatibility_checks(module, compat, p, r, reps=reps, runtime=runtime)
    try:
        manifest = module.write_outputs(
            protocol, output=output, raw=raw, weight_contrasts=weight_contrasts, reps=reps, runtime=runtime
        )
    except Exception as exc:
        status.update(status="validation_failed", error=str(exc))
        (output / "execution_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        raise
    finally:
        module.structural_checks = original_check
    sidecar = {
        "protocol": compat["protocol"],
        "execution_scope": "full" if reps == full_reps else "preflight",
        "reps": reps,
        "compatibility_source_sha256": json.loads(COMPAT_LOCK.read_text(encoding="utf-8"))["source_sha256"],
        "target_manifest": manifest,
    }
    (output / "compatibility_manifest.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    verify_compat_lock()
    status.update(status="complete")
    (output / "execution_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(output), "reps": reps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
