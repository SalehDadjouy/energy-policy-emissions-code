"""Runtime adapter for approved version-2 reporting inference only.

This module never edits the public package.  It imports a hash-checked
common ARIMA implementation and exposes the result shape expected by the
existing sensitivity programs.  Point estimates are calculated by the
existing estimator helper and are supplied unchanged to the interval routine.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
PUBLIC = PROJECTS / "energy-policy-emissions-code"
COMMON_PATH = PROJECTS / "ARIMA_Common_Implementation_2026_09_16/common_arima.py"
LOCK_PATH = HERE / "implementation_lock.json"


class ReportingInferenceUnavailable(RuntimeError):
    """A reporting interval could not be computed without changing the design."""


@dataclass(frozen=True)
class ReportingResult:
    """Compatibility shape used by sensitivity programs' ARIMA reporting rows."""

    tau: float
    pi: float
    se: float
    critical: float
    lower: float
    upper: float
    fit_key: str


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load_common():
    spec = importlib.util.spec_from_file_location("arima_v2_common_locked", COMMON_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load locked common inference source: {COMMON_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_lock() -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise RuntimeError(f"Implementation lock is missing: {LOCK_PATH}")
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def verify_lock() -> dict[str, Any]:
    lock = _load_lock()
    if Path(lock["public_root"]).resolve() != PUBLIC.resolve():
        raise RuntimeError("Implementation lock names a different public package.")
    for raw_path, expected in lock["source_sha256"].items():
        path = Path(raw_path)
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"Locked source changed or is unavailable: {path}")
    return lock


def _public_imports():
    if str(PUBLIC) not in sys.path:
        sys.path.insert(0, str(PUBLIC))
    from sim.estimators import slope_ratio
    return slope_ratio


def reporting_arima_result(
    y: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    *,
    z_process: np.ndarray,
    ar_order: int = 2,
    varz_mode: str = "window",
) -> ReportingResult:
    """Compute only the frozen-point, observed-denominator ARIMA interval.

    The arguments ar_order and varz_mode are asserted rather than reinterpreted.
    Existing sensitivity scripts pass AR(2) and the post-window variance rule.
    """
    lock = verify_lock()
    if int(ar_order) != 2 or str(varz_mode) != "window":
        raise ReportingInferenceUnavailable(
            "Version-2 reporting inference requires AR(2) and the observed post-window denominator."
        )
    slope_ratio = _public_imports()
    point = slope_ratio(np.asarray(y, dtype=float), np.asarray(w, dtype=float), np.asarray(z, dtype=float))
    common = _load_common()
    try:
        fit = common.fit_instrument(np.asarray(z_process, dtype=float))
        interval = common.interval_from_fit(
            np.asarray(y, dtype=float), np.asarray(w, dtype=float), np.asarray(z, dtype=float),
            tau_hat=float(point.tau), pi_hat=float(point.pi), fit=fit,
        )
    except common.InferenceFailure as exc:
        raise ReportingInferenceUnavailable(f"{exc.code}: {exc}") from exc
    if interval.status != "ok":
        raise ReportingInferenceUnavailable(f"{interval.reason}: {interval.detail}")
    values = (point.tau, point.pi, interval.se, interval.critical, interval.lower, interval.upper)
    if not all(math.isfinite(float(value)) for value in values):
        raise ReportingInferenceUnavailable("Reporting inference returned a nonfinite value.")
    return ReportingResult(
        tau=float(point.tau), pi=float(point.pi), se=float(interval.se),
        critical=float(interval.critical), lower=float(interval.lower), upper=float(interval.upper),
        fit_key=str(fit.cache_key),
    )


def lock_template(source_paths: list[Path]) -> dict[str, Any]:
    """Return the immutable source state to be written only after tests pass."""
    return {
        "protocol": "arima_sensitivity_implementation_v2",
        "purpose": "Replace reporting ARIMA inference only; retain all DGP, point, calibration, and comparator calculations.",
        "public_root": str(PUBLIC),
        "source_sha256": {str(path): digest(path) for path in source_paths},
        "reporting_inference": {
            "common_source": str(COMMON_PATH),
            "fitted_model": "ARIMA(2,0,0) with constant, ML state-space fit and stationarity enforcement",
            "fit_window": "full available aggregate instrument path",
            "residual": "Y - tau_hat W",
            "covariance": "raw-unit stationary AR(2) covariance, centered in the post-learning window",
            "denominator": "observed centered Z-W slope denominator",
            "critical_rule": "t(T1-2) for T1 <= 10; normal otherwise",
        },
        "invariants": {
            "no_dgp_change": True,
            "no_weight_change": True,
            "no_target_change": True,
            "no_random_draw_change": True,
            "no_hac_or_ar_change": True,
            "orientation_calibration_uses_legacy_reporting_helper": True,
        },
    }
