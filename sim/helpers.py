"""
Simulation helpers: constants, data loading, panel utilities, and shock calibration.

All functions operate on types defined in the local sim/ package.
No external pipeline dependencies.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dgp import stable_seed
from .estimators import (
    aggregate_series,
    ols_slope_with_intercept,
    tsls_weights,
    unit_level_slope_vector,
    weighted_slope_ratio,
)
from .inference import arima_residual_covariance_slope_se, raw_z_arp_design_based_se
from .model_class import (
    ModelClassComponents,
    build_branch_panel,
    estimate_branch_panel,
    first_stage_component,
)
from .robust_weights import projection_residualizer_const_z, residuals_two_way_unit_slope

# ── simulation design constants ───────────────────────────────────────────────

BRANCHES: tuple[str, ...] = ("Basic", "GFE", "Aggregate Shock", "GFE + Agg. Shock")

DESIGN_ORDER: tuple[str, ...] = ("Basic", "GFE", "Aggregate Shock", "GFE + Agg. Shock")

EVALUATION_CONFIGS: tuple[tuple[str, int, int], ...] = (
    # finite_length: mirrors the empirical panel (T₀=5, T₁=10, 2008-2022).
    ("finite_length", 5, 10),
    # longer_length: fixed-N, larger-T diagnostic. It changes T₀/T₁,
    # removes pre-period compression (PREPERIOD_SCALE=1.0), and uses a
    # different shock calibration.  It is a related simulation environment
    # under a larger time dimension.
    ("longer_length", 30, 60),
)


# ── data loading ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EmpiricalScaleConstants:
    year_start: int
    year_end: int
    n_years: int
    n_units: int
    instrument_sd: float
    first_stage_component_sd: float
    first_stage_component_rms: float
    exposure_mean: float
    exposure_sd: float


def load_exposure_profile(path: Path) -> np.ndarray:
    """Load the design-restricted TSLS exposure profile from the panel CSV.

    The exposure profile is the vector of unit-level first-stage slopes π̂_i
    estimated from the design-restricted panel (n=49 states, 2008-2022,
    excluding CA and VT).  Each π̂_i measures state i's sensitivity of
    wind+solar retail sales intensity to the federal renewable subsidy instrument.
    """
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "D_tsls" not in (reader.fieldnames or []):
            raise RuntimeError(f"{path} does not contain D_tsls column.")
        values = [float(row["D_tsls"]) for row in reader]
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise RuntimeError(f"Invalid exposure profile from {path}.")
    return arr


def load_instrument_path(path: Path, *, year_start: int = 2008, year_end: int = 2022) -> tuple[np.ndarray, np.ndarray]:
    """Load the unique annual aggregate instrument path from the panel CSV."""
    z_by_year: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"year", "Z_t_lag2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            year = int(row["year"])
            if year < int(year_start) or year > int(year_end):
                continue
            z_raw = row.get("Z_t_lag2", "")
            if z_raw in ("", None):
                continue
            z_val = float(z_raw)
            previous = z_by_year.get(year)
            if previous is not None and abs(previous - z_val) > 1e-15:
                raise RuntimeError(f"{path} has non-unique Z_t_lag2 values for year {year}.")
            z_by_year[year] = z_val

    expected_years = list(range(int(year_start), int(year_end) + 1))
    missing_years = [year for year in expected_years if year not in z_by_year]
    if missing_years:
        raise RuntimeError(f"{path} missing instrument values for years: {missing_years}")

    years = np.asarray(expected_years, dtype=int)
    z = np.asarray([z_by_year[year] for year in expected_years], dtype=float)
    if not np.all(np.isfinite(z)) or float(np.std(z, ddof=0)) <= 0.0:
        raise RuntimeError(f"Invalid instrument path from {path}.")
    return years, z


def derive_empirical_scale_constants(
    panel_path: Path,
    exposure_path: Path,
    *,
    year_start: int = 2008,
    year_end: int = 2022,
) -> EmpiricalScaleConstants:
    """Derive the raw empirical scale constants used by the simulation."""
    _, z = load_instrument_path(panel_path, year_start=year_start, year_end=year_end)
    exposure = load_exposure_profile(exposure_path)
    first_stage = np.outer(exposure, z)
    return EmpiricalScaleConstants(
        year_start=int(year_start),
        year_end=int(year_end),
        n_years=int(z.size),
        n_units=int(exposure.size),
        instrument_sd=float(np.std(z, ddof=0)),
        first_stage_component_sd=float(np.std(first_stage, ddof=0)),
        first_stage_component_rms=float(np.sqrt(np.mean(first_stage * first_stage))),
        exposure_mean=float(np.mean(exposure)),
        exposure_sd=float(np.std(exposure, ddof=0)),
    )




# ── panel construction helpers ────────────────────────────────────────────────

def components_from_common(common: dict[str, np.ndarray]) -> ModelClassComponents:
    """Wrap the dgp.simulate_common_components output into ModelClassComponents."""
    n = int(np.asarray(common["pi"]).reshape(-1).size)
    return ModelClassComponents(
        Z=np.asarray(common["Z"], dtype=float),
        D=np.asarray(common["pi"], dtype=float),
        alpha_w=np.zeros(n, dtype=float),
        alpha_y=np.zeros(n, dtype=float),
        lowrank_w=np.asarray(common["lowrank_w"], dtype=float),
        lowrank_y=np.asarray(common["lowrank_y"], dtype=float),
        shock_w=np.asarray(common["shock_w"], dtype=float),
        shock_y=np.asarray(common["shock_y"], dtype=float),
        eps_w=np.asarray(common["eps_w"], dtype=float),
        eps_y=np.asarray(common["eps_y"], dtype=float),
    )


def apply_preperiod_scale(
    components: ModelClassComponents,
    *,
    T0: int,
    scale: float,
) -> ModelClassComponents:
    """Scale non-first-stage learning-window components by `scale`.

    The first-stage column (pi_i * Z_t) and all post-window columns are
    unchanged.  This calibrates the signal-to-noise ratio seen by the
    Robust SIV optimizer during weight learning.
    """
    s = float(scale)
    if abs(s - 1.0) <= 1e-15:
        return components

    def _scaled(panel: np.ndarray) -> np.ndarray:
        out = np.array(panel, dtype=float, copy=True)
        out[:, : int(T0)] *= s
        return out

    return ModelClassComponents(
        Z=components.Z,
        D=components.D,
        alpha_w=components.alpha_w,
        alpha_y=components.alpha_y,
        lowrank_w=_scaled(components.lowrank_w),
        lowrank_y=_scaled(components.lowrank_y),
        shock_w=_scaled(components.shock_w),
        shock_y=_scaled(components.shock_y),
        eps_w=_scaled(components.eps_w),
        eps_y=_scaled(components.eps_y),
    )


def shock_bias_component(
    *,
    shock_y: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    weights: np.ndarray,
    T0: int,
    T1: int,
) -> float:
    """Direct aggregate-shock contribution to the IV estimator bias.

    Equals slope(w' shock_y, Z) / slope(w' W, Z) on the post-window.
    """
    post = slice(int(T0), int(T0) + int(T1))
    y_s = aggregate_series(shock_y[:, post], weights)
    w_s = aggregate_series(W[:, post], weights)
    z_s = np.asarray(Z, dtype=float)[post]
    return float(
        ols_slope_with_intercept(y_s, z_s)
        / ols_slope_with_intercept(w_s, z_s)
    )


# ── shock calibration ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ShockCalibration:
    """Result of calibrate_shock_loading."""
    components: ModelClassComponents
    benchmark_se: float
    target_component: float
    realized_component: float
    calibration_error: float
    achieved_ratio: float


def calibrate_shock_loading(
    components: ModelClassComponents,
    *,
    T0: int,
    T1: int,
    shock_se_scale: float,
    tau_true: float,
    arima_ar_order: int = 2,
    use_gfe_benchmark: bool = False,
) -> ShockCalibration:
    """Replace the shock-outcome loading using the calibrated Q_W direction.

    shock_se_scale expresses the shock amplitude as a multiple of the ARIMA SE
    in the benchmark environment for the realized replication.  benchmark_SE is
    computed from the same AR(arima_ar_order) Toeplitz formula used for reported
    inference, placing the shock magnitude on the same scale as the estimation
    uncertainty in that replication.

    use_gfe_benchmark selects the reference environment for benchmark_SE:
      False — pure Aggregate Shock panel (GFE loadings zeroed).  benchmark_SE
              reflects the shock-only residual structure.
      True  — GFE+Agg.Shock panel.  benchmark_SE incorporates both the GFE
              residual variance and shock residual variance in the combined
              environment.

    The preliminary theta_y drawn by sim.dgp is not the final outcome-shock
    loading in reported runs.  This function is called inside the Monte Carlo
    loop and constructs a replication-specific final loading from the realized
    treatment-side learning-window curvature:

        direction = Q_W @ w_TSLS
        theta_y = target_component × pi_t × direction
                  / (loading_slope × h_slope)

    direction and loading_slope are both derived from ref_est.weights_tsls,
    ensuring internal consistency of the calibration identity.
    """
    Z = np.asarray(components.Z, dtype=float)
    post = slice(T0, T0 + T1)

    # Recover the aggregate-shock time path as the leading right singular
    # vector of the treatment shock panel.
    _, _, vt = np.linalg.svd(np.asarray(components.shock_w, dtype=float), full_matrices=False)
    H = np.asarray(vt[0], dtype=float)
    H -= float(np.mean(H))
    _h_rms = float(np.sqrt(np.mean(H * H)))
    if _h_rms <= 1e-12:
        raise ValueError("Shock H has near-zero RMS after centering — shock_w panel is degenerate.")
    H /= _h_rms

    # Build the shock-only treatment panel for Q_W.
    # The Z-orthogonalised shock residual defines the direction SIV's pre-period
    # objective suppresses.  Note: direction and loading_slope are both computed
    # from ref_est.weights_tsls (see below) to ensure the calibration algebra
    # bias = target_component is exact.
    W_source = np.asarray(
        first_stage_component(components.D, components.Z)
        + components.shock_w
        + components.eps_w,
        dtype=float,
    )
    sig_w, _ = residuals_two_way_unit_slope(W_source[:, :T0], Z[:T0])
    Mz = projection_residualizer_const_z(Z[:T0])
    n, t0 = W_source[:, :T0].shape
    Q_W = (W_source[:, :T0] @ Mz @ W_source[:, :T0].T) / (float(t0) * sig_w * float(n) ** 2)

    # Reference panel for benchmark SE (see docstring for use_gfe_benchmark).
    _bench_branch = "GFE + Agg. Shock" if use_gfe_benchmark else "Aggregate Shock"
    ref_comps = ModelClassComponents(
        Z=components.Z, D=components.D,
        alpha_w=components.alpha_w, alpha_y=components.alpha_y,
        lowrank_w=components.lowrank_w if use_gfe_benchmark else np.zeros_like(components.lowrank_w),
        lowrank_y=components.lowrank_y if use_gfe_benchmark else np.zeros_like(components.lowrank_y),
        shock_w=components.shock_w,
        shock_y=np.zeros_like(components.shock_y),
        eps_w=components.eps_w, eps_y=components.eps_y,
    )
    ref_panel = build_branch_panel(
        ref_comps, branch=_bench_branch, tau_true=tau_true, T0=T0, T1=T1
    )
    ref_est = estimate_branch_panel(ref_panel, Z)

    # direction uses ref_est.weights_tsls — the same weights used for
    # loading_slope below — so the calibration identity bias = target_component
    # holds exactly (no mixing of pre-window and full-sample first-stage slopes).
    direction = Q_W @ ref_est.weights_tsls
    direction -= float(direction.mean())
    direction /= float(np.sqrt(np.mean(direction * direction)))
    y_ref = aggregate_series(ref_panel.Y[:, post], ref_est.weights_tsls)
    w_ref = aggregate_series(ref_panel.W[:, post], ref_est.weights_tsls)
    z_ref = Z[post]

    # Benchmark SE: AR(p) Toeplitz variance of the IV score.
    # NOTE: the DGP generates Z as AR(1) (z_process_mode="ar1"), but
    # arima_ar_order=2 in the active configuration.  The AR(2) order was
    # selected by AICc from the empirical instrument series. The inference order
    # and DGP order are distinct design objects. The AR(2) Toeplitz estimator is
    # over-parameterised for an AR(1) DGP and remains consistent as φ₂→0.
    # At short T₁ the estimated AR coefficients carry sampling uncertainty that
    # propagates into Ω; this is a parametric approximation.
    benchmark_se = float(
        raw_z_arp_design_based_se(
            y_ref, w_ref, z_ref,
            z_process=Z,
            ar_order=arima_ar_order,
            varz_mode="window",
        ).se
    )
    target_component = float(shock_se_scale) * benchmark_se

    # Scale theta_y so the TSLS direct shock component equals target_component.
    pi_t = weighted_slope_ratio(
        ref_panel.Y[:, post], ref_panel.W[:, post], z_ref, ref_est.weights_tsls
    ).pi
    h_slope = ols_slope_with_intercept(H[post], z_ref)
    loading_slope = float(np.mean(ref_est.weights_tsls * direction))
    # Guard each factor independently: a near-zero h_slope means H is nearly
    # orthogonal to Z on the post-window; a near-zero loading_slope means the
    # TSLS weights are nearly orthogonal to direction.  Checking only the
    # product denom allows one large factor to mask a near-zero other factor,
    # producing an astronomically large theta_y that silently corrupts panels.
    if abs(h_slope) <= 1e-10:
        raise ValueError("h_slope ≈ 0: shock H is nearly orthogonal to Z on the post-window.")
    if abs(loading_slope) <= 1e-10:
        raise ValueError("loading_slope ≈ 0: TSLS weights are nearly orthogonal to direction.")
    denom = loading_slope * h_slope
    theta_y = target_component * float(pi_t) * direction / denom

    calibrated = ModelClassComponents(
        Z=components.Z, D=components.D,
        alpha_w=components.alpha_w, alpha_y=components.alpha_y,
        lowrank_w=components.lowrank_w, lowrank_y=components.lowrank_y,
        shock_w=components.shock_w,
        shock_y=np.outer(theta_y, H),
        eps_w=components.eps_w, eps_y=components.eps_y,
    )
    realized_component = shock_bias_component(
        shock_y=calibrated.shock_y,
        W=ref_panel.W,
        Z=components.Z,
        weights=ref_est.weights_tsls,
        T0=T0,
        T1=T1,
    )
    calibration_error = float(realized_component - target_component)
    achieved = float(realized_component / benchmark_se) if benchmark_se > 0 else float("nan")
    return ShockCalibration(
        components=calibrated,
        benchmark_se=benchmark_se,
        target_component=target_component,
        realized_component=realized_component,
        calibration_error=calibration_error,
        achieved_ratio=achieved,
    )
