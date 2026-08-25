from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np


BRANCHES = ("Basic", "GFE", "Aggregate Shock", "GFE + Agg. Shock")
BRANCH_SWITCHES = {
    "Basic": (0, 0),
    "GFE": (1, 0),
    "Aggregate Shock": (0, 1),
    "GFE + Agg. Shock": (1, 1),
}
REQUIRED_PANEL_COMPONENTS = (
    "first_stage",
    "lowrank_w",
    "lowrank_y",
    "shock_w",
    "shock_y",
    "eps_w",
    "eps_y",
)


@dataclass(frozen=True)
class FiniteSampleParameters:
    rank: int
    z_ar: float
    h_ar: float
    factor_ar: float
    h_z_corr: float
    pi_sd: float
    lowrank_w_sd: float
    lowrank_y_sd: float
    lowrank_w_pi_corr: float
    lowrank_yw_loading_corr: float
    shock_w_sd: float
    shock_y_sd: float
    shock_w_pi_corr: float
    shock_y_pi_corr: float
    shock_yw_loading_corr: float
    eps_w_sd: float
    eps_y_sd: float
    eps_yw_corr: float = 0.0
    z_process_mode: str = "ar1"
    z_sd: float = 1.0
    exposure_profile_mode: str = "standardized"
    z_ar2_phi1: float = 0.0
    z_ar2_phi2: float = 0.0
    z_ma2_theta1: float = 0.0
    z_ma2_theta2: float = 0.0
    lowrank_w_shock_trait_corr: float = 0.0
    lowrank_y_shock_trait_corr: float = 0.0
    h_z_corr_pre: float | None = None
    h_z_corr_post: float | None = None
    shock_y_learning_marker_sd: float = 0.0
    shock_y_z_aligned_sd: float = 0.0
    aggregate_shock_mode: str = "sample_correlated"
    lowrank_factor_mode: str = "z_correlated"
    lowrank_factor_orthogonalization_weight: float = 1.0
    lowrank_factor_z_corr: float = 0.0
    loading_correlation_mode: str = "sample_orthogonalized"
    shock_y_loading_mode: str = "pi_theta_w_blend"


def stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32 - 1)


def load_parameters(raw: Mapping[str, object]) -> FiniteSampleParameters:
    return FiniteSampleParameters(
        rank=int(raw["rank"]),
        z_ar=float(raw["z_ar"]),
        z_process_mode=str(raw.get("z_process_mode", "ar1")),
        z_sd=float(raw.get("z_sd", 1.0)),
        exposure_profile_mode=str(raw.get("exposure_profile_mode", "standardized")),
        z_ar2_phi1=float(raw.get("z_ar2_phi1", 0.0)),
        z_ar2_phi2=float(raw.get("z_ar2_phi2", 0.0)),
        z_ma2_theta1=float(raw.get("z_ma2_theta1", 0.0)),
        z_ma2_theta2=float(raw.get("z_ma2_theta2", 0.0)),
        h_ar=float(raw["h_ar"]),
        factor_ar=float(raw["factor_ar"]),
        h_z_corr=float(raw["h_z_corr"]),
        pi_sd=float(raw["pi_sd"]),
        lowrank_w_sd=float(raw["lowrank_w_sd"]),
        lowrank_y_sd=float(raw["lowrank_y_sd"]),
        lowrank_w_pi_corr=float(raw["lowrank_w_pi_corr"]),
        lowrank_yw_loading_corr=float(raw["lowrank_yw_loading_corr"]),
        shock_w_sd=float(raw["shock_w_sd"]),
        shock_y_sd=float(raw["shock_y_sd"]),
        shock_w_pi_corr=float(raw["shock_w_pi_corr"]),
        shock_y_pi_corr=float(raw["shock_y_pi_corr"]),
        shock_yw_loading_corr=float(raw["shock_yw_loading_corr"]),
        eps_w_sd=float(raw["eps_w_sd"]),
        eps_y_sd=float(raw["eps_y_sd"]),
        eps_yw_corr=float(raw.get("eps_yw_corr", 0.0)),
        lowrank_w_shock_trait_corr=float(raw.get("lowrank_w_shock_trait_corr", 0.0)),
        lowrank_y_shock_trait_corr=float(raw.get("lowrank_y_shock_trait_corr", 0.0)),
        h_z_corr_pre=(
            None if raw.get("h_z_corr_pre") is None else float(raw.get("h_z_corr_pre"))
        ),
        h_z_corr_post=(
            None if raw.get("h_z_corr_post") is None else float(raw.get("h_z_corr_post"))
        ),
        shock_y_learning_marker_sd=float(raw.get("shock_y_learning_marker_sd", 0.0)),
        shock_y_z_aligned_sd=float(raw.get("shock_y_z_aligned_sd", 0.0)),
        aggregate_shock_mode=str(raw.get("aggregate_shock_mode", "sample_correlated")),
        lowrank_factor_mode=str(raw.get("lowrank_factor_mode", "full_sample_orthogonalized")),
        lowrank_factor_orthogonalization_weight=float(raw.get("lowrank_factor_orthogonalization_weight", 1.0)),
        lowrank_factor_z_corr=float(raw.get("lowrank_factor_z_corr", 0.0)),
        loading_correlation_mode=str(raw.get("loading_correlation_mode", "sample_orthogonalized")),
        shock_y_loading_mode=str(raw.get("shock_y_loading_mode", "pi_theta_w_blend")),
    )


def standardize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    out = arr - float(arr.mean())
    sd = float(out.std(ddof=0))
    if sd <= 0:
        raise ValueError("cannot standardize a zero-variance vector.")
    return out / sd


def residualize_against_const_z(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    xv = np.asarray(x, dtype=float).reshape(-1)
    zv = np.asarray(z, dtype=float).reshape(-1)
    if xv.size != zv.size:
        raise ValueError("x and z must have the same length.")
    X = np.column_stack([np.ones(zv.size), zv])
    return xv - X @ (np.linalg.pinv(X) @ xv)


def residualize_against_controls(x: np.ndarray, *controls: np.ndarray) -> np.ndarray:
    xv = np.asarray(x, dtype=float).reshape(-1)
    columns = [np.ones(xv.size)]
    for control in controls:
        cv = np.asarray(control, dtype=float).reshape(-1)
        if cv.size != xv.size:
            raise ValueError("all controls must have the same length as x.")
        columns.append(cv)
    X = np.column_stack(columns)
    return xv - X @ (np.linalg.pinv(X) @ xv)


def standardize_orthogonal_to_z(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    return standardize(residualize_against_const_z(x, z))


def standardize_orthogonal_to_z_h(x: np.ndarray, z: np.ndarray, h: np.ndarray) -> np.ndarray:
    return standardize(residualize_against_controls(x, z, h))


def standardize_split_orthogonal_to_z(x: np.ndarray, z: np.ndarray, split_index: int) -> np.ndarray:
    xv = np.asarray(x, dtype=float).reshape(-1)
    zv = np.asarray(z, dtype=float).reshape(-1)
    if xv.size != zv.size:
        raise ValueError("x and z must have the same length.")
    split = int(split_index)
    if split <= 2 or split >= xv.size - 2:
        raise ValueError("split_index must leave at least three observations on each side.")
    left = standardize_orthogonal_to_z(xv[:split], zv[:split])
    right = standardize_orthogonal_to_z(xv[split:], zv[split:])
    return standardize(np.concatenate([left, right]))


def standardize_split_orthogonal_to_z_h(
    x: np.ndarray,
    z: np.ndarray,
    h: np.ndarray,
    split_index: int,
) -> np.ndarray:
    xv = np.asarray(x, dtype=float).reshape(-1)
    zv = np.asarray(z, dtype=float).reshape(-1)
    hv = np.asarray(h, dtype=float).reshape(-1)
    if xv.size != zv.size or xv.size != hv.size:
        raise ValueError("x, z, and h must have the same length.")
    split = int(split_index)
    if split <= 3 or split >= xv.size - 3:
        raise ValueError("split_index must leave enough observations for split Z/H residualization.")
    left = standardize_orthogonal_to_z_h(xv[:split], zv[:split], hv[:split])
    right = standardize_orthogonal_to_z_h(xv[split:], zv[split:], hv[split:])
    return standardize(np.concatenate([left, right]))


def ar1_series(length: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    if int(length) <= 2:
        raise ValueError("AR(1) length must exceed two.")
    x = np.zeros(int(length), dtype=float)
    x[0] = rng.normal()
    innov_sd = math.sqrt(max(1.0 - float(phi) ** 2, 1e-8))
    for t in range(1, int(length)):
        x[t] = float(phi) * x[t - 1] + innov_sd * rng.normal()
    return standardize(x)


def ar2_series(length: int, phi1: float, phi2: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a standardized stationary AR(2) path."""
    if int(length) <= 2:
        raise ValueError("AR(2) length must exceed two.")
    roots = np.roots([1.0, -float(phi1), -float(phi2)])
    max_root = float(np.max(np.abs(roots))) if roots.size else 0.0
    if not np.isfinite(max_root) or max_root >= 1.0:
        raise ValueError(
            f"AR(2) coefficients are not stationary in recurrence form: phi1={phi1}, phi2={phi2}."
        )
    burnin = max(200, 10 * int(length))
    x = np.zeros(int(length) + burnin, dtype=float)
    x[0] = rng.normal()
    x[1] = float(phi1) * x[0] + rng.normal()
    for t in range(2, x.size):
        x[t] = float(phi1) * x[t - 1] + float(phi2) * x[t - 2] + rng.normal()
    return standardize(x[burnin:])


def ar2_population_unit_series(length: int, phi1: float, phi2: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a stationary AR(2) path with unit population variance.

    Unlike ``ar2_series``, this does not re-standardize each realized path.
    That distinction matters for finite-sample inference diagnostics because
    path-by-path normalization changes the distribution of the aggregate
    instrument process.
    """
    if int(length) <= 2:
        raise ValueError("AR(2) length must exceed two.")
    roots = np.roots([1.0, -float(phi1), -float(phi2)])
    max_root = float(np.max(np.abs(roots))) if roots.size else 0.0
    if not np.isfinite(max_root) or max_root >= 1.0:
        raise ValueError(
            f"AR(2) coefficients are not stationary in recurrence form: phi1={phi1}, phi2={phi2}."
        )
    coeff = np.asarray([float(phi1), float(phi2)], dtype=float)
    a = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    a[0, 0] = 1.0
    a[0, 1] -= coeff[0]
    a[0, 2] -= coeff[1]
    b[0] = 1.0
    for k in (1, 2):
        a[k, k] = 1.0
        for j in (1, 2):
            a[k, abs(k - j)] -= coeff[j - 1]
    gamma0 = float(np.linalg.solve(a, b)[0])
    if not math.isfinite(gamma0) or gamma0 <= 0.0:
        raise ValueError("AR(2) stationary variance calculation failed.")
    innov_sd = 1.0 / math.sqrt(gamma0)
    burnin = max(200, 10 * int(length))
    x = np.zeros(int(length) + burnin, dtype=float)
    x[0] = rng.normal()
    x[1] = float(phi1) * x[0] + innov_sd * rng.normal()
    for t in range(2, x.size):
        x[t] = float(phi1) * x[t - 1] + float(phi2) * x[t - 2] + innov_sd * rng.normal()
    return x[burnin:]


def ma2_series(length: int, theta1: float, theta2: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a standardized MA(2) path."""
    if int(length) <= 2:
        raise ValueError("MA(2) length must exceed two.")
    T = int(length)
    e = rng.normal(size=T + 2)
    x = np.empty(T, dtype=float)
    for t in range(T):
        j = t + 2
        x[t] = e[j] + float(theta1) * e[j - 1] + float(theta2) * e[j - 2]
    return standardize(x)


def instrument_series(length: int, params: FiniteSampleParameters, rng: np.random.Generator) -> np.ndarray:
    mode = str(params.z_process_mode).lower()
    if mode == "ar1":
        return ar1_series(length, params.z_ar, rng)
    if mode == "ar2":
        return ar2_series(length, params.z_ar2_phi1, params.z_ar2_phi2, rng)
    if mode == "ar2_population_unit":
        return ar2_population_unit_series(length, params.z_ar2_phi1, params.z_ar2_phi2, rng)
    if mode == "ma2":
        return ma2_series(length, params.z_ma2_theta1, params.z_ma2_theta2, rng)
    raise ValueError(f"unknown z_process_mode: {params.z_process_mode}")


def correlated_loading(
    base: np.ndarray,
    *,
    corr: float,
    sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    base_std = standardize(np.asarray(base, dtype=float).reshape(-1))
    noise = rng.normal(size=base_std.size)
    noise = noise - float(noise.mean())
    denom = float(base_std @ base_std)
    if denom <= 0:
        raise ValueError("base loading has zero usable variation.")
    noise = noise - base_std * float(noise @ base_std) / denom
    noise_std = standardize(noise)
    c = float(np.clip(corr, -0.999, 0.999))
    return float(sd) * standardize(c * base_std + math.sqrt(max(1.0 - c * c, 0.0)) * noise_std)


def parametric_correlated_series(
    reference: np.ndarray,
    innovation: np.ndarray,
    *,
    corr: float,
) -> np.ndarray:
    """Population-correlation construction without realized-sample projection."""
    ref = standardize(np.asarray(reference, dtype=float).reshape(-1))
    noise = standardize(np.asarray(innovation, dtype=float).reshape(-1))
    if ref.size != noise.size:
        raise ValueError("reference and innovation must have the same length.")
    c = float(np.clip(corr, -0.999, 0.999))
    return standardize(c * ref + math.sqrt(max(1.0 - c * c, 0.0)) * noise)


def parametric_correlated_loading(
    base: np.ndarray,
    *,
    corr: float,
    sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """State-loading analogue of the paper-style linear-combination rule."""
    base_std = standardize(np.asarray(base, dtype=float).reshape(-1))
    noise_std = standardize(rng.normal(size=base_std.size))
    c = float(np.clip(corr, -0.999, 0.999))
    return float(sd) * standardize(c * base_std + math.sqrt(max(1.0 - c * c, 0.0)) * noise_std)


def loading_matrix_from_base(
    base: np.ndarray,
    *,
    rank: int,
    corr: float,
    sd: float,
    rng: np.random.Generator,
    correlation_mode: str = "sample_orthogonalized",
) -> np.ndarray:
    if correlation_mode not in {"sample_orthogonalized", "z_correlated"}:
        raise ValueError(f"unknown loading correlation mode: {correlation_mode}")
    loading_fn = parametric_correlated_loading if correlation_mode == "z_correlated" else correlated_loading
    base_arr = np.asarray(base, dtype=float)
    if base_arr.ndim == 1:
        return np.column_stack(
            [loading_fn(base_arr, corr=corr, sd=sd, rng=rng) for _ in range(int(rank))]
        )
    if base_arr.ndim != 2 or base_arr.shape[1] != int(rank):
        raise ValueError("base matrix must have one column per rank component.")
    return np.column_stack(
        [
            loading_fn(base_arr[:, k], corr=corr, sd=sd, rng=rng)
            for k in range(int(rank))
        ]
    )


def align_loading_matrix_with_trait(
    loadings: np.ndarray,
    trait: np.ndarray,
    *,
    corr: float,
    sd: float,
    rng: np.random.Generator | None = None,
    correlation_mode: str = "sample_orthogonalized",
) -> np.ndarray:
    L = np.asarray(loadings, dtype=float)
    s = standardize(np.asarray(trait, dtype=float).reshape(-1))
    if L.ndim != 2 or L.shape[0] != s.size:
        raise ValueError("loadings must be n x rank and aligned with the trait.")
    c = float(np.clip(corr, -0.999, 0.999))
    if abs(c) <= 0.0:
        return np.column_stack([float(sd) * standardize(L[:, k]) for k in range(L.shape[1])])
    if correlation_mode not in {"sample_orthogonalized", "z_correlated"}:
        raise ValueError(f"unknown loading correlation mode: {correlation_mode}")
    out = []
    for k in range(L.shape[1]):
        base = standardize(L[:, k])
        if correlation_mode == "z_correlated":
            if rng is None:
                raise ValueError("rng is required for z_correlated trait alignment.")
            innovation = standardize(rng.normal(size=s.size))
            out.append(float(sd) * standardize(c * s + math.sqrt(max(1.0 - c * c, 0.0)) * innovation))
        else:
            out.append(float(sd) * standardize(math.sqrt(max(1.0 - c * c, 0.0)) * base + c * s))
    return np.column_stack(out)


def simulate_common_components(
    *,
    n: int,
    T: int,
    params: FiniteSampleParameters,
    rng: np.random.Generator,
    exposure_profile: np.ndarray | None = None,
    instrument_profile: np.ndarray | None = None,
    split_index: int | None = None,
    shock_split_index: int | None = None,
    fixed_cross_section: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """
    fixed_cross_section: optional dict with keys "Lw", "Ly", "theta_w", "theta_y", "S".
    When supplied, the spatial cross-section components (loading matrices and shock
    loadings) are taken from this dict. Only time-series components (Z, H, F, eps)
    are generated fresh.
    This follows the A-K (2019) design principle of holding the panel's structural
    spatial geometry fixed while varying the time-series draws across replications.
    """
    if int(n) <= 2 or int(T) <= 2:
        raise ValueError("n and T must exceed two.")
    if params.rank <= 0:
        raise ValueError("rank must be positive.")
    if split_index is not None and shock_split_index is not None and int(split_index) != int(shock_split_index):
        raise ValueError("split_index and legacy shock_split_index disagree.")
    resolved_split_index = split_index if split_index is not None else shock_split_index

    if instrument_profile is None:
        Z = float(params.z_sd) * instrument_series(T, params, rng)
    else:
        Z = np.asarray(instrument_profile, dtype=float).reshape(-1)
        if Z.size != int(T):
            raise ValueError("instrument_profile length must match T.")
        if not np.all(np.isfinite(Z)):
            raise ValueError("instrument_profile contains nonfinite values.")
        if float(np.std(Z, ddof=0)) <= 0.0:
            raise ValueError("instrument_profile must have nonzero variation.")
    Z_latent = standardize(Z)
    eta_h_raw = ar1_series(T, params.h_ar, rng)
    if params.h_z_corr_pre is not None or params.h_z_corr_post is not None:
        if params.aggregate_shock_mode != "sample_correlated":
            raise ValueError("split h_z correlations are only supported under sample_correlated aggregate_shock_mode.")
        if params.h_z_corr_pre is None or params.h_z_corr_post is None:
            raise ValueError("h_z_corr_pre and h_z_corr_post must be supplied together.")
        if resolved_split_index is None:
            raise ValueError("split_index is required for split h_z correlation.")
        split = int(resolved_split_index)
        if split <= 2 or split >= int(T) - 2:
            raise ValueError("split_index must leave at least three observations on each side.")
        H_parts = []
        for segment, corr in ((slice(0, split), params.h_z_corr_pre), (slice(split, int(T)), params.h_z_corr_post)):
            z_seg = standardize(Z_latent[segment])
            eta_seg = standardize_orthogonal_to_z(eta_h_raw[segment], z_seg)
            c = float(np.clip(corr, -0.999, 0.999))
            H_parts.append(standardize(c * z_seg + math.sqrt(max(1.0 - c * c, 0.0)) * eta_seg))
        H = standardize(np.concatenate(H_parts))
    else:
        if params.aggregate_shock_mode == "sample_correlated":
            eta_h = standardize_orthogonal_to_z(eta_h_raw, Z_latent)
            H = standardize(
                float(params.h_z_corr) * Z_latent
                + math.sqrt(max(1.0 - float(params.h_z_corr) ** 2, 0.0)) * eta_h
            )
        elif params.aggregate_shock_mode == "z_correlated":
            # Alternative: H drawn as a linear combination of Z and independent
            # noise with exact target correlation.  Active config uses "sample_correlated".
            H = parametric_correlated_series(Z_latent, eta_h_raw, corr=float(params.h_z_corr))
        else:
            raise ValueError(f"unknown aggregate_shock_mode: {params.aggregate_shock_mode}")
    if params.lowrank_factor_mode == "full_sample_orthogonalized":
        F = np.vstack(
            [
                standardize_orthogonal_to_z(ar1_series(T, params.factor_ar, rng), Z_latent)
                for _ in range(params.rank)
            ]
        )
    elif params.lowrank_factor_mode == "split_sample_orthogonalized":
        if resolved_split_index is None:
            raise ValueError("split_index is required for split_sample_orthogonalized low-rank factors.")
        F = np.vstack(
            [
                standardize_split_orthogonal_to_z(ar1_series(T, params.factor_ar, rng), Z_latent, resolved_split_index)
                for _ in range(params.rank)
            ]
        )
    elif params.lowrank_factor_mode == "split_sample_orthogonalized_to_z_h":
        if resolved_split_index is None:
            raise ValueError("split_index is required for split_sample_orthogonalized_to_z_h low-rank factors.")
        F = np.vstack(
            [
                standardize_split_orthogonal_to_z_h(
                    ar1_series(T, params.factor_ar, rng),
                    Z_latent,
                    H,
                    resolved_split_index,
                )
                for _ in range(params.rank)
            ]
        )
    elif params.lowrank_factor_mode == "independent_not_orthogonalized":
        F = np.vstack([ar1_series(T, params.factor_ar, rng) for _ in range(params.rank)])
    elif params.lowrank_factor_mode == "z_correlated":
        F = np.vstack(
            [
                parametric_correlated_series(
                    Z_latent,
                    ar1_series(T, params.factor_ar, rng),
                    corr=float(params.lowrank_factor_z_corr),
                )
                for _ in range(params.rank)
            ]
        )
    elif params.lowrank_factor_mode == "partial_sample_orthogonalized":
        weight = float(np.clip(params.lowrank_factor_orthogonalization_weight, 0.0, 1.0))
        factors = []
        for _ in range(params.rank):
            raw_factor = ar1_series(T, params.factor_ar, rng)
            orth_factor = standardize_orthogonal_to_z(raw_factor, Z_latent)
            factors.append(standardize(weight * orth_factor + (1.0 - weight) * raw_factor))
        F = np.vstack(factors)
    elif params.lowrank_factor_mode == "partial_split_sample_orthogonalized":
        if resolved_split_index is None:
            raise ValueError("split_index is required for partial_split_sample_orthogonalized low-rank factors.")
        weight = float(np.clip(params.lowrank_factor_orthogonalization_weight, 0.0, 1.0))
        factors = []
        for _ in range(params.rank):
            raw_factor = ar1_series(T, params.factor_ar, rng)
            orth_factor = standardize_split_orthogonal_to_z(raw_factor, Z_latent, resolved_split_index)
            factors.append(standardize(weight * orth_factor + (1.0 - weight) * raw_factor))
        F = np.vstack(factors)
    elif params.lowrank_factor_mode == "partial_split_sample_orthogonalized_to_z_h":
        if resolved_split_index is None:
            raise ValueError("split_index is required for partial_split_sample_orthogonalized_to_z_h low-rank factors.")
        weight = float(np.clip(params.lowrank_factor_orthogonalization_weight, 0.0, 1.0))
        factors = []
        for _ in range(params.rank):
            raw_factor = ar1_series(T, params.factor_ar, rng)
            orth_factor = standardize_split_orthogonal_to_z_h(
                raw_factor,
                Z_latent,
                H,
                resolved_split_index,
            )
            factors.append(standardize(weight * orth_factor + (1.0 - weight) * raw_factor))
        F = np.vstack(factors)
    else:
        raise ValueError(f"unknown lowrank_factor_mode: {params.lowrank_factor_mode}")

    if exposure_profile is None:
        pi_base = rng.normal(size=n)
        pi = float(params.pi_sd) * standardize(pi_base)
    else:
        pi_base = np.asarray(exposure_profile, dtype=float).reshape(-1)
        if pi_base.size != int(n):
            raise ValueError("exposure_profile length must match n.")
        if not np.all(np.isfinite(pi_base)):
            raise ValueError("exposure_profile contains nonfinite values.")
        mode = str(params.exposure_profile_mode).lower()
        if mode == "standardized":
            pi = float(params.pi_sd) * standardize(pi_base)
        elif mode == "raw":
            pi = float(params.pi_sd) * pi_base
        else:
            raise ValueError(f"unknown exposure_profile_mode: {params.exposure_profile_mode}")

    # ── cross-section: loadings and shock-loading seeds ────────────────────────
    # This simulator creates preliminary theta_y values.  The active Monte Carlo
    # runner calls helpers.calibrate_shock_loading() after applying the
    # learning/post split; that step replaces the final shock_y panel with a
    # Q_W-based calibrated direction before branch panels are estimated.
    # When fixed_cross_section is supplied (spatial panel geometry held fixed
    # across replications), skip sampling and use the provided matrices directly.
    # Only time-series components (Z, H, F, eps) continue to be drawn fresh.
    if fixed_cross_section is not None:
        Lw = np.asarray(fixed_cross_section["Lw"], dtype=float)
        Ly = np.asarray(fixed_cross_section["Ly"], dtype=float)
        theta_w = np.asarray(fixed_cross_section["theta_w"], dtype=float)
        theta_y = np.asarray(fixed_cross_section["theta_y"], dtype=float)
        S = np.asarray(fixed_cross_section["S"], dtype=float)
        if Lw.shape != (int(n), int(params.rank)) or Ly.shape != (int(n), int(params.rank)):
            raise ValueError("fixed_cross_section loading matrices have wrong shape.")
        if any(v.shape != (int(n),) for v in (theta_w, theta_y, S)):
            raise ValueError("fixed_cross_section shock vectors have wrong shape.")
    elif params.loading_correlation_mode == "sample_orthogonalized":
        loading_corr_mode = "sample_orthogonalized"
        theta_w = correlated_loading(
            pi,
            corr=params.shock_w_pi_corr,
            sd=params.shock_w_sd,
            rng=rng,
        )
        theta_y_pi = correlated_loading(
            pi,
            corr=params.shock_y_pi_corr,
            sd=1.0,
            rng=rng,
        )
        theta_y_w = correlated_loading(
            theta_w,
            corr=params.shock_yw_loading_corr,
            sd=1.0,
            rng=rng,
        )
        S = standardize(0.5 * standardize(theta_y_pi) + 0.5 * standardize(theta_y_w))
        theta_y = float(params.shock_y_sd) * S
        Lw_raw = loading_matrix_from_base(
            pi, rank=params.rank, corr=params.lowrank_w_pi_corr,
            sd=params.lowrank_w_sd, rng=rng, correlation_mode=loading_corr_mode,
        )
        Lw = align_loading_matrix_with_trait(
            Lw_raw, S, corr=params.lowrank_w_shock_trait_corr,
            sd=params.lowrank_w_sd, rng=rng, correlation_mode=loading_corr_mode,
        )
        Ly_raw = loading_matrix_from_base(
            Lw, rank=params.rank, corr=params.lowrank_yw_loading_corr,
            sd=params.lowrank_y_sd, rng=rng, correlation_mode=loading_corr_mode,
        )
        Ly = align_loading_matrix_with_trait(
            Ly_raw, S, corr=params.lowrank_y_shock_trait_corr,
            sd=params.lowrank_y_sd, rng=rng, correlation_mode=loading_corr_mode,
        )
    elif params.loading_correlation_mode == "z_correlated":
        # Alternative loading construction using parametric Z-correlation.
        # Not the active configuration (active: "sample_orthogonalized").
        loading_corr_mode = "z_correlated"
        theta_w = parametric_correlated_loading(
            pi, corr=params.shock_w_pi_corr, sd=params.shock_w_sd, rng=rng,
        )
        if params.shock_y_loading_mode == "pi_correlated_only":
            theta_y = parametric_correlated_loading(
                pi, corr=params.shock_y_pi_corr, sd=params.shock_y_sd, rng=rng,
            )
            S = standardize(theta_y)
        elif params.shock_y_loading_mode == "pi_theta_w_blend":
            theta_y_pi = parametric_correlated_loading(
                pi, corr=params.shock_y_pi_corr, sd=1.0, rng=rng,
            )
            theta_y_w = parametric_correlated_loading(
                theta_w, corr=params.shock_yw_loading_corr, sd=1.0, rng=rng,
            )
            S = standardize(0.5 * standardize(theta_y_pi) + 0.5 * standardize(theta_y_w))
            theta_y = float(params.shock_y_sd) * S
        else:
            raise ValueError(f"unknown shock_y_loading_mode: {params.shock_y_loading_mode}")
        Lw_raw = loading_matrix_from_base(
            pi, rank=params.rank, corr=params.lowrank_w_pi_corr,
            sd=params.lowrank_w_sd, rng=rng, correlation_mode=loading_corr_mode,
        )
        Lw = align_loading_matrix_with_trait(
            Lw_raw, S, corr=params.lowrank_w_shock_trait_corr,
            sd=params.lowrank_w_sd, rng=rng, correlation_mode=loading_corr_mode,
        )
        Ly_raw = loading_matrix_from_base(
            Lw, rank=params.rank, corr=params.lowrank_yw_loading_corr,
            sd=params.lowrank_y_sd, rng=rng, correlation_mode=loading_corr_mode,
        )
        Ly = align_loading_matrix_with_trait(
            Ly_raw, S, corr=params.lowrank_y_shock_trait_corr,
            sd=params.lowrank_y_sd, rng=rng, correlation_mode=loading_corr_mode,
        )
    else:
        raise ValueError(f"unknown loading_correlation_mode: {params.loading_correlation_mode}")
    shock_y = np.outer(theta_y, H)
    if float(params.shock_y_z_aligned_sd) != 0.0:
        shock_y = shock_y + float(params.shock_y_z_aligned_sd) * np.outer(theta_y, Z_latent)
    if float(params.shock_y_learning_marker_sd) != 0.0:
        if resolved_split_index is None:
            raise ValueError("split_index is required for shock_y_learning_marker_sd.")
        split = int(resolved_split_index)
        if split <= 2 or split >= int(T):
            raise ValueError("split_index must define a nonempty learning window.")
        marker_time = standardize_orthogonal_to_z(ar1_series(split, params.h_ar, rng), Z_latent[:split])
        shock_y[:, :split] = shock_y[:, :split] + float(params.shock_y_learning_marker_sd) * np.outer(
            theta_y,
            marker_time,
        )

    eps_corr = float(np.clip(params.eps_yw_corr, -0.999, 0.999))
    eps_w_base = rng.normal(size=(n, T))
    eps_y_base = eps_corr * eps_w_base + math.sqrt(max(1.0 - eps_corr * eps_corr, 0.0)) * rng.normal(size=(n, T))

    return {
        "Z": Z,
        "H": H,
        "F": F,
        "pi": pi,
        "Lw": Lw,
        "Ly": Ly,
        "S": S,
        "theta_w": theta_w,
        "theta_y": theta_y,
        "first_stage": np.outer(pi, Z),
        "lowrank_w": Lw @ F,
        "lowrank_y": Ly @ F,
        "shock_w": np.outer(theta_w, H),
        "shock_y": shock_y,
        "eps_w": float(params.eps_w_sd) * eps_w_base,
        "eps_y": float(params.eps_y_sd) * eps_y_base,
    }


def branch_switches(branch: str) -> tuple[int, int]:
    try:
        return BRANCH_SWITCHES[branch]
    except KeyError as exc:
        raise ValueError(f"unknown branch: {branch}") from exc


def validate_panel_components(components: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = [key for key in REQUIRED_PANEL_COMPONENTS if key not in components]
    if missing:
        raise KeyError(f"missing required DGP components: {missing}")
    arrays = {key: np.asarray(components[key], dtype=float) for key in REQUIRED_PANEL_COMPONENTS}
    shape = arrays["first_stage"].shape
    if len(shape) != 2:
        raise ValueError("first_stage component must be a two-dimensional n x T array.")
    for key, value in arrays.items():
        if value.shape != shape:
            raise ValueError(f"{key} shape {value.shape} does not match first_stage shape {shape}.")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{key} contains nonfinite values.")
    return arrays


def build_branch_panel(
    components: Mapping[str, np.ndarray],
    *,
    branch: str,
    tau_true: float,
) -> tuple[np.ndarray, np.ndarray]:
    gfe_on, shock_on = branch_switches(branch)
    arrays = validate_panel_components(components)
    W = np.array(arrays["first_stage"], copy=True)
    W = W + gfe_on * arrays["lowrank_w"]
    W = W + shock_on * arrays["shock_w"]
    W = W + arrays["eps_w"]
    Y = float(tau_true) * W + arrays["eps_y"]
    Y = Y + gfe_on * arrays["lowrank_y"]
    Y = Y + shock_on * arrays["shock_y"]
    return Y, W
