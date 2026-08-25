from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FIRST_STAGE_TOL = 1e-12


@dataclass(frozen=True)
class SlopeRatio:
    delta: float
    pi: float
    tau: float


def _as_vector(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be nonempty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains nonfinite values.")
    return arr


def aggregate_series(panel: np.ndarray, weights: np.ndarray) -> np.ndarray:
    X = np.asarray(panel, dtype=float)
    w = _as_vector(weights, "weights")
    if X.ndim != 2:
        raise ValueError("panel must be a two-dimensional n x T array.")
    if X.shape[0] != w.size:
        raise ValueError("weight dimension does not match panel rows.")
    return (w @ X) / float(X.shape[0])


def unweighted_state_average(panel: np.ndarray) -> np.ndarray:
    X = np.asarray(panel, dtype=float)
    if X.ndim != 2:
        raise ValueError("panel must be a two-dimensional n x T array.")
    if not np.all(np.isfinite(X)):
        raise ValueError("panel contains nonfinite values.")
    return X.mean(axis=0)


def ols_slope_with_intercept(y: np.ndarray, z: np.ndarray) -> float:
    yv = _as_vector(y, "y")
    zv = _as_vector(z, "z")
    if yv.size != zv.size:
        raise ValueError("y and z must have the same length.")
    zc = zv - float(zv.mean())
    denom = float(zc @ zc)
    if denom <= 0:
        raise ValueError("z has zero variance.")
    yc = yv - float(yv.mean())
    return float((zc @ yc) / denom)


def slope_ratio(y: np.ndarray, w: np.ndarray, z: np.ndarray) -> SlopeRatio:
    delta = ols_slope_with_intercept(y, z)
    pi = ols_slope_with_intercept(w, z)
    if abs(pi) <= FIRST_STAGE_TOL:
        raise ValueError("first-stage denominator is zero or too close to zero.")
    tau = float(delta / pi)
    return SlopeRatio(delta=delta, pi=pi, tau=tau)


def weighted_slope_ratio(y_panel: np.ndarray, w_panel: np.ndarray, z: np.ndarray, weights: np.ndarray) -> SlopeRatio:
    Y = np.asarray(y_panel, dtype=float)
    W = np.asarray(w_panel, dtype=float)
    if Y.shape != W.shape or Y.ndim != 2:
        raise ValueError("y_panel and w_panel must be aligned n x T arrays.")
    y_series = aggregate_series(Y, weights)
    w_series = aggregate_series(W, weights)
    return slope_ratio(y_series, w_series, z)


def tsls_estimate(y_panel: np.ndarray, w_panel: np.ndarray, z: np.ndarray) -> SlopeRatio:
    Y = np.asarray(y_panel, dtype=float)
    W = np.asarray(w_panel, dtype=float)
    if Y.shape != W.shape or Y.ndim != 2:
        raise ValueError("y_panel and w_panel must be aligned n x T arrays.")
    D = unit_level_slope_vector(W, z)
    return tsls_estimate_from_d(Y, W, z, D)


def unit_level_slope_vector(panel: np.ndarray, z: np.ndarray) -> np.ndarray:
    X = np.asarray(panel, dtype=float)
    zv = _as_vector(z, "z")
    if X.ndim != 2:
        raise ValueError("panel must be a two-dimensional n x T array.")
    if X.shape[1] != zv.size:
        raise ValueError("panel time dimension does not match z.")
    zc = zv - float(zv.mean())
    denom = float(zc @ zc)
    if denom <= 0:
        raise ValueError("z has zero variance.")
    Xc = X - X.mean(axis=1, keepdims=True)
    return (Xc @ zc) / denom


def tsls_weights(first_stage_slopes: np.ndarray) -> np.ndarray:
    D = _as_vector(first_stage_slopes, "first_stage_slopes")
    Dc = D - float(D.mean())
    denom = float(np.mean(Dc * D))
    if abs(denom) <= FIRST_STAGE_TOL:
        raise ValueError("first-stage exposure vector has zero usable variation.")
    return Dc / denom


def raw_centered_tsls_weights(first_stage_slopes: np.ndarray) -> np.ndarray:
    D = _as_vector(first_stage_slopes, "first_stage_slopes")
    Dc = D - float(D.mean())
    if float(Dc @ Dc) <= FIRST_STAGE_TOL:
        raise ValueError("first-stage exposure vector has zero usable variation.")
    return Dc


def tsls_estimate_from_d(
    y_panel: np.ndarray,
    w_panel: np.ndarray,
    z: np.ndarray,
    first_stage_slopes: np.ndarray,
) -> SlopeRatio:
    weights = tsls_weights(first_stage_slopes)
    return weighted_slope_ratio(y_panel, w_panel, z, weights)


def check_weight_constraints(weights: np.ndarray, first_stage_slopes: np.ndarray) -> dict[str, float]:
    w = _as_vector(weights, "weights")
    D = _as_vector(first_stage_slopes, "first_stage_slopes")
    if w.size != D.size:
        raise ValueError("weights and first-stage slopes must have the same length.")
    return {
        "mean_weight": float(np.mean(w)),
        "mean_weight_times_D": float(np.mean(w * D)),
    }


def safe_weight_correlation(a: np.ndarray, b: np.ndarray) -> float:
    av = _as_vector(a, "a")
    bv = _as_vector(b, "b")
    if av.size != bv.size:
        raise ValueError("weight vectors must have the same length.")
    ac = av - float(av.mean())
    bc = bv - float(bv.mean())
    denom = float((ac @ ac) * (bc @ bc))
    return float((ac @ bc) / math.sqrt(denom)) if denom > 0 else float("nan")
