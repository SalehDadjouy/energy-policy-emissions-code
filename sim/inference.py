"""
Inference methods for the aggregate IV slope-ratio estimator.

Active functions
----------------
  centered_slope_ratio_moments     core IV moments (delta, pi, tau, SZZ)
  hac_delta_se                     Newey-West delta-method SE  [run.py, empirical.py]
  ar_style_test                    Anderson-Rubin-style HAC orthogonality inversion  [run.py]
  arima_residual_covariance_slope_se  AR(p) residual-covariance SE  [helpers.py]
  raw_z_arp_design_based_se        AR(p) on Z, closed-form Toeplitz SE  [run.py]

Additional helpers
------------------
The remaining functions are documented sensitivity or diagnostic helpers.
They are not used by the active final simulation unless imported explicitly
by a runner.
"""
from __future__ import annotations

import math
import hashlib
import warnings
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


FIRST_STAGE_TOL = 1e-12
_ARIMA_FIT_CACHE_MAX = 20000
_ARIMA_FIT_CACHE: dict[tuple[int, str, tuple[int, int, int]], object] = {}


def _statsmodels_arima(*args, **kwargs):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as exc:  # pragma: no cover - exercised only if dependency is missing.
        raise ImportError("statsmodels is required for Algorithm-2 ARIMA parity inference.") from exc
    return ARIMA(*args, **kwargs)


def _fit_statsmodels_arima_cached(z_fit: np.ndarray, order: tuple[int, int, int]) -> object:
    zv = _as_vector(z_fit, "z_fit")
    ord_tuple = tuple(int(v) for v in order)
    key = (
        int(zv.size),
        hashlib.sha256(np.ascontiguousarray(zv).view(np.uint8)).hexdigest(),
        ord_tuple,
    )
    cached = _ARIMA_FIT_CACHE.get(key)
    if cached is not None:
        return cached
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _statsmodels_arima(
            zv,
            order=ord_tuple,
            trend="c",
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit()
    if len(_ARIMA_FIT_CACHE) >= _ARIMA_FIT_CACHE_MAX:
        _ARIMA_FIT_CACHE.clear()
    _ARIMA_FIT_CACHE[key] = res
    return res


def _standardize_for_arima_fit(z: np.ndarray) -> tuple[np.ndarray, float, float]:
    zv = _as_vector(z, "z_process")
    mean = float(np.mean(zv))
    scale = float(np.std(zv, ddof=0))
    if not math.isfinite(scale) or scale <= FIRST_STAGE_TOL:
        raise ValueError("ARIMA process has nonpositive scale.")
    return (zv - mean) / scale, mean, scale


@dataclass(frozen=True)
class SlopeRatioMoments:
    delta: float
    pi: float
    tau: float
    szz: float
    t: int


@dataclass(frozen=True)
class InferenceResult:
    se: float
    tau: float
    pi: float
    scale_multiplier: float
    diagnostics: dict[str, float | int | str]


def _as_vector(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be nonempty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains nonfinite values.")
    return arr


def centered_slope_ratio_moments(y: np.ndarray, w: np.ndarray, z: np.ndarray) -> SlopeRatioMoments:
    yv = _as_vector(y, "y")
    wv = _as_vector(w, "w")
    zv = _as_vector(z, "z")
    if not (yv.size == wv.size == zv.size):
        raise ValueError("y, w, and z must have the same length.")
    zc = zv - float(zv.mean())
    szz = float(np.mean(zc * zc))
    if szz <= FIRST_STAGE_TOL:
        raise ValueError("z has zero usable variance.")
    delta = float(np.mean((zc / szz) * yv))
    pi = float(np.mean((zc / szz) * wv))
    if abs(pi) <= FIRST_STAGE_TOL:
        raise ValueError("first-stage denominator is zero or too close to zero.")
    return SlopeRatioMoments(delta=delta, pi=pi, tau=float(delta / pi), szz=szz, t=int(zv.size))


def _hac_long_run_covariance(x: np.ndarray, maxlags: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError("x must be T x k with at least two rows.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x contains nonfinite values.")
    t = arr.shape[0]
    lags = max(0, min(int(maxlags), t - 1))
    xc = arr - np.mean(arr, axis=0, keepdims=True)
    omega = (xc.T @ xc) / float(t)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / float(lags + 1)
        gamma = (xc[lag:].T @ xc[:-lag]) / float(t)
        omega = omega + weight * (gamma + gamma.T)
    return 0.5 * (omega + omega.T)


def _ar1_long_run_variance(score: np.ndarray) -> tuple[float, dict[str, float | int | str]]:
    x = _as_vector(score, "score")
    if x.size < 6:
        raise ValueError("AR(1) score fit requires at least six observations.")
    lag = x[:-1]
    cur = x[1:]
    denom = float(lag @ lag)
    if denom <= FIRST_STAGE_TOL:
        lrv = float(np.mean(x * x))
        return lrv, {
            "lrv_rule": "score_var_fallback",
            "arma_p": 0,
            "arma_q": 0,
            "ar1_phi": 0.0,
            "ar1_sigma2": lrv,
            "arma_fallback": 1,
        }
    phi = float((lag @ cur) / denom)
    if not math.isfinite(phi):
        raise ValueError("AR(1) score fit produced nonfinite phi.")
    phi_clipped = float(np.clip(phi, -0.98, 0.98))
    resid = cur - phi_clipped * lag
    sigma2 = float(np.mean(resid * resid))
    if not math.isfinite(sigma2) or sigma2 <= 0.0:
        sigma2 = float(np.mean(x * x))
        phi_clipped = 0.0
        fallback = 1
    else:
        fallback = int(abs(phi_clipped - phi) > 1e-12)
    lrv = sigma2 / ((1.0 - phi_clipped) ** 2)
    return float(lrv), {
        "lrv_rule": "fixed_ar1_score",
        "arma_p": 1,
        "arma_q": 0,
        "ar1_phi": phi_clipped,
        "ar1_sigma2": sigma2,
        "arma_fallback": fallback,
    }


def _default_ar_sieve_max_order(t: int) -> int:
    if int(t) < 8:
        return 0
    return max(1, min(int(math.floor(float(t) ** (1.0 / 3.0))), int(t) // 4))


def _ar_stationary(phi: np.ndarray) -> bool:
    coeff = np.asarray(phi, dtype=float).reshape(-1)
    if coeff.size == 0:
        return True
    # statsmodels and numpy use roots of phi_p z^p + ... + phi_1 z - 1 = 0
    # for the stationarity check of 1 - phi_1 L - ... - phi_p L^p.
    roots = np.roots(np.r_[-coeff[::-1], 1.0])
    return bool(np.all(np.abs(roots) > 1.0 + 1e-8))


def _ar_inverse_root_min_modulus(phi: np.ndarray) -> float:
    coeff = np.asarray(phi, dtype=float).reshape(-1)
    if coeff.size == 0:
        return float("inf")
    roots = np.roots(np.r_[-coeff[::-1], 1.0])
    if roots.size == 0:
        return float("inf")
    return float(np.min(np.abs(roots)))


def _ma_inverse_root_min_modulus(theta: np.ndarray) -> float:
    coeff = np.asarray(theta, dtype=float).reshape(-1)
    if coeff.size == 0:
        return float("inf")
    roots = np.roots(np.r_[coeff[::-1], 1.0])
    if roots.size == 0:
        return float("inf")
    return float(np.min(np.abs(roots)))


def _ma_invertible(theta: np.ndarray) -> bool:
    return bool(_ma_inverse_root_min_modulus(theta) > 1.0 + 1e-8)


def _aicc_from_fit(arima_res: object, nobs: int, k_params: int) -> float:
    aic = float(getattr(arima_res, "aic"))
    n = int(nobs)
    k = int(k_params)
    if n <= k + 1:
        return float("inf")
    return float(aic + (2.0 * k * (k + 1)) / float(n - k - 1))


def _fit_ar_rootguard(
    z_fit: np.ndarray,
    *,
    max_p: int = 2,
    root_guard: float = 1.05,
) -> tuple[object, tuple[int, int, int], float, float]:
    zv = _as_vector(z_fit, "z_fit")
    best: tuple[float, object, tuple[int, int, int], float] | None = None
    for p in range(0, int(max_p) + 1):
        order = (p, 0, 0)
        try:
            res = _fit_statsmodels_arima_cached(zv, order)
            phi = np.asarray(getattr(res, "arparams", np.array([], dtype=float)), dtype=float).reshape(-1)
            min_root = _ar_inverse_root_min_modulus(phi)
            if p > 0 and (not _ar_stationary(phi) or min_root < float(root_guard)):
                continue
            k_params = int(len(np.asarray(getattr(res, "params", []), dtype=float)))
            aicc = _aicc_from_fit(res, nobs=int(zv.size), k_params=k_params)
            if not math.isfinite(aicc):
                continue
        except Exception:
            continue
        if best is None or aicc < best[0]:
            best = (aicc, res, order, min_root)
    if best is None:
        raise RuntimeError("no stationary root-guard AR candidate was available for Algorithm-2 Z process.")
    aicc, res, order, min_root = best
    return res, order, float(aicc), float(min_root)


def _fit_arma_rootguard(
    z_fit: np.ndarray,
    *,
    max_p: int = 2,
    max_q: int = 2,
    root_guard: float = 1.05,
) -> tuple[object, tuple[int, int, int], float, float, float]:
    zv = _as_vector(z_fit, "z_fit")
    best: tuple[float, object, tuple[int, int, int], float, float] | None = None
    for p in range(0, int(max_p) + 1):
        for q in range(0, int(max_q) + 1):
            order = (p, 0, q)
            try:
                res = _fit_statsmodels_arima_cached(zv, order)
                phi = np.asarray(getattr(res, "arparams", np.array([], dtype=float)), dtype=float).reshape(-1)
                theta = np.asarray(getattr(res, "maparams", np.array([], dtype=float)), dtype=float).reshape(-1)
                min_ar_root = _ar_inverse_root_min_modulus(phi)
                min_ma_root = _ma_inverse_root_min_modulus(theta)
                if p > 0 and (not _ar_stationary(phi) or min_ar_root < float(root_guard)):
                    continue
                if q > 0 and (not _ma_invertible(theta) or min_ma_root < float(root_guard)):
                    continue
                k_params = int(len(np.asarray(getattr(res, "params", []), dtype=float)))
                aicc = _aicc_from_fit(res, nobs=int(zv.size), k_params=k_params)
                if not math.isfinite(aicc):
                    continue
            except Exception:
                continue
            if best is None or aicc < best[0]:
                best = (aicc, res, order, min_ar_root, min_ma_root)
    if best is None:
        raise RuntimeError("no stationary/invertible root-guard ARMA candidate was available for Algorithm-2 Z process.")
    aicc, res, order, min_ar_root, min_ma_root = best
    return res, order, float(aicc), float(min_ar_root), float(min_ma_root)


def _ar_sieve_long_run_variance(
    score: np.ndarray,
    *,
    max_order: int | None = None,
) -> tuple[float, dict[str, float | int | str]]:
    x = _as_vector(score, "score")
    t = int(x.size)
    pmax = _default_ar_sieve_max_order(t) if max_order is None else int(max_order)
    pmax = max(0, min(pmax, max(t - 3, 0)))
    start = int(pmax)
    if t - start < 3:
        pmax = 0
        start = 0
    common_y = x[start:]
    n_common = int(common_y.size)
    candidates: list[dict[str, float | int]] = []
    best: dict[str, float | int | np.ndarray] | None = None
    fallback_count = 0
    for p in range(pmax + 1):
        if p == 0:
            resid = common_y - float(np.mean(common_y))
            n_eff = n_common
            sigma2 = float(np.mean(resid * resid))
            phi = np.array([], dtype=float)
            k = 1
        else:
            y = common_y
            X = np.column_stack(
                [np.ones(n_common)] + [x[start - j - 1 : t - j - 1] for j in range(p)]
            )
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            n_eff = n_common
            sigma2 = float(np.mean(resid * resid))
            phi = np.asarray(beta[1:], dtype=float)
            k = p + 1
        if not math.isfinite(sigma2) or sigma2 <= FIRST_STAGE_TOL:
            fallback_count += 1
            continue
        if not _ar_stationary(phi):
            fallback_count += 1
            continue
        denom = float(1.0 - np.sum(phi))
        if not math.isfinite(denom) or abs(denom) <= 1e-8:
            fallback_count += 1
            continue
        bic = math.log(sigma2) + float(k) * math.log(float(n_eff)) / float(n_eff)
        lrv = sigma2 / (denom * denom)
        candidate = {
            "p": int(p),
            "sigma2": float(sigma2),
            "bic": float(bic),
            "lrv": float(lrv),
            "n_eff": int(n_eff),
            "k": int(k),
        }
        candidates.append(candidate)
        if best is None or float(candidate["bic"]) < float(best["bic"]):
            best = {**candidate, "phi": phi}
    if best is None:
        lrv = float(np.mean((x - float(np.mean(x))) ** 2))
        return lrv, {
            "lrv_rule": "ar_sieve_bic_score_var_fallback",
            "arma_p": 0,
            "arma_q": 0,
            "ar_sieve_max_order": int(pmax),
            "ar_sieve_common_start": int(start),
            "ar_sieve_common_n": int(n_common),
            "ar_sieve_candidate_count": 0,
            "ar_sieve_rejected_count": int(fallback_count),
            "arma_fallback": 1,
        }
    return float(best["lrv"]), {
        "lrv_rule": "ar_sieve_bic_score",
        "arma_p": int(best["p"]),
        "arma_q": 0,
        "ar_sieve_max_order": int(pmax),
        "ar_sieve_common_start": int(start),
        "ar_sieve_common_n": int(n_common),
        "ar_sieve_candidate_count": int(len(candidates)),
        "ar_sieve_rejected_count": int(fallback_count),
        "ar_sieve_bic": float(best["bic"]),
        "ar_sieve_sigma2": float(best["sigma2"]),
        "arma_fallback": 0,
    }


def _score_long_run_variance(
    score: np.ndarray,
    *,
    ar_order: int | str,
    t: int,
) -> tuple[float, dict[str, float | int | str]]:
    ar_order_key = str(ar_order).lower()
    if ar_order_key in {"finite_t_guard", "finite-t-guard", "finite_t", "finite-t"}:
        if int(t) <= 10:
            lrv = float(np.mean(score * score))
            return lrv, {
                "lrv_rule": "finite_t_guard_arima_0_0_0_score",
                "arma_p": 0,
                "arma_q": 0,
                "finite_t_guard_active": 1,
                "finite_t_guard_reason": "T1_leq_10_no_data_dependent_ar_order_selection",
                "arma_fallback": 0,
            }
        lrv, lrv_diag = _ar_sieve_long_run_variance(score)
        lrv_diag["finite_t_guard_active"] = 0
        lrv_diag["finite_t_guard_reason"] = "T1_gt_10_ar_sieve_allowed"
        return lrv, lrv_diag
    if ar_order_key in {"bic", "sieve", "ar_sieve", "ar-sieve"}:
        return _ar_sieve_long_run_variance(score)
    if int(ar_order) == 0:
        return float(np.mean(score * score)), {
            "lrv_rule": "arma_0_0_smoke",
            "arma_p": 0,
            "arma_q": 0,
            "arma_fallback": 0,
        }
    if int(ar_order) == 1:
        return _ar1_long_run_variance(score)
    raise ValueError("clean ARIMA score helper supports ar_order 0, 1, or BIC-selected AR-sieve.")


def _structural_residual_after_const_z(
    y: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    tau: float,
) -> np.ndarray:
    yv = _as_vector(y, "y")
    wv = _as_vector(w, "w")
    zv = _as_vector(z, "z")
    if not (yv.size == wv.size == zv.size):
        raise ValueError("y, w, and z must have the same length.")
    residual = yv - float(tau) * wv
    x = np.column_stack([np.ones(zv.size), zv])
    beta = np.linalg.lstsq(x, residual, rcond=None)[0]
    return residual - x @ beta


def _ar_autocovariances(phi: np.ndarray, sigma2: float, max_lag: int) -> np.ndarray:
    coeff = np.asarray(phi, dtype=float).reshape(-1)
    lag_count = int(max_lag)
    if lag_count < 0:
        raise ValueError("max_lag must be nonnegative.")
    if coeff.size == 0:
        gamma = np.zeros(lag_count + 1, dtype=float)
        gamma[0] = float(sigma2)
        return gamma
    if not _ar_stationary(coeff):
        raise ValueError("AR coefficients are not stationary.")
    p = int(coeff.size)
    dim = p + 1
    a = np.zeros((dim, dim), dtype=float)
    b = np.zeros(dim, dtype=float)
    # k = 0: gamma_0 - sum_j phi_j gamma_j = sigma2.
    a[0, 0] = 1.0
    for j in range(1, p + 1):
        a[0, j] -= coeff[j - 1]
    b[0] = float(sigma2)
    # k = 1, ..., p: gamma_k - sum_j phi_j gamma_|k-j| = 0.
    for k in range(1, p + 1):
        a[k, k] = 1.0
        for j in range(1, p + 1):
            a[k, abs(k - j)] -= coeff[j - 1]
    base = np.linalg.solve(a, b)
    gamma = np.zeros(lag_count + 1, dtype=float)
    gamma[: min(dim, lag_count + 1)] = base[: min(dim, lag_count + 1)]
    for k in range(dim, lag_count + 1):
        gamma[k] = float(sum(coeff[j - 1] * gamma[k - j] for j in range(1, p + 1)))
    return gamma


def _arma_autocovariances(phi: np.ndarray, theta: np.ndarray, sigma2: float, max_lag: int) -> np.ndarray:
    ar = np.asarray(phi, dtype=float).reshape(-1)
    ma = np.asarray(theta, dtype=float).reshape(-1)
    lag_count = int(max_lag)
    if lag_count < 0:
        raise ValueError("max_lag must be nonnegative.")
    if ar.size == 0:
        psi = np.zeros(max(lag_count + ma.size + 1, 1), dtype=float)
        psi[0] = 1.0
        for j, value in enumerate(ma, start=1):
            if j < psi.size:
                psi[j] = float(value)
        gamma = np.zeros(lag_count + 1, dtype=float)
        for lag in range(lag_count + 1):
            gamma[lag] = float(sigma2) * float(np.dot(psi[: psi.size - lag], psi[lag:]))
        return gamma
    if not _ar_stationary(ar):
        raise ValueError("AR coefficients are not stationary.")
    if ma.size and not _ma_invertible(ma):
        raise ValueError("MA coefficients are not invertible.")
    impulse_count = max(2000, 200 * (lag_count + 1))
    psi = np.zeros(impulse_count + lag_count + 1, dtype=float)
    psi[0] = 1.0
    for k in range(1, psi.size):
        value = ma[k - 1] if k <= ma.size else 0.0
        for j, ar_value in enumerate(ar, start=1):
            if k - j >= 0:
                value += float(ar_value) * psi[k - j]
        psi[k] = float(value)
    gamma = np.zeros(lag_count + 1, dtype=float)
    for lag in range(lag_count + 1):
        gamma[lag] = float(sigma2) * float(np.dot(psi[: impulse_count], psi[lag : lag + impulse_count]))
    return gamma


def _toeplitz_from_autocovariances(gamma: np.ndarray, t: int) -> np.ndarray:
    g = np.asarray(gamma, dtype=float).reshape(-1)
    if g.size < int(t):
        raise ValueError("not enough autocovariances for Toeplitz covariance.")
    idx = np.abs(np.subtract.outer(np.arange(int(t)), np.arange(int(t))))
    return g[idx]


def _sigma2_from_statsmodels_arima(arima_res) -> float:
    for attr in ("sigma2", "sigma2_"):
        if hasattr(arima_res, attr):
            value = float(getattr(arima_res, attr))
            if math.isfinite(value) and value > 0.0:
                return value
    if hasattr(arima_res, "param_names") and hasattr(arima_res, "params"):
        names = list(getattr(arima_res, "param_names"))
        if "sigma2" in names:
            idx = names.index("sigma2")
            value = float(np.asarray(arima_res.params, dtype=float)[idx])
            if math.isfinite(value) and value > 0.0:
                return value
    if hasattr(arima_res, "resid"):
        resid = np.asarray(arima_res.resid, dtype=float).reshape(-1)
        resid = resid[np.isfinite(resid)]
        if resid.size > 1:
            value = float(np.mean(resid * resid))
            if math.isfinite(value) and value > 0.0:
                return value
    raise RuntimeError("Could not extract a positive ARIMA innovation variance.")


def _fit_ar_residual_covariance(
    residual: np.ndarray,
    *,
    ar_order: int | str,
    max_order: int | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    u = _as_vector(residual, "residual")
    t = int(u.size)
    order_key = str(ar_order).lower()
    if order_key in {"finite_t_guard", "finite-t-guard", "finite_t", "finite-t"}:
        if t <= 10:
            omega, diag = _fit_ar_residual_covariance(u, ar_order=0, max_order=max_order)
            diag["finite_t_guard_active"] = 1
            diag["finite_t_guard_reason"] = "T1_leq_10_no_data_dependent_ar_order_selection"
            return omega, diag
        omega, diag = _fit_ar_residual_covariance(u, ar_order="bic", max_order=max_order)
        diag["finite_t_guard_active"] = 0
        diag["finite_t_guard_reason"] = "T1_gt_10_ar_sieve_allowed"
        return omega, diag
    if order_key in {"bic", "sieve", "ar_sieve", "ar-sieve"}:
        pmax = _default_ar_sieve_max_order(t) if max_order is None else int(max_order)
        pmax = max(0, min(pmax, max(t - 3, 0)))
        best: dict[str, float | int | np.ndarray] | None = None
        candidate_count = 0
        rejected_count = 0
        for p in range(pmax + 1):
            try:
                if p == 0:
                    df = max(t - 2, 1)
                    sigma2 = float(u @ u) / float(df)
                    phi = np.array([], dtype=float)
                    n_eff = t
                    k = 1
                else:
                    if t - p < max(p + 2, 3):
                        rejected_count += 1
                        continue
                    y_ar = u[p:]
                    x_ar = np.column_stack([u[p - j - 1 : t - j - 1] for j in range(p)])
                    phi = np.linalg.lstsq(x_ar, y_ar, rcond=None)[0]
                    if not _ar_stationary(phi):
                        rejected_count += 1
                        continue
                    resid_ar = y_ar - x_ar @ phi
                    n_eff = int(y_ar.size)
                    df = max(n_eff - p, 1)
                    sigma2 = float(resid_ar @ resid_ar) / float(df)
                    k = p + 1
                if not math.isfinite(sigma2) or sigma2 <= FIRST_STAGE_TOL:
                    rejected_count += 1
                    continue
                bic = math.log(sigma2) + float(k) * math.log(float(n_eff)) / float(n_eff)
                candidate_count += 1
                if best is None or bic < float(best["bic"]):
                    best = {
                        "p": int(p),
                        "phi": phi,
                        "sigma2": float(sigma2),
                        "bic": float(bic),
                        "n_eff": int(n_eff),
                        "df": int(df),
                    }
            except (np.linalg.LinAlgError, ValueError):
                rejected_count += 1
        if best is None:
            omega, diag = _fit_ar_residual_covariance(u, ar_order=0, max_order=max_order)
            diag["residual_ar_fallback"] = 1
            diag["residual_ar_rejected_count"] = int(rejected_count)
            return omega, diag
        gamma = _ar_autocovariances(np.asarray(best["phi"], dtype=float), float(best["sigma2"]), t - 1)
        omega = _toeplitz_from_autocovariances(gamma, t)
        return omega, {
            "lrv_rule": "ar_residual_covariance_bic_sieve",
            "residual_covariance_rule": "stationary_ar_toeplitz",
            "arma_p": int(best["p"]),
            "arma_q": 0,
            "ar_sieve_max_order": int(pmax),
            "ar_sieve_candidate_count": int(candidate_count),
            "ar_sieve_rejected_count": int(rejected_count),
            "ar_sieve_bic": float(best["bic"]),
            "ar_sieve_sigma2": float(best["sigma2"]),
            "ar_sieve_n_eff": int(best["n_eff"]),
            "residual_df": int(best["df"]),
            "arma_fallback": 0,
        }
    p = int(ar_order)
    if p == 0:
        df = max(t - 2, 1)
        sigma2 = float(u @ u) / float(df)
        if not math.isfinite(sigma2) or sigma2 <= FIRST_STAGE_TOL:
            raise ValueError("AR(0) residual covariance has nonpositive variance.")
        return np.eye(t, dtype=float) * sigma2, {
            "lrv_rule": "ar0_residual_covariance_model_based_no_hc",
            "residual_covariance_rule": "sigma2_identity",
            "arma_p": 0,
            "arma_q": 0,
            "arma_fallback": 0,
            "residual_df": int(df),
            "ar_sieve_sigma2": float(sigma2),
        }
    if p < 0:
        raise ValueError("AR order must be nonnegative.")
    if t - p < max(p + 2, 3):
        raise ValueError("not enough observations to fit requested AR residual covariance.")
    y_ar = u[p:]
    x_ar = np.column_stack([u[p - j - 1 : t - j - 1] for j in range(p)])
    phi = np.linalg.lstsq(x_ar, y_ar, rcond=None)[0]
    if not _ar_stationary(phi):
        raise ValueError("requested AR residual covariance is nonstationary.")
    resid_ar = y_ar - x_ar @ phi
    df = max(int(y_ar.size) - p, 1)
    sigma2 = float(resid_ar @ resid_ar) / float(df)
    if not math.isfinite(sigma2) or sigma2 <= FIRST_STAGE_TOL:
        raise ValueError("requested AR residual covariance has nonpositive innovation variance.")
    gamma = _ar_autocovariances(phi, sigma2, t - 1)
    return _toeplitz_from_autocovariances(gamma, t), {
        "lrv_rule": "fixed_ar_residual_covariance",
        "residual_covariance_rule": "stationary_ar_toeplitz",
        "arma_p": int(p),
        "arma_q": 0,
        "arma_fallback": 0,
        "residual_df": int(df),
        "ar_sieve_sigma2": float(sigma2),
    }


def arima_residual_covariance_slope_se(
    y: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    *,
    ar_order: int | str = "finite_t_guard",
    max_order: int | None = None,
) -> InferenceResult:
    """AR residual-covariance slope-ratio standard error.

    This path fits the time-series model to the post-window structural
    residual after removing the intercept and instrument projection.  The
    estimated residual covariance is then mapped through the intercept-included
    IV slope-ratio formula.  The AR(0) case is therefore the homoskedastic
    no-HC slope-ratio standard error.
    """
    moments = centered_slope_ratio_moments(y, w, z)
    zv = _as_vector(z, "z")
    zc = zv - float(zv.mean())
    sum_z2 = float(zc @ zc)
    if sum_z2 <= FIRST_STAGE_TOL:
        raise ValueError("z has zero usable variance.")
    residual = _structural_residual_after_const_z(y, w, z, moments.tau)
    omega, cov_diag = _fit_ar_residual_covariance(
        residual,
        ar_order=ar_order,
        max_order=max_order,
    )
    slope_map = zc / sum_z2
    var_slope = float(slope_map @ omega @ slope_map)
    if not math.isfinite(var_slope) or var_slope <= 0.0:
        raise ValueError("AR residual-covariance slope variance is nonpositive.")
    se = math.sqrt(var_slope) / abs(moments.pi)
    diagnostics = {
        "method": "ARIMA",
        "score_rule": "residual_covariance_slope_map",
        "residual_model_object": "structural_residual_after_const_and_Z",
        "first_stage_division_count": 1,
        "residual_mean": float(np.mean(residual)),
        "residual_z_projection": float(residual @ zc),
        "slope_variance": float(var_slope),
        "T1": moments.t,
    }
    diagnostics.update(cov_diag)
    return InferenceResult(
        se=float(se),
        tau=moments.tau,
        pi=moments.pi,
        scale_multiplier=1.0,
        diagnostics=diagnostics,
    )


def raw_z_ar1_design_based_se(
    y: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    *,
    z_process: np.ndarray,
    varz_mode: str = "window",
) -> InferenceResult:
    """Algorithm-2 raw-Z AR(1) design-based slope-ratio SE.

    This sensitivity helper implements the raw-instrument Algorithm-2
    structure for an AR(1) law: estimate the slope-ratio residual
    ``u_t = Y_t - tau_hat W_t``, model the raw instrument process as AR(1)
    with an intercept, center the simulated post-learning instrument paths,
    and map their implied covariance through ``u'(Z - mean(Z))``.  The active
    simulation runner uses ``raw_z_arp_design_based_se`` with AR order 2.

    The implementation uses the closed-form AR(1) covariance for feasible
    Monte Carlo coverage runs.
    """
    moments = centered_slope_ratio_moments(y, w, z)
    yv = _as_vector(y, "y")
    wv = _as_vector(w, "w")
    zv = _as_vector(z, "z")
    zp = _as_vector(z_process, "z_process")
    zp_fit, _zp_mean, zp_scale = _standardize_for_arima_fit(zp)
    if not (yv.size == wv.size == zv.size):
        raise ValueError("y, w, and z must have the same length.")
    if zp.size < 4:
        raise ValueError("z_process must have at least four observations for AR(1).")
    lag = zp_fit[:-1]
    cur = zp_fit[1:]
    X = np.column_stack([np.ones(lag.size), lag])
    beta = np.linalg.lstsq(X, cur, rcond=None)[0]
    phi_raw = float(beta[1])
    phi = float(np.clip(phi_raw, -0.98, 0.98))
    resid = cur - X @ np.array([float(beta[0]), phi])
    df = max(int(resid.size) - 2, 1)
    sigma2 = float(resid @ resid) / float(df)
    if not math.isfinite(sigma2) or sigma2 <= FIRST_STAGE_TOL:
        raise ValueError("raw-Z AR(1) innovation variance is nonpositive.")
    gamma0 = sigma2 / max(1.0 - phi * phi, FIRST_STAGE_TOL)
    t = int(zv.size)
    gamma = gamma0 * (phi ** np.arange(t, dtype=float))
    omega = _toeplitz_from_autocovariances(gamma, t)
    center = np.eye(t, dtype=float) - np.ones((t, t), dtype=float) / float(t)
    omega_centered = (zp_scale * zp_scale) * (center @ omega @ center)
    u = yv - moments.tau * wv
    varz_key = str(varz_mode).lower()
    if varz_key == "full":
        z_for_var = zp
    elif varz_key == "window":
        z_for_var = zv
    else:
        raise ValueError("varz_mode must be 'full' or 'window'.")
    varz = float(np.mean((z_for_var - float(np.mean(z_for_var))) ** 2))
    if varz <= FIRST_STAGE_TOL:
        raise ValueError("instrument variance denominator is nonpositive.")
    sd_dot = math.sqrt(max(float(u @ omega_centered @ u), 0.0))
    se = sd_dot / (abs(moments.pi) * varz * float(t))
    if not math.isfinite(se) or se <= 0.0:
        raise ValueError("raw-Z AR(1) design-based SE is nonpositive.")
    return InferenceResult(
        se=float(se),
        tau=moments.tau,
        pi=moments.pi,
        scale_multiplier=1.0,
        diagnostics={
            "method": "ARIMA-Z",
            "score_rule": "algorithm2_centered_post_z_slope_score_fast_approximation",
            "z_process_rule": "raw_Z_OLS_AR1_intercept_centered_post_covariance",
            "arima_fit_scale_rule": "standardized_Z_mapped_to_raw_units",
            "arima_fit_raw_scale": float(zp_scale),
            "first_stage_division_count": 1,
            "arma_p": 1,
            "arma_q": 0,
            "arma_fallback": int(abs(phi - phi_raw) > 1e-12),
            "ar1_phi": float(phi),
            "ar1_phi_raw": float(phi_raw),
            "ar1_sigma2": float(sigma2),
            "sd_dot": float(sd_dot),
            "varZ_denom": float(varz),
            "varZ_mode": varz_key,
            "simulated_z_centered_within_post_window": 1,
            "residual_df": int(df),
            "T1": t,
        },
    )


def raw_z_arp_design_based_se(
    y: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    *,
    z_process: np.ndarray,
    ar_order: int = 2,
    varz_mode: str = "window",
) -> InferenceResult:
    """AR(p) design-based slope-ratio SE for any AR order p ≥ 0.

    Fits OLS AR(p) with intercept to the standardised Z process, computes
    the implied Toeplitz post-window covariance analytically via Yule-Walker,
    and maps it through the IV slope-ratio formula.  No statsmodels required.

    For p=1 this is numerically equivalent to raw_z_ar1_design_based_se.
    """
    moments = centered_slope_ratio_moments(y, w, z)
    yv = _as_vector(y, "y")
    wv = _as_vector(w, "w")
    zv = _as_vector(z, "z")
    zp = _as_vector(z_process, "z_process")
    if not (yv.size == wv.size == zv.size):
        raise ValueError("y, w, and z must have the same length.")
    p = int(ar_order)
    if p < 0:
        raise ValueError("ar_order must be non-negative.")
    zp_fit, _zp_mean, zp_scale = _standardize_for_arima_fit(zp)
    n_fit = int(zp_fit.size)
    if n_fit < p + 3:
        raise ValueError(f"z_process has {n_fit} obs — too short for AR({p}).")

    _ar_fallback_fired = False
    if p == 0:
        sigma2 = float(np.var(zp_fit, ddof=1))
        phi = np.array([], dtype=float)
    else:
        cur = zp_fit[p:]
        X_ar = np.column_stack(
            [np.ones(cur.size)] + [zp_fit[p - j - 1 : n_fit - j - 1] for j in range(p)]
        )
        beta_ar = np.linalg.lstsq(X_ar, cur, rcond=None)[0]
        phi_raw = np.asarray(beta_ar[1:], dtype=float)
        if _ar_stationary(phi_raw):
            # Normal path: use the fitted AR(p) parameters and their residuals.
            phi = phi_raw
            resid_ar = cur - X_ar @ beta_ar
            df = max(int(cur.size) - p - 1, 1)
            sigma2 = float(resid_ar @ resid_ar) / float(df)
        else:
            # AR(p) OLS gave non-stationary roots.  Fall back to AR(0).
            # Use ddof=1 to match the explicit p=0 path so both AR(0) code
            # paths produce the same sigma2 for identical input.
            phi = np.array([], dtype=float)
            sigma2 = float(np.var(zp_fit, ddof=1))
            _ar_fallback_fired = True

    if not math.isfinite(sigma2) or sigma2 <= FIRST_STAGE_TOL:
        raise ValueError(f"AR({p}) innovation variance is nonpositive.")

    t = int(zv.size)
    gamma = _ar_autocovariances(phi, sigma2, t - 1)
    omega = _toeplitz_from_autocovariances(gamma, t)
    center = np.eye(t, dtype=float) - np.ones((t, t), dtype=float) / float(t)
    omega_centered = (zp_scale * zp_scale) * (center @ omega @ center)

    u = yv - moments.tau * wv
    varz_key = str(varz_mode).lower()
    z_for_var = zv if varz_key == "window" else zp
    varz = float(np.mean((z_for_var - float(np.mean(z_for_var))) ** 2))
    if varz <= FIRST_STAGE_TOL:
        raise ValueError("Instrument variance is nonpositive.")
    sd_dot = math.sqrt(max(float(u @ omega_centered @ u), 0.0))
    se = sd_dot / (abs(moments.pi) * varz * float(t))
    if not math.isfinite(se) or se <= 0.0:
        raise ValueError(f"AR({p}) design-based SE is nonpositive.")
    # Fitted AR coefficients as a fixed-length dict (up to 4 lags; NaN-padded).
    _phi_arr = np.asarray(phi, dtype=float)
    _phi_dict = {
        f"ar_phi_{j+1}": float(_phi_arr[j]) if j < len(_phi_arr) else float("nan")
        for j in range(4)
    }
    return InferenceResult(
        se=float(se),
        tau=moments.tau,
        pi=moments.pi,
        scale_multiplier=1.0,
        diagnostics={
            "method": "ARIMA-Z",
            "score_rule": f"raw_z_ar{p}_design_based",
            "arma_p": int(p),
            "arma_q": 0,
            "arma_fallback": int(_ar_fallback_fired),
            "sd_dot": float(sd_dot),
            "varZ_denom": float(varz),
            "varZ_mode": varz_key,
            "T1": t,
            **_phi_dict,
        },
    )


def algorithm2_raw_z_arima_design_based_se(
    y: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    *,
    z_process: np.ndarray,
    arima_order: tuple[int, int, int] = (2, 0, 0),
    varz_mode: str = "window",
) -> InferenceResult:
    """Statsmodels-ARIMA Algorithm-2 SE used for empirical/simulation parity.

    This is the simulation-side implementation of the empirical
    Algorithm-2 rule:

    * fit the raw aggregate instrument path with an intercept-included ARIMA
      model;
    * use the implied post-learning covariance of centered simulated
      instrument paths;
    * divide by the post-learning centered variance of the instrument, the
      first-stage slope, and T1 exactly once.

    The implementation supports stationary/invertible ARMA(p,q) models with
    d=0. This keeps the simulation-side Algorithm-2 covariance object aligned
    with the order-selection diagnostics and allows non-AR covariance laws.
    """
    moments = centered_slope_ratio_moments(y, w, z)
    yv = _as_vector(y, "y")
    wv = _as_vector(w, "w")
    zv = _as_vector(z, "z")
    zp = _as_vector(z_process, "z_process")
    zp_fit, _zp_mean, zp_scale = _standardize_for_arima_fit(zp)
    order = tuple(int(v) for v in arima_order)
    if len(order) != 3:
        raise ValueError("arima_order must be a (p,d,q) tuple.")
    p, d, q = order
    if d != 0:
        raise NotImplementedError("Algorithm-2 parity helper currently supports d=0 only.")
    if p < 0 or q < 0:
        raise ValueError("ARMA orders p and q must be nonnegative.")
    arima_res = _fit_statsmodels_arima_cached(zp_fit, order)
    phi = np.asarray(getattr(arima_res, "arparams", np.array([], dtype=float)), dtype=float).reshape(-1)
    theta = np.asarray(getattr(arima_res, "maparams", np.array([], dtype=float)), dtype=float).reshape(-1)
    if phi.size != p:
        raise RuntimeError("ARIMA result did not return the expected number of AR parameters.")
    if theta.size != q:
        raise RuntimeError("ARIMA result did not return the expected number of MA parameters.")
    sigma2 = _sigma2_from_statsmodels_arima(arima_res)
    t = int(zv.size)
    gamma = _arma_autocovariances(phi, theta, sigma2, t - 1)
    omega = _toeplitz_from_autocovariances(gamma, t)
    center = np.eye(t, dtype=float) - np.ones((t, t), dtype=float) / float(t)
    omega_centered = (zp_scale * zp_scale) * (center @ omega @ center)
    residual = yv - moments.tau * wv
    varz_key = str(varz_mode).lower()
    if varz_key == "window":
        z_for_var = zv
    elif varz_key == "full":
        z_for_var = zp
    else:
        raise ValueError("varz_mode must be 'window' or 'full'.")
    varz = float(np.mean((z_for_var - float(np.mean(z_for_var))) ** 2))
    if varz <= FIRST_STAGE_TOL:
        raise ValueError("instrument variance denominator is nonpositive.")
    sd_dot = math.sqrt(max(float(residual @ omega_centered @ residual), 0.0))
    se = sd_dot / (abs(moments.pi) * varz * float(t))
    if not math.isfinite(se) or se <= 0.0:
        raise ValueError("Algorithm-2 ARIMA design-based SE is nonpositive.")
    return InferenceResult(
        se=float(se),
        tau=moments.tau,
        pi=moments.pi,
        scale_multiplier=1.0,
        diagnostics={
            "method": "ARIMA-Z",
            "score_rule": "algorithm2_centered_post_z_slope_score",
            "z_process_rule": f"raw_Z_ARIMA_{p}_0_{q}_intercept_centered_post_covariance",
            "arima_fit_scale_rule": "standardized_Z_mapped_to_raw_units",
            "arima_fit_raw_scale": float(zp_scale),
            "first_stage_division_count": 1,
            "arma_p": int(p),
            "arma_d": 0,
            "arma_q": int(q),
            "arima_aic": float(arima_res.aic),
            "arima_bic": float(arima_res.bic),
            "ar_params": ",".join(f"{float(v):.17g}" for v in phi),
            "ma_params": ",".join(f"{float(v):.17g}" for v in theta),
            "arima_sigma2": float(sigma2),
            "sd_dot": float(sd_dot),
            "varZ_denom": float(varz),
            "varZ_mode": varz_key,
            "simulated_z_centered_within_post_window": 1,
            "T1": int(t),
        },
    )


def algorithm2_raw_z_ar_rootguard_design_based_se(
    y: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    *,
    z_process: np.ndarray,
    max_p: int = 2,
    root_guard: float = 1.05,
    varz_mode: str = "window",
) -> InferenceResult:
    """Algorithm-2 SE with a root-guard AR order selected on the observed Z path."""
    zp = _as_vector(z_process, "z_process")
    zp_fit, _zp_mean, _zp_scale = _standardize_for_arima_fit(zp)
    arima_res, order, selected_aicc, min_root = _fit_ar_rootguard(
        zp_fit,
        max_p=int(max_p),
        root_guard=float(root_guard),
    )
    result = algorithm2_raw_z_arima_design_based_se(
        y,
        w,
        z,
        z_process=z_process,
        arima_order=order,
        varz_mode=varz_mode,
    )
    diagnostics = {
        **result.diagnostics,
        "z_process_rule": (
            f"raw_Z_AR_rootguard_p0_{int(max_p)}_intercept_centered_post_covariance"
        ),
        "arima_order_selection_rule": "stationary_invertible_AR_p0_pmax_AICc_root_guard",
        "arima_selected_order": str(order),
        "arima_selected_aicc": float(selected_aicc),
        "arima_selected_min_inverse_root_modulus": float(min_root),
        "arima_root_guard": float(root_guard),
        "arima_refit_aic": float(getattr(arima_res, "aic")),
    }
    return InferenceResult(
        se=result.se,
        tau=result.tau,
        pi=result.pi,
        scale_multiplier=result.scale_multiplier,
        diagnostics=diagnostics,
    )


def algorithm2_raw_z_arma_rootguard_design_based_se(
    y: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    *,
    z_process: np.ndarray,
    max_p: int = 2,
    max_q: int = 2,
    root_guard: float = 1.05,
    varz_mode: str = "window",
) -> InferenceResult:
    """Algorithm-2 SE with root-guard ARMA order selection on the observed Z path."""
    zp = _as_vector(z_process, "z_process")
    zp_fit, _zp_mean, _zp_scale = _standardize_for_arima_fit(zp)
    arima_res, order, selected_aicc, min_ar_root, min_ma_root = _fit_arma_rootguard(
        zp_fit,
        max_p=int(max_p),
        max_q=int(max_q),
        root_guard=float(root_guard),
    )
    result = algorithm2_raw_z_arima_design_based_se(
        y,
        w,
        z,
        z_process=z_process,
        arima_order=order,
        varz_mode=varz_mode,
    )
    diagnostics = {
        **result.diagnostics,
        "z_process_rule": (
            f"raw_Z_ARMA_rootguard_p0_{int(max_p)}_q0_{int(max_q)}_intercept_centered_post_covariance"
        ),
        "arima_order_selection_rule": "stationary_invertible_ARMA_p0_pmax_q0_qmax_AICc_root_guard",
        "arima_selected_order": str(order),
        "arima_selected_aicc": float(selected_aicc),
        "arima_selected_min_ar_inverse_root_modulus": float(min_ar_root),
        "arima_selected_min_ma_inverse_root_modulus": float(min_ma_root),
        "arima_root_guard": float(root_guard),
        "arima_refit_aic": float(getattr(arima_res, "aic")),
    }
    return InferenceResult(
        se=result.se,
        tau=result.tau,
        pi=result.pi,
        scale_multiplier=result.scale_multiplier,
        diagnostics=diagnostics,
    )


def algorithm2_known_z_covariance_se(
    y: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    *,
    omega_centered: np.ndarray,
    varz_mode: str = "window",
) -> InferenceResult:
    """Algorithm-2 slope-ratio SE using a supplied centered Z covariance.

    This is a simulation diagnostic. It uses the Algorithm-2 slope-ratio
    normalization with a centered post-window instrument covariance supplied
    by the simulation design.
    """

    moments = centered_slope_ratio_moments(y, w, z)
    yv = _as_vector(y, "y")
    wv = _as_vector(w, "w")
    zv = _as_vector(z, "z")
    omega = np.asarray(omega_centered, dtype=float)
    t = int(zv.size)
    if omega.shape != (t, t):
        raise ValueError("omega_centered must be T1 by T1.")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega_centered contains nonfinite values.")
    residual = yv - moments.tau * wv
    varz_key = str(varz_mode).lower()
    if varz_key != "window":
        raise ValueError("known covariance diagnostic currently uses post-window varZ.")
    varz = float(np.mean((zv - float(np.mean(zv))) ** 2))
    if varz <= FIRST_STAGE_TOL:
        raise ValueError("instrument variance denominator is nonpositive.")
    sd_dot = math.sqrt(max(float(residual @ omega @ residual), 0.0))
    se = sd_dot / (abs(moments.pi) * varz * float(t))
    if not math.isfinite(se) or se <= 0.0:
        raise ValueError("Algorithm-2 known-Z-covariance SE is nonpositive.")
    return InferenceResult(
        se=float(se),
        tau=moments.tau,
        pi=moments.pi,
        scale_multiplier=1.0,
        diagnostics={
            "method": "ARIMA-Z",
            "score_rule": "algorithm2_centered_post_z_slope_score",
            "z_process_rule": "known_simulated_centered_post_Z_covariance_diagnostic",
            "first_stage_division_count": 1,
            "arma_p": -2,
            "arma_d": 0,
            "arma_q": 0,
            "arima_aic": float("nan"),
            "arima_bic": float("nan"),
            "ar_params": "",
            "arima_sigma2": float("nan"),
            "sd_dot": float(sd_dot),
            "varZ_denom": float(varz),
            "varZ_mode": varz_key,
            "known_z_covariance_trace": float(np.trace(omega)),
            "simulated_z_centered_within_post_window": 1,
            "T1": int(t),
        },
    )


def arima_score_se(y: np.ndarray, w: np.ndarray, z: np.ndarray, *, ar_order: int | str = 0) -> InferenceResult:
    moments = centered_slope_ratio_moments(y, w, z)
    yv = _as_vector(y, "y")
    wv = _as_vector(w, "w")
    zv = _as_vector(z, "z")
    zc = zv - float(zv.mean())
    structural_residual = yv - moments.tau * wv
    structural_residual = structural_residual - float(np.mean(structural_residual))
    score = (zc / moments.szz) * structural_residual
    score = score - float(np.mean(score))
    lrv, lrv_diag = _score_long_run_variance(score, ar_order=ar_order, t=moments.t)
    if lrv <= 0.0:
        raise ValueError("ARIMA smoke score has nonpositive long-run variance.")
    se = math.sqrt(lrv / float(moments.t)) / abs(moments.pi)
    diagnostics = {
        "method": "ARIMA-Z",
        "score_rule": "centered_residual_slope_score",
        "first_stage_division_count": 1,
        "score_mean": float(np.mean(score)),
        "score_lrv": lrv,
        "T1": moments.t,
    }
    diagnostics.update(lrv_diag)
    return InferenceResult(
        se=float(se),
        tau=moments.tau,
        pi=moments.pi,
        scale_multiplier=1.0,
        diagnostics=diagnostics,
    )


def ar0_homoskedastic_slope_se(y: np.ndarray, w: np.ndarray, z: np.ndarray) -> InferenceResult:
    """No-HC AR(0) model-based slope-ratio standard error.

    This is the homoskedastic intercept-included regression SE for the
    post-window structural residual, divided by the first-stage slope exactly
    once. It is diagnostic for the AR(0) no-HC model; it does not apply HC
    leverage corrections or interval scale multipliers.
    """
    moments = centered_slope_ratio_moments(y, w, z)
    yv = _as_vector(y, "y")
    wv = _as_vector(w, "w")
    zv = _as_vector(z, "z")
    residual = yv - moments.tau * wv
    X = np.column_stack([np.ones(zv.size), zv])
    beta = np.linalg.lstsq(X, residual, rcond=None)[0]
    u = residual - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)
    df = max(int(zv.size) - 2, 1)
    sigma2 = float(u @ u) / float(df)
    se_slope = math.sqrt(max(sigma2 * float(xtx_inv[1, 1]), 0.0))
    if se_slope <= 0.0:
        raise ValueError("homoskedastic AR(0) slope standard error is nonpositive.")
    return InferenceResult(
        se=float(se_slope / abs(moments.pi)),
        tau=moments.tau,
        pi=moments.pi,
        scale_multiplier=1.0,
        diagnostics={
            "method": "ARIMA-Z",
            "score_rule": "ar0_homoskedastic_intercept_included_slope",
            "lrv_rule": "ar0_homoskedastic_model_based_no_hc",
            "first_stage_division_count": 1,
            "arma_p": 0,
            "arma_q": 0,
            "arma_fallback": 0,
            "residual_df": int(df),
            "T1": moments.t,
        },
    )


def leverage_adjusted_arima_score_se(
    y: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    *,
    ar_order: int | str = 0,
    adjustment: str = "HC3",
) -> InferenceResult:
    moments = centered_slope_ratio_moments(y, w, z)
    yv = _as_vector(y, "y")
    wv = _as_vector(w, "w")
    zv = _as_vector(z, "z")
    residual = yv - moments.tau * wv
    x = np.column_stack([np.ones(zv.size), zv])
    beta = np.linalg.lstsq(x, residual, rcond=None)[0]
    u = residual - x @ beta
    xtx_inv = np.linalg.inv(x.T @ x)
    leverage = np.clip(np.diag(x @ xtx_inv @ x.T), 0.0, 0.999999)
    adj_key = str(adjustment).upper()
    if adj_key == "HC0":
        denom_power = 0.0
    elif adj_key == "HC2":
        denom_power = 0.5
    elif adj_key == "HC3":
        denom_power = 1.0
    else:
        raise ValueError("leverage-adjusted ARIMA score supports HC0, HC2, or HC3.")
    adjusted_residual = u / ((1.0 - leverage) ** denom_power)
    zc = zv - float(zv.mean())
    score = (zc / moments.szz) * adjusted_residual
    lrv, lrv_diag = _score_long_run_variance(score, ar_order=ar_order, t=moments.t)
    if lrv <= 0.0:
        raise ValueError("leverage-adjusted ARIMA score has nonpositive long-run variance.")
    se = math.sqrt(lrv / float(moments.t)) / abs(moments.pi)
    diagnostics = {
        "method": "ARIMA-Z",
        "score_rule": "leverage_adjusted_centered_instrument_slope_score",
        "leverage_adjustment": adj_key,
        "leverage_denominator_power": float(denom_power),
        "first_stage_division_count": 1,
        "score_mean": float(np.mean(score)),
        "score_lrv": lrv,
        "T1": moments.t,
    }
    diagnostics.update(lrv_diag)
    return InferenceResult(
        se=float(se),
        tau=moments.tau,
        pi=moments.pi,
        scale_multiplier=1.0,
        diagnostics=diagnostics,
    )


def hac_delta_se(y: np.ndarray, w: np.ndarray, z: np.ndarray, maxlags: int = 0) -> InferenceResult:
    moments = centered_slope_ratio_moments(y, w, z)
    yv = _as_vector(y, "y")
    wv = _as_vector(w, "w")
    zv = _as_vector(z, "z")
    zc = zv - float(zv.mean())
    x = np.column_stack([(zc / moments.szz) * wv, (zc / moments.szz) * yv])
    omega = _hac_long_run_covariance(x, maxlags=maxlags)
    gradient = np.array([-moments.tau / moments.pi, 1.0 / moments.pi], dtype=float)
    var_tau = float(gradient @ omega @ gradient / float(moments.t))
    if var_tau <= 0.0:
        raise ValueError("HAC delta smoke variance is nonpositive.")
    return InferenceResult(
        se=math.sqrt(var_tau),
        tau=moments.tau,
        pi=moments.pi,
        scale_multiplier=1.0,
        diagnostics={
            "method": "HAC",
            "moment_rule": "centered_bivariate_slope_moment",
            "maxlags": int(maxlags),
            "T1": moments.t,
        },
    )


def ar_style_test(
    y: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    tau0: float,
    *,
    maxlags: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | int | str | bool]:
    yv = _as_vector(y, "y")
    wv = _as_vector(w, "w")
    zv = _as_vector(z, "z")
    if not (yv.size == wv.size == zv.size):
        raise ValueError("y, w, and z must have the same length.")
    residual = yv - float(tau0) * wv
    t = int(zv.size)
    x = np.column_stack([np.ones(t), zv])
    q = (x.T @ x) / float(t)
    q_inv = np.linalg.inv(q)
    beta = np.linalg.lstsq(x, residual, rcond=None)[0]
    fitted_resid = residual - x @ beta
    score = x * fitted_resid[:, None]
    omega = _hac_long_run_covariance(score, maxlags=maxlags)
    v_beta = q_inv @ omega @ q_inv / float(t)
    se_slope = math.sqrt(max(float(v_beta[1, 1]), 0.0))
    if se_slope <= 0.0:
        raise ValueError("AR-style smoke slope standard error is nonpositive.")
    t_stat = float(beta[1] / se_slope)
    crit = float(NormalDist().inv_cdf(1.0 - float(alpha) / 2.0))
    return {
        "method": "AR",
        "test": "intercept_included_residual_slope",
        "tau0": float(tau0),
        "slope": float(beta[1]),
        "se_slope": float(se_slope),
        "t_stat": t_stat,
        "critical_rule": "standard_normal",
        "critical_value": crit,
        "critical_scale": 1.0,
        "accepted": bool(abs(t_stat) <= crit),
        "maxlags": int(maxlags),
        "T1": t,
    }


def _ar_style_slope_variance(
    y: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    tau0: float,
    *,
    maxlags: int,
) -> tuple[float, float]:
    yv = _as_vector(y, "y")
    wv = _as_vector(w, "w")
    zv = _as_vector(z, "z")
    if not (yv.size == wv.size == zv.size):
        raise ValueError("y, w, and z must have the same length.")
    residual = yv - float(tau0) * wv
    t = int(zv.size)
    x = np.column_stack([np.ones(t), zv])
    q = (x.T @ x) / float(t)
    q_inv = np.linalg.inv(q)
    beta = np.linalg.lstsq(x, residual, rcond=None)[0]
    fitted_resid = residual - x @ beta
    score = x * fitted_resid[:, None]
    omega = _hac_long_run_covariance(score, maxlags=maxlags)
    v_beta = q_inv @ omega @ q_inv / float(t)
    return float(beta[1]), max(float(v_beta[1, 1]), 0.0)


def ar_style_interval(
    y: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    *,
    maxlags: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | int | str | bool]:
    """Invert the intercept-included AR-style residual-slope test analytically.

    The residual-slope numerator is linear in tau and the HAC slope variance is
    quadratic in tau because the regression residual is linear in tau.  The
    accepted set is therefore determined by one quadratic inequality.
    """
    moments = centered_slope_ratio_moments(y, w, z)
    crit = float(NormalDist().inv_cdf(1.0 - float(alpha) / 2.0))
    slope0, var0 = _ar_style_slope_variance(y, w, z, 0.0, maxlags=maxlags)
    _, var1 = _ar_style_slope_variance(y, w, z, 1.0, maxlags=maxlags)
    _, varm1 = _ar_style_slope_variance(y, w, z, -1.0, maxlags=maxlags)
    # v(tau) = a tau^2 + b tau + c.
    a_var = 0.5 * (var1 + varm1 - 2.0 * var0)
    b_var = 0.5 * (var1 - varm1)
    c_var = var0
    # (delta - pi tau)^2 - crit^2 v(tau) <= 0.
    A = moments.pi * moments.pi - crit * crit * a_var
    B = -2.0 * moments.delta * moments.pi - crit * crit * b_var
    C = moments.delta * moments.delta - crit * crit * c_var
    tol = 1e-12
    interval_type = "empty"
    low = float("nan")
    high = float("nan")
    root_low = float("nan")
    root_high = float("nan")
    excluded_low = float("nan")
    excluded_high = float("nan")
    if abs(A) <= tol:
        if abs(B) <= tol:
            if C <= 0.0:
                interval_type = "all_real"
                low = float("-inf")
                high = float("inf")
        else:
            root = -C / B
            if B > 0:
                interval_type = "lower_ray"
                low = float("-inf")
                high = float(root)
                root_high = float(root)
            else:
                interval_type = "upper_ray"
                low = float(root)
                high = float("inf")
                root_low = float(root)
    else:
        disc = B * B - 4.0 * A * C
        if disc < -tol:
            if A < 0.0:
                interval_type = "all_real"
                low = float("-inf")
                high = float("inf")
        else:
            disc = max(disc, 0.0)
            root1 = (-B - math.sqrt(disc)) / (2.0 * A)
            root2 = (-B + math.sqrt(disc)) / (2.0 * A)
            rlo = min(root1, root2)
            rhi = max(root1, root2)
            root_low = float(rlo)
            root_high = float(rhi)
            if A > 0.0:
                interval_type = "bounded"
                low = float(rlo)
                high = float(rhi)
            else:
                interval_type = "two_rays"
                low = float("-inf")
                high = float("inf")
                excluded_low = float(rlo)
                excluded_high = float(rhi)
    descriptions = {
        "empty": "no tau values accepted",
        "all_real": "all real tau values accepted",
        "bounded": "accepted tau values lie between interval_low and interval_high",
        "lower_ray": "accepted tau values are less than or equal to interval_high",
        "upper_ray": "accepted tau values are greater than or equal to interval_low",
        "two_rays": "accepted tau values lie outside the finite excluded interval",
    }
    return {
        "method": "AR",
        "test": "analytic_intercept_included_residual_slope_inversion",
        "tau_hat": float(moments.tau),
        "pi": float(moments.pi),
        "critical_rule": "standard_normal",
        "critical_value": crit,
        "critical_scale": 1.0,
        "interval_low": low,
        "interval_high": high,
        "interval_type": interval_type,
        "accepted_set_description": descriptions[interval_type],
        "root_low": root_low,
        "root_high": root_high,
        "excluded_interval_low": excluded_low,
        "excluded_interval_high": excluded_high,
        "quadratic_A": float(A),
        "quadratic_B": float(B),
        "quadratic_C": float(C),
        "variance_quadratic_a": float(a_var),
        "variance_quadratic_b": float(b_var),
        "variance_quadratic_c": float(c_var),
        "maxlags": int(maxlags),
        "T1": int(moments.t),
    }
