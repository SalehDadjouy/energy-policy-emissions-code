#!/usr/bin/env python3
"""
Empirical analysis: TSLS and Robust estimation on the state-year panel.

Panel: U.S. states, 2008-2022, two-year lags.  CO2e emissions (Y),
wind-and-solar retail electricity sales (W), national renewable subsidy (Z).
Baseline: N=49 (drops CA and VT); sensitivity: N=51 (full sample).
Learning window T0=5 (2008-2012); post-window T1=10 (2013-2022).

Three inference procedures:
  ARIMA  — primary design-based interval: fit ARIMA(2,0,0) to Z, simulate Z paths,
            compute SD of u'Z_sim, SE = SD / (|π| · VarZ_post · T1)
  HAC    — Newey-West delta method, L=1
  AR     — Anderson-Rubin-style HAC orthogonality inversion, L=1

Usage:
  python empirical.py                     # full run, 80 000 ARIMA sims
  python empirical.py --n-sims 5000       # fast check
  python empirical.py --out results/      # custom output directory
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import t as _t_dist
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── locked study design ───────────────────────────────────────────────────────
LAG          = 2
YEAR_START   = 2008
YEAR_END     = 2022
RESTRICTED   = ["CA", "VT"]
T0           = 5              # learning window length
HAC_LAGS     = 1              # Newey-West bandwidth
ARIMA_ORDER  = (2, 0, 0)      # AR(2) — AICc-selected from T=15 annual Z series
AR_ALPHA     = 0.05           # AR test level
AR_GRID_MULT = 10.0           # initial half-width = grid_mult × HAC SE
AR_GRID_PTS  = 2001           # grid resolution
AR_MAX_ITER  = 6              # adaptive grid expansion cap
AR_EXPAND    = 2.0            # grid expansion factor
AR_MAX_MULT  = 2000.0         # maximum grid half-width multiple


# ── data loading ──────────────────────────────────────────────────────────────

def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {f"Y_it_lag{LAG}": "Y", f"W_it_lag{LAG}": "W", f"Z_t_lag{LAG}": "Z",
              "state": "unit", "year": "time"}
    df = df.rename(columns=rename)[["unit", "time", "Y", "W", "Z"]]
    return df.dropna(subset=["Y", "W", "Z"]).copy()


def pivot(df: pd.DataFrame):
    units = sorted(df["unit"].unique())
    times = sorted(df["time"].unique())
    df = df.copy()
    df["unit"] = pd.Categorical(df["unit"], categories=units, ordered=True)
    df["time"] = pd.Categorical(df["time"], categories=times, ordered=True)
    Y = df.pivot(index="unit", columns="time", values="Y").to_numpy(dtype=float)
    W = df.pivot(index="unit", columns="time", values="W").to_numpy(dtype=float)
    Z = df[["time", "Z"]].drop_duplicates().sort_values("time")["Z"].to_numpy(dtype=float)
    return Y, W, Z, units, times


# ── estimators ────────────────────────────────────────────────────────────────

def exposure_profile(W: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Unit-level first-stage slopes D_i = slope(W_{it}, Z_t) after unit FE."""
    Zc = Z - Z.mean()
    Wc = W - W.mean(axis=1, keepdims=True)
    return (Wc @ Zc) / float(Zc @ Zc)


def tsls_weights(D: np.ndarray) -> np.ndarray:
    Dc = D - D.mean()
    denom = float(np.mean(Dc * D))
    if abs(denom) <= 1e-12:
        raise ValueError("first-stage exposure vector has zero usable variation.")
    return Dc / denom


def _proj_const_z(Z0: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(Z0)), Z0])
    return np.eye(len(Z0)) - X @ np.linalg.solve(X.T @ X, X.T)


def _residual_two_way(X: np.ndarray, Z0: np.ndarray):
    """Residual from X ~ unit FE + time FE + unit slope on Z. Returns (σ², E)."""
    n, T0 = X.shape
    N = n * T0
    y = X.reshape(N)
    ui = np.repeat(np.arange(n), T0)
    ti = np.tile(np.arange(T0), n)
    U = np.zeros((N, n)); U[np.arange(N), ui] = 1.0
    Tm = np.zeros((N, T0)); Tm[np.arange(N), ti] = 1.0
    G = np.zeros((N, n)); G[np.arange(N), ui] = Z0[ti]
    Xd = np.concatenate([U, Tm, G], axis=1)
    b = np.linalg.lstsq(Xd, y, rcond=None)[0]
    E = (y - Xd @ b).reshape(n, T0)
    return float(np.mean(E ** 2)), E


def robust_siv_weights(Y: np.ndarray, W: np.ndarray, Z: np.ndarray, D: np.ndarray):
    """Robust weights via constrained QP with the RMS zeta rule."""
    n, T = Y.shape
    Y0, W0, Z0 = Y[:, :T0], W[:, :T0], Z[:T0]
    sigma_y2, Ey = _residual_two_way(Y0, Z0)   # two-way FE residuals
    sigma_w2, Ew = _residual_two_way(W0, Z0)
    op_y = float(np.linalg.svd(Ey, compute_uv=False)[0])   # operator norm of FE residual
    op_w = float(np.linalg.svd(Ew, compute_uv=False)[0])
    sigma_hat = max(op_y, op_w) / math.sqrt(n * T0)
    zeta = math.sqrt(math.log(T0)) * sigma_hat
    P = _proj_const_z(Z0)
    Q = (zeta ** 2 / (n * T0)) * np.eye(n)
    Q += (Y0 @ P @ Y0.T) / (T0 * sigma_y2 * n ** 2)
    Q += (W0 @ P @ W0.T) / (T0 * sigma_w2 * n ** 2)
    A = np.vstack([D.reshape(1, -1), np.ones((1, n))])
    b = np.array([float(n), 0.0])
    Qi_AT = np.linalg.solve(Q, A.T)
    w = (Qi_AT @ np.linalg.solve(A @ Qi_AT, b)).reshape(-1)
    return w


def slope_ratio(Y_t: np.ndarray, W_t: np.ndarray, Z_t: np.ndarray) -> dict:
    T = len(Z_t)
    X = np.column_stack([np.ones(T), Z_t])
    beta_y = np.linalg.lstsq(X, Y_t, rcond=None)[0]
    beta_w = np.linalg.lstsq(X, W_t, rcond=None)[0]
    delta, pi = float(beta_y[1]), float(beta_w[1])
    return {"tau": delta / pi, "delta": delta, "pi": pi,
            "u": Y_t - X @ beta_y, "v": W_t - X @ beta_w}


def aggregate(panel: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (w @ panel) / panel.shape[0]


# ── ARIMA simulation SE ───────────────────────────────────────────────────────

def _fit_arima(Z: np.ndarray):
    """Fit ARIMA(p,0,0) to standardised Z using ARIMA_ORDER; return result + raw scale."""
    mean, scale = float(Z.mean()), float(Z.std(ddof=0))
    if scale <= 0:
        raise ValueError("Z has zero variance.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA((Z - mean) / scale, order=ARIMA_ORDER, trend="c",
                    enforce_stationarity=True, enforce_invertibility=True).fit()
    return res, mean, scale


def _sigma2(res) -> float:
    for attr in ("sigma2", "sigma2_"):
        v = getattr(res, attr, None)
        if v is not None and np.isfinite(float(v)) and float(v) > 0:
            return float(v)
    resid = np.asarray(getattr(res, "resid", np.array([])), float)
    resid = resid[np.isfinite(resid)]
    if resid.size:
        return float(np.var(resid, ddof=0))
    raise RuntimeError("Cannot extract σ² from ARIMA result.")


def _long_run_mean(res, calibration_mean: float) -> float:
    phi = np.asarray(getattr(res, "arparams", []), float)
    const = getattr(res, "params", np.array([]))[0] if hasattr(res, "params") else None
    if const is None:
        return calibration_mean
    denom = 1.0 - float(phi.sum()) if phi.size else 1.0
    return calibration_mean if abs(denom) < 1e-12 else float(const) / denom


def arima_simulation_se(
    u: np.ndarray,
    res,
    raw_mean: float,
    raw_scale: float,
    pi_hat: float,
    varZ_post: float,
    n_sims: int,
    seed: int = 12345,
    burnin: int = 500,
    batch_size: int = 5000,
) -> float:
    """SE = SD(u' Z_sim_centered) / (|π| · VarZ_post · T1), Algorithm-2 style."""
    u = np.asarray(u, float).reshape(-1)
    T1 = u.size
    ar = np.asarray(getattr(res, "arparams", []), float)
    ma = np.asarray(getattr(res, "maparams", []), float)
    sigma = math.sqrt(_sigma2(res))
    mu = _long_run_mean(res, float(np.asarray(res.model.endog, float).mean()))

    rng = np.random.default_rng(seed)
    count, mean_acc, M2 = 0, 0.0, 0.0
    remaining = int(n_sims)
    p, q = len(ar), len(ma)

    while remaining > 0:
        m = min(batch_size, remaining)
        remaining -= m
        n_total = burnin + T1
        e = rng.normal(0.0, sigma, size=(m, n_total))
        y = np.full((m, n_total), mu, float)
        for t in range(n_total):
            yt = np.full(m, mu, float)
            for j in range(p):
                if t - j - 1 >= 0:
                    yt += ar[j] * (y[:, t - j - 1] - mu)
            for j in range(q):
                if t - j - 1 >= 0:
                    yt += ma[j] * e[:, t - j - 1]
            yt += e[:, t]
            y[:, t] = yt
        z_sim = y[:, burnin:] * raw_scale      # back to raw units
        z_centered = z_sim - z_sim.mean(axis=1, keepdims=True)
        dots = (z_centered * u).sum(axis=1)
        for x in dots:
            count += 1
            d = x - mean_acc
            mean_acc += d / count
            M2 += d * (x - mean_acc)

    sd_dot = math.sqrt(M2 / (count - 1))
    return sd_dot / (abs(pi_hat) * varZ_post * T1)


# ── HAC delta SE ─────────────────────────────────────────────────────────────
# Standard sandwich: V(β̂) = Q⁻¹ Ω Q⁻¹ / T, Newey-West kernel, L=HAC_LAGS.
# Delta method for τ = δ/π.

def _nw_cov(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Newey-West long-run covariance (Bartlett kernel, L=HAC_LAGS)."""
    T = A.shape[0]
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    S = (Ac.T @ Bc) / T
    for lag in range(1, min(HAC_LAGS, T - 1) + 1):
        w = 1.0 - lag / (HAC_LAGS + 1.0)
        S += w * ((Ac[lag:].T @ Bc[:-lag] + Bc[:-lag].T @ Ac[lag:]) / T)
    return S


def hac_delta_se(Y_t: np.ndarray, W_t: np.ndarray, Z_t: np.ndarray) -> float:
    """HAC delta-method SE for τ = δ/π (Newey-West, L=HAC_LAGS, normal critical)."""
    T = len(Z_t)
    X = np.column_stack([np.ones(T), Z_t])
    Q_inv = np.linalg.inv((X.T @ X) / T)          # = (X'X/T)⁻¹
    by = np.linalg.lstsq(X, Y_t, rcond=None)[0]
    bw = np.linalg.lstsq(X, W_t, rcond=None)[0]
    delta, pi = float(by[1]), float(bw[1])
    sy = X * (Y_t - X @ by)[:, None]              # T×2 score for Y regression
    sw = X * (W_t - X @ bw)[:, None]              # T×2 score for W regression

    # Sandwich variances for the slopes (index 1 of 2-vector)
    var_d = float((Q_inv @ _nw_cov(sy, sy) @ Q_inv)[1, 1]) / T
    var_p = float((Q_inv @ _nw_cov(sw, sw) @ Q_inv)[1, 1]) / T

    # Cross-covariance: S_yw[i,j] = NW-Cov(sy[:,i], sw[:,j])
    S_yw = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            pair = np.column_stack([sy[:, i], sw[:, j]])
            S_yw[i, j] = float(_nw_cov(pair, pair)[0, 1])
    cov_dp = float((Q_inv @ S_yw @ Q_inv)[1, 1]) / T

    # Delta method: Var(τ) = (1/π)² Var(δ) + (δ/π²)² Var(π) - 2(δ/π³) Cov(δ,π)
    g = np.array([1.0 / pi, -delta / pi ** 2])
    var_tau = float(g @ np.array([[var_d, cov_dp], [cov_dp, var_p]]) @ g)
    return math.sqrt(max(var_tau, 0.0))


# ── Anderson-Rubin inversion ──────────────────────────────────────────────────

def _ar_tstat(Y_t, W_t, Z_t, tau0: float, *, maxlags: int) -> float:
    R = Y_t - tau0 * W_t
    hac = sm.OLS(R, sm.add_constant(Z_t)).fit().get_robustcov_results(
        cov_type="HAC", maxlags=int(maxlags), use_correction=True
    )
    se = float(hac.bse[1])
    return float(hac.params[1] / se) if se > 0 else float("nan")


def ar_confset(
    Y_t,
    W_t,
    Z_t,
    tau_center: float,
    tau_scale: float,
    crit: Optional[float] = None,
    *,
    maxlags: int = HAC_LAGS,
) -> dict:
    """Invert AR test over a grid; expand adaptively if CI hits boundary.

    crit: critical value for the slope t-test. Defaults to t(T1-2) for
    T1<=10, standard normal otherwise, consistent with ARIMA.
    """
    if crit is None:
        t1 = len(Y_t)
        crit = float(_t_dist.ppf(0.975, df=t1 - 2)) if t1 <= 10 else float(NormalDist().inv_cdf(0.975))
    mult = AR_GRID_MULT
    result = None

    for _ in range(AR_MAX_ITER):
        half = mult * max(float(tau_scale), 1e-6)
        grid = np.linspace(tau_center - half, tau_center + half, AR_GRID_PTS)
        accept = np.array(
            [abs(_ar_tstat(Y_t, W_t, Z_t, g, maxlags=maxlags)) <= crit for g in grid]
        )
        n_acc = int(accept.sum())
        if n_acc == 0:
            return {"ci_lo": float("nan"), "ci_hi": float("nan"),
                    "lo_open": False, "hi_open": False}
        idx = np.where(accept)[0]
        lo_open = idx[0] == 0
        hi_open = idx[-1] == AR_GRID_PTS - 1
        result = {
            "ci_lo": float("-inf") if lo_open else float(grid[idx[0]]),
            "ci_hi": float("inf")  if hi_open else float(grid[idx[-1]]),
            "lo_open": lo_open, "hi_open": hi_open,
        }
        if not lo_open and not hi_open:
            return result
        if mult >= AR_MAX_MULT:
            return result
        mult = min(mult * AR_EXPAND, AR_MAX_MULT)

    return result or {"ci_lo": float("nan"), "ci_hi": float("nan"),
                      "lo_open": False, "hi_open": False}


# ── p-values ──────────────────────────────────────────────────────────────────

def p_two_sided(t: float) -> float:
    """Two-sided p-value, standard normal reference (used for HAC)."""
    if not math.isfinite(t):
        return float("nan")
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0)))))


def p_two_sided_t(t: float, df: int) -> float:
    """Two-sided p-value from Student-t(df) reference (used for ARIMA and AR)."""
    if not math.isfinite(t):
        return float("nan")
    return float(2.0 * float(_t_dist.sf(abs(t), df=max(int(df), 1))))


# ── per-sample runner ─────────────────────────────────────────────────────────

def run_sample(
    Y: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    units: list,
    label: str,
    n_sims: int,
    out_dir: Path,
) -> dict:
    n, T = Y.shape
    Z_post = Z[T0:]
    T1 = len(Z_post)
    varZ_post = float(np.mean((Z_post - Z_post.mean()) ** 2))

    # ── estimators ──────────────────────────────────────────────────────────
    D = exposure_profile(W, Z)
    w_t = tsls_weights(D)
    w_r = robust_siv_weights(Y, W, Z, D)

    Yt_t = aggregate(Y[:, T0:], w_t)
    Wt_t = aggregate(W[:, T0:], w_t)
    Yt_r = aggregate(Y[:, T0:], w_r)
    Wt_r = aggregate(W[:, T0:], w_r)

    est_t = slope_ratio(Yt_t, Wt_t, Z_post)
    est_r = slope_ratio(Yt_r, Wt_r, Z_post)

    # ── first-stage ─────────────────────────────────────────────────────────
    def first_stage(Wt, Zt):
        ols = sm.OLS(Wt, sm.add_constant(Zt)).fit()
        hac = ols.get_robustcov_results(
            cov_type="HAC", maxlags=HAC_LAGS, use_correction=True
        )
        return {
            "pi": float(hac.params[1]),
            "se_ols": float(ols.bse[1]),
            "se_hac": float(hac.bse[1]),
            "f_hac": float(np.asarray(hac.f_test([[0, 1]]).fvalue).squeeze()),
        }

    fs_t = first_stage(Wt_t, Z_post)
    fs_r = first_stage(Wt_r, Z_post)

    # ── ARIMA simulation SE ──────────────────────────────────────────────────
    arima_res, raw_mean, raw_scale = _fit_arima(Z)
    print(f"  [{label}] ARIMA({ARIMA_ORDER}) fitted, running {n_sims:,} simulations ...")
    se_arima_t = arima_simulation_se(est_t["u"], arima_res, raw_mean, raw_scale,
                                     est_t["pi"], varZ_post, n_sims, seed=12345)
    se_arima_r = arima_simulation_se(est_r["u"], arima_res, raw_mean, raw_scale,
                                     est_r["pi"], varZ_post, n_sims, seed=12346)

    # ── HAC delta SE ────────────────────────────────────────────────────────
    se_hac_t = hac_delta_se(Yt_t, Wt_t, Z_post)
    se_hac_r = hac_delta_se(Yt_r, Wt_r, Z_post)

    # ── Anderson-Rubin confidence sets ──────────────────────────────────────
    print(f"  [{label}] AR inversion ...")
    ar_t_l0 = ar_confset(Yt_t, Wt_t, Z_post, est_t["tau"], se_hac_t, maxlags=0)
    ar_r_l0 = ar_confset(Yt_r, Wt_r, Z_post, est_r["tau"], se_hac_r, maxlags=0)
    ar_t_l1 = ar_confset(Yt_t, Wt_t, Z_post, est_t["tau"], se_hac_t, maxlags=1)
    ar_r_l1 = ar_confset(Yt_r, Wt_r, Z_post, est_r["tau"], se_hac_r, maxlags=1)

    # ── weights CSV ──────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"unit": units, "w_tsls": w_t, "w_siv": w_r,
                  "D_tsls": D, "D_siv": D}).to_csv(
        out_dir / f"weights_{label}.csv", index=False
    )

    # critical value for ARIMA and AR: t(T1-2) for T1<=10, normal for T1>10
    df_arima = T1 - 2
    c_arima = float(_t_dist.ppf(0.975, df=df_arima)) if T1 <= 10 else float(NormalDist().inv_cdf(0.975))

    return {
        "label": label, "n": n, "T": T, "T0": T0, "T1": T1,
        "tsls_tau":  round(float(est_t["tau"]), 6),
        "siv_tau":   round(float(est_r["tau"]), 6),
        # ARIMA simulation SE — t(T1-2) critical for T1<=10, normal otherwise
        "tsls_se_arima":  round(se_arima_t, 6),
        "siv_se_arima":   round(se_arima_r, 6),
        "tsls_p_arima":   round(p_two_sided_t(est_t["tau"] / se_arima_t, df_arima), 4),
        "siv_p_arima":    round(p_two_sided_t(est_r["tau"] / se_arima_r, df_arima), 4),
        "arima_critical": round(c_arima, 4),
        # HAC delta SE — standard normal critical
        "tsls_se_hac":    round(se_hac_t, 6),
        "siv_se_hac":     round(se_hac_r, 6),
        "tsls_p_hac":     round(p_two_sided(est_t["tau"] / se_hac_t), 4),
        "siv_p_hac":      round(p_two_sided(est_r["tau"] / se_hac_r), 4),
        # Orthogonality-inversion confidence sets at lag zero and lag one
        "tsls_ar_l0_lo": round(ar_t_l0["ci_lo"], 4) if math.isfinite(ar_t_l0["ci_lo"]) else ar_t_l0["ci_lo"],
        "tsls_ar_l0_hi": round(ar_t_l0["ci_hi"], 4) if math.isfinite(ar_t_l0["ci_hi"]) else ar_t_l0["ci_hi"],
        "siv_ar_l0_lo": round(ar_r_l0["ci_lo"], 4) if math.isfinite(ar_r_l0["ci_lo"]) else ar_r_l0["ci_lo"],
        "siv_ar_l0_hi": round(ar_r_l0["ci_hi"], 4) if math.isfinite(ar_r_l0["ci_hi"]) else ar_r_l0["ci_hi"],
        "tsls_ar_l1_lo": round(ar_t_l1["ci_lo"], 4) if math.isfinite(ar_t_l1["ci_lo"]) else ar_t_l1["ci_lo"],
        "tsls_ar_l1_hi": round(ar_t_l1["ci_hi"], 4) if math.isfinite(ar_t_l1["ci_hi"]) else ar_t_l1["ci_hi"],
        "siv_ar_l1_lo": round(ar_r_l1["ci_lo"], 4) if math.isfinite(ar_r_l1["ci_lo"]) else ar_r_l1["ci_lo"],
        "siv_ar_l1_hi": round(ar_r_l1["ci_hi"], 4) if math.isfinite(ar_r_l1["ci_hi"]) else ar_r_l1["ci_hi"],
        # First stage
        "tsls_pi": round(fs_t["pi"], 6),
        "tsls_pi_se_ols": round(fs_t["se_ols"], 6),
        "tsls_pi_se_hac": round(fs_t["se_hac"], 6),
        "tsls_f_hac": round(fs_t["f_hac"], 3),
        "siv_pi": round(fs_r["pi"], 6),
        "siv_pi_se_ols": round(fs_r["se_ols"], 6),
        "siv_pi_se_hac": round(fs_r["se_hac"], 6),
        "siv_f_hac": round(fs_r["f_hac"], 3),
        "arima_order": str(ARIMA_ORDER),
    }


# ── output ────────────────────────────────────────────────────────────────────

def _fmt(x) -> str:
    if x == float("inf"):  return "+inf"
    if x == float("-inf"): return "-inf"
    if isinstance(x, float) and math.isnan(x): return "nan"
    return str(x)


def print_results(results: list[dict]) -> None:
    for r in results:
        print(f"\n[{r['label']}]  n={r['n']}  T={r['T']}  T0={r['T0']}  T1={r['T1']}")
        c_arima = float(r.get("arima_critical", 1.96))
        for est, key in (("TSLS", "tsls"), ("SIV", "siv")):
            tau = r[f"{key}_tau"]
            se_a = r[f"{key}_se_arima"]
            se_h = r[f"{key}_se_hac"]
            print(f"  {est}  τ = {tau:.4f}")
            print(f"    ARIMA:  SE = {se_a:.4f}  p = {r[f'{key}_p_arima']:.4f}"
                  f"  CI ≈ [{tau - c_arima*se_a:.3f}, {tau + c_arima*se_a:.3f}]"
                  f"  (crit={c_arima:.3f})")
            print(f"    HAC:    SE = {se_h:.4f}  p = {r[f'{key}_p_hac']:.4f}"
                  f"  CI ≈ [{tau - 1.96*se_h:.3f}, {tau + 1.96*se_h:.3f}]")
            print(
                f"    AR L0 set: [{_fmt(r[f'{key}_ar_l0_lo'])}, {_fmt(r[f'{key}_ar_l0_hi'])}]"
            )
            print(
                f"    AR L1 set: [{_fmt(r[f'{key}_ar_l1_lo'])}, {_fmt(r[f'{key}_ar_l1_hi'])}]"
            )
            print(f"    π = {r[f'{key}_pi']:.4f}  F_HAC = {r[f'{key}_f_hac']:.2f}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Empirical TSLS and Robust analysis")
    ap.add_argument("--data",   type=Path, default=ROOT / "data" / "panel_lag2.csv")
    ap.add_argument("--out",    type=Path, default=ROOT / "outputs" / "empirical")
    ap.add_argument("--n-sims", type=int,  default=80000,
                    help="ARIMA simulation draws (use 5000 for a quick check)")
    args = ap.parse_args()

    print(f"Empirical analysis  panel={args.data.name}  n_sims={args.n_sims:,}")

    df_all = load_panel(args.data)
    df = df_all[(df_all["time"] >= YEAR_START) & (df_all["time"] <= YEAR_END)].copy()
    if df.empty:
        raise RuntimeError("No data after year filter.")

    results = []
    for label, drop in [("restricted", RESTRICTED), ("full", None)]:
        sub = df[~df["unit"].isin(drop)].copy() if drop else df.copy()
        Y, W, Z, units, _ = pivot(sub)
        r = run_sample(Y, W, Z, units, label, args.n_sims, args.out / f"lag{LAG}")
        results.append(r)

    # write summary CSV
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "results_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {summary_path}")

    print_results(results)


if __name__ == "__main__":
    main()
