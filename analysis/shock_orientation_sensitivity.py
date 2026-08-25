#!/usr/bin/env python3
"""Shock-orientation sensitivity suite for shock-active simulation designs.

This analysis varies only the direct outcome-shock loading direction. Its
outputs remain separate from the principal exposure-correlated simulation.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np
from scipy.stats import t as _t_dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paper_config as active  # noqa: E402
from sim.dgp import simulate_common_components, stable_seed, standardize  # noqa: E402
from sim.estimators import (  # noqa: E402
    aggregate_series,
    ols_slope_with_intercept,
    safe_weight_correlation,
    weighted_slope_ratio,
)
from sim.helpers import (  # noqa: E402
    EVALUATION_CONFIGS,
    apply_preperiod_scale,
    components_from_common,
    load_exposure_profile,
    shock_bias_component,
)
from sim.inference import (  # noqa: E402
    ar_style_test,
    hac_delta_se,
    raw_z_arp_design_based_se,
)
from sim.model_class import (  # noqa: E402
    ModelClassComponents,
    build_branch_panel,
    estimate_branch_panel,
    first_stage_component,
)
from sim.robust_weights import projection_residualizer_const_z, residuals_two_way_unit_slope  # noqa: E402


SHOCK_BRANCHES = ("Aggregate Shock", "GFE + Agg. Shock")
CORE_ARMS = ("A_TSLS_targeted", "B_SIV_targeted", "C_random_geometry")
OPTIONAL_ARM_D = "D_TSLS_shielded"
DEFAULT_SEVERITIES = (0.0, 0.6, 1.4, 2.0)
DEFAULT_OUT = ROOT / "outputs" / "shock_orientation_sensitivity"
K_TOL = 1e-8


def _crit(method: str, t1: int) -> float:
    if method == "HAC":
        return float(NormalDist().inv_cdf(0.975))
    return float(_t_dist.ppf(0.975, df=int(t1) - 2)) if int(t1) <= 10 else float(NormalDist().inv_cdf(0.975))


def _centered_corr(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=float).reshape(-1)
    bv = np.asarray(b, dtype=float).reshape(-1)
    if av.size != bv.size:
        raise ValueError("correlation vectors must have the same length.")
    av = av - float(av.mean())
    bv = bv - float(bv.mean())
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(av @ bv / denom) if denom > 1e-15 else float("nan")


def _with_shock_y(components: ModelClassComponents, shock_y: np.ndarray) -> ModelClassComponents:
    return ModelClassComponents(
        Z=components.Z,
        D=components.D,
        alpha_w=components.alpha_w,
        alpha_y=components.alpha_y,
        lowrank_w=components.lowrank_w,
        lowrank_y=components.lowrank_y,
        shock_w=components.shock_w,
        shock_y=np.asarray(shock_y, dtype=float),
        eps_w=components.eps_w,
        eps_y=components.eps_y,
    )


def _zero_direct_outcome_shock(components: ModelClassComponents) -> ModelClassComponents:
    return _with_shock_y(components, np.zeros_like(components.shock_y))


def _recover_shock_path(components: ModelClassComponents) -> np.ndarray:
    _, _, vt = np.linalg.svd(np.asarray(components.shock_w, dtype=float), full_matrices=False)
    h = np.asarray(vt[0], dtype=float)
    h = h - float(h.mean())
    rms = float(np.sqrt(np.mean(h * h)))
    if rms <= 1e-12:
        raise ValueError("shock path is degenerate.")
    return h / rms


def _shock_qw(components: ModelClassComponents, *, T0: int) -> np.ndarray:
    z = np.asarray(components.Z, dtype=float)
    w_source = np.asarray(
        first_stage_component(components.D, components.Z)
        + components.shock_w
        + components.eps_w,
        dtype=float,
    )
    sig_w, _ = residuals_two_way_unit_slope(w_source[:, : int(T0)], z[: int(T0)])
    mz = projection_residualizer_const_z(z[: int(T0)])
    n, t0 = w_source[:, : int(T0)].shape
    return (w_source[:, : int(T0)] @ mz @ w_source[:, : int(T0)].T) / (
        float(t0) * float(sig_w) * float(n) ** 2
    )


def _benchmark(
    components: ModelClassComponents,
    *,
    branch: str,
    T0: int,
    T1: int,
) -> tuple[object, object]:
    bench_components = _zero_direct_outcome_shock(components)
    panel = build_branch_panel(
        bench_components,
        branch=branch,
        tau_true=active.TAU,
        T0=T0,
        T1=T1,
    )
    est = estimate_branch_panel(panel, bench_components.Z)
    return panel, est


def _direction_for_arm(
    *,
    arm: str,
    q_w: np.ndarray,
    weights_tsls: np.ndarray,
    weights_siv: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if arm == "A_TSLS_targeted":
        base = q_w @ weights_tsls
    elif arm == "B_SIV_targeted":
        base = q_w @ weights_siv
    elif arm == "C_random_geometry":
        base = q_w @ rng.normal(size=weights_tsls.size)
    elif arm == OPTIONAL_ARM_D:
        raw = q_w @ rng.normal(size=weights_tsls.size)
        controls = np.column_stack([
            np.ones(weights_tsls.size, dtype=float),
            np.asarray(weights_tsls, dtype=float),
        ])
        base = raw - controls @ (np.linalg.pinv(controls) @ raw)
    else:
        raise ValueError(f"unknown arm: {arm}")
    return standardize(base - float(np.mean(base)))


def _se_for_weights(
    panel,
    weights: np.ndarray,
    z_process: np.ndarray,
) -> tuple[float, float, float]:
    post = slice(int(panel.T0), int(panel.T0) + int(panel.T1))
    y = aggregate_series(panel.Y[:, post], weights)
    w = aggregate_series(panel.W[:, post], weights)
    z = np.asarray(z_process, dtype=float)[post]
    inf = raw_z_arp_design_based_se(
        y,
        w,
        z,
        z_process=np.asarray(z_process, dtype=float),
        ar_order=active.ARIMA_AR_ORDER,
        varz_mode="window",
    )
    return float(inf.se), float(inf.pi), float(inf.tau)


def _impact_terms(
    *,
    direction: np.ndarray,
    h_path: np.ndarray,
    z: np.ndarray,
    T0: int,
    T1: int,
    weights: np.ndarray,
    pi: float,
) -> tuple[float, float, float]:
    post = slice(int(T0), int(T0) + int(T1))
    lam = float(np.mean(np.asarray(weights, dtype=float) * np.asarray(direction, dtype=float)))
    h_slope = float(ols_slope_with_intercept(h_path[post], np.asarray(z, dtype=float)[post]))
    psi = float(lam * h_slope / float(pi)) if abs(float(pi)) > 1e-15 else float("nan")
    return lam, h_slope, psi


def _scale_for_direction(
    *,
    arm: str,
    direction: np.ndarray,
    h_path: np.ndarray,
    z: np.ndarray,
    T0: int,
    T1: int,
    weights_tsls: np.ndarray,
    weights_siv: np.ndarray,
    pi_tsls: float,
    pi_siv: float,
    se_tsls: float,
    se_siv: float,
    severity: float,
    scale_cap: float,
) -> dict[str, float | int | str | np.ndarray]:
    lam_t, h_slope, psi_t = _impact_terms(
        direction=direction, h_path=h_path, z=z, T0=T0, T1=T1,
        weights=weights_tsls, pi=pi_tsls,
    )
    lam_r, _, psi_r = _impact_terms(
        direction=direction, h_path=h_path, z=z, T0=T0, T1=T1,
        weights=weights_siv, pi=pi_siv,
    )
    kappa_t = abs(psi_t) / float(se_tsls) if float(se_tsls) > 0 else float("nan")
    kappa_r = abs(psi_r) / float(se_siv) if float(se_siv) > 0 else float("nan")

    # Sign orientation is done before severity scaling so signed-bias summaries
    # have a stable interpretation. Absolute-bias and RMSE are the primary
    # cross-arm metrics.
    sign = 1.0
    if arm == "A_TSLS_targeted":
        sign = 1.0 if psi_t >= 0.0 else -1.0
    elif arm == "B_SIV_targeted":
        sign = 1.0 if psi_r >= 0.0 else -1.0
    elif arm in {"C_random_geometry", OPTIONAL_ARM_D}:
        target = psi_t if kappa_t >= kappa_r else psi_r
        sign = 1.0 if target >= 0.0 else -1.0
    direction = sign * np.asarray(direction, dtype=float)
    lam_t *= sign
    lam_r *= sign
    psi_t *= sign
    psi_r *= sign

    kappa_t = abs(psi_t) / float(se_tsls) if float(se_tsls) > 0 else float("nan")
    kappa_r = abs(psi_r) / float(se_siv) if float(se_siv) > 0 else float("nan")
    k_value = max(kappa_t, kappa_r)

    severity = float(severity)
    excluded = int(False)
    cap_bound = int(False)
    if severity == 0.0:
        scale = 0.0
    elif (not math.isfinite(k_value)) or k_value < K_TOL:
        excluded = int(True)
        scale = float("nan")
    else:
        uncapped = severity / k_value
        if abs(uncapped) > float(scale_cap):
            cap_bound = int(True)
            scale = math.copysign(float(scale_cap), uncapped)
        else:
            scale = float(uncapped)

    achieved = abs(scale) * k_value if math.isfinite(scale) and math.isfinite(k_value) else float("nan")
    return {
        "direction": direction,
        "direction_sign": sign,
        "lambda_tsls": lam_t,
        "lambda_siv": lam_r,
        "h_slope": h_slope,
        "psi_tsls": psi_t,
        "psi_siv": psi_r,
        "kappa_tsls": kappa_t,
        "kappa_siv": kappa_r,
        "K": k_value,
        "scale_factor": scale,
        "achieved_max_kappa": achieved,
        "excluded": excluded,
        "cap_bound": cap_bound,
    }


def _iv_moment_fields(y: np.ndarray, w: np.ndarray, z: np.ndarray) -> dict[str, float]:
    zc = np.asarray(z, dtype=float) - float(np.mean(z))
    wc = np.asarray(w, dtype=float) - float(np.mean(w))
    residual = np.asarray(y, dtype=float) - float(active.TAU) * np.asarray(w, dtype=float)
    moment = float(np.mean(zc * residual))
    cov_zw = float(np.mean(zc * wc))
    return {
        "iv_moment_at_tau_true": moment,
        "abs_iv_moment_at_tau_true": abs(moment),
        "iv_cov_zw": cov_zw,
        "iv_moment_bias_ratio": moment / cov_zw if abs(cov_zw) > 1e-15 else float("nan"),
    }


def _inference_records(
    *,
    base: dict[str, object],
    y: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    z_process: np.ndarray,
    T1: int,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    crit_arima = _crit("ARIMA-Z", T1)
    crit_hac = _crit("HAC", T1)
    crit_ar = _crit("AR", T1)

    arima = raw_z_arp_design_based_se(
        y, w, z, z_process=z_process, ar_order=active.ARIMA_AR_ORDER, varz_mode="window"
    )
    bias = float(arima.tau - active.TAU)
    out.append({
        **base,
        "method": "ARIMA-Z",
        "tau": float(arima.tau),
        "bias": bias,
        "abs_bias": abs(bias),
        "squared_error": bias * bias,
        "se": float(arima.se),
        "covered": int(abs(bias) <= crit_arima * float(arima.se)),
        "critical": crit_arima,
        "pi": float(arima.pi),
    })

    hac = hac_delta_se(y, w, z, maxlags=active.HAC_LAGS)
    bias = float(hac.tau - active.TAU)
    out.append({
        **base,
        "method": "HAC",
        "tau": float(hac.tau),
        "bias": bias,
        "abs_bias": abs(bias),
        "squared_error": bias * bias,
        "se": float(hac.se),
        "covered": int(abs(bias) <= crit_hac * float(hac.se)),
        "critical": crit_hac,
        "pi": float(hac.pi),
    })

    ar = ar_style_test(y, w, z, active.TAU, maxlags=active.HAC_LAGS)
    out.append({
        **base,
        "method": "AR",
        "tau": float("nan"),
        "bias": float("nan"),
        "abs_bias": float("nan"),
        "squared_error": float("nan"),
        "se": float("nan"),
        "covered": int(abs(float(ar["t_stat"])) <= crit_ar),
        "critical": crit_ar,
        "pi": float("nan"),
    })
    return out


def run_sensitivity(
    *,
    reps: int,
    seed: int,
    severities: tuple[float, ...],
    arms: tuple[str, ...],
    scale_cap: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    exposure = load_exposure_profile(active.DATA_DIR / "exposure_restricted.csv")
    raw_records: list[dict[str, object]] = []
    cell_records: list[dict[str, object]] = []

    for config, T0, T1 in EVALUATION_CONFIGS:
        cs_rng = np.random.default_rng(stable_seed("fixed_cross_section", seed, config))
        cs_tmp = simulate_common_components(
            n=int(exposure.size),
            T=T0 + T1,
            params=active.PARAMS,
            rng=cs_rng,
            exposure_profile=exposure,
            split_index=T0,
        )
        fixed_cs = {k: cs_tmp[k] for k in ("Lw", "Ly", "theta_w", "theta_y", "S")}

        for rep in range(int(reps)):
            rng = np.random.default_rng(
                stable_seed("current_source_inference_method_comparison", seed, config, rep)
            )
            common = simulate_common_components(
                n=int(exposure.size),
                T=T0 + T1,
                params=active.PARAMS,
                rng=rng,
                exposure_profile=exposure,
                split_index=T0,
                fixed_cross_section=fixed_cs,
            )
            components = apply_preperiod_scale(
                components_from_common(common),
                T0=T0,
                scale=active.PREPERIOD_SCALES[config],
            )
            components = _zero_direct_outcome_shock(components)
            q_w = _shock_qw(components, T0=T0)
            q_eig = np.linalg.eigvalsh(0.5 * (q_w + q_w.T))
            q_eig = np.sort(np.clip(q_eig, 0.0, None))[::-1]
            q_trace = float(np.sum(q_eig))
            q_rank1_share = float(q_eig[0] / q_trace) if q_trace > 1e-15 else float("nan")
            q_rank2_share = float(np.sum(q_eig[:2]) / q_trace) if q_trace > 1e-15 else float("nan")
            h_path = _recover_shock_path(components)
            z = np.asarray(components.Z, dtype=float)

            for branch in SHOCK_BRANCHES:
                bench_panel, bench_est = _benchmark(
                    components,
                    branch=branch,
                    T0=T0,
                    T1=T1,
                )
                se_t, pi_t, tau_t0 = _se_for_weights(bench_panel, bench_est.weights_tsls, z)
                se_r, pi_r, tau_r0 = _se_for_weights(bench_panel, bench_est.weights_robust, z)

                for arm in arms:
                    dir_rng = np.random.default_rng(
                        stable_seed("shock_direction_arm", seed, config, branch, arm, rep)
                    )
                    direction = _direction_for_arm(
                        arm=arm,
                        q_w=q_w,
                        weights_tsls=bench_est.weights_tsls,
                        weights_siv=bench_est.weights_robust,
                        rng=dir_rng,
                    )

                    for severity in severities:
                        scale_info = _scale_for_direction(
                            arm=arm,
                            direction=direction,
                            h_path=h_path,
                            z=z,
                            T0=T0,
                            T1=T1,
                            weights_tsls=bench_est.weights_tsls,
                            weights_siv=bench_est.weights_robust,
                            pi_tsls=pi_t,
                            pi_siv=pi_r,
                            se_tsls=se_t,
                            se_siv=se_r,
                            severity=float(severity),
                            scale_cap=float(scale_cap),
                        )
                        cell_base = {
                            "config": config,
                            "T0": T0,
                            "T1": T1,
                            "rep": rep,
                            "branch": branch,
                            "arm": arm,
                            "severity": float(severity),
                            "benchmark_tau_tsls": tau_t0,
                            "benchmark_tau_siv": tau_r0,
                            "benchmark_se_tsls": se_t,
                            "benchmark_se_siv": se_r,
                            "benchmark_pi_tsls": pi_t,
                            "benchmark_pi_siv": pi_r,
                            "benchmark_weight_corr": float(bench_est.weight_correlation),
                            "q_w_rank1_trace_share": q_rank1_share,
                            "q_w_rank2_trace_share": q_rank2_share,
                            "direction_corr_tsls": safe_weight_correlation(
                                np.asarray(scale_info["direction"], dtype=float),
                                bench_est.weights_tsls,
                            ),
                            "direction_corr_siv": safe_weight_correlation(
                                np.asarray(scale_info["direction"], dtype=float),
                                bench_est.weights_robust,
                            ),
                            "scale_cap": float(scale_cap),
                            "excluded": int(scale_info["excluded"]),
                            "cap_bound": int(scale_info["cap_bound"]),
                            "lambda_tsls": float(scale_info["lambda_tsls"]),
                            "lambda_siv": float(scale_info["lambda_siv"]),
                            "h_slope": float(scale_info["h_slope"]),
                            "psi_tsls": float(scale_info["psi_tsls"]),
                            "psi_siv": float(scale_info["psi_siv"]),
                            "kappa_tsls": float(scale_info["kappa_tsls"]),
                            "kappa_siv": float(scale_info["kappa_siv"]),
                            "K": float(scale_info["K"]),
                            "scale_factor": float(scale_info["scale_factor"]),
                            "achieved_max_kappa": float(scale_info["achieved_max_kappa"]),
                            "direction_sign": float(scale_info["direction_sign"]),
                        }
                        cell_records.append(cell_base)
                        if int(scale_info["excluded"]):
                            continue

                        shock_y = np.outer(
                            float(scale_info["scale_factor"]) * np.asarray(scale_info["direction"], dtype=float),
                            h_path,
                        )
                        shocked_components = _with_shock_y(components, shock_y)
                        panel = build_branch_panel(
                            shocked_components,
                            branch=branch,
                            tau_true=active.TAU,
                            T0=T0,
                            T1=T1,
                        )
                        est = estimate_branch_panel(panel, shocked_components.Z)
                        post = slice(T0, T0 + T1)
                        active_shock = shocked_components.shock_y

                        for estimator, weights in (("TSLS", est.weights_tsls), ("SIV", est.weights_robust)):
                            y_post = aggregate_series(panel.Y[:, post], weights)
                            w_post = aggregate_series(panel.W[:, post], weights)
                            z_post = z[post]
                            moment = _iv_moment_fields(y_post, w_post, z_post)
                            direct = shock_bias_component(
                                shock_y=active_shock,
                                W=panel.W,
                                Z=z,
                                weights=weights,
                                T0=T0,
                                T1=T1,
                            )
                            base = {
                                **cell_base,
                                "estimator": estimator,
                                "weight_corr": float(est.weight_correlation),
                                "direct_shock_component": float(direct),
                                **moment,
                            }
                            raw_records.extend(
                                _inference_records(
                                    base=base,
                                    y=y_post,
                                    w=w_post,
                                    z=z_post,
                                    z_process=z,
                                    T1=T1,
                                )
                            )

        print(f"{config}: completed {reps} reps")

    return raw_records, cell_records


def summarise(raw_records: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    for row in raw_records:
        key = (
            row["method"],
            row["config"],
            row["T0"],
            row["T1"],
            row["branch"],
            row["arm"],
            row["severity"],
            row["estimator"],
        )
        groups[key].append(row)

    out: list[dict[str, object]] = []
    for (method, config, T0, T1, branch, arm, severity, estimator), rows in sorted(groups.items()):
        bias = np.asarray([float(r["bias"]) for r in rows if math.isfinite(float(r["bias"]))], dtype=float)
        se = np.asarray([float(r["se"]) for r in rows if math.isfinite(float(r["se"]))], dtype=float)
        direct = np.asarray([float(r["direct_shock_component"]) for r in rows], dtype=float)
        moment_ratio = np.asarray([float(r["iv_moment_bias_ratio"]) for r in rows], dtype=float)
        out.append({
            "method": method,
            "config": config,
            "T0": int(T0),
            "T1": int(T1),
            "branch": branch,
            "arm": arm,
            "severity": float(severity),
            "estimator": estimator,
            "n": int(len(rows)),
            "coverage": float(np.mean([float(r["covered"]) for r in rows])),
            "mean_bias": float(np.mean(bias)) if bias.size else float("nan"),
            "mean_abs_bias": float(np.mean(np.abs(bias))) if bias.size else float("nan"),
            "rmse": float(np.sqrt(np.mean(bias * bias))) if bias.size else float("nan"),
            "mean_se": float(np.mean(se)) if se.size else float("nan"),
            "mean_direct_shock_component": float(np.mean(direct)),
            "mean_iv_moment_bias_ratio": float(np.mean(moment_ratio)),
            "mean_weight_corr": float(np.mean([float(r["weight_corr"]) for r in rows])),
            "cap_bound_fraction": float(np.mean([float(r["cap_bound"]) for r in rows])),
        })
    return out


def geometry_summary(cell_records: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    for row in cell_records:
        groups[(row["config"], row["T0"], row["T1"], row["branch"], row["arm"], row["severity"])].append(row)
    out: list[dict[str, object]] = []
    for (config, T0, T1, branch, arm, severity), rows in sorted(groups.items()):
        out.append({
            "config": config,
            "T0": int(T0),
            "T1": int(T1),
            "branch": branch,
            "arm": arm,
            "severity": float(severity),
            "attempted_cells": int(len(rows)),
            "retained_cells": int(sum(1 for r in rows if int(r["excluded"]) == 0)),
            "excluded_cells": int(sum(1 for r in rows if int(r["excluded"]) != 0)),
            "cap_bound_cells": int(sum(1 for r in rows if int(r["cap_bound"]) != 0)),
            "mean_K": float(np.mean([float(r["K"]) for r in rows if math.isfinite(float(r["K"]))])),
            "mean_q_w_rank1_trace_share": float(np.mean([
                float(r["q_w_rank1_trace_share"]) for r in rows
                if math.isfinite(float(r["q_w_rank1_trace_share"]))
            ])),
            "mean_q_w_rank2_trace_share": float(np.mean([
                float(r["q_w_rank2_trace_share"]) for r in rows
                if math.isfinite(float(r["q_w_rank2_trace_share"]))
            ])),
            "mean_achieved_max_kappa": float(np.mean([
                float(r["achieved_max_kappa"]) for r in rows
                if math.isfinite(float(r["achieved_max_kappa"]))
            ])),
            "mean_direction_corr_tsls": float(np.mean([float(r["direction_corr_tsls"]) for r in rows])),
            "mean_direction_corr_siv": float(np.mean([float(r["direction_corr_siv"]) for r in rows])),
            "mean_lambda_tsls": float(np.mean([float(r["lambda_tsls"]) for r in rows])),
            "mean_lambda_siv": float(np.mean([float(r["lambda_siv"]) for r in rows])),
            "mean_kappa_tsls": float(np.mean([float(r["kappa_tsls"]) for r in rows])),
            "mean_kappa_siv": float(np.mean([float(r["kappa_siv"]) for r in rows])),
        })
    return out


def paired_differences(raw_records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_cell: dict[tuple, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in raw_records:
        key = (
            row["method"],
            row["config"],
            row["T0"],
            row["T1"],
            row["branch"],
            row["arm"],
            row["severity"],
            row["rep"],
        )
        by_cell[key][str(row["estimator"])] = row

    groups: dict[tuple, list[tuple[dict[str, object], dict[str, object]]]] = defaultdict(list)
    for key, pair in by_cell.items():
        if "TSLS" in pair and "SIV" in pair:
            method, config, T0, T1, branch, arm, severity, _rep = key
            groups[(method, config, T0, T1, branch, arm, severity)].append((pair["TSLS"], pair["SIV"]))

    out: list[dict[str, object]] = []
    for (method, config, T0, T1, branch, arm, severity), pairs in sorted(groups.items()):
        n = len(pairs)
        cov_diff = np.asarray([float(r["covered"]) - float(t["covered"]) for t, r in pairs], dtype=float)
        row: dict[str, object] = {
            "method": method,
            "config": config,
            "T0": int(T0),
            "T1": int(T1),
            "branch": branch,
            "arm": arm,
            "severity": float(severity),
            "n_pairs": int(n),
            "delta_coverage_siv_minus_tsls": float(np.mean(cov_diff)),
            "mcse_delta_coverage": float(np.std(cov_diff, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        }
        if all(math.isfinite(float(t["bias"])) and math.isfinite(float(r["bias"])) for t, r in pairs):
            b_t = np.asarray([float(t["bias"]) for t, _r in pairs], dtype=float)
            b_r = np.asarray([float(r["bias"]) for _t, r in pairs], dtype=float)
            d_bias = b_r - b_t
            d_abs = np.abs(b_r) - np.abs(b_t)
            mse_t = float(np.mean(b_t * b_t))
            mse_r = float(np.mean(b_r * b_r))
            rmse_t = math.sqrt(mse_t)
            rmse_r = math.sqrt(mse_r)
            if rmse_t > 0 and rmse_r > 0:
                influence = (b_r * b_r) / (2.0 * rmse_r) - (b_t * b_t) / (2.0 * rmse_t)
                mcse_rmse = float(np.std(influence, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
            else:
                mcse_rmse = float("nan")
            row.update({
                "bias_tsls": float(np.mean(b_t)),
                "bias_siv": float(np.mean(b_r)),
                "delta_signed_bias_siv_minus_tsls": float(np.mean(d_bias)),
                "mcse_delta_signed_bias": float(np.std(d_bias, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
                "delta_abs_bias_siv_minus_tsls": float(np.mean(d_abs)),
                "mcse_delta_abs_bias": float(np.std(d_abs, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
                "rmse_tsls": rmse_t,
                "rmse_siv": rmse_r,
                "delta_rmse_siv_minus_tsls": rmse_r - rmse_t,
                "mcse_delta_rmse": mcse_rmse,
            })
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    *,
    out: Path,
    reps: int,
    seed: int,
    severities: tuple[float, ...],
    arms: tuple[str, ...],
    scale_cap: float,
    raw_records: list[dict[str, object]],
    cell_records: list[dict[str, object]],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary = summarise(raw_records)
    geometry = geometry_summary(cell_records)
    paired = paired_differences(raw_records)
    _write_csv(out / "raw.csv", raw_records)
    _write_csv(out / "cell_status.csv", cell_records)
    _write_csv(out / "summary.csv", summary)
    _write_csv(out / "geometry_summary.csv", geometry)
    _write_csv(out / "paired_differences.csv", paired)
    manifest = {
        "analysis": "shock_orientation_sensitivity",
        "seed": int(seed),
        "reps": int(reps),
        "arms": list(arms),
        "shock_branches": list(SHOCK_BRANCHES),
        "severities": [float(v) for v in severities],
        "scale_cap": float(scale_cap),
        "k_tolerance": K_TOL,
        "tau_true": active.TAU,
        "arima_z_ar_order": active.ARIMA_AR_ORDER,
        "hac_lags": active.HAC_LAGS,
        "preperiod_scales": active.PREPERIOD_SCALES,
        "component_scale_multipliers": active.COMPONENT_SCALE_MULTIPLIERS,
        "params": dc.asdict(active.PARAMS),
        "outputs": {
            "raw": "raw.csv",
            "cell_status": "cell_status.csv",
            "summary": "summary.csv",
            "geometry_summary": "geometry_summary.csv",
            "paired_differences": "paired_differences.csv",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote {out}")
    print("ARIMA-Z paired differences, severity > 0:")
    for row in paired:
        if row["method"] != "ARIMA-Z" or float(row["severity"]) <= 0:
            continue
        print(
            f"  {row['config']} | {row['branch']} | {row['arm']} | c={row['severity']}: "
            f"d_abs={row.get('delta_abs_bias_siv_minus_tsls', float('nan')):+.4f}, "
            f"d_rmse={row.get('delta_rmse_siv_minus_tsls', float('nan')):+.4f}, "
            f"d_cov={row['delta_coverage_siv_minus_tsls']:+.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--severities",
        type=float,
        nargs="+",
        default=list(DEFAULT_SEVERITIES),
        help="Severity grid for symmetric max-impact scaling.",
    )
    parser.add_argument("--scale-cap", type=float, default=100.0)
    parser.add_argument(
        "--include-arm-d",
        action="store_true",
        help="Include the optional TSLS-shielded directional mechanism arm.",
    )
    args = parser.parse_args()

    severities = tuple(float(v) for v in args.severities)
    arms = tuple(list(CORE_ARMS) + ([OPTIONAL_ARM_D] if args.include_arm_d else []))
    print(
        f"Shock-orientation sensitivity reps={args.reps} seed={args.seed} "
        f"arms={arms} severities={severities} scale_cap={args.scale_cap}"
    )
    raw_records, cell_records = run_sensitivity(
        reps=int(args.reps),
        seed=int(args.seed),
        severities=severities,
        arms=arms,
        scale_cap=float(args.scale_cap),
    )
    write_outputs(
        out=args.out,
        reps=int(args.reps),
        seed=int(args.seed),
        severities=severities,
        arms=arms,
        scale_cap=float(args.scale_cap),
        raw_records=raw_records,
        cell_records=cell_records,
    )


if __name__ == "__main__":
    main()
