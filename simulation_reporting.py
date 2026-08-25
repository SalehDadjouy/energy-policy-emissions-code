"""Simulation constants and Monte Carlo summaries used by the paper."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import NormalDist

import numpy as np
from scipy.stats import t as t_dist

from paper_config import TAU_TRUE


BRANCHES = ("Basic", "GFE", "Aggregate Shock", "GFE + Agg. Shock")
H_Z_CORR = 0.50
THETA_W_EXPOSURE_CORR = 0.20
THETA_Y_EXPOSURE_CORR = 0.30
THETA_Y_SCALE_MULTIPLIER = 3.00


def critical_value(method: str, t1: int) -> float:
    """Return the two-sided 95 percent critical value used in the paper."""
    if method == "HAC":
        return float(NormalDist().inv_cdf(0.975))
    if int(t1) <= 10:
        return float(t_dist.ppf(0.975, df=int(t1) - 2))
    return float(NormalDist().inv_cdf(0.975))


def iv_moment_fields(y: np.ndarray, w: np.ndarray, z: np.ndarray) -> dict[str, float]:
    """Evaluate the covariance-form IV moment at the imposed causal response."""
    zc = np.asarray(z, dtype=float) - float(np.mean(z))
    wc = np.asarray(w, dtype=float) - float(np.mean(w))
    residual = np.asarray(y, dtype=float) - TAU_TRUE * np.asarray(w, dtype=float)
    moment = float(np.mean(zc * residual))
    cov_zw = float(np.mean(zc * wc))
    return {
        "iv_moment_tau": moment,
        "iv_moment_abs_tau": abs(moment),
        "iv_cov_zw": cov_zw,
        "iv_moment_bias_ratio": moment / cov_zw if abs(cov_zw) > 1e-15 else float("nan"),
    }


# Internal aliases used by the prespecified sensitivity runners.
_crit = critical_value
_iv_moment_fields = iv_moment_fields


def summarise(records: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        key = (row["method"], row["config"], row["T0"], row["T1"], row["branch"], row["estimator"])
        groups[key].append(row)

    rows: list[dict[str, object]] = []
    for (method, config, t0, t1, branch, estimator), group in sorted(groups.items()):
        bias = np.asarray(
            [float(row["bias"]) for row in group if math.isfinite(float(row["bias"]))],
            dtype=float,
        )
        se = np.asarray(
            [float(row["se"]) for row in group if math.isfinite(float(row["se"]))],
            dtype=float,
        )
        moment_ratio = np.asarray(
            [
                float(row["iv_moment_bias_ratio"])
                for row in group
                if math.isfinite(float(row["iv_moment_bias_ratio"]))
            ],
            dtype=float,
        )
        rows.append(
            {
                "method": method,
                "config": config,
                "T0": t0,
                "T1": t1,
                "branch": branch,
                "estimator": estimator,
                "n": len(group),
                "coverage": float(np.mean([int(row["covered"]) for row in group])),
                "mean_bias": float(np.mean(bias)) if bias.size else float("nan"),
                "rmse": float(math.sqrt(float(np.mean(bias * bias)))) if bias.size else float("nan"),
                "mean_se": float(np.mean(se)) if se.size else float("nan"),
                "sd_bias": float(np.std(bias, ddof=1)) if bias.size > 1 else float("nan"),
                "mean_weight_corr": float(np.mean([float(row["weight_corr"]) for row in group])),
                "mean_shock": float(np.mean([float(row["shock"]) for row in group])),
                "mean_iv_moment_bias_ratio": float(np.mean(moment_ratio)),
                "critical": float(group[0]["critical"]),
            }
        )
    return rows


def paired_differences(records: list[dict[str, object]]) -> list[dict[str, object]]:
    cells: dict[tuple[object, ...], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in records:
        if row["method"] != "ARIMA-Z":
            continue
        key = (row["config"], row["T0"], row["T1"], row["branch"], row["rep"])
        cells[key][str(row["estimator"])] = row

    groups: dict[tuple[object, ...], list[tuple[dict[str, object], dict[str, object]]]] = defaultdict(list)
    for (config, t0, t1, branch, _), pair in cells.items():
        if "TSLS" in pair and "SIV" in pair:
            groups[(config, t0, t1, branch)].append((pair["TSLS"], pair["SIV"]))

    def mcse(values: np.ndarray) -> float:
        return float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else float("nan")

    rows: list[dict[str, object]] = []
    for (config, t0, t1, branch), pairs in sorted(groups.items()):
        signed = np.asarray([float(s["bias"]) - float(t["bias"]) for t, s in pairs], dtype=float)
        absolute = np.asarray([abs(float(s["bias"])) - abs(float(t["bias"])) for t, s in pairs], dtype=float)
        coverage = np.asarray([int(s["covered"]) - int(t["covered"]) for t, s in pairs], dtype=float)
        rmse_t = math.sqrt(float(np.mean([float(t["bias"]) ** 2 for t, _ in pairs])))
        rmse_s = math.sqrt(float(np.mean([float(s["bias"]) ** 2 for _, s in pairs])))
        rows.append(
            {
                "method": "ARIMA-Z",
                "config": config,
                "T0": t0,
                "T1": t1,
                "branch": branch,
                "n_pairs": len(pairs),
                "delta_coverage_siv_minus_tsls": float(np.mean(coverage)),
                "mcse_delta_coverage": mcse(coverage),
                "bias_tsls": float(np.mean([float(t["bias"]) for t, _ in pairs])),
                "bias_siv": float(np.mean([float(s["bias"]) for _, s in pairs])),
                "delta_signed_bias_siv_minus_tsls": float(np.mean(signed)),
                "mcse_delta_signed_bias": mcse(signed),
                "delta_abs_bias_siv_minus_tsls": float(np.mean(absolute)),
                "mcse_delta_abs_bias": mcse(absolute),
                "rmse_tsls": rmse_t,
                "rmse_siv": rmse_s,
                "delta_rmse_siv_minus_tsls": float(rmse_s - rmse_t),
            }
        )
    return rows
