#!/usr/bin/env python3
"""Run the prespecified state-panel and aggregate-confounding sensitivities.

This analysis imports the estimator and inference implementation used by the
principal exposure-correlated simulation. It evaluates the same design on the
full and restricted state panels and over a fixed confounding-parameter grid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paper_config as active  # noqa: E402
import run_simulation as principal  # noqa: E402
from simulation_reporting import (  # noqa: E402
    BRANCHES,
    THETA_W_EXPOSURE_CORR as THETA_W_PI_CORR,
    THETA_Y_EXPOSURE_CORR as THETA_Y_PI_CORR,
    _crit,
    _iv_moment_fields,
)
from sim.dgp import (  # noqa: E402
    instrument_series,
    parametric_correlated_loading,
    stable_seed,
    standardize,
)
from sim.estimators import aggregate_series, unit_level_slope_vector  # noqa: E402
from sim.helpers import shock_bias_component  # noqa: E402
from sim.inference import ar_style_test, hac_delta_se, raw_z_arp_design_based_se  # noqa: E402
from sim.model_class import (  # noqa: E402
    BRANCH_SWITCHES,
    ModelClassComponents,
    build_branch_panel,
    estimate_branch_panel,
)

PROTOCOL = ROOT / "analysis" / "protocols" / "state_panel_and_confounding_sensitivity_v1.json"
PANEL = active.DATA_DIR / "panel_lag2.csv"
RESTRICTED_EXPOSURE = active.DATA_DIR / "exposure_restricted.csv"
FULL_EXPOSURE = ROOT / "data" / "exposure_full.csv"
YEAR_START = 2008
YEAR_END = 2022
WINDOWS = {
    "finite_length": (5, 10),
    "longer_length": (30, 60),
}
SAMPLES = ("restricted", "full")
GRID_H_Z = (0.25, 0.50, 0.75)
GRID_THETA_Y_SCALE = (1.50, 3.00, 4.50)
BASE_H_Z = 0.50
BASE_THETA_Y_SCALE = 3.00
GFE_W_MULTIPLIER = 1.0
GFE_Y_MULTIPLIER = 15.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def load_protocol() -> dict[str, object]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["protocol"] != "state_panel_and_confounding_sensitivity_v1":
        raise RuntimeError("Unexpected sensitivity protocol identifier.")
    return protocol


def load_empirical_panel(sample: str) -> dict[str, object]:
    if sample not in SAMPLES:
        raise ValueError(f"Unknown sample: {sample}")
    exposure_path = RESTRICTED_EXPOSURE if sample == "restricted" else FULL_EXPOSURE
    exposure_df = pd.read_csv(exposure_path)
    states = list(exposure_df["unit"].astype(str))
    exposure = exposure_df["D_tsls"].to_numpy(dtype=float)
    expected_n = 49 if sample == "restricted" else 51
    if len(states) != expected_n or len(set(states)) != expected_n:
        raise RuntimeError(f"{sample} exposure input does not contain {expected_n} unique jurisdictions.")
    if sample == "restricted" and ({"CA", "VT"} & set(states)):
        raise RuntimeError("Restricted exposure input contains California or Vermont.")
    if sample == "full" and not {"CA", "VT"}.issubset(states):
        raise RuntimeError("Full exposure input omits California or Vermont.")

    panel = pd.read_csv(PANEL)
    panel = panel[
        panel["state"].isin(states)
        & panel["year"].between(YEAR_START, YEAR_END)
    ].copy()
    years = np.arange(YEAR_START, YEAR_END + 1, dtype=int)

    def matrix(column: str) -> np.ndarray:
        pivot = panel.pivot(index="state", columns="year", values=column).reindex(index=states, columns=years)
        values = pivot.to_numpy(dtype=float)
        if values.shape != (expected_n, years.size) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"{sample} panel is incomplete for {column}.")
        return values

    z_by_year = panel.drop_duplicates("year").set_index("year").reindex(years)["Z_t_lag2"].to_numpy(dtype=float)
    if z_by_year.size != years.size or not np.all(np.isfinite(z_by_year)):
        raise RuntimeError(f"{sample} instrument path is incomplete.")
    return {
        "sample": sample,
        "states": states,
        "years": years,
        "exposure": exposure,
        "Y": matrix("Y_it_lag2"),
        "W": matrix("W_it_lag2"),
        "Z": z_by_year,
        "exposure_source": relative(exposure_path),
    }


def two_way_residual(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    return values - values.mean(axis=1, keepdims=True) - values.mean(axis=0, keepdims=True) + float(values.mean())


def svd_parts(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(np.asarray(matrix, dtype=float), full_matrices=False)
    k = min(int(rank), s.size)
    loadings = u[:, :k] * s[:k]
    scores = vt[:k, :]
    return loadings, scores, loadings @ scores


def empirical_svd_base(sample: str) -> dict[str, object]:
    panel = load_empirical_panel(sample)
    exposure = np.asarray(panel["exposure"], dtype=float)
    z = np.asarray(panel["Z"], dtype=float)
    w_obs = np.asarray(panel["W"], dtype=float)
    y_obs = np.asarray(panel["Y"], dtype=float)
    w_resid = two_way_residual(w_obs - np.outer(exposure, z))
    y_resid = two_way_residual(y_obs - float(active.TAU) * w_obs)
    lw_loadings, lw_scores, lowrank_w = svd_parts(w_resid, int(active.PARAMS.rank))
    ly_loadings, ly_scores, lowrank_y = svd_parts(y_resid, int(active.PARAMS.rank))
    return {
        **panel,
        "lw_loadings": lw_loadings,
        "lw_scores": lw_scores,
        "ly_loadings": ly_loadings,
        "ly_scores": ly_scores,
        "lowrank_w_emp": lowrank_w,
        "lowrank_y_emp": lowrank_y,
        "stats": {
            "n_units": len(panel["states"]),
            "empirical_t": int(z.size),
            "rank": int(active.PARAMS.rank),
            "sd_lowrank_w_emp": float(np.std(lowrank_w, ddof=0)),
            "sd_lowrank_y_emp": float(np.std(lowrank_y, ddof=0)),
            "sd_w_residual": float(np.std(w_resid, ddof=0)),
            "sd_y_residual": float(np.std(y_resid, ddof=0)),
            "share_w_residual_rank2": float(np.sum(lowrank_w * lowrank_w) / np.sum(w_resid * w_resid)),
            "share_y_residual_rank2": float(np.sum(lowrank_y * lowrank_y) / np.sum(y_resid * y_resid)),
        },
    }


def fixed_geometry(
    *,
    sample: str,
    config: str,
    T0: int,
    T1: int,
    seed: int,
    empirical: dict[str, object],
    theta_y_scale: float,
) -> dict[str, object]:
    exposure = np.asarray(empirical["exposure"], dtype=float)
    total_t = int(T0) + int(T1)
    # The restricted baseline intentionally uses the principal runner's seed tag.
    rng = np.random.default_rng(stable_seed("original_paper_empirical_svd_fixed_geometry", seed, config))
    if total_t == int(np.asarray(empirical["Z"]).size):
        lowrank_w = np.asarray(empirical["lowrank_w_emp"], dtype=float)
        lowrank_y = np.asarray(empirical["lowrank_y_emp"], dtype=float)
        source = f"{sample}_direct_empirical_rank2_svd"
    else:
        lowrank_w = np.asarray(empirical["lw_loadings"], dtype=float) @ principal._extend_scores(
            np.asarray(empirical["lw_scores"], dtype=float), T=total_t, rng=rng
        )
        lowrank_y = np.asarray(empirical["ly_loadings"], dtype=float) @ principal._extend_scores(
            np.asarray(empirical["ly_scores"], dtype=float), T=total_t, rng=rng
        )
        source = f"{sample}_empirical_rank2_loadings_with_synthetic_scores"

    lowrank_w = principal._scale_to_sd(lowrank_w, GFE_W_MULTIPLIER * float(active.PARAMS.lowrank_w_sd))
    lowrank_y = principal._scale_to_sd(lowrank_y, GFE_Y_MULTIPLIER * float(active.PARAMS.lowrank_y_sd))
    theta_w = parametric_correlated_loading(
        exposure,
        corr=THETA_W_PI_CORR,
        sd=float(active.PARAMS.shock_w_sd),
        rng=rng,
    )
    theta_y = parametric_correlated_loading(
        exposure,
        corr=THETA_Y_PI_CORR,
        sd=float(theta_y_scale) * float(active.PARAMS.shock_w_sd),
        rng=rng,
    )
    return {
        "lowrank_w": lowrank_w,
        "lowrank_y": lowrank_y,
        "theta_w": theta_w,
        "theta_y": theta_y,
        "source": source,
        "stats": {
            **dict(empirical["stats"]),
            "sd_lowrank_w_used": float(np.std(lowrank_w, ddof=0)),
            "sd_lowrank_y_used": float(np.std(lowrank_y, ddof=0)),
            "sd_theta_w_used": float(np.std(theta_w, ddof=0)),
            "sd_theta_y_used": float(np.std(theta_y, ddof=0)),
        },
    }


def draw_components(
    *,
    config: str,
    rep: int,
    seed: int,
    T0: int,
    T1: int,
    exposure: np.ndarray,
    geometry: dict[str, object],
    h_z_corr: float,
) -> ModelClassComponents:
    rng = np.random.default_rng(stable_seed("original_paper_empirical_svd_rep", seed, config, rep))
    total_t = int(T0) + int(T1)
    n = int(exposure.size)
    z_latent = instrument_series(total_t, active.PARAMS, rng)
    z_independent = instrument_series(total_t, active.PARAMS, rng)
    z = float(active.PARAMS.z_sd) * z_latent
    h = float(h_z_corr) * standardize(z_latent) + math.sqrt(1.0 - float(h_z_corr) ** 2) * standardize(z_independent)
    cov = np.array(
        [
            [float(active.PARAMS.eps_w_sd) ** 2, float(active.PARAMS.eps_yw_corr) * float(active.PARAMS.eps_w_sd) * float(active.PARAMS.eps_y_sd)],
            [float(active.PARAMS.eps_yw_corr) * float(active.PARAMS.eps_w_sd) * float(active.PARAMS.eps_y_sd), float(active.PARAMS.eps_y_sd) ** 2],
        ],
        dtype=float,
    )
    eps = rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=(n, total_t))
    return ModelClassComponents(
        Z=np.asarray(z, dtype=float),
        D=np.asarray(exposure, dtype=float),
        alpha_w=np.zeros(n, dtype=float),
        alpha_y=np.zeros(n, dtype=float),
        lowrank_w=np.asarray(geometry["lowrank_w"], dtype=float),
        lowrank_y=np.asarray(geometry["lowrank_y"], dtype=float),
        shock_w=np.outer(np.asarray(geometry["theta_w"], dtype=float), h),
        shock_y=np.outer(np.asarray(geometry["theta_y"], dtype=float), h),
        eps_w=eps[:, :, 0],
        eps_y=eps[:, :, 1],
    )


def evaluate_components(
    *,
    records: list[dict[str, object]],
    dimensions: dict[str, object],
    components: ModelClassComponents,
    rep: int,
    config: str,
    T0: int,
    T1: int,
) -> None:
    for branch in BRANCHES:
        panel = build_branch_panel(components, branch=branch, tau_true=active.TAU, T0=T0, T1=T1)
        first_stage_slopes = unit_level_slope_vector(panel.W[:, :T0], components.Z[:T0])
        estimate = estimate_branch_panel(panel, components.Z, first_stage_slopes=first_stage_slopes)
        post = slice(T0, T0 + T1)
        shock_on = BRANCH_SWITCHES[branch][1]
        active_shock = components.shock_y if shock_on else np.zeros_like(components.shock_y)
        for estimator, weights in (("TSLS", estimate.weights_tsls), ("SIV", estimate.weights_robust)):
            y = aggregate_series(panel.Y[:, post], weights)
            w = aggregate_series(panel.W[:, post], weights)
            z = np.asarray(components.Z, dtype=float)[post]
            moment = _iv_moment_fields(y, w, z)
            component_fields = principal._component_biases(components, panel, weights, T0, T1)
            shock = shock_bias_component(
                shock_y=active_shock,
                W=panel.W,
                Z=components.Z,
                weights=weights,
                T0=T0,
                T1=T1,
            )
            base = {
                **dimensions,
                "rep": rep,
                "config": config,
                "T0": T0,
                "T1": T1,
                "branch": branch,
                "estimator": estimator,
                "weight_corr": float(estimate.weight_correlation),
                "shock": float(shock),
                **moment,
                **component_fields,
            }
            arima = raw_z_arp_design_based_se(
                y,
                w,
                z,
                z_process=np.asarray(components.Z, dtype=float),
                ar_order=active.ARIMA_AR_ORDER,
                varz_mode="window",
            )
            bias = float(arima.tau - active.TAU)
            records.append({
                **base,
                "method": "ARIMA-Z",
                "tau": float(arima.tau),
                "bias": bias,
                "se": float(arima.se),
                "covered": int(abs(bias) <= _crit("ARIMA-Z", T1) * float(arima.se)),
                "pi": float(arima.pi),
                "critical": _crit("ARIMA-Z", T1),
            })
            hac = hac_delta_se(y, w, z, maxlags=active.HAC_LAGS)
            bias = float(hac.tau - active.TAU)
            records.append({
                **base,
                "method": "HAC",
                "tau": float(hac.tau),
                "bias": bias,
                "se": float(hac.se),
                "covered": int(abs(bias) <= _crit("HAC", T1) * float(hac.se)),
                "pi": float(hac.pi),
                "critical": _crit("HAC", T1),
            })
            ar = ar_style_test(y, w, z, active.TAU, maxlags=active.HAC_LAGS)
            records.append({
                **base,
                "method": "AR",
                "tau": float("nan"),
                "bias": float("nan"),
                "se": float("nan"),
                "covered": int(abs(float(ar["t_stat"])) <= _crit("AR", T1)),
                "pi": float("nan"),
                "critical": _crit("AR", T1),
            })


def summary_rows(records: list[dict[str, object]], dimensions: tuple[str, ...]) -> list[dict[str, object]]:
    keys = (*dimensions, "method", "config", "T0", "T1", "branch", "estimator")
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        groups[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, object]] = []
    for values, rows in sorted(groups.items()):
        bias = np.asarray([float(row["bias"]) for row in rows if math.isfinite(float(row["bias"]))])
        se = np.asarray([float(row["se"]) for row in rows if math.isfinite(float(row["se"]))])
        finite_moment = [float(row["iv_moment_bias_ratio"]) for row in rows if math.isfinite(float(row["iv_moment_bias_ratio"]))]
        result = dict(zip(keys, values, strict=True))
        result.update({
            "n": len(rows),
            "coverage": float(np.mean([int(row["covered"]) for row in rows])),
            "mean_bias": float(np.mean(bias)) if bias.size else float("nan"),
            "mean_abs_error": float(np.mean(np.abs(bias))) if bias.size else float("nan"),
            "rmse": float(math.sqrt(float(np.mean(bias * bias)))) if bias.size else float("nan"),
            "mean_se": float(np.mean(se)) if se.size else float("nan"),
            "sd_bias": float(np.std(bias, ddof=1)) if bias.size > 1 else float("nan"),
            "mean_weight_corr": float(np.mean([float(row["weight_corr"]) for row in rows])),
            "mean_shock": float(np.mean([float(row["shock"]) for row in rows])),
            "mean_iv_moment_bias_ratio": float(np.mean(finite_moment)),
            "critical": float(rows[0]["critical"]),
        })
        output.append(result)
    return output


def paired_rows(records: list[dict[str, object]], dimensions: tuple[str, ...]) -> list[dict[str, object]]:
    cell_keys = (*dimensions, "config", "T0", "T1", "branch", "rep")
    by_cell: dict[tuple[object, ...], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in records:
        if row["method"] == "ARIMA-Z":
            by_cell[tuple(row[key] for key in cell_keys)][str(row["estimator"])] = row
    group_keys = (*dimensions, "config", "T0", "T1", "branch")
    groups: dict[tuple[object, ...], list[tuple[dict[str, object], dict[str, object]]]] = defaultdict(list)
    for values, pair in by_cell.items():
        if set(pair) == {"TSLS", "SIV"}:
            groups[values[:-1]].append((pair["TSLS"], pair["SIV"]))

    def mcse(values: np.ndarray) -> float:
        return float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else float("nan")

    output: list[dict[str, object]] = []
    for values, pairs in sorted(groups.items()):
        signed = np.asarray([float(siv["bias"]) - float(tsls["bias"]) for tsls, siv in pairs])
        absolute = np.asarray([abs(float(siv["bias"])) - abs(float(tsls["bias"])) for tsls, siv in pairs])
        coverage = np.asarray([int(siv["covered"]) - int(tsls["covered"]) for tsls, siv in pairs], dtype=float)
        rmse_t = math.sqrt(float(np.mean([float(tsls["bias"]) ** 2 for tsls, _ in pairs])))
        rmse_s = math.sqrt(float(np.mean([float(siv["bias"]) ** 2 for _, siv in pairs])))
        result = dict(zip(group_keys, values, strict=True))
        result.update({
            "method": "ARIMA-Z",
            "n_pairs": len(pairs),
            "bias_tsls": float(np.mean([float(tsls["bias"]) for tsls, _ in pairs])),
            "bias_siv": float(np.mean([float(siv["bias"]) for _, siv in pairs])),
            "delta_signed_bias_siv_minus_tsls": float(np.mean(signed)),
            "mcse_delta_signed_bias": mcse(signed),
            "delta_abs_error_siv_minus_tsls": float(np.mean(absolute)),
            "mcse_delta_abs_error": mcse(absolute),
            "rmse_tsls": rmse_t,
            "rmse_siv": rmse_s,
            "delta_rmse_siv_minus_tsls": rmse_s - rmse_t,
            "delta_coverage_siv_minus_tsls": float(np.mean(coverage)),
            "mcse_delta_coverage": mcse(coverage),
        })
        output.append(result)
    return output


def component_rows(records: list[dict[str, object]], dimensions: tuple[str, ...]) -> list[dict[str, object]]:
    fields = ("bias_component_total", "bias_component_gfe", "bias_component_shock", "bias_component_eps")
    keys = (*dimensions, "config", "T0", "T1", "branch", "estimator")
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        if row["method"] == "ARIMA-Z":
            groups[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, object]] = []
    for values, rows in sorted(groups.items()):
        result = dict(zip(keys, values, strict=True))
        result["n"] = len(rows)
        for field in fields:
            data = np.asarray([float(row[field]) for row in rows])
            result[f"mean_{field}"] = float(np.mean(data))
            result[f"mean_abs_{field}"] = float(np.mean(np.abs(data)))
            result[f"rmse_{field}"] = float(math.sqrt(float(np.mean(data * data))))
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows available for {path.name}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_sample_sensitivity(reps: int, seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    geometry_manifest: dict[str, object] = {}
    for sample in SAMPLES:
        empirical = empirical_svd_base(sample)
        exposure = np.asarray(empirical["exposure"], dtype=float)
        for config, (T0, T1) in WINDOWS.items():
            geometry = fixed_geometry(
                sample=sample,
                config=config,
                T0=T0,
                T1=T1,
                seed=seed,
                empirical=empirical,
                theta_y_scale=BASE_THETA_Y_SCALE,
            )
            geometry_manifest[f"{sample}:{config}"] = {
                "source": geometry["source"],
                "stats": geometry["stats"],
                "exposure_source": empirical["exposure_source"],
            }
            for rep in range(reps):
                components = draw_components(
                    config=config,
                    rep=rep,
                    seed=seed,
                    T0=T0,
                    T1=T1,
                    exposure=exposure,
                    geometry=geometry,
                    h_z_corr=BASE_H_Z,
                )
                evaluate_components(
                    records=records,
                    dimensions={
                        "sample": sample,
                        "n_units": exposure.size,
                        "h_z_corr": BASE_H_Z,
                        "theta_y_scale_multiplier": BASE_THETA_Y_SCALE,
                    },
                    components=components,
                    rep=rep,
                    config=config,
                    T0=T0,
                    T1=T1,
                )
            print(f"sample sensitivity: {sample} {config} completed ({reps} replications)", flush=True)
    return records, geometry_manifest


def run_confounding_grid(reps: int, seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    sample = "restricted"
    config = "finite_length"
    T0, T1 = WINDOWS[config]
    empirical = empirical_svd_base(sample)
    exposure = np.asarray(empirical["exposure"], dtype=float)
    records: list[dict[str, object]] = []
    geometry_manifest: dict[str, object] = {}
    for theta_y_scale in GRID_THETA_Y_SCALE:
        geometry = fixed_geometry(
            sample=sample,
            config=config,
            T0=T0,
            T1=T1,
            seed=seed,
            empirical=empirical,
            theta_y_scale=theta_y_scale,
        )
        geometry_manifest[f"theta_y_scale:{theta_y_scale:g}"] = {
            "source": geometry["source"],
            "stats": geometry["stats"],
            "exposure_source": empirical["exposure_source"],
        }
        for h_z_corr in GRID_H_Z:
            for rep in range(reps):
                components = draw_components(
                    config=config,
                    rep=rep,
                    seed=seed,
                    T0=T0,
                    T1=T1,
                    exposure=exposure,
                    geometry=geometry,
                    h_z_corr=h_z_corr,
                )
                evaluate_components(
                    records=records,
                    dimensions={
                        "sample": sample,
                        "n_units": exposure.size,
                        "h_z_corr": h_z_corr,
                        "theta_y_scale_multiplier": theta_y_scale,
                    },
                    components=components,
                    rep=rep,
                    config=config,
                    T0=T0,
                    T1=T1,
                )
            print(
                f"confounding grid: h_z={h_z_corr:.2f} theta_y_scale={theta_y_scale:.2f} completed ({reps} replications)",
                flush=True,
            )
    return records, geometry_manifest


def write_outputs(
    *,
    mode: str,
    output: Path,
    records: list[dict[str, object]],
    dimensions: tuple[str, ...],
    geometry_manifest: dict[str, object],
    reps: int,
    seed: int,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    files = {
        "raw.csv": records,
        "summary.csv": summary_rows(records, dimensions),
        "paired_differences.csv": paired_rows(records, dimensions),
        "component_summary.csv": component_rows(records, dimensions),
    }
    for name, rows in files.items():
        write_csv(output / name, rows)
    input_paths = [
        Path(__file__).resolve(),
        PROTOCOL,
        ROOT / "paper_config.py",
        ROOT / "simulation_reporting.py",
        ROOT / "sim" / "dgp.py",
        ROOT / "sim" / "helpers.py",
        ROOT / "sim" / "model_class.py",
        ROOT / "sim" / "estimators.py",
        ROOT / "sim" / "inference.py",
        ROOT / "run_simulation.py",
        PANEL,
        RESTRICTED_EXPOSURE,
        FULL_EXPOSURE,
    ]
    manifest = {
        "analysis": "state_panel_and_confounding_sensitivity_v1",
        "mode": mode,
        "reps": reps,
        "seed": seed,
        "tau_true": float(active.TAU),
        "protocol": relative(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "dimensions": list(dimensions),
        "geometry": geometry_manifest,
        "environment": {
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scipy", "statsmodels")
            },
        },
        "input_sha256": {relative(path): sha256(path) for path in input_paths},
        "output_sha256": {name: sha256(output / name) for name in sorted(files)},
        "skipped_replications": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sample", "grid"), required=True)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    if args.reps != int(protocol["common"]["reps"]):
        print(f"nonproduction replication count requested: {args.reps}", file=sys.stderr)
    if args.seed != int(protocol["common"]["seed"]):
        raise ValueError("The prespecified protocol requires seed 20261024.")
    if args.out.exists():
        raise FileExistsError(f"Output directory already exists: {args.out}")
    if args.mode == "sample":
        records, geometry = run_sample_sensitivity(args.reps, args.seed)
    else:
        records, geometry = run_confounding_grid(args.reps, args.seed)
    dimensions = ("sample", "n_units", "h_z_corr", "theta_y_scale_multiplier")
    write_outputs(
        mode=args.mode,
        output=args.out,
        records=records,
        dimensions=dimensions,
        geometry_manifest=geometry,
        reps=args.reps,
        seed=args.seed,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
