#!/usr/bin/env python3
"""Monte Carlo comparison reported in the paper.

The simulation uses the restricted empirical exposure profile, rank-two
generalized fixed-effect components extracted from the empirical panel, and an
aggregate shock correlated with the instrument. State shock loadings are drawn
around the exposure profile. The same draws are used for TSLS and Robust.
Within every replication and design, state exposure is estimated from the
five-year learning window before either estimator constructs its weights.
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

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import paper_config as settings  # noqa: E402
from simulation_reporting import (  # noqa: E402
    BRANCHES,
    H_Z_CORR,
    THETA_W_EXPOSURE_CORR,
    THETA_Y_EXPOSURE_CORR,
    THETA_Y_SCALE_MULTIPLIER,
    critical_value,
    iv_moment_fields,
    paired_differences,
    summarise,
)
from sim.dgp import (  # noqa: E402
    instrument_series,
    parametric_correlated_loading,
    stable_seed,
    standardize,
)
from sim.estimators import aggregate_series, ols_slope_with_intercept, unit_level_slope_vector  # noqa: E402
from sim.helpers import EVALUATION_CONFIGS, load_exposure_profile, shock_bias_component  # noqa: E402
from sim.inference import ar_style_test, hac_delta_se, raw_z_arp_design_based_se  # noqa: E402
from sim.model_class import BRANCH_SWITCHES, ModelClassComponents, build_branch_panel, estimate_branch_panel  # noqa: E402


OUT_DIR = ROOT / "outputs" / "simulation"
EMPIRICAL_YEAR_START = 2008
EMPIRICAL_YEAR_END = 2022


def _load_restricted_empirical_panel() -> dict[str, np.ndarray | list[str]]:
    import pandas as pd

    exposure_df = pd.read_csv(settings.DATA_DIR / "exposure_restricted.csv")
    states = list(exposure_df["unit"].astype(str))
    exposure = exposure_df["D_tsls"].to_numpy(dtype=float)

    panel = pd.read_csv(settings.DATA_DIR / "panel_lag2.csv")
    panel = panel[
        panel["state"].isin(states)
        & panel["year"].between(EMPIRICAL_YEAR_START, EMPIRICAL_YEAR_END)
    ].copy()
    years = np.arange(EMPIRICAL_YEAR_START, EMPIRICAL_YEAR_END + 1, dtype=int)

    def matrix(column: str) -> np.ndarray:
        pivot = panel.pivot(index="state", columns="year", values=column).reindex(index=states, columns=years)
        values = pivot.to_numpy(dtype=float)
        if values.shape != (len(states), years.size) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"Empirical restricted panel is incomplete for {column}.")
        return values

    z_by_year = panel.drop_duplicates("year").set_index("year").reindex(years)["Z_t_lag2"].to_numpy(dtype=float)
    if z_by_year.size != years.size or not np.all(np.isfinite(z_by_year)):
        raise RuntimeError("Empirical instrument path is incomplete.")
    return {
        "states": states,
        "years": years,
        "exposure": exposure,
        "Y": matrix("Y_it_lag2"),
        "W": matrix("W_it_lag2"),
        "Z": z_by_year,
    }


def _two_way_residual(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    return values - values.mean(axis=1, keepdims=True) - values.mean(axis=0, keepdims=True) + float(values.mean())


def _svd_parts(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    u, s, vt = np.linalg.svd(values, full_matrices=False)
    k = min(int(rank), s.size)
    loadings = u[:, :k] * s[:k]
    scores = vt[:k, :]
    approx = loadings @ scores
    return loadings, scores, approx


def _empirical_svd_base() -> dict[str, np.ndarray | list[str] | dict[str, float]]:
    panel = _load_restricted_empirical_panel()
    exposure = np.asarray(panel["exposure"], dtype=float)
    z = np.asarray(panel["Z"], dtype=float)
    W_obs = np.asarray(panel["W"], dtype=float)
    Y_obs = np.asarray(panel["Y"], dtype=float)

    # The treatment GFE component is extracted after removing the exposure-driven
    # first-stage part. The outcome GFE component is extracted after removing the
    # scalar treatment effect used as the simulation target.
    W_resid = _two_way_residual(W_obs - np.outer(exposure, z))
    Y_resid = _two_way_residual(Y_obs - float(settings.TAU_TRUE) * W_obs)
    lw_loadings, lw_scores, lowrank_w_emp = _svd_parts(W_resid, int(settings.PARAMS.rank))
    ly_loadings, ly_scores, lowrank_y_emp = _svd_parts(Y_resid, int(settings.PARAMS.rank))

    return {
        **panel,
        "lw_loadings": lw_loadings,
        "lw_scores": lw_scores,
        "ly_loadings": ly_loadings,
        "ly_scores": ly_scores,
        "lowrank_w_emp": lowrank_w_emp,
        "lowrank_y_emp": lowrank_y_emp,
        "stats": {
            "empirical_t": float(z.size),
            "rank": float(settings.PARAMS.rank),
            "sd_lowrank_w_emp": float(np.std(lowrank_w_emp, ddof=0)),
            "sd_lowrank_y_emp": float(np.std(lowrank_y_emp, ddof=0)),
            "sd_w_residual": float(np.std(W_resid, ddof=0)),
            "sd_y_residual": float(np.std(Y_resid, ddof=0)),
            "share_w_residual_rank2": float(np.sum(lowrank_w_emp * lowrank_w_emp) / np.sum(W_resid * W_resid)),
            "share_y_residual_rank2": float(np.sum(lowrank_y_emp * lowrank_y_emp) / np.sum(Y_resid * Y_resid)),
        },
    }


def _extend_scores(scores_emp: np.ndarray, *, T: int, rng: np.random.Generator) -> np.ndarray:
    """Create fixed longer-window factor scores with empirical score scale."""
    scores_emp = np.asarray(scores_emp, dtype=float)
    out = np.zeros((scores_emp.shape[0], int(T)), dtype=float)
    for k in range(scores_emp.shape[0]):
        sd = float(np.std(scores_emp[k], ddof=0))
        if sd <= 0.0:
            raise RuntimeError("Empirical SVD score has zero scale.")
        draw = rng.normal(size=int(T))
        draw = draw - float(draw.mean())
        draw_sd = float(np.std(draw, ddof=0))
        if draw_sd <= 0.0:
            raise RuntimeError("Generated longer-window SVD score has zero scale.")
        out[k] = (draw / draw_sd) * sd
    return out


def _scale_to_sd(matrix: np.ndarray, target_sd: float) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    current_sd = float(np.std(values, ddof=0))
    if current_sd <= 0.0:
        raise RuntimeError("Cannot rescale a zero-SD GFE matrix.")
    return values * (float(target_sd) / current_sd)


def _fixed_geometry(
    *,
    config: str,
    T0: int,
    T1: int,
    seed: int,
    empirical: dict[str, np.ndarray | list[str] | dict[str, float]],
    gfe_scale_mode: str,
    gfe_scale_multiplier: float,
    gfe_w_multiplier: float | None,
    gfe_y_multiplier: float | None,
    shock_w_multiplier: float,
    shock_y_multiplier: float,
) -> dict[str, np.ndarray | dict[str, float] | str]:
    exposure = np.asarray(empirical["exposure"], dtype=float)
    T = int(T0) + int(T1)
    rng = np.random.default_rng(stable_seed("original_paper_empirical_svd_fixed_geometry", seed, config))

    if T == int(np.asarray(empirical["Z"]).size):
        lowrank_w = np.asarray(empirical["lowrank_w_emp"], dtype=float)
        lowrank_y = np.asarray(empirical["lowrank_y_emp"], dtype=float)
        gfe_source = "direct_empirical_rank2_svd"
    else:
        lowrank_w = np.asarray(empirical["lw_loadings"], dtype=float) @ _extend_scores(
            np.asarray(empirical["lw_scores"], dtype=float),
            T=T,
            rng=rng,
        )
        lowrank_y = np.asarray(empirical["ly_loadings"], dtype=float) @ _extend_scores(
            np.asarray(empirical["ly_scores"], dtype=float),
            T=T,
            rng=rng,
        )
        gfe_source = "empirical_rank2_svd_loadings_with_fixed_synthetic_scores"

    if gfe_scale_mode == "calibrated":
        common = float(gfe_scale_multiplier)
        w_multiplier = common if gfe_w_multiplier is None else float(gfe_w_multiplier)
        y_multiplier = common if gfe_y_multiplier is None else float(gfe_y_multiplier)
        for name, multiplier in (("gfe_w_multiplier", w_multiplier), ("gfe_y_multiplier", y_multiplier)):
            if not math.isfinite(multiplier) or multiplier <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        lowrank_w = _scale_to_sd(lowrank_w, w_multiplier * float(settings.PARAMS.lowrank_w_sd))
        lowrank_y = _scale_to_sd(lowrank_y, y_multiplier * float(settings.PARAMS.lowrank_y_sd))
        gfe_source = (
            f"{gfe_source}_scaled_to_paper_gfe_sd_"
            f"w{w_multiplier:g}_y{y_multiplier:g}"
        )
    elif gfe_scale_mode != "raw":
        raise ValueError("gfe_scale_mode must be 'raw' or 'calibrated'.")

    for name, multiplier in (("shock_w_multiplier", float(shock_w_multiplier)), ("shock_y_multiplier", float(shock_y_multiplier))):
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")

    theta_w = parametric_correlated_loading(
        exposure,
        corr=THETA_W_EXPOSURE_CORR,
        sd=float(shock_w_multiplier) * float(settings.PARAMS.shock_w_sd),
        rng=rng,
    )
    theta_y = parametric_correlated_loading(
        exposure,
        corr=THETA_Y_EXPOSURE_CORR,
        sd=float(shock_y_multiplier) * THETA_Y_SCALE_MULTIPLIER * float(settings.PARAMS.shock_w_sd),
        rng=rng,
    )
    return {
        "lowrank_w": lowrank_w,
        "lowrank_y": lowrank_y,
        "theta_w": theta_w,
        "theta_y": theta_y,
        "gfe_source": gfe_source,
        "stats": {
            **dict(empirical["stats"]),  # type: ignore[arg-type]
            "sd_lowrank_w_used": float(np.std(lowrank_w, ddof=0)),
            "sd_lowrank_y_used": float(np.std(lowrank_y, ddof=0)),
        },
    }


def _draw_components(
    *,
    config: str,
    rep: int,
    seed: int,
    T0: int,
    T1: int,
    exposure: np.ndarray,
    geometry: dict[str, np.ndarray | dict[str, float] | str],
) -> ModelClassComponents:
    rng = np.random.default_rng(stable_seed("original_paper_empirical_svd_rep", seed, config, rep))
    T = int(T0) + int(T1)
    n = int(exposure.size)

    z_latent = instrument_series(T, settings.PARAMS, rng)
    z_tilde = instrument_series(T, settings.PARAMS, rng)
    Z = float(settings.PARAMS.z_sd) * z_latent
    H = H_Z_CORR * standardize(z_latent) + math.sqrt(max(1.0 - H_Z_CORR**2, 0.0)) * standardize(z_tilde)

    cov = np.array(
        [
            [float(settings.PARAMS.eps_w_sd) ** 2, float(settings.PARAMS.eps_yw_corr) * float(settings.PARAMS.eps_w_sd) * float(settings.PARAMS.eps_y_sd)],
            [float(settings.PARAMS.eps_yw_corr) * float(settings.PARAMS.eps_w_sd) * float(settings.PARAMS.eps_y_sd), float(settings.PARAMS.eps_y_sd) ** 2],
        ],
        dtype=float,
    )
    eps = rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=(n, T))

    return ModelClassComponents(
        Z=np.asarray(Z, dtype=float),
        D=np.asarray(exposure, dtype=float),
        alpha_w=np.zeros(n, dtype=float),
        alpha_y=np.zeros(n, dtype=float),
        lowrank_w=np.asarray(geometry["lowrank_w"], dtype=float),
        lowrank_y=np.asarray(geometry["lowrank_y"], dtype=float),
        shock_w=np.outer(np.asarray(geometry["theta_w"], dtype=float), H),
        shock_y=np.outer(np.asarray(geometry["theta_y"], dtype=float), H),
        eps_w=eps[:, :, 0],
        eps_y=eps[:, :, 1],
    )


def _component_biases(components: ModelClassComponents, panel, weights: np.ndarray, T0: int, T1: int) -> dict[str, float]:
    post = slice(int(T0), int(T0) + int(T1))
    z = np.asarray(components.Z, dtype=float)[post]
    gfe_on, shock_on = BRANCH_SWITCHES[panel.branch]

    def slope(component: np.ndarray) -> float:
        return ols_slope_with_intercept(aggregate_series(component[:, post], weights), z)

    denom = ols_slope_with_intercept(aggregate_series(panel.W[:, post], weights), z)
    if abs(denom) <= 1e-15:
        raise ValueError("post-learning weighted denominator is too close to zero.")

    gfe_y = components.lowrank_y if gfe_on else np.zeros_like(components.lowrank_y)
    shock_y = components.shock_y if shock_on else np.zeros_like(components.shock_y)
    eps_y = components.eps_y
    total_y = panel.Y - float(settings.TAU_TRUE) * panel.W
    return {
        "bias_component_total": float(slope(total_y) / denom),
        "bias_component_gfe": float(slope(gfe_y) / denom),
        "bias_component_shock": float(slope(shock_y) / denom),
        "bias_component_eps": float(slope(eps_y) / denom),
    }


def run(reps: int, seed: int) -> tuple[list[dict[str, object]], dict[str, int], dict[str, object]]:
    return run_with_options(
        reps=reps,
        seed=seed,
        gfe_scale_mode="calibrated",
        gfe_w_multiplier=1.0,
        gfe_y_multiplier=15.0,
    )


def run_with_options(
    *,
    reps: int,
    seed: int,
    gfe_scale_mode: str,
    gfe_scale_multiplier: float = 1.0,
    gfe_w_multiplier: float | None = None,
    gfe_y_multiplier: float | None = None,
    shock_w_multiplier: float = 1.0,
    shock_y_multiplier: float = 1.0,
) -> tuple[list[dict[str, object]], dict[str, int], dict[str, object]]:
    empirical = _empirical_svd_base()
    exposure = np.asarray(empirical["exposure"], dtype=float)
    records: list[dict[str, object]] = []
    skipped: dict[str, int] = {}
    geometry_manifest: dict[str, object] = {}

    for config, T0, T1 in EVALUATION_CONFIGS:
        geometry = _fixed_geometry(
            config=config,
            T0=T0,
            T1=T1,
            seed=seed,
            empirical=empirical,
            gfe_scale_mode=gfe_scale_mode,
            gfe_scale_multiplier=gfe_scale_multiplier,
            gfe_w_multiplier=gfe_w_multiplier,
            gfe_y_multiplier=gfe_y_multiplier,
            shock_w_multiplier=shock_w_multiplier,
            shock_y_multiplier=shock_y_multiplier,
        )
        common = float(gfe_scale_multiplier)
        geometry_manifest[config] = {
            "gfe_source": str(geometry["gfe_source"]),
            "stats": geometry["stats"],
            "gfe_scale_mode": str(gfe_scale_mode),
            "gfe_scale_multiplier": common,
            "gfe_w_multiplier": common if gfe_w_multiplier is None else float(gfe_w_multiplier),
            "gfe_y_multiplier": common if gfe_y_multiplier is None else float(gfe_y_multiplier),
            "shock_w_multiplier": float(shock_w_multiplier),
            "shock_y_multiplier": float(shock_y_multiplier),
        }
        skipped_config = 0
        for rep in range(int(reps)):
            try:
                components = _draw_components(
                    config=config,
                    rep=rep,
                    seed=seed,
                    T0=T0,
                    T1=T1,
                    exposure=exposure,
                    geometry=geometry,
                )
                for branch in BRANCHES:
                    panel = build_branch_panel(components, branch=branch, tau_true=settings.TAU_TRUE, T0=T0, T1=T1)
                    first_stage_slopes = unit_level_slope_vector(
                        panel.W[:, :T0],
                        components.Z[:T0],
                    )
                    est = estimate_branch_panel(panel, components.Z, first_stage_slopes=first_stage_slopes)
                    post = slice(T0, T0 + T1)
                    shock_on = BRANCH_SWITCHES[branch][1]
                    active_shock = components.shock_y if shock_on else np.zeros_like(components.shock_y)

                    for estimator, weights in (("TSLS", est.weights_tsls), ("SIV", est.weights_robust)):
                        y = aggregate_series(panel.Y[:, post], weights)
                        w = aggregate_series(panel.W[:, post], weights)
                        z = np.asarray(components.Z, dtype=float)[post]
                        moment_fields = iv_moment_fields(y, w, z)
                        component_fields = _component_biases(components, panel, weights, T0, T1)
                        shock = shock_bias_component(
                            shock_y=active_shock,
                            W=panel.W,
                            Z=components.Z,
                            weights=weights,
                            T0=T0,
                            T1=T1,
                        )
                        base = {
                            "rep": rep,
                            "config": config,
                            "T0": T0,
                            "T1": T1,
                            "branch": branch,
                            "estimator": estimator,
                            "weight_corr": float(est.weight_correlation),
                            "shock": float(shock),
                            **moment_fields,
                            **component_fields,
                        }

                        arima = raw_z_arp_design_based_se(
                            y,
                            w,
                            z,
                            z_process=np.asarray(components.Z, dtype=float),
                            ar_order=settings.ARIMA_AR_ORDER,
                            varz_mode="window",
                        )
                        bias = float(arima.tau - settings.TAU_TRUE)
                        records.append(
                            {
                                **base,
                                "method": "ARIMA-Z",
                                "tau": float(arima.tau),
                                "bias": bias,
                                "se": float(arima.se),
                                "covered": int(abs(bias) <= critical_value("ARIMA-Z", T1) * float(arima.se)),
                                "pi": float(arima.pi),
                                "critical": critical_value("ARIMA-Z", T1),
                            }
                        )

                        hac = hac_delta_se(y, w, z, maxlags=settings.HAC_LAGS)
                        bias = float(hac.tau - settings.TAU_TRUE)
                        records.append(
                            {
                                **base,
                                "method": "HAC",
                                "tau": float(hac.tau),
                                "bias": bias,
                                "se": float(hac.se),
                                "covered": int(abs(bias) <= critical_value("HAC", T1) * float(hac.se)),
                                "pi": float(hac.pi),
                                "critical": critical_value("HAC", T1),
                            }
                        )

                        ar = ar_style_test(y, w, z, settings.TAU_TRUE, maxlags=settings.HAC_LAGS)
                        records.append(
                            {
                                **base,
                                "method": "AR",
                                "tau": float("nan"),
                                "bias": float("nan"),
                                "se": float("nan"),
                                "covered": int(abs(float(ar["t_stat"])) <= critical_value("AR", T1)),
                                "pi": float("nan"),
                                "critical": critical_value("AR", T1),
                            }
                        )
            except ValueError as exc:
                skipped_config += 1
                print(f"  skipped {config} rep {rep}: {exc}", file=sys.stderr)
                continue
        skipped[config] = skipped_config
        print(f"  {config}: {int(reps) - skipped_config} usable reps" + (f" ({skipped_config} skipped)" if skipped_config else ""))
    return records, skipped, geometry_manifest


def component_summary(records: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        if row["method"] != "ARIMA-Z":
            continue
        groups[(row["config"], row["T0"], row["T1"], row["branch"], row["estimator"])].append(row)

    fields = ["bias_component_total", "bias_component_gfe", "bias_component_shock", "bias_component_eps"]
    rows: list[dict[str, object]] = []
    for (config, T0, T1, branch, estimator), rs in sorted(groups.items()):
        out: dict[str, object] = {
            "config": config,
            "T0": T0,
            "T1": T1,
            "branch": branch,
            "estimator": estimator,
            "n": len(rs),
        }
        for field in fields:
            values = np.asarray([float(r[field]) for r in rs], dtype=float)
            out[f"mean_{field}"] = float(np.mean(values))
            out[f"rmse_{field}"] = float(math.sqrt(float(np.mean(values * values))))
            out[f"mean_abs_{field}"] = float(np.mean(np.abs(values)))
        rows.append(out)
    return rows


def write_outputs(
    *,
    out_dir: Path,
    records: list[dict[str, object]],
    summary: list[dict[str, object]],
    paired: list[dict[str, object]],
    components: list[dict[str, object]],
    reps: int,
    seed: int,
    skipped: dict[str, int],
    geometry_manifest: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in {
        "raw.csv": records,
        "summary.csv": summary,
        "paired_differences.csv": paired,
        "component_summary.csv": components,
    }.items():
        with (out_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    manifest = {
        "simulation": "exposure_correlated_aggregate_shock_learning_window_exposure",
        "reps_requested": int(reps),
        "seed": int(seed),
        "tau_true": float(settings.TAU_TRUE),
        "empirical_panel_window": [EMPIRICAL_YEAR_START, EMPIRICAL_YEAR_END],
        "exposure_estimation": {
            "window": "learning_window",
            "description": (
                "State exposure is estimated from the first T0 simulated years "
                "within each replication and design before estimator weights are formed."
            ),
        },
        "gfe_construction": geometry_manifest,
        "aggregate_shock_construction": {
            "h_z_corr": H_Z_CORR,
            "theta_w_exposure_corr": THETA_W_EXPOSURE_CORR,
            "theta_y_exposure_corr": THETA_Y_EXPOSURE_CORR,
            "theta_y_scale_multiplier": THETA_Y_SCALE_MULTIPLIER,
            "estimator_oriented": False,
            "severity_normalized_to_tsls_se": False,
        },
        "inference": {
            "arima_z_ar_order": int(settings.ARIMA_AR_ORDER),
            "hac_lags": int(settings.HAC_LAGS),
        },
        "derived_scale_constants": dc.asdict(settings.EMPIRICAL_SCALE),
        "skipped": skipped,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the paper's exposure-correlated simulation.")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--gfe-scale-mode",
        choices=("raw", "calibrated"),
        default="calibrated",
        help="calibrated applies the reported GFE scales; raw keeps the empirical SVD amplitude.",
    )
    parser.add_argument(
        "--gfe-scale-multiplier",
        type=float,
        default=1.0,
        help="Common multiplier applied to the paper's GFE component scales.",
    )
    parser.add_argument(
        "--gfe-w-multiplier",
        type=float,
        default=1.0,
        help="Treatment-side GFE multiplier. Overrides --gfe-scale-multiplier for lowrank_w when provided.",
    )
    parser.add_argument(
        "--gfe-y-multiplier",
        type=float,
        default=15.0,
        help="Outcome-side GFE multiplier. Overrides --gfe-scale-multiplier for lowrank_y when provided.",
    )
    parser.add_argument(
        "--shock-w-multiplier",
        type=float,
        default=1.0,
        help="Treatment-side aggregate-shock loading multiplier.",
    )
    parser.add_argument(
        "--shock-y-multiplier",
        type=float,
        default=1.0,
        help="Outcome-side aggregate-shock loading multiplier.",
    )
    args = parser.parse_args()

    print(
        "Exposure-correlated simulation  "
        f"reps={args.reps}  seed={args.seed}  gfe_scale_mode={args.gfe_scale_mode}  "
        f"gfe_scale_multiplier={args.gfe_scale_multiplier:g}  "
        f"gfe_w_multiplier={args.gfe_w_multiplier}  gfe_y_multiplier={args.gfe_y_multiplier}  "
        f"shock_w_multiplier={args.shock_w_multiplier:g}  shock_y_multiplier={args.shock_y_multiplier:g}"
    )
    records, skipped, geometry_manifest = run_with_options(
        reps=args.reps,
        seed=args.seed,
        gfe_scale_mode=args.gfe_scale_mode,
        gfe_scale_multiplier=args.gfe_scale_multiplier,
        gfe_w_multiplier=args.gfe_w_multiplier,
        gfe_y_multiplier=args.gfe_y_multiplier,
        shock_w_multiplier=args.shock_w_multiplier,
        shock_y_multiplier=args.shock_y_multiplier,
    )
    summary = summarise(records)
    paired = paired_differences(records)
    components = component_summary(records)
    write_outputs(
        out_dir=args.out,
        records=records,
        summary=summary,
        paired=paired,
        components=components,
        reps=args.reps,
        seed=args.seed,
        skipped=skipped,
        geometry_manifest=geometry_manifest,
    )
    print(f"  wrote {args.out}")
    print("  ARIMA-Z summary:")
    for row in summary:
        if row["method"] != "ARIMA-Z":
            continue
        print(
            f"    {row['config']:<14} {row['branch']:<22} {row['estimator']:<4} "
            f"rate={row['coverage']:.3f} bias={row['mean_bias']:+.4f} rmse={row['rmse']:.4f}"
        )


if __name__ == "__main__":
    main()
