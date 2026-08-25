#!/usr/bin/env python3
"""Finite-window Robust SIV weight-learning and inference analysis.

The analysis constructs branch-specific finite-DGP population-objective
weights and evaluates exposure and objective learning with paired
common-random-number comparisons at T0=5 and T1=10.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paper_config as active  # noqa: E402
import run_simulation as production  # noqa: E402
from simulation_reporting import BRANCHES, _crit  # noqa: E402
from sim.estimators import (  # noqa: E402
    aggregate_series,
    safe_weight_correlation,
    slope_ratio,
    tsls_weights,
    unit_level_slope_vector,
)
from sim.inference import ar_style_test, hac_delta_se, raw_z_arp_design_based_se  # noqa: E402
from sim.model_class import build_branch_panel  # noqa: E402
from sim.robust_weights import robust_q_components, robust_weights_from_q  # noqa: E402


PROTOCOL_PATH = ROOT / "analysis" / "protocols" / "finite_window_weight_learning_v1.json"
VARIANT_ORDER = (
    "robust_population_true_d",
    "robust_learned_q_true_d",
    "robust_population_q_full_dhat",
    "robust_full_empirical",
    "robust_reported_simulation",
    "tsls_true_d",
    "tsls_full_dhat",
    "tsls_learning_dhat",
)
ROBUST_VARIANTS = set(VARIANT_ORDER[:5])
PAIR_REFERENCES = {
    "robust_learned_q_true_d": "robust_population_true_d",
    "robust_population_q_full_dhat": "robust_population_true_d",
    "robust_full_empirical": "robust_population_true_d",
    "robust_reported_simulation": "robust_population_true_d",
    "tsls_full_dhat": "tsls_true_d",
    "tsls_learning_dhat": "tsls_true_d",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_reproduction_context(protocol: dict[str, object]) -> dict[str, object]:
    observed_environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "statsmodels": statsmodels.__version__,
    }
    reference = protocol["production_reference"]
    reference_dir = ROOT / reference["output_directory"]
    required = tuple(reference["reference_files"])
    missing = [filename for filename in required if not (reference_dir / filename).is_file()]
    if missing:
        raise RuntimeError(f"Principal simulation references are missing: {missing}.")

    return {
        "environment": observed_environment,
        "reference_sha256": {
            filename: sha256_file(reference_dir / filename)
            for filename in required
        },
    }


def production_objects(protocol: dict[str, object]):
    reference = protocol["production_reference"]
    empirical = production._empirical_svd_base()
    exposure = np.asarray(empirical["exposure"], dtype=float)
    if exposure.size != int(reference["n_states"]):
        raise RuntimeError("Restricted exposure vector does not match the prespecified state count.")
    geometry = production._fixed_geometry(
        config=str(reference["configuration"]),
        T0=int(reference["T0"]),
        T1=int(reference["T1"]),
        seed=int(reference["baseline_seed"]),
        empirical=empirical,
        gfe_scale_mode=str(reference["gfe_scale_mode"]),
        gfe_scale_multiplier=1.0,
        gfe_w_multiplier=float(reference["gfe_w_multiplier"]),
        gfe_y_multiplier=float(reference["gfe_y_multiplier"]),
        shock_w_multiplier=float(reference["shock_w_multiplier"]),
        shock_y_multiplier=float(reference["shock_y_multiplier"]),
    )
    return empirical, exposure, geometry


def draw_components(
    protocol: dict[str, object],
    *,
    rep: int,
    seed: int,
    exposure: np.ndarray,
    geometry: dict[str, object],
):
    reference = protocol["production_reference"]
    return production._draw_components(
        config=str(reference["configuration"]),
        rep=int(rep),
        seed=int(seed),
        T0=int(reference["T0"]),
        T1=int(reference["T1"]),
        exposure=exposure,
        geometry=geometry,
    )


def branch_panel(protocol: dict[str, object], components, branch: str):
    reference = protocol["production_reference"]
    return build_branch_panel(
        components,
        branch=branch,
        tau_true=float(active.TAU),
        T0=int(reference["T0"]),
        T1=int(reference["T1"]),
    )


def q_objects(panel, z: np.ndarray, t0: int) -> tuple[np.ndarray, np.ndarray, float, dict[str, object]]:
    q = robust_q_components(
        y0=panel.Y[:, :t0],
        w0=panel.W[:, :t0],
        z0=np.asarray(z, dtype=float)[:t0],
    )
    return (
        np.asarray(q["Q_Y"], dtype=float),
        np.asarray(q["Q_W"], dtype=float),
        float(q["rho"]),
        q,
    )


def solve_robust(d: np.ndarray, qy: np.ndarray, qw: np.ndarray, rho: float) -> np.ndarray:
    weights, diagnostics = robust_weights_from_q(
        first_stage_slopes=np.asarray(d, dtype=float),
        Q_Y=np.asarray(qy, dtype=float),
        Q_W=np.asarray(qw, dtype=float),
        rho=float(rho),
    )
    if abs(float(np.mean(weights))) > 1e-8:
        raise RuntimeError("Robust weight centering constraint failed.")
    if abs(float(np.mean(weights * np.asarray(d, dtype=float))) - 1.0) > 1e-8:
        raise RuntimeError("Robust exposure constraint failed.")
    if float(diagnostics["admissible"]) != 1.0:
        raise RuntimeError("Robust weight problem was not admissible.")
    return weights


def branch_key(branch: str) -> str:
    return branch.lower().replace(" + ", "_plus_").replace(" ", "_").replace(".", "")


def rms(x: np.ndarray) -> float:
    values = np.asarray(x, dtype=float)
    return float(math.sqrt(float(np.mean(values * values))))


def effective_abs_count(weights: np.ndarray) -> float:
    absolute = np.abs(np.asarray(weights, dtype=float))
    total = float(np.sum(absolute))
    if total <= 0.0:
        return float("nan")
    share = absolute / total
    return float(1.0 / np.sum(share * share))


def compare_weights(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    distance = rms(np.asarray(candidate) - np.asarray(reference))
    reference_scale = rms(reference)
    return {
        "rms_distance": distance,
        "relative_rms_distance": distance / reference_scale if reference_scale > 0.0 else float("nan"),
        "weight_correlation": safe_weight_correlation(candidate, reference),
        "max_abs_difference": float(np.max(np.abs(np.asarray(candidate) - np.asarray(reference)))),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows available for {path.name}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def construct_population_objective(
    protocol: dict[str, object],
    *,
    output_dir: Path,
    draws: int | None = None,
    checkpoints: tuple[int, ...] | None = None,
) -> dict[str, dict[str, np.ndarray | float]]:
    bank = protocol["population_objective_bank"]
    reference = protocol["production_reference"]
    total_draws = int(bank["draws"] if draws is None else draws)
    checkpoint_values = tuple(int(v) for v in (bank["checkpoints"] if checkpoints is None else checkpoints))
    if checkpoint_values[-1] != total_draws:
        raise ValueError("Final checkpoint must equal the population-objective draw count.")

    _, exposure, geometry = production_objects(protocol)
    n = exposure.size
    t0 = int(reference["T0"])
    bank_seed = int(bank["seed"])
    sums = {
        branch: {
            "Q_Y": np.zeros((n, n), dtype=float),
            "Q_W": np.zeros((n, n), dtype=float),
            "rho": 0.0,
        }
        for branch in BRANCHES
    }
    first_half = {
        branch: {
            "Q_Y": np.zeros((n, n), dtype=float),
            "Q_W": np.zeros((n, n), dtype=float),
            "rho": 0.0,
        }
        for branch in BRANCHES
    }
    second_half = {
        branch: {
            "Q_Y": np.zeros((n, n), dtype=float),
            "Q_W": np.zeros((n, n), dtype=float),
            "rho": 0.0,
        }
        for branch in BRANCHES
    }
    checkpoint_weights: dict[int, dict[str, np.ndarray]] = {}

    half = total_draws // 2
    for rep in range(total_draws):
        components = draw_components(
            protocol,
            rep=rep,
            seed=bank_seed,
            exposure=exposure,
            geometry=geometry,
        )
        for branch in BRANCHES:
            panel = branch_panel(protocol, components, branch)
            qy, qw, rho, _ = q_objects(panel, components.Z, t0)
            sums[branch]["Q_Y"] += qy
            sums[branch]["Q_W"] += qw
            sums[branch]["rho"] += rho
            destination = first_half if rep < half else second_half
            destination[branch]["Q_Y"] += qy
            destination[branch]["Q_W"] += qw
            destination[branch]["rho"] += rho

        count = rep + 1
        if count in checkpoint_values:
            checkpoint_weights[count] = {}
            for branch in BRANCHES:
                checkpoint_weights[count][branch] = solve_robust(
                    exposure,
                    sums[branch]["Q_Y"] / float(count),
                    sums[branch]["Q_W"] / float(count),
                    float(sums[branch]["rho"]) / float(count),
                )
        if count % max(min(1000, total_draws), 1) == 0 or count == total_draws:
            print(f"  population-objective bank: {count}/{total_draws}", flush=True)

    final: dict[str, dict[str, np.ndarray | float]] = {}
    matrix_payload: dict[str, np.ndarray] = {}
    weight_rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    final_weights = checkpoint_weights[total_draws]
    for branch in BRANCHES:
        qy = sums[branch]["Q_Y"] / float(total_draws)
        qw = sums[branch]["Q_W"] / float(total_draws)
        rho = float(sums[branch]["rho"]) / float(total_draws)
        weights = final_weights[branch]
        final[branch] = {"Q_Y": qy, "Q_W": qw, "rho": rho, "weights": weights}
        key = branch_key(branch)
        matrix_payload[f"{key}__Q_Y"] = qy
        matrix_payload[f"{key}__Q_W"] = qw
        matrix_payload[f"{key}__rho"] = np.asarray([rho], dtype=float)
        matrix_payload[f"{key}__weights"] = weights
        for state_index, weight in enumerate(weights):
            weight_rows.append(
                {"branch": branch, "state_index": state_index, "weight": float(weight)}
            )

        for checkpoint in checkpoint_values:
            comparison = compare_weights(checkpoint_weights[checkpoint][branch], weights)
            convergence_rows.append(
                {
                    "branch": branch,
                    "comparison": f"checkpoint_{checkpoint}_vs_{total_draws}",
                    "draws": checkpoint,
                    **comparison,
                }
            )

        first_count = half
        second_count = total_draws - half
        first_weights = solve_robust(
            exposure,
            first_half[branch]["Q_Y"] / float(first_count),
            first_half[branch]["Q_W"] / float(first_count),
            float(first_half[branch]["rho"]) / float(first_count),
        )
        second_weights = solve_robust(
            exposure,
            second_half[branch]["Q_Y"] / float(second_count),
            second_half[branch]["Q_W"] / float(second_count),
            float(second_half[branch]["rho"]) / float(second_count),
        )
        convergence_rows.append(
            {
                "branch": branch,
                "comparison": "split_half",
                "draws": total_draws,
                **compare_weights(first_weights, second_weights),
            }
        )

    np.savez_compressed(output_dir / "population_objective.npz", **matrix_payload)
    write_csv(output_dir / "population_weights.csv", weight_rows)
    write_csv(output_dir / "population_convergence.csv", convergence_rows)
    return final


def load_population_objective(output_dir: Path) -> dict[str, dict[str, np.ndarray | float]]:
    path = output_dir / "population_objective.npz"
    if not path.exists():
        raise RuntimeError("Population-objective file is missing.")
    payload = np.load(path)
    final: dict[str, dict[str, np.ndarray | float]] = {}
    for branch in BRANCHES:
        key = branch_key(branch)
        final[branch] = {
            "Q_Y": np.asarray(payload[f"{key}__Q_Y"], dtype=float),
            "Q_W": np.asarray(payload[f"{key}__Q_W"], dtype=float),
            "rho": float(np.asarray(payload[f"{key}__rho"], dtype=float)[0]),
            "weights": np.asarray(payload[f"{key}__weights"], dtype=float),
        }
    return final


def variant_weights(
    *,
    true_d: np.ndarray,
    full_dhat: np.ndarray,
    learning_dhat: np.ndarray,
    qy: np.ndarray,
    qw: np.ndarray,
    rho: float,
    population: dict[str, np.ndarray | float],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    pop_qy = np.asarray(population["Q_Y"], dtype=float)
    pop_qw = np.asarray(population["Q_W"], dtype=float)
    pop_rho = float(population["rho"])
    return {
        "robust_population_true_d": (
            np.asarray(population["weights"], dtype=float),
            true_d,
        ),
        "robust_learned_q_true_d": (solve_robust(true_d, qy, qw, rho), true_d),
        "robust_population_q_full_dhat": (
            solve_robust(full_dhat, pop_qy, pop_qw, pop_rho),
            full_dhat,
        ),
        "robust_full_empirical": (solve_robust(full_dhat, qy, qw, rho), full_dhat),
        "robust_reported_simulation": (
            solve_robust(learning_dhat, qy, qw, rho),
            learning_dhat,
        ),
        "tsls_true_d": (tsls_weights(true_d), true_d),
        "tsls_full_dhat": (tsls_weights(full_dhat), full_dhat),
        "tsls_learning_dhat": (tsls_weights(learning_dhat), learning_dhat),
    }


WEIGHT_FIELDS = (
    "rep",
    "branch",
    "variant",
    "family",
    "rms_weight",
    "max_abs_weight",
    "effective_abs_count",
    "mean_weight",
    "constraint_own_d",
    "constraint_true_d",
    "rms_distance_reference",
    "relative_rms_distance_reference",
    "weight_correlation_reference",
    "max_abs_difference_reference",
    "rho_learned",
    "zeta_learned",
)

ESTIMATE_FIELDS = (
    "rep",
    "branch",
    "calibration_scope",
    "variant",
    "family",
    "method",
    "tau",
    "bias",
    "abs_error",
    "squared_error",
    "se",
    "pi",
    "contains_target",
    "critical",
)


def evaluate_paired(
    protocol: dict[str, object],
    *,
    output_dir: Path,
    population: dict[str, dict[str, np.ndarray | float]],
    reps: int | None = None,
) -> None:
    reference = protocol["production_reference"]
    evaluation = protocol["paired_evaluation"]
    total_reps = int(evaluation["reps"] if reps is None else reps)
    eval_seed = int(evaluation["seed"])
    t0 = int(reference["T0"])
    t1 = int(reference["T1"])
    _, exposure, geometry = production_objects(protocol)

    weight_partial = output_dir / "weight_replications.csv.partial"
    estimate_partial = output_dir / "estimate_replications.csv.partial"
    if weight_partial.exists() or estimate_partial.exists():
        raise RuntimeError("Partial evaluation output already exists.")

    with weight_partial.open("w", newline="", encoding="utf-8") as weight_handle, estimate_partial.open(
        "w", newline="", encoding="utf-8"
    ) as estimate_handle:
        weight_writer = csv.DictWriter(weight_handle, fieldnames=WEIGHT_FIELDS)
        estimate_writer = csv.DictWriter(estimate_handle, fieldnames=ESTIMATE_FIELDS)
        weight_writer.writeheader()
        estimate_writer.writeheader()

        for rep in range(total_reps):
            components = draw_components(
                protocol,
                rep=rep,
                seed=eval_seed,
                exposure=exposure,
                geometry=geometry,
            )
            for branch in BRANCHES:
                panel = branch_panel(protocol, components, branch)
                qy, qw, rho, qdiag = q_objects(panel, components.Z, t0)
                full_dhat = unit_level_slope_vector(panel.W, components.Z)
                learning_dhat = unit_level_slope_vector(panel.W[:, :t0], components.Z[:t0])
                weights = variant_weights(
                    true_d=exposure,
                    full_dhat=full_dhat,
                    learning_dhat=learning_dhat,
                    qy=qy,
                    qw=qw,
                    rho=rho,
                    population=population[branch],
                )

                post = slice(t0, t0 + t1)
                z_post = np.asarray(components.Z, dtype=float)[post]
                for variant in VARIANT_ORDER:
                    weight, own_d = weights[variant]
                    family = "Robust SIV" if variant in ROBUST_VARIANTS else "TSLS"
                    reference_variant = (
                        "robust_population_true_d" if family == "Robust SIV" else "tsls_true_d"
                    )
                    weight_reference = weights[reference_variant][0]
                    comparison = compare_weights(weight, weight_reference)
                    weight_writer.writerow(
                        {
                            "rep": rep,
                            "branch": branch,
                            "variant": variant,
                            "family": family,
                            "rms_weight": rms(weight),
                            "max_abs_weight": float(np.max(np.abs(weight))),
                            "effective_abs_count": effective_abs_count(weight),
                            "mean_weight": float(np.mean(weight)),
                            "constraint_own_d": float(np.mean(weight * own_d)),
                            "constraint_true_d": float(np.mean(weight * exposure)),
                            "rms_distance_reference": comparison["rms_distance"],
                            "relative_rms_distance_reference": comparison["relative_rms_distance"],
                            "weight_correlation_reference": comparison["weight_correlation"],
                            "max_abs_difference_reference": comparison["max_abs_difference"],
                            "rho_learned": float(qdiag["rho"]),
                            "zeta_learned": float(qdiag["zeta"]),
                        }
                    )

                    y = aggregate_series(panel.Y[:, post], weight)
                    w = aggregate_series(panel.W[:, post], weight)
                    ratio = slope_ratio(y, w, z_post)
                    calibration_scope = "nominal_coverage" if branch in {"Basic", "GFE"} else "shock_active_containment"

                    arima = raw_z_arp_design_based_se(
                        y,
                        w,
                        z_post,
                        z_process=np.asarray(components.Z, dtype=float),
                        ar_order=int(active.ARIMA_AR_ORDER),
                        varz_mode="window",
                    )
                    hac = hac_delta_se(y, w, z_post, maxlags=int(active.HAC_LAGS))
                    ar = ar_style_test(y, w, z_post, float(active.TAU), maxlags=int(active.HAC_LAGS))
                    method_values = {
                        "ARIMA-Z": (float(arima.se), int(abs(ratio.tau - active.TAU) <= _crit("ARIMA-Z", t1) * arima.se)),
                        "HAC": (float(hac.se), int(abs(ratio.tau - active.TAU) <= _crit("HAC", t1) * hac.se)),
                        "AR": (float("nan"), int(abs(float(ar["t_stat"])) <= _crit("AR", t1))),
                    }
                    for method, (se, contains) in method_values.items():
                        bias = float(ratio.tau - active.TAU)
                        estimate_writer.writerow(
                            {
                                "rep": rep,
                                "branch": branch,
                                "calibration_scope": calibration_scope,
                                "variant": variant,
                                "family": family,
                                "method": method,
                                "tau": float(ratio.tau),
                                "bias": bias,
                                "abs_error": abs(bias),
                                "squared_error": bias * bias,
                                "se": se,
                                "pi": float(ratio.pi),
                                "contains_target": contains,
                                "critical": _crit(method, t1),
                            }
                        )

            count = rep + 1
            if count % max(min(250, total_reps), 1) == 0 or count == total_reps:
                print(f"  paired evaluation: {count}/{total_reps}", flush=True)

    weight_partial.replace(output_dir / "weight_replications.csv")
    estimate_partial.replace(output_dir / "estimate_replications.csv")


def finite_quantile(values: pd.Series, q: float) -> float:
    clean = values[np.isfinite(values.to_numpy(dtype=float))]
    return float(clean.quantile(q)) if len(clean) else float("nan")


def summarize_outputs(output_dir: Path) -> None:
    estimates = pd.read_csv(output_dir / "estimate_replications.csv")
    weights = pd.read_csv(output_dir / "weight_replications.csv")

    summary_rows: list[dict[str, object]] = []
    for keys, group in estimates.groupby(
        ["branch", "calibration_scope", "variant", "family", "method"], sort=True
    ):
        branch, scope, variant, family, method = keys
        n = int(len(group))
        tau_sd = float(group["tau"].std(ddof=1))
        se_values = group["se"].to_numpy(dtype=float)
        finite_se = se_values[np.isfinite(se_values)]
        containment = float(group["contains_target"].mean())
        summary_rows.append(
            {
                "branch": branch,
                "calibration_scope": scope,
                "variant": variant,
                "family": family,
                "method": method,
                "n": n,
                "mean_bias": float(group["bias"].mean()),
                "mean_abs_error": float(group["abs_error"].mean()),
                "median_abs_error": float(group["abs_error"].median()),
                "rmse": float(math.sqrt(float(group["squared_error"].mean()))),
                "tau_mc_sd": tau_sd,
                "mean_se": float(np.mean(finite_se)) if finite_se.size else float("nan"),
                "median_se": float(np.median(finite_se)) if finite_se.size else float("nan"),
                "mean_se_over_tau_mc_sd": (
                    float(np.mean(finite_se)) / tau_sd if finite_se.size and tau_sd > 0.0 else float("nan")
                ),
                "target_containment": containment,
                "containment_mcse": math.sqrt(containment * (1.0 - containment) / float(n)),
                "pi_q10": finite_quantile(group["pi"], 0.10),
                "pi_median": finite_quantile(group["pi"], 0.50),
                "pi_q90": finite_quantile(group["pi"], 0.90),
            }
        )
    write_csv(output_dir / "estimate_summary.csv", summary_rows)

    paired_rows: list[dict[str, object]] = []
    indexed = estimates.set_index(["rep", "branch", "method", "variant"]).sort_index()
    for candidate, reference in PAIR_REFERENCES.items():
        candidate_rows = indexed.xs(candidate, level="variant")
        reference_rows = indexed.xs(reference, level="variant")
        joined = candidate_rows.join(reference_rows, lsuffix="_candidate", rsuffix="_reference", how="inner")
        for (branch, method), group in joined.groupby(level=["branch", "method"], sort=True):
            n = int(len(group))
            containment_difference = (
                group["contains_target_candidate"].to_numpy(dtype=float)
                - group["contains_target_reference"].to_numpy(dtype=float)
            )
            candidate_rmse = math.sqrt(float(group["squared_error_candidate"].mean()))
            reference_rmse = math.sqrt(float(group["squared_error_reference"].mean()))
            paired_rows.append(
                {
                    "branch": branch,
                    "method": method,
                    "candidate": candidate,
                    "reference": reference,
                    "n": n,
                    "mean_tau_difference": float(
                        np.mean(group["tau_candidate"].to_numpy(dtype=float) - group["tau_reference"].to_numpy(dtype=float))
                    ),
                    "mean_abs_error_difference": float(
                        np.mean(
                            group["abs_error_candidate"].to_numpy(dtype=float)
                            - group["abs_error_reference"].to_numpy(dtype=float)
                        )
                    ),
                    "candidate_rmse": candidate_rmse,
                    "reference_rmse": reference_rmse,
                    "rmse_difference": candidate_rmse - reference_rmse,
                    "candidate_containment": float(group["contains_target_candidate"].mean()),
                    "reference_containment": float(group["contains_target_reference"].mean()),
                    "containment_difference": float(np.mean(containment_difference)),
                    "paired_containment_mcse": float(np.std(containment_difference, ddof=1) / math.sqrt(float(n))),
                    "mean_se_difference": float(
                        np.nanmean(
                            group["se_candidate"].to_numpy(dtype=float)
                            - group["se_reference"].to_numpy(dtype=float)
                        )
                    ) if method != "AR" else float("nan"),
                }
            )
    write_csv(output_dir / "paired_effects.csv", paired_rows)

    weight_rows: list[dict[str, object]] = []
    for (branch, variant, family), group in weights.groupby(["branch", "variant", "family"], sort=True):
        weight_rows.append(
            {
                "branch": branch,
                "variant": variant,
                "family": family,
                "n": int(len(group)),
                "median_rms_weight": float(group["rms_weight"].median()),
                "median_max_abs_weight": float(group["max_abs_weight"].median()),
                "median_effective_abs_count": float(group["effective_abs_count"].median()),
                "median_rms_distance_reference": float(group["rms_distance_reference"].median()),
                "q90_rms_distance_reference": finite_quantile(group["rms_distance_reference"], 0.90),
                "median_relative_rms_distance_reference": float(group["relative_rms_distance_reference"].median()),
                "median_weight_correlation_reference": float(group["weight_correlation_reference"].median()),
                "max_abs_mean_weight": float(np.max(np.abs(group["mean_weight"].to_numpy(dtype=float)))),
                "max_abs_own_constraint_error": float(
                    np.max(np.abs(group["constraint_own_d"].to_numpy(dtype=float) - 1.0))
                ),
                "median_true_d_constraint": float(group["constraint_true_d"].median()),
            }
        )
    write_csv(output_dir / "weight_summary.csv", weight_rows)


def output_hashes(output_dir: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }


def write_manifest(
    protocol: dict[str, object],
    *,
    output_dir: Path,
    provenance: dict[str, object],
    stages: list[str],
) -> None:
    manifest = {
        "analysis": protocol["protocol"],
        "status": "complete",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "analysis_source_sha256": sha256_file(Path(__file__).resolve()),
        "production_provenance": provenance,
        "effective_population_objective_draws": int(protocol["population_objective_bank"]["draws"]),
        "effective_evaluation_reps": int(protocol["paired_evaluation"]["reps"]),
        "completed_stages": stages,
        "fixed_decisions": protocol["fixed_decisions"],
        "output_sha256": output_hashes(output_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("oracle", "evaluate", "all"), default="all")
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument(
        "--population-draws",
        type=int,
        default=None,
        help="Use fewer population-objective draws for a noninterpretive structural smoke test.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    protocol = load_protocol()
    if args.population_draws is not None:
        if args.population_draws < 2:
            parser.error("--population-draws must be at least 2.")
        protocol["population_objective_bank"]["draws"] = int(args.population_draws)
        protocol["population_objective_bank"]["checkpoints"] = [int(args.population_draws)]
    if args.reps is not None:
        protocol["paired_evaluation"]["reps"] = int(args.reps)
    provenance = verify_reproduction_context(protocol)
    output_dir = args.out if args.out is not None else ROOT / protocol["output_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    stages: list[str] = []
    if (output_dir / "population_objective.npz").exists():
        stages.append("population_objective")

    if args.stage in {"oracle", "all"}:
        if (output_dir / "population_objective.npz").exists():
            raise RuntimeError("Population-objective output already exists; refusing to overwrite it.")
        construct_population_objective(protocol, output_dir=output_dir)
        if "population_objective" not in stages:
            stages.append("population_objective")

    if args.stage in {"evaluate", "all"}:
        if (output_dir / "estimate_replications.csv").exists():
            raise RuntimeError("Paired evaluation output already exists; refusing to overwrite it.")
        population = load_population_objective(output_dir)
        evaluate_paired(protocol, output_dir=output_dir, population=population, reps=args.reps)
        summarize_outputs(output_dir)
        stages.append("paired_evaluation")

    write_manifest(protocol, output_dir=output_dir, provenance=provenance, stages=stages)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
