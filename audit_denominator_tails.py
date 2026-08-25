#!/usr/bin/env python3
"""Summarize first-stage denominator tails in the paper simulation."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUT = Path("outputs/simulation")


def _rmse(x: np.ndarray) -> float:
    values = np.asarray(x, dtype=float)
    return float(math.sqrt(float(np.mean(values * values)))) if values.size else float("nan")


def _trimmed_rmse(x: np.ndarray, trim_share: float) -> float:
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        return float("nan")
    keep = int(math.floor(values.size * (1.0 - float(trim_share))))
    keep = max(1, min(keep, values.size))
    order = np.argsort(np.abs(values))
    return _rmse(values[order[:keep]])


def _winsorized_rmse(x: np.ndarray, upper_quantile: float) -> float:
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        return float("nan")
    cap = float(np.quantile(np.abs(values), float(upper_quantile)))
    return _rmse(np.clip(values, -cap, cap))


def _cell_summary(group: pd.DataFrame) -> dict[str, float | int]:
    bias = group["bias"].to_numpy(dtype=float)
    se = group["se"].to_numpy(dtype=float)
    pi = group["pi"].to_numpy(dtype=float)
    abs_pi = np.abs(pi)
    abs_bias = np.abs(bias)
    covered = group["covered"].to_numpy(dtype=float) if "covered" in group.columns else np.full_like(bias, np.nan)
    return {
        "n": int(len(group)),
        "coverage": float(np.nanmean(covered)),
        "mean_bias": float(np.mean(bias)),
        "rmse": _rmse(bias),
        "trim1_rmse": _trimmed_rmse(bias, 0.01),
        "trim5_rmse": _trimmed_rmse(bias, 0.05),
        "winsor99_rmse": _winsorized_rmse(bias, 0.99),
        "winsor95_rmse": _winsorized_rmse(bias, 0.95),
        "median_bias": float(np.median(bias)),
        "median_abs_bias": float(np.median(abs_bias)),
        "p90_abs_bias": float(np.quantile(abs_bias, 0.90)),
        "p95_abs_bias": float(np.quantile(abs_bias, 0.95)),
        "p99_abs_bias": float(np.quantile(abs_bias, 0.99)),
        "max_abs_bias": float(np.max(abs_bias)),
        "mean_se": float(np.mean(se)),
        "median_se": float(np.median(se)),
        "p95_se": float(np.quantile(se, 0.95)),
        "p99_se": float(np.quantile(se, 0.99)),
        "max_se": float(np.max(se)),
        "mean_pi": float(np.mean(pi)),
        "median_pi": float(np.median(pi)),
        "p01_pi": float(np.quantile(pi, 0.01)),
        "p05_pi": float(np.quantile(pi, 0.05)),
        "p95_pi": float(np.quantile(pi, 0.95)),
        "p99_pi": float(np.quantile(pi, 0.99)),
        "min_pi": float(np.min(pi)),
        "share_pi_le_zero": float(np.mean(pi <= 0.0)),
        "share_abs_pi_lt_0_05": float(np.mean(abs_pi < 0.05)),
        "share_abs_pi_lt_0_10": float(np.mean(abs_pi < 0.10)),
        "share_abs_pi_lt_0_20": float(np.mean(abs_pi < 0.20)),
        "share_abs_pi_lt_0_30": float(np.mean(abs_pi < 0.30)),
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    arima = raw[raw["method"].eq("ARIMA-Z")].copy()
    rows: list[dict[str, object]] = []
    for key, group in arima.groupby(["config", "branch", "estimator"], sort=True):
        config, branch, estimator = key
        rows.append(
            {
                "config": config,
                "branch": branch,
                "estimator": estimator,
                **_cell_summary(group),
            }
        )
    return pd.DataFrame(rows)


def top_tail_records(raw: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    arima = raw[raw["method"].eq("ARIMA-Z")].copy()
    arima["abs_bias"] = arima["bias"].abs()
    arima["abs_pi"] = arima["pi"].abs()
    columns = [
        "config",
        "branch",
        "estimator",
        "rep",
        "bias",
        "se",
        "pi",
        "abs_pi",
        "bias_component_gfe",
        "bias_component_shock",
        "bias_component_eps",
        "weight_corr",
    ]
    available = [c for c in columns if c in arima.columns]
    return arima.sort_values("abs_bias", ascending=False).head(int(top_n))[available]


def conditional_summary(raw: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    arima = raw[raw["method"].eq("ARIMA-Z")].copy()
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        kept = arima[arima["pi"].abs() >= float(threshold)]
        for key, group in kept.groupby(["config", "branch", "estimator"], sort=True):
            config, branch, estimator = key
            rows.append(
                {
                    "abs_pi_floor": float(threshold),
                    "config": config,
                    "branch": branch,
                    "estimator": estimator,
                    **_cell_summary(group),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit denominator-tail behavior in ARIMA-Z simulation outputs.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--floors", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    args = parser.parse_args()

    raw_path = args.out_dir / "raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    raw = pd.read_csv(raw_path)

    summary = summarize(raw)
    tail = top_tail_records(raw, top_n=int(args.top_n))
    conditional = conditional_summary(raw, thresholds=[float(x) for x in args.floors])

    summary.to_csv(args.out_dir / "denominator_tail_summary.csv", index=False)
    tail.to_csv(args.out_dir / "denominator_tail_top_records.csv", index=False)
    conditional.to_csv(args.out_dir / "denominator_tail_conditional_summary.csv", index=False)

    print(f"wrote denominator-tail audit files to {args.out_dir}")
    focus = summary[
        summary["branch"].eq("GFE + Agg. Shock")
        & summary["config"].eq("finite_length")
    ]
    print("\nFinite-window GFE + Aggregate Shock:")
    print(
        focus[
            [
                "estimator",
                "n",
                "rmse",
                "trim1_rmse",
                "trim5_rmse",
                "winsor99_rmse",
                "winsor95_rmse",
                "median_abs_bias",
                "p95_abs_bias",
                "p99_abs_bias",
                "max_abs_bias",
                "mean_se",
                "median_se",
                "p99_se",
                "max_se",
                "share_pi_le_zero",
                "share_abs_pi_lt_0_20",
            ]
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
