from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimators import (
    SlopeRatio,
    safe_weight_correlation,
    tsls_weights,
    unit_level_slope_vector,
    weighted_slope_ratio,
)
from .robust_weights import robust_weights


BRANCH_SWITCHES: dict[str, tuple[int, int]] = {
    "Basic": (0, 0),
    "GFE": (1, 0),
    "Aggregate Shock": (0, 1),
    "GFE + Agg. Shock": (1, 1),
}


@dataclass(frozen=True)
class ModelClassComponents:
    """Deterministic components for one evaluation of M(T0,T1)."""

    Z: np.ndarray
    D: np.ndarray
    alpha_w: np.ndarray
    alpha_y: np.ndarray
    lowrank_w: np.ndarray
    lowrank_y: np.ndarray
    shock_w: np.ndarray
    shock_y: np.ndarray
    eps_w: np.ndarray
    eps_y: np.ndarray


@dataclass(frozen=True)
class BranchPanel:
    branch: str
    T0: int
    T1: int
    Y: np.ndarray
    W: np.ndarray
    first_stage: np.ndarray
    gfe_on: int
    shock_on: int


@dataclass(frozen=True)
class BranchEstimate:
    branch: str
    T0: int
    T1: int
    D_hat: np.ndarray
    weights_tsls: np.ndarray
    weights_robust: np.ndarray
    tsls: SlopeRatio
    robust: SlopeRatio
    weight_correlation: float
    robust_diagnostics: dict[str, float | str]


def _as_vector(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be nonempty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains nonfinite values.")
    return arr


def _as_panel(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional n x T array.")
    if arr.size == 0:
        raise ValueError(f"{name} must be nonempty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains nonfinite values.")
    return arr


def branch_switches(branch: str) -> tuple[int, int]:
    try:
        return BRANCH_SWITCHES[branch]
    except KeyError as exc:
        raise ValueError(f"unknown branch: {branch}") from exc


def validate_time_dimensions(T0: int, T1: int) -> tuple[int, int, int]:
    t0 = int(T0)
    t1 = int(T1)
    if t0 <= 2:
        raise ValueError("T0 must exceed two so Robust SIV learning residuals are defined.")
    if t1 <= 2:
        raise ValueError("T1 must exceed two so post-learning slope ratios are defined.")
    return t0, t1, t0 + t1


def validate_components(components: ModelClassComponents, *, T0: int, T1: int) -> ModelClassComponents:
    _, _, T = validate_time_dimensions(T0, T1)
    Z = _as_vector(components.Z, "Z")
    D = _as_vector(components.D, "D")
    alpha_w = _as_vector(components.alpha_w, "alpha_w")
    alpha_y = _as_vector(components.alpha_y, "alpha_y")
    if Z.size != T:
        raise ValueError("Z length must equal T0 + T1.")
    n = D.size
    if alpha_w.size != n or alpha_y.size != n:
        raise ValueError("state intercept vectors must align with D.")
    panels = {
        "lowrank_w": _as_panel(components.lowrank_w, "lowrank_w"),
        "lowrank_y": _as_panel(components.lowrank_y, "lowrank_y"),
        "shock_w": _as_panel(components.shock_w, "shock_w"),
        "shock_y": _as_panel(components.shock_y, "shock_y"),
        "eps_w": _as_panel(components.eps_w, "eps_w"),
        "eps_y": _as_panel(components.eps_y, "eps_y"),
    }
    for name, panel in panels.items():
        if panel.shape != (n, T):
            raise ValueError(f"{name} shape must be n x (T0 + T1).")
    return ModelClassComponents(
        Z=Z,
        D=D,
        alpha_w=alpha_w,
        alpha_y=alpha_y,
        lowrank_w=panels["lowrank_w"],
        lowrank_y=panels["lowrank_y"],
        shock_w=panels["shock_w"],
        shock_y=panels["shock_y"],
        eps_w=panels["eps_w"],
        eps_y=panels["eps_y"],
    )


def first_stage_component(D: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return np.outer(_as_vector(D, "D"), _as_vector(Z, "Z"))


def build_branch_panel(
    components: ModelClassComponents,
    *,
    branch: str,
    tau_true: float,
    T0: int,
    T1: int,
) -> BranchPanel:
    t0, t1, _ = validate_time_dimensions(T0, T1)
    c = validate_components(components, T0=t0, T1=t1)
    gfe_on, shock_on = branch_switches(branch)
    first_stage = first_stage_component(c.D, c.Z)
    W = (
        c.alpha_w[:, None]
        + first_stage
        + gfe_on * c.lowrank_w
        + shock_on * c.shock_w
        + c.eps_w
    )
    Y = (
        c.alpha_y[:, None]
        + float(tau_true) * W
        + gfe_on * c.lowrank_y
        + shock_on * c.shock_y
        + c.eps_y
    )
    return BranchPanel(
        branch=branch,
        T0=t0,
        T1=t1,
        Y=Y,
        W=W,
        first_stage=first_stage,
        gfe_on=gfe_on,
        shock_on=shock_on,
    )


def estimate_branch_panel(
    panel: BranchPanel,
    Z: np.ndarray,
    first_stage_slopes: np.ndarray | None = None,
) -> BranchEstimate:
    zv = _as_vector(Z, "Z")
    if panel.Y.shape != panel.W.shape:
        raise ValueError("Y and W panels must be aligned.")
    if panel.Y.shape[1] != zv.size:
        raise ValueError("panel time dimension must match Z.")
    t0 = int(panel.T0)
    D_hat = (
        unit_level_slope_vector(panel.W, zv)
        if first_stage_slopes is None
        else _as_vector(first_stage_slopes, "first_stage_slopes")
    )
    if D_hat.size != panel.Y.shape[0]:
        raise ValueError("first_stage_slopes must align with panel rows.")
    weights_t = tsls_weights(D_hat)
    weights_r, robust_diag = robust_weights(
        y0=panel.Y[:, :t0],
        w0=panel.W[:, :t0],
        z0=zv[:t0],
        first_stage_slopes=D_hat,
    )
    robust_diag = {
        **robust_diag,
        "robust_qy_mode": "raw",
    }
    post = slice(t0, t0 + int(panel.T1))
    tsls = weighted_slope_ratio(panel.Y[:, post], panel.W[:, post], zv[post], weights_t)
    robust = weighted_slope_ratio(panel.Y[:, post], panel.W[:, post], zv[post], weights_r)
    return BranchEstimate(
        branch=panel.branch,
        T0=panel.T0,
        T1=panel.T1,
        D_hat=D_hat,
        weights_tsls=weights_t,
        weights_robust=weights_r,
        tsls=tsls,
        robust=robust,
        weight_correlation=safe_weight_correlation(weights_t, weights_r),
        robust_diagnostics=robust_diag,
    )


def run_model_class(
    *,
    T0: int,
    T1: int,
    components: ModelClassComponents,
    tau_true: float,
    branch: str,
) -> BranchEstimate:
    panel = build_branch_panel(
        components,
        branch=branch,
        tau_true=float(tau_true),
        T0=T0,
        T1=T1,
    )
    return estimate_branch_panel(panel, components.Z)
