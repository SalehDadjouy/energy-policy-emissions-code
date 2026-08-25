from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZetaResult:
    zeta: float
    sigma_hat: float
    denominator: float
    zeta_sq: float
    rho: float


@dataclass(frozen=True)
class RhoRule:
    rule: str = "rms"
    kappa: float = 2.0


def zeta_denominator(n: int, t0: int) -> float:
    if n <= 0 or t0 <= 1:
        raise ValueError("n must be positive and t0 must exceed one.")
    return math.sqrt(float(n) * float(t0))


def zeta_value(
    *,
    opnorm_y: float,
    opnorm_w: float,
    n: int,
    t0: int,
) -> ZetaResult:
    denom = zeta_denominator(n, t0)
    sigma_hat = max(float(opnorm_y), float(opnorm_w)) / denom
    multiplier = math.sqrt(math.log(float(t0)))
    zeta = multiplier * sigma_hat
    zeta_sq = zeta * zeta
    return ZetaResult(
        zeta=zeta,
        sigma_hat=sigma_hat,
        denominator=denom,
        zeta_sq=zeta_sq,
        rho=zeta_sq / (float(n) * float(t0)),
    )


def projection_residualizer_const_z(z: np.ndarray) -> np.ndarray:
    zv = np.asarray(z, dtype=float).reshape(-1)
    if zv.size <= 2:
        raise ValueError("z must contain more than two observations.")
    X = np.column_stack([np.ones(zv.size), zv])
    return np.eye(zv.size) - X @ np.linalg.pinv(X)


def residuals_two_way_unit_slope(panel: np.ndarray, z: np.ndarray) -> tuple[float, np.ndarray]:
    K = np.asarray(panel, dtype=float)
    zv = np.asarray(z, dtype=float).reshape(-1)
    if K.ndim != 2:
        raise ValueError("panel must be n x T0.")
    n, t0 = K.shape
    if t0 != zv.size:
        raise ValueError("panel time dimension must match z.")
    obs = n * t0
    y = K.reshape(obs)
    unit_idx = np.repeat(np.arange(n), t0)
    time_idx = np.tile(np.arange(t0), n)
    U = np.zeros((obs, n), dtype=float)
    U[np.arange(obs), unit_idx] = 1.0
    T = np.zeros((obs, t0), dtype=float)
    T[np.arange(obs), time_idx] = 1.0
    G = np.zeros((obs, n), dtype=float)
    G[np.arange(obs), unit_idx] = zv[time_idx]
    X = np.concatenate([U, T, G], axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = (y - X @ beta).reshape(n, t0)
    return float(np.mean(resid * resid)), resid


def robust_q_components(
    *,
    y0: np.ndarray,
    w0: np.ndarray,
    z0: np.ndarray,
    rho_rule: RhoRule | None = None,
) -> dict[str, np.ndarray | float | str]:
    Y = np.asarray(y0, dtype=float)
    W = np.asarray(w0, dtype=float)
    Z = np.asarray(z0, dtype=float).reshape(-1)
    if Y.shape != W.shape or Y.ndim != 2:
        raise ValueError("y0 and w0 must be aligned n x T0 arrays.")
    n, t0 = Y.shape
    if t0 != Z.size:
        raise ValueError("z0 length must match learning-window columns.")
    sig_y, Ey = residuals_two_way_unit_slope(Y, Z)
    sig_w, Ew = residuals_two_way_unit_slope(W, Z)
    if sig_y <= 0 or sig_w <= 0:
        raise ValueError("two-way residual variance must be positive.")
    op_y = float(np.linalg.svd(Ey, compute_uv=False)[0])
    op_w = float(np.linalg.svd(Ew, compute_uv=False)[0])
    zeta = zeta_value(
        opnorm_y=op_y,
        opnorm_w=op_w,
        n=n,
        t0=t0,
    )
    Mz = projection_residualizer_const_z(Z)
    QY = (Y @ Mz @ Y.T) / (float(t0) * sig_y * float(n) ** 2)
    QW = (W @ Mz @ W.T) / (float(t0) * sig_w * float(n) ** 2)
    selected_rule = rho_rule or RhoRule()
    if selected_rule.rule == "rms":
        rho = zeta.rho
        rho_rule_name = "rms"
        rho_kappa = float("nan")
    elif selected_rule.rule == "trace":
        rho = float(selected_rule.kappa) * float(np.trace(QY + QW)) / float(n)
        rho_rule_name = "trace"
        rho_kappa = float(selected_rule.kappa)
    else:
        raise ValueError(f"unknown rho rule: {selected_rule.rule}")
    return {
        "Q_Y": QY,
        "Q_W": QW,
        "rho": rho,
        "zeta": zeta.zeta,
        "zeta_sq": zeta.zeta_sq,
        "sigma_hat": zeta.sigma_hat,
        "zeta_denominator_value": zeta.denominator,
        "zeta_denominator_rule": "rms",
        "rho_rule": rho_rule_name,
        "rho_kappa": rho_kappa,
        "rms_rho": zeta.rho,
        "trace_Qdata_per_n": float(np.trace(QY + QW)) / float(n),
        "trace_QY_per_n": float(np.trace(QY)) / float(n),
        "trace_QW_per_n": float(np.trace(QW)) / float(n),
        "trace_QW_over_QY": float(np.trace(QW) / np.trace(QY)) if float(np.trace(QY)) > 0.0 else float("inf"),
        "sigma_y2_mean": sig_y,
        "sigma_w2_mean": sig_w,
        "sigma_y2_used": sig_y,
        "sigma_w2_used": sig_w,
        "opnorm_y": op_y,
        "opnorm_w": op_w,
        "Ey": Ey,
        "Ew": Ew,
    }


def _nullspace(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    _, s, vh = np.linalg.svd(np.asarray(A, dtype=float), full_matrices=True)
    rank = int(np.sum(s > tol))
    return vh[rank:].T


def robust_q_admissibility(first_stage_slopes: np.ndarray, Q: np.ndarray) -> dict[str, float | bool]:
    D = np.asarray(first_stage_slopes, dtype=float).reshape(-1)
    Qv = np.asarray(Q, dtype=float)
    if D.size == 0:
        raise ValueError("first-stage slopes must be nonempty.")
    n = D.size
    if Qv.shape != (n, n):
        raise ValueError("Q must be n x n.")
    A = np.vstack([D.reshape(1, -1), np.ones((1, n))])
    rank = int(np.linalg.matrix_rank(A))
    B = _nullspace(A)
    tangent = B.T @ Qv @ B
    eig = np.linalg.eigvalsh(tangent) if tangent.size else np.array([], dtype=float)
    symmetric = bool(np.allclose(Qv, Qv.T, atol=1e-10))
    finite = bool(np.all(np.isfinite(Qv)))
    tangent_pd = bool(eig.size == max(n - rank, 0) and (eig.size == 0 or float(np.min(eig)) > 0.0))
    return {
        "symmetric": symmetric,
        "finite": finite,
        "constraint_rank": float(rank),
        "min_tangent_eigenvalue": float(np.min(eig)) if eig.size else float("inf"),
        "tangent_positive_definite": tangent_pd,
        "admissible": bool(symmetric and finite and rank == 2 and tangent_pd),
    }


def robust_weights_from_q(
    *,
    first_stage_slopes: np.ndarray,
    Q_Y: np.ndarray,
    Q_W: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, dict[str, float]]:
    D = np.asarray(first_stage_slopes, dtype=float).reshape(-1)
    if D.size == 0:
        raise ValueError("first-stage slopes must be nonempty.")
    n = D.size
    QY = np.asarray(Q_Y, dtype=float)
    QW = np.asarray(Q_W, dtype=float)
    if QY.shape != (n, n) or QW.shape != (n, n):
        raise ValueError("Q blocks must be n x n.")
    Q_data = QY + QW
    Q = np.asarray(float(rho) * np.eye(n) + Q_data, dtype=float)
    # Check conditioning before solve; a near-singular Q (rho ≪ 1/n) can
    # produce NaN weights that propagate silently through aggregation.
    _eig_min = float(np.linalg.eigvalsh(Q).min())
    if _eig_min < 1e-10:
        import warnings
        warnings.warn(
            f"Robust SIV Q matrix is near-singular (λ_min={_eig_min:.2e}); "
            "rho may be too small for this (n, t0) configuration.",
            RuntimeWarning, stacklevel=2,
        )
    admissibility = robust_q_admissibility(D, Q)
    if not bool(admissibility["admissible"]):
        raise ValueError("Robust SIV quadratic problem is not admissible.")
    A = np.vstack([D.reshape(1, -1), np.ones((1, n))])
    b = np.array([float(n), 0.0], dtype=float)
    Qi_AT = np.linalg.solve(Q, A.T)
    lam = np.linalg.solve(A @ Qi_AT, b)
    weights = (Qi_AT @ lam).reshape(-1)
    diag = {
        "constraint_D": float((weights @ D) / float(n)),
        "constraint_ones": float(np.mean(weights)),
        "rho": float(rho),
        "objective_data": float(weights.T @ Q_data @ weights),
        "objective_ridge": float(weights.T @ (float(rho) * np.eye(n)) @ weights),
        "admissible": float(bool(admissibility["admissible"])),
        "min_tangent_eigenvalue": float(admissibility["min_tangent_eigenvalue"]),
    }
    return weights, diag


def robust_weights(
    *,
    y0: np.ndarray,
    w0: np.ndarray,
    z0: np.ndarray,
    first_stage_slopes: np.ndarray,
    rho_rule: RhoRule | None = None,
) -> tuple[np.ndarray, dict[str, float | str]]:
    q = robust_q_components(y0=y0, w0=w0, z0=z0, rho_rule=rho_rule)
    weights, diag = robust_weights_from_q(
        first_stage_slopes=first_stage_slopes,
        Q_Y=np.asarray(q["Q_Y"], dtype=float),
        Q_W=np.asarray(q["Q_W"], dtype=float),
        rho=float(q["rho"]),
    )
    return weights, {
        **diag,
        # Full zeta path used to audit the regularization scale.
        "zeta":                   float(q["zeta"]),
        "zeta_sq":                float(q["zeta_sq"]),
        "sigma_hat":              float(q["sigma_hat"]),
        "zeta_denominator_value": float(q["zeta_denominator_value"]),
        "zeta_denominator_rule":  str(q["zeta_denominator_rule"]),
        "rho_rule":               str(q["rho_rule"]),
        "zeta_mode": "ln",
        "rho_kappa": float(q["rho_kappa"]),
        "rms_rho": float(q["rms_rho"]),
        "trace_Qdata_per_n": float(q["trace_Qdata_per_n"]),
        "trace_QY_per_n": float(q["trace_QY_per_n"]),
        "trace_QW_per_n": float(q["trace_QW_per_n"]),
        "trace_QW_over_QY": float(q["trace_QW_over_QY"]),
    }
