"""Configuration shared by the empirical-length and longer-window simulations."""

from __future__ import annotations

from pathlib import Path

from sim.dgp import load_parameters
from sim.helpers import derive_empirical_scale_constants


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

TAU_TRUE = -1.8747459297146785
TAU = TAU_TRUE
HAC_LAGS = 1
ARIMA_AR_ORDER = 2

EMPIRICAL_SCALE = derive_empirical_scale_constants(
    DATA_DIR / "panel_lag2.csv",
    DATA_DIR / "exposure_restricted.csv",
)
W_UNIT = EMPIRICAL_SCALE.first_stage_component_sd

COMPONENT_SCALE_MULTIPLIERS = {
    "lowrank_w_sd": 0.30,
    "lowrank_y_sd": 0.40,
    "shock_w_sd": 1.00,
    "eps_w_sd": 0.40,
    "eps_y_sd": 0.30,
}

PREPERIOD_SCALES = {
    "finite_length": 0.54,
    "longer_length": 1.00,
}

PARAMS = load_parameters(
    {
        "rank": 2,
        "z_ar": 0.77,
        "z_process_mode": "ar1",
        "z_sd": EMPIRICAL_SCALE.instrument_sd,
        "h_ar": 0.35,
        "factor_ar": 0.0,
        "h_z_corr": 0.60,
        "h_z_corr_pre": 0.60,
        "h_z_corr_post": 0.60,
        "pi_sd": 1.0,
        "exposure_profile_mode": "raw",
        "lowrank_w_sd": COMPONENT_SCALE_MULTIPLIERS["lowrank_w_sd"] * W_UNIT,
        "lowrank_y_sd": COMPONENT_SCALE_MULTIPLIERS["lowrank_y_sd"] * W_UNIT,
        "lowrank_w_pi_corr": 0.65,
        "lowrank_yw_loading_corr": 0.45,
        "lowrank_w_shock_trait_corr": 0.0,
        "lowrank_y_shock_trait_corr": 0.0,
        "lowrank_factor_mode": "z_correlated",
        "lowrank_factor_orthogonalization_weight": 0.15,
        "lowrank_factor_z_corr": 0.01,
        "shock_w_sd": COMPONENT_SCALE_MULTIPLIERS["shock_w_sd"] * W_UNIT,
        "shock_y_sd": 1.0,
        "shock_w_pi_corr": 0.0,
        "shock_y_pi_corr": -0.25,
        "shock_yw_loading_corr": 0.995,
        "eps_w_sd": COMPONENT_SCALE_MULTIPLIERS["eps_w_sd"] * W_UNIT,
        "eps_y_sd": COMPONENT_SCALE_MULTIPLIERS["eps_y_sd"] * W_UNIT,
        "eps_yw_corr": 0.0,
        "aggregate_shock_mode": "sample_correlated",
        "loading_correlation_mode": "sample_orthogonalized",
        "shock_y_loading_mode": "pi_theta_w_blend",
    }
)
