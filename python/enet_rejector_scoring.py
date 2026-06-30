"""Shared elastic-net rejector scoring (matches R export_enet_rejector_coef_df layout)."""

from __future__ import annotations

import numpy as np
import pandas as pd

ELASTICNET_TWO_HEAD_PARAMS_KEY = "two_head"

# Params CSV basename (matches R calibration_reject_deploy_config.R params_file_key).
def enet_params_file_key(rejector_key: str) -> str:
    return rejector_key


def enet_params_rejector_key(rejector_key: str) -> str:
    """Backward-compatible alias: deployment uses rejector_key for params paths."""
    return enet_params_file_key(rejector_key)


def score_logistic_head(feature_map: dict[str, np.ndarray], params: dict[str, float], feature_scales=None):
    if not params:
        raise ValueError("GLM parameters are empty.")
    if not feature_map:
        raise ValueError("Rejection feature map is empty.")
    n_rows = len(next(iter(feature_map.values())))
    linear = np.full(n_rows, float(params["(Intercept)"]), dtype=np.float64)
    for term, coef in params.items():
        if term == "(Intercept)":
            continue
        if term not in feature_map:
            raise ValueError(f"GLM requires feature '{term}' but it was not computed.")
        x_val = np.asarray(feature_map[term], dtype=np.float64)
        if feature_scales is not None:
            if term not in feature_scales:
                raise ValueError(f"Elastic-net params missing glmnet scale for feature '{term}'.")
            mean_x, sd_x = feature_scales[term]
            use_raw = (
                not np.isfinite(mean_x)
                or not np.isfinite(sd_x)
                or (abs(mean_x) < 1e-12 and abs(sd_x - 1.0) < 1e-12)
            )
            if not use_raw:
                if sd_x <= 0.0:
                    raise ValueError(f"Invalid glmnet sd_x for feature '{term}': {sd_x}")
                x_val = (x_val - mean_x) / sd_x
        linear += float(coef) * x_val
    return 1.0 / (1.0 + np.exp(-linear))


def score_enet_head_from_feature_df(test_df: pd.DataFrame, head_params_df: pd.DataFrame) -> np.ndarray:
    """Score rows from a feature DataFrame using one exported head CSV block."""
    if head_params_df.empty or test_df.empty:
        return np.array([], dtype=np.float64)
    intercept_rows = head_params_df[head_params_df["term"] == "(Intercept)"]
    if len(intercept_rows) != 1:
        raise ValueError("Exported elastic-net head missing one (Intercept) row.")
    linear = np.full(len(test_df), float(intercept_rows["estimate"].iloc[0]), dtype=np.float64)
    feat_rows = head_params_df[head_params_df["term"] != "(Intercept)"]
    for _, row in feat_rows.iterrows():
        term = row["term"]
        if term not in test_df.columns:
            raise ValueError(f"Exported elastic-net head requires feature '{term}' in test data.")
        mean_x = float(row["mean_x"])
        sd_x = float(row["sd_x"])
        x_val = test_df[term].to_numpy(dtype=np.float64)
        use_raw = (
            not np.isfinite(mean_x)
            or not np.isfinite(sd_x)
            or (abs(mean_x) < 1e-12 and abs(sd_x - 1.0) < 1e-12)
        )
        if use_raw:
            linear += float(row["estimate"]) * x_val
        else:
            if sd_x <= 0.0:
                raise ValueError(f"Invalid exported sd_x for feature '{term}': {sd_x}")
            linear += float(row["estimate"]) * (x_val - mean_x) / sd_x
    return 1.0 / (1.0 + np.exp(-linear))
