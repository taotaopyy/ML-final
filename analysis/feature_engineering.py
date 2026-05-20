"""
Feature engineering for target lesion prediction.

Creates derived features from the raw baseline predictors to capture
non-linear relationships and domain-relevant interactions that improve
discriminative power beyond raw features alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Build engineered features on top of the raw predictor columns.

    Returns (X_engineered, new_col_list) where X_engineered contains
    both the original and the new columns.
    """
    X = df[cols].copy()

    # --- Plaque burden interactions ---
    _safe_ratio(X, "stenosis_percent", "minimum_luminal_area_mm3", "stenosis_mla_ratio")
    _safe_product(X, "stenosis_percent", "whole_lesion_plaque_burden", "stenosis_x_burden")
    _safe_product(X, "stenosis_percent", "High-Risk Plaque Flag", "stenosis_x_hrp")
    _safe_product(X, "stenosis_percent", "non_calcified_plaque_burden", "stenosis_x_nc_burden")
    _safe_product(X, "non_calcified_plaque_burden", "fibrous_fatty_plaque_burden", "nc_x_ff_burden")

    # --- Composite plaque scores ---
    _safe_sum(X, [
        "napkin_ring_sign_flag", "spotty_calcification_flag",
        "positive_remodeling_flag", "low_attenuation_volume_ratio",
        "High-Risk Plaque Flag"
    ], "hrp_composite_score")

    _safe_sum(X, [
        "whole_lesion_plaque_burden", "non_calcified_plaque_burden",
        "fibrous_fatty_plaque_burden", "fibrotic_plaque_burden",
        "low_attenuation_plaque_burden", "calcified_plaque_burden"
    ], "total_burden_sum")

    # --- Volume ratios ---
    _safe_ratio(X, "non_calcified_volume_mm3", "whole_lesion_volume_mm3", "nc_vol_fraction")
    _safe_ratio(X, "low_attenuation_volume_mm3", "whole_lesion_volume_mm3", "la_vol_fraction")
    _safe_ratio(X, "fibrous_fatty_volume_mm3", "whole_lesion_volume_mm3", "ff_vol_fraction")
    _safe_ratio(X, "calcified_volume_mm3", "whole_lesion_volume_mm3", "calc_vol_fraction")

    # --- Stenosis severity features ---
    _safe_product(X, "maximum_area_stenosis_percent", "maximum_diameter_stenosis_percent",
                  "area_x_diam_stenosis")
    if "stenosis_percent" in X.columns:
        X["stenosis_sq"] = X["stenosis_percent"] ** 2
        X["stenosis_log"] = np.log1p(X["stenosis_percent"].clip(lower=0))

    # --- FFR-related ---
    _safe_product(X, "ffrct_value", "stenosis_percent", "ffr_x_stenosis")
    _safe_product(X, "ffrct_risk_class", "stenosis_percent", "ffr_risk_x_stenosis")
    if "ffrct_value" in X.columns:
        X["ffr_below_08"] = (X["ffrct_value"] < 0.80).astype(float)
        X["ffr_below_075"] = (X["ffrct_value"] < 0.75).astype(float)

    # --- Lesion geometry ---
    _safe_ratio(X, "whole_lesion_volume_mm3", "lesion_range_mm", "vol_per_mm_length")
    _safe_product(X, "lesion_range_mm", "stenosis_percent", "length_x_stenosis")

    # --- Lab value interactions ---
    _safe_ratio(X, "baseline_ldl", "baseline_hdl_cholesterol", "ldl_hdl_ratio")
    _safe_ratio(X, "baseline_triglycerides", "baseline_hdl_cholesterol", "tg_hdl_ratio")
    _safe_ratio(X, "baseline_neutrophil_pct", "baseline_lymphocyte_pct", "nlr_ratio")

    # --- Remodeling interactions ---
    _safe_product(X, "remodeling_index", "stenosis_percent", "remodel_x_stenosis")
    _safe_product(X, "remodeling_index", "non_calcified_plaque_burden", "remodel_x_nc_burden")

    # --- MLA-based features ---
    if "minimum_luminal_area_mm3" in X.columns:
        X["mla_below_4"] = (X["minimum_luminal_area_mm3"] < 4.0).astype(float)
        X["mla_below_3"] = (X["minimum_luminal_area_mm3"] < 3.0).astype(float)

    # --- Plaque feature count interactions ---
    _safe_product(X, "plaque_feature_count", "stenosis_percent", "feat_count_x_stenosis")

    # --- Follow-up / diff features (for all_variables set) ---
    _safe_ratio(X, "followup_ldl", "followup_hdl_cholesterol", "fu_ldl_hdl_ratio")
    _safe_ratio(X, "followup_triglycerides", "followup_hdl_cholesterol", "fu_tg_hdl_ratio")
    _safe_ratio(X, "followup_neutrophil_pct", "followup_lymphocyte_pct", "fu_nlr_ratio")
    _safe_product(X, "diff_ldl", "stenosis_percent", "diff_ldl_x_stenosis")
    _safe_product(X, "diff_hba1c", "stenosis_percent", "diff_hba1c_x_stenosis")
    _safe_product(X, "followup_ldl", "stenosis_percent", "fu_ldl_x_stenosis")

    new_cols = [c for c in X.columns if c not in cols] + list(cols)
    return X[new_cols], new_cols


def _safe_product(X: pd.DataFrame, a: str, b: str, name: str) -> None:
    if a in X.columns and b in X.columns:
        X[name] = X[a] * X[b]


def _safe_ratio(X: pd.DataFrame, num: str, den: str, name: str) -> None:
    if num in X.columns and den in X.columns:
        X[name] = X[num] / X[den].replace(0, np.nan)


def _safe_sum(X: pd.DataFrame, cols_list: list[str], name: str) -> None:
    present = [c for c in cols_list if c in X.columns]
    if present:
        X[name] = X[present].sum(axis=1)
