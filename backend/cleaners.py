"""
cleaners.py — Scikit-learn / scipy ML cleaning pipelines.

Supported methods per category
--------------------------------
Imputation (numeric):
  - iterative       : sklearn IterativeImputer (linear regression internally)
  - knn             : sklearn KNNImputer
  - median          : column median
  - mean            : column mean

Imputation (categorical):
  - mode            : most-frequent value

Outlier handling (numeric only):
  - isolation_forest : IsolationForest; rows marked as outliers → NaN or removed
  - zscore           : |z| > threshold → NaN or removed

Encoding (categorical):
  - onehot          : sklearn OneHotEncoder (sparse=False)
  - target          : mean-target encoding (requires numeric target col param)
  - label           : simple ordinal encoding

Scaling (numeric):
  - standard        : sklearn StandardScaler
  - robust          : sklearn RobustScaler
  - minmax          : sklearn MinMaxScaler
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder,
)

warnings.filterwarnings("ignore")


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(exclude=[np.number]).columns.tolist()


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def impute_iterative(df: pd.DataFrame, cols: List[str], max_iter: int = 10) -> pd.DataFrame:
    """IterativeImputer (MICE-like linear regression)."""
    num_cols = [c for c in cols if c in _numeric_cols(df)]
    if not num_cols:
        return df
    imp = IterativeImputer(max_iter=max_iter, random_state=42)
    df = df.copy()
    df[num_cols] = imp.fit_transform(df[num_cols])
    return df


def impute_knn(df: pd.DataFrame, cols: List[str], n_neighbors: int = 5) -> pd.DataFrame:
    """KNN Imputer."""
    num_cols = [c for c in cols if c in _numeric_cols(df)]
    if not num_cols:
        return df
    imp = KNNImputer(n_neighbors=n_neighbors)
    df = df.copy()
    df[num_cols] = imp.fit_transform(df[num_cols])
    return df


def impute_median(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in _numeric_cols(df):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
    return df


def impute_mean(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in _numeric_cols(df):
            df[col] = df[col].fillna(df[col].mean())
    return df


def impute_mode(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val.iloc[0])
    return df


# ---------------------------------------------------------------------------
# Outlier detection & removal
# ---------------------------------------------------------------------------

def outlier_isolation_forest(
    df: pd.DataFrame,
    cols: List[str],
    contamination: float = 0.05,
    action: str = "remove",
) -> pd.DataFrame:
    """
    Detect outliers with IsolationForest on selected numeric columns.
    action='remove' → drop rows; action='nan' → set outlier cells to NaN.
    """
    num_cols = [c for c in cols if c in _numeric_cols(df)]
    if not num_cols:
        return df

    df = df.copy()
    sub = df[num_cols].dropna()
    if sub.empty:
        return df

    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = clf.fit_predict(sub)  # -1 = outlier, 1 = inlier
    outlier_idx = sub.index[preds == -1]

    if action == "remove":
        df = df.drop(index=outlier_idx).reset_index(drop=True)
    else:  # nan
        for col in num_cols:
            df.loc[df.index.isin(outlier_idx), col] = np.nan

    return df


def outlier_zscore(
    df: pd.DataFrame,
    cols: List[str],
    threshold: float = 3.0,
    action: str = "remove",
) -> pd.DataFrame:
    """Remove/NaN rows where |z-score| > threshold for any selected column."""
    num_cols = [c for c in cols if c in _numeric_cols(df)]
    if not num_cols:
        return df

    df = df.copy()
    mask_any_outlier = pd.Series(False, index=df.index)

    for col in num_cols:
        col_data = df[col].dropna()
        if col_data.empty:
            continue
        z = np.abs(scipy_stats.zscore(col_data))
        outlier_idx = col_data.index[z > threshold]
        if action == "nan":
            df.loc[outlier_idx, col] = np.nan
        else:
            mask_any_outlier.loc[outlier_idx] = True

    if action == "remove":
        df = df[~mask_any_outlier].reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_onehot(df: pd.DataFrame, cols: List[str], drop_first: bool = False) -> pd.DataFrame:
    """One-Hot Encode categorical columns."""
    cat_cols = [c for c in cols if c in df.columns]
    if not cat_cols:
        return df

    df = df.copy()
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first" if drop_first else None)
    encoded = enc.fit_transform(df[cat_cols].astype(str))
    feature_names = enc.get_feature_names_out(cat_cols)
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded_df], axis=1)
    return df


def encode_label(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Ordinal / label encode categorical columns."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            le = LabelEncoder()
            non_null_mask = df[col].notna()
            df.loc[non_null_mask, col] = le.fit_transform(df.loc[non_null_mask, col].astype(str))
    return df


def encode_target(df: pd.DataFrame, cols: List[str], target_col: str) -> pd.DataFrame:
    """
    Mean target encoding.
    Replaces each category with the mean of target_col for that category.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    df = df.copy()
    for col in cols:
        if col not in df.columns or col == target_col:
            continue
        means = df.groupby(col)[target_col].mean()
        df[col] = df[col].map(means)

    return df


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def scale_standard(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    num_cols = [c for c in cols if c in _numeric_cols(df)]
    if not num_cols:
        return df
    df = df.copy()
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def scale_robust(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    num_cols = [c for c in cols if c in _numeric_cols(df)]
    if not num_cols:
        return df
    df = df.copy()
    scaler = RobustScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def scale_minmax(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    num_cols = [c for c in cols if c in _numeric_cols(df)]
    if not num_cols:
        return df
    df = df.copy()
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

METHOD_MAP = {
    # Imputation
    "iterative": lambda df, cols, params: impute_iterative(df, cols, **params),
    "knn": lambda df, cols, params: impute_knn(df, cols, **params),
    "median": lambda df, cols, params: impute_median(df, cols),
    "mean": lambda df, cols, params: impute_mean(df, cols),
    "mode": lambda df, cols, params: impute_mode(df, cols),
    # Outliers
    "isolation_forest": lambda df, cols, params: outlier_isolation_forest(df, cols, **params),
    "zscore": lambda df, cols, params: outlier_zscore(df, cols, **params),
    # Encoding
    "onehot": lambda df, cols, params: encode_onehot(df, cols, **params),
    "label": lambda df, cols, params: encode_label(df, cols),
    "target": lambda df, cols, params: encode_target(df, cols, **params),
    # Scaling
    "standard": lambda df, cols, params: scale_standard(df, cols),
    "robust": lambda df, cols, params: scale_robust(df, cols),
    "minmax": lambda df, cols, params: scale_minmax(df, cols),
}


class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def apply(
        self,
        global_config: Optional[Dict[str, Any]],
        column_configs: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Apply cleaning operations. Column-level configs are applied first,
        then global config is applied to remaining columns not yet touched.

        Each config entry: {"method": str, "params": dict}
        """
        df = self.df.copy()
        all_cols = list(df.columns)
        handled_cols = set()

        # 1. Column-level operations (grouped by method for efficiency)
        method_to_cols: Dict[str, Dict[str, Any]] = {}
        for col, cfg in column_configs.items():
            if col not in df.columns:
                continue
            method = cfg.get("method")
            if not method or method not in METHOD_MAP:
                continue
            if method not in method_to_cols:
                method_to_cols[method] = {"cols": [], "params": cfg.get("params", {})}
            method_to_cols[method]["cols"].append(col)
            handled_cols.add(col)

        for method, info in method_to_cols.items():
            df = METHOD_MAP[method](df, info["cols"], info["params"])

        # 2. Global config — applied to remaining columns
        if global_config:
            method = global_config.get("method")
            params = global_config.get("params", {})
            if method and method in METHOD_MAP:
                remaining = [c for c in all_cols if c in df.columns and c not in handled_cols]
                if remaining:
                    df = METHOD_MAP[method](df, remaining, params)

        return df
