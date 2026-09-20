"""
Module A - Dynamic / Population-Aware Anomaly Detection.

Instead of comparing a component only to a fixed datasheet limit, we learn
the behavior of its own manufacturing lot (peer group = lot_id x parameter)
and flag components that deviate significantly from that peer group, at
EVERY test stage (0h/24h/96h/168h) - not just at the end.

Methods implemented (all lot-wise, i.e. computed independently per
lot_id x parameter group so that legitimate lot-to-lot process differences
are not confused with anomalies):

  1. Z-score                (mean / std)
  2. Robust Z-score         (median / MAD - resistant to the very outliers
                              we are trying to detect)
  3. IQR method              (Q1, Q3, 1.5*IQR fences)
  4. Percentile rank         (where does this component sit in its lot?)
  5. Isolation Forest        (multivariate, lot-wise, on the full 4-point
                              trajectory so shape - not just final value -
                              matters)

All signals are combined into a single 0-1 `population_anomaly_score`.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dynamic_detection")

VALUE_COLS = [f"parameter_value_{h}h" for h in config.TEST_HOURS]
MAD_SCALE = 1.4826  # scales MAD to be comparable to std under normality


def _robust_z(values: pd.Series) -> pd.Series:
    med = values.median()
    mad = (values - med).abs().median() * MAD_SCALE
    if mad == 0 or np.isnan(mad):
        mad = values.std(ddof=0) or 1e-6
    return (values - med) / mad


def _iqr_flag(values: pd.Series) -> pd.Series:
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(0, index=values.index)
    lower = q1 - config.IQR_MULTIPLIER * iqr
    upper = q3 + config.IQR_MULTIPLIER * iqr
    return ((values < lower) | (values > upper)).astype(int)


def _percentile_rank(values: pd.Series) -> pd.Series:
    # distance from the median expressed as a percentile (0 = at median,
    # 1 = at the most extreme edge of the lot's distribution)
    ranks = values.rank(pct=True)
    return (ranks - 0.5).abs() * 2


def _lot_isolation_forest(group: pd.DataFrame) -> pd.Series:
    """Multivariate anomaly detection on the full trajectory shape within
    one lot x parameter peer group. Falls back to all-normal if the group
    is too small for a meaningful forest."""
    if len(group) < 6:
        return pd.Series(0.0, index=group.index)
    X = group[VALUE_COLS].values
    # Standardize within-group so that different parameters/scales are
    # comparable before feeding the forest.
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1e-6
    Xs = (X - mu) / sigma
    iso = IsolationForest(
        n_estimators=200,
        contamination=config.ISOLATION_FOREST_CONTAMINATION,
        random_state=config.RANDOM_SEED,
    )
    iso.fit(Xs)
    # decision_function: higher = more normal. Convert to 0-1 anomaly score.
    raw = -iso.decision_function(Xs)
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return pd.Series(raw, index=group.index)


def detect(wide: pd.DataFrame) -> pd.DataFrame:
    df = wide.copy()
    for col in ["z_final", "robust_z_final", "iqr_flag_final", "percentile_dev_final",
                "iso_forest_score", "population_anomaly_score"]:
        df[col] = 0.0

    out_parts = []
    for (lot_id, parameter), g in df.groupby(["lot_id", "parameter"]):
        g = g.copy()
        final_vals = g["parameter_value_168h"]

        std = final_vals.std(ddof=0) or 1e-6
        g["z_final"] = (final_vals - final_vals.mean()) / std
        g["robust_z_final"] = _robust_z(final_vals)
        g["iqr_flag_final"] = _iqr_flag(final_vals)
        g["percentile_dev_final"] = _percentile_rank(final_vals)
        g["iso_forest_score"] = _lot_isolation_forest(g)

        out_parts.append(g)

    result = pd.concat(out_parts).sort_index()

    # --- Combine the five signals into one 0-1 population anomaly score ---
    z_component = np.clip(result["robust_z_final"].abs() / config.ROBUST_Z_CRITICAL, 0, 1)
    percentile_component = result["percentile_dev_final"].clip(0, 1)
    iqr_component = result["iqr_flag_final"].astype(np.float64)
    iso_component = result["iso_forest_score"].clip(0, 1)

    result["population_anomaly_score"] = (
        0.35 * z_component
        + 0.20 * percentile_component
        + 0.15 * iqr_component
        + 0.30 * iso_component
    ).clip(0, 1)

    result["population_flag"] = (
        (result["robust_z_final"].abs() >= config.ROBUST_Z_WARN)
        | (result["iqr_flag_final"] == 1)
        | (result["iso_forest_score"] >= 0.6)
    ).astype(int)

    logger.info(
        "Module A complete: %d/%d components flagged as population anomalies",
        result["population_flag"].sum(), len(result),
    )
    return result


if __name__ == "__main__":
    from . import preprocessing
    wide, _ = preprocessing.run()
    scored = detect(wide)
    print(scored[["component_id", "lot_id", "parameter", "population_anomaly_score", "population_flag"]].head(10))
