"""
Module B - Time-Series / Temporal Anomaly Detection.

For each component x parameter, the burn-in trajectory
(0h -> 24h -> 96h -> 168h) is turned into interpretable temporal features,
then scored for abnormal drift / jumps / acceleration - independent of
whether the raw values ever cross the datasheet limit.

Features extracted per component:
  - initial_value, final_value, abs_change, pct_change
  - mean, std, max, min
  - slope (linear regression over all 4 points, µ-unit/hour)
  - early_slope (0h->24h), late_slope (96h->168h) and their difference
    (acceleration - is degradation speeding up?)
  - rate_of_change per step (max absolute step-to-step slope)
  - n_abnormal_jumps: number of step-to-step changes whose EWMA-normalized
    z-score exceeds SUDDEN_JUMP_Z

These features feed:
  (a) a rule-based drift/acceleration/jump score, and
  (b) an Isolation Forest trained on the temporal-feature space (lot-wise,
      falling back to component_type-wise if a lot group is too small).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("temporal_detection")

MAD_SCALE = 1.4826

TEMPORAL_FEATURE_COLS = [
    "abs_change", "pct_change", "mean_val", "std_val", "max_val", "min_val",
    "slope", "early_slope", "late_slope", "acceleration", "max_step_rate", "n_abnormal_jumps",
]


def _linear_slope(times: np.ndarray, values: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    reg = LinearRegression().fit(times.reshape(-1, 1), values)
    return float(reg.coef_[0])


def extract_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """Build one feature row per component_id x parameter from the sorted
    long-format time series."""
    rows = []
    for (comp_id, parameter), g in long_df.groupby(["component_id", "parameter"]):
        g = g.sort_values("time_h")
        t = g["time_h"].values.astype(float)
        v = g["parameter_value"].values.astype(float)
        if len(v) < 2:
            continue

        initial, final = v[0], v[-1]
        abs_change = final - initial
        pct_change = (abs_change / initial * 100.0) if initial != 0 else np.nan

        step_changes = np.diff(v)
        step_times = np.diff(t)
        step_slopes = step_changes / np.where(step_times == 0, 1e-6, step_times)

        slope = _linear_slope(t, v)

        # early slope (0h->24h) vs late slope (last two available points)
        early_slope = step_slopes[0] if len(step_slopes) >= 1 else 0.0
        late_slope = step_slopes[-1] if len(step_slopes) >= 1 else 0.0
        acceleration = late_slope - early_slope

        # Per-step percentage changes (used later for a PEER-relative jump
        # test - with only 4 points per component, comparing a component's
        # own EWMA residual to itself is unreliable; comparing its step
        # changes to the same step across its LOT peers is far more robust).
        step_pct = np.divide(
            step_changes, np.where(v[:-1] == 0, 1e-6, v[:-1]),
        ) * 100.0
        step_pct = np.pad(step_pct, (0, 3 - len(step_pct)), constant_values=np.nan)

        rows.append({
            "component_id": comp_id,
            "parameter": parameter,
            "lot_id": g["lot_id"].iloc[0],
            "component_type": g["component_type"].iloc[0],
            "initial_value": initial,
            "final_value": final,
            "abs_change": abs_change,
            "pct_change": pct_change,
            "mean_val": v.mean(),
            "std_val": v.std(ddof=0),
            "max_val": v.max(),
            "min_val": v.min(),
            "slope": slope,
            "early_slope": early_slope,
            "late_slope": late_slope,
            "acceleration": acceleration,
            "max_step_rate": np.max(np.abs(step_slopes)) if len(step_slopes) else 0.0,
            "step_pct_1": step_pct[0], "step_pct_2": step_pct[1], "step_pct_3": step_pct[2],
        })

    feats = pd.DataFrame(rows)
    logger.info("Extracted temporal features for %d component/parameter series", len(feats))
    return feats


def _group_isolation_forest(group: pd.DataFrame) -> pd.Series:
    if len(group) < 6:
        return pd.Series(0.0, index=group.index)
    X = group[TEMPORAL_FEATURE_COLS].fillna(0).values
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1e-6
    Xs = (X - mu) / sigma
    iso = IsolationForest(
        n_estimators=200,
        contamination=config.ISOLATION_FOREST_CONTAMINATION_TEMPORAL,
        random_state=config.RANDOM_SEED,
    )
    iso.fit(Xs)
    raw = -iso.decision_function(Xs)
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return pd.Series(raw, index=group.index)


def _peer_step_jump_flags(group: pd.DataFrame) -> pd.Series:
    """Peer-relative sudden-jump detection: for each of the 3 burn-in
    transitions (0->24h, 24->96h, 96->168h), compare this component's step
    change to the SAME step across its lot/parameter peers using a robust
    z-score. With only 4 points per component this is far more reliable
    than judging a component's jump against its own short history."""
    n_jumps = pd.Series(0, index=group.index)
    for step_col in ["step_pct_1", "step_pct_2", "step_pct_3"]:
        vals = group[step_col]
        med = vals.median()
        mad = (vals - med).abs().median() * MAD_SCALE
        if mad == 0 or np.isnan(mad):
            mad = vals.std(ddof=0) or 1e-6
        rz = ((vals - med) / mad).abs()
        n_jumps = n_jumps + (rz >= config.SUDDEN_JUMP_Z).astype(int).fillna(0)
    return n_jumps


def detect(long_df: pd.DataFrame) -> pd.DataFrame:
    feats = extract_features(long_df)

    parts = []
    for (lot_id, parameter), g in feats.groupby(["lot_id", "parameter"]):
        g = g.copy()
        g["n_abnormal_jumps"] = _peer_step_jump_flags(g) if len(g) >= 6 else 0
        g["iso_forest_temporal_score"] = _group_isolation_forest(g)
        parts.append(g)
    feats = pd.concat(parts).sort_index()

    # Rule-based sub-scores (0-1), each capturing a distinct temporal pattern
    pct_change_score = np.clip(feats["pct_change"].abs() / 100.0, 0, 1)          # >=100% change -> maxed
    jump_score = np.clip(feats["n_abnormal_jumps"] / 2.0, 0, 1)                    # >=2 jumps -> maxed
    accel_score = np.clip(feats["acceleration"].abs() / (feats["std_val"].replace(0, np.nan) + 1e-6), 0, 1).fillna(0)
    iso_score = feats["iso_forest_temporal_score"].clip(0, 1)

    feats["temporal_anomaly_score"] = (
        0.30 * pct_change_score
        + 0.25 * jump_score
        + 0.20 * accel_score
        + 0.25 * iso_score
    ).clip(0, 1)

    feats["temporal_flag"] = (
        (feats["n_abnormal_jumps"] >= 1)
        | (pct_change_score >= 0.5)
        | (feats["iso_forest_temporal_score"] >= 0.6)
    ).astype(int)

    logger.info(
        "Module B complete: %d/%d series flagged with temporal anomalies",
        feats["temporal_flag"].sum(), len(feats),
    )
    return feats


if __name__ == "__main__":
    from . import preprocessing
    _, long_df = preprocessing.run()
    feats = detect(long_df)
    print(feats[["component_id", "parameter", "temporal_anomaly_score", "temporal_flag"]].head(10))
