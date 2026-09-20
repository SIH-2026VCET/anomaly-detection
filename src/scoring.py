"""
Combined Anomaly Scoring.

Final Score = w_pop * Population Score + w_temporal * Temporal Score
              + w_spec * Specification Score

Weights and risk thresholds live in src/config.py and are explicitly
PROTOTYPE values, not scientifically calibrated ones.

This module also implements the CRITICAL DESIGN REQUIREMENT: distinguishing
NORMAL / LATENT ANOMALY / DEGRADATION RISK / SPECIFICATION FAILURE /
HIGH-RISK COMPONENT, so the system can show cases such as:

    Datasheet Status: PASS
    AI Status: ANOMALOUS
"""

import logging

import numpy as np
import pandas as pd

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scoring")


def _spec_violation_and_proximity(row: pd.Series) -> tuple[int, float]:
    """Return (violation: 0/1, proximity: 0-1) using the final (168h) value
    against the datasheet [min, max] window."""
    value = row["parameter_value_168h"]
    lo, hi = row["datasheet_min"], row["datasheet_max"]
    if hi == lo:
        return 0, 0.0
    if value < lo or value > hi:
        return 1, 1.0
    pos = (value - lo) / (hi - lo)  # 0..1 within the allowed window
    proximity = float(np.clip(2 * abs(pos - 0.5), 0, 1))  # 0=center, 1=at a limit
    return 0, proximity


def _risk_level(score: float) -> str:
    for level, (low, high) in config.RISK_THRESHOLDS.items():
        if low <= score < high:
            return level
    return "CRITICAL"


def _case_classification(row: pd.Series) -> str:
    if row["spec_violation"] == 1:
        return "SPECIFICATION_FAILURE"
    pop_bad = row["population_flag"] == 1
    temp_bad = row["temporal_flag"] == 1
    near_limit = row["spec_proximity"] >= 0.6
    if pop_bad and temp_bad and near_limit:
        return "HIGH_RISK_COMPONENT"
    if pop_bad and temp_bad:
        return "HIGH_RISK_COMPONENT"
    if pop_bad and not temp_bad:
        return "LATENT_ANOMALY"
    if temp_bad and not pop_bad:
        return "DEGRADATION_RISK"
    return "NORMAL"


def combine(pop_scored: pd.DataFrame, temporal_scored: pd.DataFrame) -> pd.DataFrame:
    key = ["component_id", "parameter"]
    merged = pop_scored.merge(
        temporal_scored[key + [
            "temporal_anomaly_score", "temporal_flag", "slope", "acceleration",
            "n_abnormal_jumps", "pct_change", "abs_change", "std_val",
            "iso_forest_temporal_score",
        ]],
        on=key, how="left",
    )
    merged["temporal_anomaly_score"] = merged["temporal_anomaly_score"].fillna(0.0)
    merged["temporal_flag"] = merged["temporal_flag"].fillna(0).astype(int)

    spec = merged.apply(_spec_violation_and_proximity, axis=1, result_type="expand")
    merged["spec_violation"] = spec[0].astype(int)
    merged["spec_proximity"] = spec[1].astype(float)
    merged["specification_score"] = np.where(
        merged["spec_violation"] == 1, 1.0, merged["spec_proximity"]
    )

    w = config.SCORE_WEIGHTS
    merged["final_anomaly_score"] = (
        w["population"] * merged["population_anomaly_score"]
        + w["temporal"] * merged["temporal_anomaly_score"]
        + w["specification"] * merged["specification_score"]
    ).clip(0, 1)
    # A hard specification failure always caps the score at CRITICAL.
    merged.loc[merged["spec_violation"] == 1, "final_anomaly_score"] = merged.loc[
        merged["spec_violation"] == 1, "final_anomaly_score"
    ].clip(lower=config.RISK_THRESHOLDS["CRITICAL"][0])

    merged["risk_level"] = merged["final_anomaly_score"].apply(_risk_level)
    merged["case_classification"] = merged.apply(_case_classification, axis=1)
    merged["qa_recommendation"] = merged["risk_level"].map(config.QA_RECOMMENDATION)
    merged["datasheet_status"] = np.where(merged["spec_violation"] == 1, "FAIL", "PASS")
    merged["ai_status"] = np.where(merged["risk_level"] == "NORMAL", "NORMAL", "ANOMALOUS")

    logger.info(
        "Scoring complete. Risk level counts:\n%s",
        merged["risk_level"].value_counts().to_string(),
    )
    logger.info(
        "Case classification counts:\n%s",
        merged["case_classification"].value_counts().to_string(),
    )
    n_latent_like = ((merged["datasheet_status"] == "PASS") & (merged["ai_status"] == "ANOMALOUS")).sum()
    logger.info(
        "Components that PASS the datasheet but are flagged ANOMALOUS by the AI system: %d / %d",
        n_latent_like, len(merged),
    )
    return merged


if __name__ == "__main__":
    from . import preprocessing, dynamic_detection, temporal_detection
    wide, long_df = preprocessing.run()
    pop = dynamic_detection.detect(wide)
    temp = temporal_detection.detect(long_df)
    scored = combine(pop, temp)
    scored.to_csv(config.PROCESSED_WIDE_CSV, index=False)
    print(scored[[
        "component_id", "datasheet_status", "ai_status", "risk_level", "case_classification",
        "final_anomaly_score",
    ]].head(15))
