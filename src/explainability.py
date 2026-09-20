"""
Explainable AI.

The dynamic (Module A) and temporal (Module B) detectors are UNSUPERVISED
(Isolation Forest + classical statistics). SHAP is not a natural fit for
Isolation Forest's anomaly score directly, so, following the brief's
instruction to use SHAP "where appropriate, particularly for the
supervised/ML component", we:

  1. Train a supervised RandomForestClassifier as an auxiliary / diagnostic
     model that predicts `future_failure_label_168h` (ground truth,
     available because this is a labeled synthetic dataset) from the
     interpretable Module A + Module B features. This model is NOT what
     produces the final anomaly score (that stays purely
     statistical/unsupervised, as required) - it exists so we can run SHAP
     and show *global* feature importance for what tends to predict failure.
  2. For the *per-component* explanation shown in the dashboard, we use a
     transparent rule-based reason generator directly on the Module A/B
     signals (robust z, drift %, jumps, spec proximity). This is fully
     interpretable and does not depend on the auxiliary model, avoiding the
     "explain a black box with another black box" problem for the
     unsupervised detectors.
"""

import logging

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("explainability")

MODEL_FEATURES = [
    "robust_z_final", "population_anomaly_score", "iso_forest_score",
    "temporal_anomaly_score", "iso_forest_temporal_score", "pct_change",
    "n_abnormal_jumps", "acceleration", "spec_proximity",
]


def train_supervised_proxy(scored: pd.DataFrame):
    """Train a RandomForest on the TRAIN split only (no leakage) to predict
    the ground-truth future_failure_label_168h from Module A/B features, and
    return the fitted model + a SHAP TreeExplainer."""
    train = scored[scored["split"] == "train"].copy()
    X_train = train[MODEL_FEATURES].fillna(0)
    y_train = train["future_failure_label_168h"].astype(int)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=3,
        random_state=config.RANDOM_SEED, class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    test = scored[scored["split"] == "test"].copy()
    if len(test) and test["future_failure_label_168h"].nunique() > 1:
        X_test = test[MODEL_FEATURES].fillna(0)
        y_test = test["future_failure_label_168h"].astype(int)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        logger.info("Auxiliary SHAP proxy model held-out test ROC-AUC: %.3f", auc)

    explainer = shap.TreeExplainer(clf)
    joblib.dump({"model": clf, "features": MODEL_FEATURES}, config.MODEL_PATH)
    return clf, explainer


def global_feature_importance(clf, explainer, scored: pd.DataFrame) -> pd.DataFrame:
    X = scored[MODEL_FEATURES].fillna(0)
    shap_values = explainer.shap_values(X)
    # shap_values can be (n, features) or list [class0, class1] depending on version
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    if sv.ndim == 3:  # (n, features, classes) in newer shap versions
        sv = sv[:, :, 1]
    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.DataFrame({"feature": MODEL_FEATURES, "mean_abs_shap": mean_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return imp


def _reason_lines(row: pd.Series) -> list[str]:
    """Rule-based, fully transparent explanation for ONE component, built
    directly from Module A / Module B signals."""
    reasons = []
    rz = row.get("robust_z_final", 0)
    if abs(rz) >= config.ROBUST_Z_CRITICAL:
        reasons.append(f"{row['parameter']} is {abs(rz):.1f}\u03c3 above/below the lot median (robust z-score)")
    elif abs(rz) >= config.ROBUST_Z_WARN:
        reasons.append(f"{row['parameter']} deviates {abs(rz):.1f}\u03c3 from the lot median")

    pct = row.get("pct_change", np.nan)
    if pd.notna(pct) and abs(pct) >= 50:
        reasons.append(f"{row['parameter']} changed {pct:.0f}% during burn-in (0h\u2192168h)")

    if row.get("n_abnormal_jumps", 0) >= 1:
        reasons.append(f"{int(row['n_abnormal_jumps'])} sudden jump(s) detected in the burn-in trajectory")

    accel = row.get("acceleration", 0)
    std = row.get("std_val", None)
    if pd.notna(accel) and abs(accel) > 0 and row.get("temporal_flag", 0) == 1:
        direction = "accelerating" if accel > 0 else "decelerating"
        reasons.append(f"Degradation rate is {direction} (late-stage slope differs from early-stage slope)")

    if row.get("iso_forest_score", 0) >= 0.6:
        reasons.append("Multivariate lot comparison (Isolation Forest) flags the full trajectory shape as unusual")

    if row.get("spec_violation", 0) == 1:
        reasons.append(f"Measurement is OUTSIDE the datasheet limit [{row['datasheet_min']}, {row['datasheet_max']}]")
    elif row.get("spec_proximity", 0) >= 0.6:
        reasons.append("Measurement is approaching the datasheet limit")

    if not reasons:
        reasons.append("No significant population or temporal deviation detected")
    return reasons


def explain_component(scored: pd.DataFrame, component_id: str) -> dict:
    row = scored[scored["component_id"] == component_id]
    if row.empty:
        raise KeyError(f"component_id {component_id} not found")
    row = row.iloc[0]
    return {
        "component_id": component_id,
        "lot_id": row["lot_id"],
        "parameter": row["parameter"],
        "risk_level": row["risk_level"],
        "case_classification": row["case_classification"],
        "final_anomaly_score": round(float(row["final_anomaly_score"]), 3),
        "datasheet_status": row["datasheet_status"],
        "ai_status": row["ai_status"],
        "qa_recommendation": row["qa_recommendation"],
        "reasons": _reason_lines(row),
    }


def add_explanations(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    scored["ai_reasons"] = scored.apply(lambda r: " | ".join(_reason_lines(r)), axis=1)
    return scored


def run(scored: pd.DataFrame) -> pd.DataFrame:
    clf, explainer = train_supervised_proxy(scored)
    imp = global_feature_importance(clf, explainer, scored)
    imp.to_csv(config.REPORTS_DIR / "shap_global_feature_importance.csv", index=False)
    logger.info("Global SHAP feature importance:\n%s", imp.to_string(index=False))
    scored = add_explanations(scored)
    return scored


if __name__ == "__main__":
    import pandas as pd
    scored = pd.read_csv(config.PROCESSED_WIDE_CSV)
    scored = run(scored)
    print(explain_component(scored, scored["component_id"].iloc[0]))
