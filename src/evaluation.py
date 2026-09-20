"""
Evaluation: Static Datasheet Screening vs Dynamic AI Screening.

Because this is a LABELED SYNTHETIC dataset (failure_mode + the
future_failure_label_168h ground-truth column), we can objectively measure
how many injected defects each approach catches - most importantly, LATENT
DEFECTS that stay within the datasheet limit the whole time.

Ground truth used: `future_failure_label_168h` (1 = genuinely unsafe
component by the dataset's construction rule).

Static prediction:  spec_violation      (raw datasheet-limit crossing only)
Dynamic prediction:  risk_level != NORMAL (population + temporal + spec)

All numbers below are computed directly from the dataset - nothing is
hardcoded or "made to look good".
"""

import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score,
)

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluation")


def _binary_metrics(y_true, y_pred, y_score=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = round(roc_auc_score(y_true, y_score), 4)
    else:
        out["roc_auc"] = None
    return out


def evaluate(scored: pd.DataFrame, split: str | None = "test") -> dict:
    df = scored if split is None else scored[scored["split"] == split]
    y_true = df["future_failure_label_168h"].astype(int).values

    static_pred = df["spec_violation"].astype(int).values
    dynamic_pred = (df["risk_level"] != "NORMAL").astype(int).values
    dynamic_score = df["final_anomaly_score"].values

    results = {
        "split": split or "all",
        "n_components": int(len(df)),
        "n_positive_ground_truth": int(y_true.sum()),
        "static_screening": _binary_metrics(y_true, static_pred),
        "dynamic_ai_screening": _binary_metrics(y_true, dynamic_pred, dynamic_score),
    }

    # --- Latent-defect focused comparison ---------------------------------
    # Latent = truly unsafe (y_true==1) but did NOT cross the datasheet limit
    latent_mask = (y_true == 1) & (static_pred == 0)
    n_latent = int(latent_mask.sum())
    n_latent_caught_by_dynamic = int(((dynamic_pred == 1) & latent_mask).sum())
    results["latent_defect_analysis"] = {
        "n_latent_defects_in_split": n_latent,
        "n_caught_by_static_screening": 0,  # by definition, static cannot catch these
        "n_caught_by_dynamic_ai_screening": n_latent_caught_by_dynamic,
        "dynamic_recall_on_latent_defects": round(
            n_latent_caught_by_dynamic / n_latent, 4
        ) if n_latent else None,
    }

    # --- Breakdown by injected failure_mode --------------------------------
    if "failure_mode" in df.columns:
        breakdown = []
        for mode, g in df.groupby("failure_mode"):
            yt = g["future_failure_label_168h"].astype(int).values
            dp = (g["risk_level"] != "NORMAL").astype(int).values
            sp = g["spec_violation"].astype(int).values
            breakdown.append({
                "failure_mode": mode,
                "n": int(len(g)),
                "static_detected": int((sp == 1).sum()),
                "dynamic_detected": int((dp == 1).sum()),
                "ground_truth_positive": int(yt.sum()),
            })
        results["by_failure_mode"] = breakdown

    return results


def run(scored: pd.DataFrame) -> dict:
    test_results = evaluate(scored, split="test")
    all_results = evaluate(scored, split=None)
    report = {"test_split": test_results, "full_dataset": all_results}
    with open(config.EVAL_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved evaluation report to %s", config.EVAL_REPORT_PATH)
    logger.info(json.dumps(test_results, indent=2))
    return report


if __name__ == "__main__":
    scored = pd.read_csv(config.PROCESSED_WIDE_CSV)
    run(scored)
