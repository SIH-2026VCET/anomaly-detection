"""
Run the complete AI-Driven Dynamic Anomaly Detection pipeline end-to-end:

    preprocessing -> Module A (population) -> Module B (temporal)
    -> combined scoring -> explainability (SHAP) -> evaluation
    -> visualizations

Usage:
    python run.py
"""

import json
import logging
import time

from src import (
    config, preprocessing, dynamic_detection, temporal_detection,
    scoring, explainability, evaluation, visualization
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run")


def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("AI-Driven Dynamic Anomaly Detection - full pipeline run")
    logger.info("=" * 70)

    logger.info("[1/7] Preprocessing raw data ...")
    wide, long_df = preprocessing.run()

    logger.info("[2/7] Module A - Population-aware dynamic anomaly detection ...")
    pop_scored = dynamic_detection.detect(wide)

    logger.info("[3/7] Module B - Temporal / time-series anomaly detection ...")
    temporal_scored = temporal_detection.detect(long_df)

    logger.info("[4/7] Combined scoring & risk classification ...")
    scored = scoring.combine(pop_scored, temporal_scored)

    logger.info("[5/7] Explainability (SHAP + rule-based reasons) ...")
    scored = explainability.run(scored)
    scored.to_csv(config.PROCESSED_WIDE_CSV, index=False)
    long_df.to_csv(config.PROCESSED_LONG_CSV, index=False)

    logger.info("[6/7] Evaluation - static vs dynamic screening ...")
    report = evaluation.run(scored)

    logger.info("[7/7] Generating visualizations ...")
    visualization.run(scored, long_df)

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("Pipeline complete in %.1fs", elapsed)
    logger.info("Processed data:      %s", config.PROCESSED_WIDE_CSV)
    logger.info("Evaluation report:   %s", config.EVAL_REPORT_PATH)
    logger.info("Figures:             %s", config.FIGURES_DIR)
    logger.info("Model:               %s", config.MODEL_PATH)
    logger.info("=" * 70)
    logger.info(
        "Summary: %d components | %d flagged ANOMALOUS by AI while PASSING "
        "the datasheet | dynamic latent-defect recall=%.2f vs static=0.00",
        len(scored),
        int(((scored["datasheet_status"] == "PASS") & (scored["ai_status"] == "ANOMALOUS")).sum()),
        report["full_dataset"]["latent_defect_analysis"]["dynamic_recall_on_latent_defects"] or 0.0,
    )
    logger.info("Next: run the dashboard -> streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
