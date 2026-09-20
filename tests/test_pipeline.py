"""
Basic tests for the anomaly-detection pipeline. Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import config, preprocessing, dynamic_detection, temporal_detection, scoring, evaluation


@pytest.fixture(scope="module")
def pipeline_data():
    wide, long_df = preprocessing.run()
    pop = dynamic_detection.detect(wide)
    temp = temporal_detection.detect(long_df)
    scored = scoring.combine(pop, temp)
    return wide, long_df, scored


def test_raw_data_loads():
    wide, long_df = preprocessing.load_raw()
    assert len(wide) > 0
    assert len(long_df) > 0
    assert "component_id" in wide.columns
    assert "time_h" in long_df.columns


def test_no_duplicate_components(pipeline_data):
    wide, _, _ = pipeline_data
    assert wide["component_id"].duplicated().sum() == 0


def test_chronological_sort(pipeline_data):
    _, long_df, _ = pipeline_data
    for (_, _), g in long_df.groupby(["component_id", "parameter"]):
        assert list(g["time_h"]) == sorted(g["time_h"])


def test_population_scores_bounded(pipeline_data):
    _, _, scored = pipeline_data
    assert scored["population_anomaly_score"].between(0, 1).all()


def test_temporal_scores_bounded(pipeline_data):
    _, _, scored = pipeline_data
    assert scored["temporal_anomaly_score"].between(0, 1).all()


def test_final_score_bounded(pipeline_data):
    _, _, scored = pipeline_data
    assert scored["final_anomaly_score"].between(0, 1).all()


def test_risk_levels_valid(pipeline_data):
    _, _, scored = pipeline_data
    assert set(scored["risk_level"].unique()) <= set(config.RISK_THRESHOLDS.keys())


def test_spec_failure_implies_critical_or_high_score(pipeline_data):
    _, _, scored = pipeline_data
    failures = scored[scored["spec_violation"] == 1]
    assert (failures["final_anomaly_score"] >= config.RISK_THRESHOLDS["CRITICAL"][0]).all()


def test_latent_anomaly_case_means_datasheet_pass(pipeline_data):
    _, _, scored = pipeline_data
    latent = scored[scored["case_classification"] == "LATENT_ANOMALY"]
    assert (latent["datasheet_status"] == "PASS").all()


def test_static_screening_never_catches_latent_defects(pipeline_data):
    """Structural sanity check: static screening's true positive count on
    latent (by-definition within-spec) defects must be exactly zero."""
    _, _, scored = pipeline_data
    report = evaluation.evaluate(scored, split=None)
    assert report["latent_defect_analysis"]["n_caught_by_static_screening"] == 0


def test_dynamic_screening_catches_some_latent_defects(pipeline_data):
    _, _, scored = pipeline_data
    report = evaluation.evaluate(scored, split=None)
    assert report["latent_defect_analysis"]["n_caught_by_dynamic_ai_screening"] > 0


def test_case_classification_covers_all_rows(pipeline_data):
    _, _, scored = pipeline_data
    valid = {"NORMAL", "LATENT_ANOMALY", "DEGRADATION_RISK", "SPECIFICATION_FAILURE", "HIGH_RISK_COMPONENT"}
    assert set(scored["case_classification"].unique()) <= valid
    assert scored["case_classification"].notna().all()
