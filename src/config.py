"""
Central configuration for the Dynamic Anomaly Detection prototype.

IMPORTANT: The scoring weights and risk thresholds below are PROTOTYPE
choices for a hackathon-quality demonstration. They are NOT derived from
a scientific / reliability-engineering study and should be re-validated
with real ESS/burn-in data and domain experts before any production use.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATA_SYNTHETIC_DIR = ROOT_DIR / "data" / "synthetic"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_WIDE_CSV = DATA_RAW_DIR / "SIH26170_solution_aligned_component_wide.csv"
RAW_LONG_CSV = DATA_RAW_DIR / "SIH26170_solution_aligned_timeseries_long.csv"

PROCESSED_WIDE_CSV = DATA_PROCESSED_DIR / "components_scored_wide.csv"
PROCESSED_LONG_CSV = DATA_PROCESSED_DIR / "components_timeseries_processed.csv"

MODEL_PATH = MODELS_DIR / "risk_classifier.joblib"
SHAP_EXPLAINER_PATH = MODELS_DIR / "shap_background.joblib"
EVAL_REPORT_PATH = REPORTS_DIR / "evaluation_report.json"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Burn-in test timeline (hours)
# ---------------------------------------------------------------------------
TEST_HOURS = [0, 24, 96, 168]
EARLY_HOURS = [0, 24]  # measurements available "early" for prediction, no leakage

# ---------------------------------------------------------------------------
# Module A - Population / Dynamic Anomaly Detection thresholds
# ---------------------------------------------------------------------------
ROBUST_Z_WARN = 2.5        # |robust z| above this => population deviation flag
ROBUST_Z_CRITICAL = 4.0
IQR_MULTIPLIER = 1.5
PERCENTILE_LOW = 5
PERCENTILE_HIGH = 95
ISOLATION_FOREST_CONTAMINATION = 0.12  # expected anomaly fraction per lot (prototype guess)

# ---------------------------------------------------------------------------
# Module B - Temporal Anomaly Detection
# ---------------------------------------------------------------------------
EWMA_SPAN = 2
SUDDEN_JUMP_Z = 3.0          # z-score of a single-step change considered a "jump"
ISOLATION_FOREST_CONTAMINATION_TEMPORAL = 0.15

# ---------------------------------------------------------------------------
# Combined scoring weights (must sum to 1.0). Configurable / adjustable.
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = {
    "population": 0.40,
    "temporal": 0.40,
    "specification": 0.20,
}

# Risk level thresholds on the final 0-1 combined anomaly score
RISK_THRESHOLDS = {
    "NORMAL": (0.00, 0.30),
    "WARNING": (0.30, 0.60),
    "SUSPICIOUS": (0.60, 0.80),
    "CRITICAL": (0.80, 1.01),  # 1.01 so a perfect 1.0 score is inclusive
}

QA_RECOMMENDATION = {
    "NORMAL": "PASS",
    "WARNING": "QA REVIEW",
    "SUSPICIOUS": "QA REVIEW",
    "CRITICAL": "HOLD / REJECT",
}

for _name, _dir in [
    (DATA_PROCESSED_DIR, DATA_PROCESSED_DIR),
    (DATA_SYNTHETIC_DIR, DATA_SYNTHETIC_DIR),
    (MODELS_DIR, MODELS_DIR),
    (REPORTS_DIR, REPORTS_DIR),
    (FIGURES_DIR, FIGURES_DIR),
]:
    _dir.mkdir(parents=True, exist_ok=True)
