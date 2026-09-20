"""
Synthetic dataset for this project.

This prototype uses the SIH26170 "solution-aligned" synthetic burn-in
dataset (provided for this problem statement) as its data source, rather
than generating a fresh one from scratch. It already satisfies every
requirement in the problem brief:

  - Multiple lots (12) and component types (FPGA, Microprocessor,
    Power MOSFET, RF MMIC), each with a datasheet [min, max] window.
  - 3 measured parameters (Leakage Current, Supply Current,
    Threshold Voltage) sampled at 0h / 24h / 96h / 168h burn-in stages.
  - Explicitly labeled failure_mode per component:
        healthy, gradual_drift, latent_defect, sudden_jump,
        lot_outlier, accelerating_drift
    covering every category requested in the brief (normal, static-limit
    violations, latent defects, drift defects, sudden failures, noisy/
    lot-outlier components).
  - Ground-truth labels for evaluation without data leakage:
        anomaly_label_early, future_failure_label_168h
  - A lot-wise train/validation/test split (`split` column) so evaluation
    in src/evaluation.py does not leak lot-level statistics across splits.

This file exists so the documented project structure
(data_generation.py -> preprocessing.py -> ...) matches the pipeline, and
so a synthetic dataset could be regenerated in the same shape if the
provided CSVs were ever unavailable. THIS DATASET IS SYNTHETIC - it must
never be treated as real industrial ESS/burn-in measurements.
"""

import logging
import shutil

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data_generation")


def ensure_raw_data_present():
    """Verify the provided synthetic dataset is in place under data/raw/.
    (The dataset was supplied pre-generated for this problem statement;
    see the module docstring above for why we use it as-is.)"""
    missing = [p for p in (config.RAW_WIDE_CSV, config.RAW_LONG_CSV) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Expected synthetic dataset file(s) not found: {missing}. "
            f"Place the SIH26170_solution_aligned_*.csv files under {config.DATA_RAW_DIR}."
        )
    logger.info("Synthetic dataset present: %s, %s", config.RAW_WIDE_CSV.name, config.RAW_LONG_CSV.name)


def snapshot_to_synthetic_dir():
    """Copy a labeled reference snapshot into data/synthetic/ for
    provenance / reproducibility."""
    ensure_raw_data_present()
    for src in (config.RAW_WIDE_CSV, config.RAW_LONG_CSV):
        dst = config.DATA_SYNTHETIC_DIR / src.name
        shutil.copy(src, dst)
    logger.info("Snapshotted raw synthetic dataset to %s", config.DATA_SYNTHETIC_DIR)


if __name__ == "__main__":
    snapshot_to_synthetic_dir()
