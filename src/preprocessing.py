"""
Preprocessing: data validation, missing-value handling, duplicate detection,
timestamp/chronology handling and lot-wise grouping.

Input: the SIH26170 solution-aligned synthetic burn-in dataset (wide + long
format). This is a SYNTHETIC dataset built for this hackathon problem
statement - it is clearly not real industrial ESS/burn-in data.

We intentionally do NOT drop statistical outliers here: outliers are the
very components Module A/B are trying to find, so aggressive cleaning would
destroy the signal we care about. We only fix structural data-quality
issues (duplicates, missing/invalid rows, dtype problems).
"""

import logging

import numpy as np
import pandas as pd

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocessing")

WIDE_REQUIRED_COLS = [
    "component_id", "lot_id", "component_type", "parameter", "unit",
    "datasheet_min", "datasheet_max",
    "parameter_value_0h", "parameter_value_24h", "parameter_value_96h", "parameter_value_168h",
]
LONG_REQUIRED_COLS = [
    "component_id", "lot_id", "component_type", "parameter",
    "time_h", "parameter_value", "datasheet_min", "datasheet_max",
]


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw wide-format and long-format (time series) CSVs."""
    if not config.RAW_WIDE_CSV.exists() or not config.RAW_LONG_CSV.exists():
        raise FileNotFoundError(
            f"Raw data not found under {config.DATA_RAW_DIR}. "
            "Run src/data_generation.py first, or place the provided CSVs there."
        )
    wide = pd.read_csv(config.RAW_WIDE_CSV)
    long = pd.read_csv(config.RAW_LONG_CSV)
    logger.info("Loaded raw wide=%s long=%s", wide.shape, long.shape)
    return wide, long


def _validate(df: pd.DataFrame, required_cols: list[str], name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")
    n_dupe = df.duplicated().sum()
    if n_dupe:
        logger.warning("%s: found %d fully-duplicated rows (will be dropped)", name, n_dupe)


def clean_wide(wide: pd.DataFrame) -> pd.DataFrame:
    _validate(wide, WIDE_REQUIRED_COLS, "wide")
    df = wide.drop_duplicates().copy()

    # Duplicate component_id x parameter should not exist - flag and keep first
    key = ["component_id", "parameter"]
    dupe_mask = df.duplicated(subset=key, keep="first")
    if dupe_mask.any():
        logger.warning("wide: dropping %d duplicate component/parameter rows", dupe_mask.sum())
        df = df[~dupe_mask]

    value_cols = [f"parameter_value_{h}h" for h in config.TEST_HOURS]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Missing-value handling: a component missing an *early* (0h/24h) reading
    # cannot be scored by Module B early-warning logic -> drop with a warning.
    # A missing *late* reading (96h/168h) is imputed via forward-fill from the
    # component's own trajectory (conservative - does not fabricate a trend).
    early_cols = [f"parameter_value_{h}h" for h in config.EARLY_HOURS]
    n_before = len(df)
    df = df.dropna(subset=early_cols)
    if len(df) < n_before:
        logger.warning("wide: dropped %d rows missing early (0h/24h) readings", n_before - len(df))

    late_cols = ["parameter_value_96h", "parameter_value_168h"]
    df[late_cols] = df[late_cols].apply(lambda row: row.ffill(), axis=1)

    for c in ["datasheet_min", "datasheet_max"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.reset_index(drop=True)
    logger.info("clean_wide: %d -> %d rows after cleaning", len(wide), len(df))
    return df


def clean_long(long: pd.DataFrame) -> pd.DataFrame:
    _validate(long, LONG_REQUIRED_COLS, "long")
    df = long.drop_duplicates().copy()

    df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
    df["parameter_value"] = pd.to_numeric(df["parameter_value"], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["time_h", "parameter_value"])
    if len(df) < n_before:
        logger.warning("long: dropped %d rows with missing time/value", n_before - len(df))

    # Sort chronologically within each component/parameter (required for
    # correct slope / drift / EWMA computation in Module B).
    df = df.sort_values(["component_id", "parameter", "time_h"]).reset_index(drop=True)
    logger.info("clean_long: %d -> %d rows after cleaning", len(long), len(df))
    return df


def lot_groups(wide: pd.DataFrame):
    """Yield (lot_id, parameter) -> sub-dataframe, the peer-group unit used
    throughout Module A. We deliberately group by lot AND parameter because
    different parameters have different scales/units and different lots can
    have genuine manufacturing-process differences."""
    for (lot_id, parameter), g in wide.groupby(["lot_id", "parameter"]):
        yield (lot_id, parameter), g


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    wide, long = load_raw()
    wide_clean = clean_wide(wide)
    long_clean = clean_long(long)
    wide_clean.to_csv(config.DATA_PROCESSED_DIR / "wide_clean.csv", index=False)
    long_clean.to_csv(config.DATA_PROCESSED_DIR / "long_clean.csv", index=False)
    return wide_clean, long_clean


if __name__ == "__main__":
    run()
