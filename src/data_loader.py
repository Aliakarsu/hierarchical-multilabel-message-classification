"""Data loading and validation.

Responsibilities:
- read the raw CSV files listed in config.RAW_DATA_FILES
- normalise column names (Type 1..4 -> y1..y4, strip whitespace)
- drop accidental unnamed columns produced by trailing commas in the CSVs
- validate the loaded frame before it enters the pipeline
"""
import logging

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    """Load and concatenate all raw CSV files defined in the config."""
    frames = []
    for path in config.RAW_DATA_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Expected data file not found: {path}")
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns=config.LABEL_RENAME_MAP)
        # AppGallery.csv contains trailing commas -> 'Unnamed: 11/12' ghost columns
        ghost_cols = [c for c in df.columns if c.startswith("Unnamed")]
        if ghost_cols:
            logger.warning("Dropping ghost columns %s from %s", ghost_cols, path.name)
            df = df.drop(columns=ghost_cols)
        df["source_file"] = path.name
        frames.append(df)
        logger.info("Loaded %s: %d rows", path.name, len(df))
    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined dataset: %d rows", len(combined))
    return combined


def validate_input_data(df: pd.DataFrame) -> None:
    """Fail fast if the dataset violates basic assumptions.

    Reports offending Ticket ids so problems can be traced back to rows,
    not just detected.
    """
    problems = []

    required = [
        config.TICKET_ID_COL,
        config.TICKET_SUMMARY_COL,
        config.INTERACTION_CONTENT_COL,
        "y1", "y2", "y3", "y4",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        problems.append(f"Missing required columns: {missing_cols}")

    if not missing_cols:
        empty_text = df[
            df[config.INTERACTION_CONTENT_COL].fillna("").astype(str).str.strip().eq("")
        ]
        if len(empty_text) > 0:
            ids = empty_text[config.TICKET_ID_COL].tolist()[:10]
            problems.append(
                f"{len(empty_text)} rows with empty interaction content "
                f"(first ticket ids: {ids})"
            )

        missing_y2 = df[df["y2"].isna() | df["y2"].astype(str).str.strip().eq("")]
        if len(missing_y2) > 0:
            ids = missing_y2[config.TICKET_ID_COL].tolist()[:10]
            problems.append(
                f"{len(missing_y2)} rows missing the primary label y2 "
                f"(first ticket ids: {ids})"
            )

    if problems:
        raise ValueError("Input data validation failed:\n- " + "\n- ".join(problems))
    logger.info("Input data validation passed.")


def load_and_validate() -> pd.DataFrame:
    """Convenience wrapper: load raw data, drop rows unusable for training,
    then validate the result."""
    df = load_raw_data()

    # Rows without the primary label y2 cannot be used for supervised training.
    before = len(df)
    df = df[df["y2"].notna() & df["y2"].astype(str).str.strip().ne("")]
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows without a y2 label.", dropped)

    # Rows with completely empty message text carry no signal.
    before = len(df)
    df = df[df[config.INTERACTION_CONTENT_COL].fillna("").astype(str).str.strip().ne("")]
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with empty interaction content.", dropped)

    validate_input_data(df)
    return df.reset_index(drop=True)
