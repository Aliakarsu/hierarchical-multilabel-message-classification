"""Central configuration for the customer message classification system.

All paths, column names, and hyperparameters live here so that no other
module contains hardcoded values (an issue identified in the starter code,
where file paths were embedded inside preprocess.py).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to project root, resolved at import time)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RAW_DATA_FILES = [
    DATA_DIR / "AppGallery.csv",
    DATA_DIR / "Purchasing.csv",
]

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
TICKET_ID_COL = "Ticket id"
TICKET_SUMMARY_COL = "Ticket Summary"
INTERACTION_CONTENT_COL = "Interaction content"

# Original label columns in the raw CSVs -> internal names
LABEL_RENAME_MAP = {
    "Type 1": "y1",
    "Type 2": "y2",
    "Type 3": "y3",
    "Type 4": "y4",
}

# The multi-label target: three related label levels predicted together.
LABEL_COLUMNS = ["y2", "y3", "y4"]

# Placeholder used when a label level was left empty by annotators.
UNLABELLED_TOKEN = "Unknown"

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
RANDOM_SEED = 0
TEST_SIZE = 0.25

TFIDF_PARAMS = {
    "max_features": 2000,
    "min_df": 2,
    "max_df": 0.90,
    "ngram_range": (1, 2),
}

RF_PARAMS = {
    "n_estimators": 300,
    "random_state": RANDOM_SEED,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
}

# Batch inference
MODEL_VERSION = "v1.0"
REVIEW_THRESHOLD = 0.50  # below this confidence a prediction is flagged
