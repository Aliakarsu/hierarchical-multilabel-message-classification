"""Fixed (non-learned) preprocessing.

Everything in this module is a deterministic rule: it does not learn any
state from the data, so the *same functions* can safely be applied at
training time and at prediction time. This is one half of the
training-serving consistency guarantee (the other half is reusing the
saved TF-IDF vectorizer, see features.py / predict.py).
"""
import html
import logging
import re

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# Email boilerplate / noise patterns inherited from the domain
# (reply prefixes in several languages, anonymised placeholders, dates).
_NOISE_PATTERNS = [
    r"(sv|wg|ynt|fw(d)?|re|r)\s*:",                      # reply/forward prefixes
    r"xxxxx@xxxx\.com",                                   # anonymised emails
    r"\*{5}\([a-z]+\)",                                   # *****(PERSON) etc.
    r"dear (customer|user)|dear\b",
    r"(hello|hallo|hi there|hi )",
    r"good morning",
    r"thank you( very much)?( for [a-z ]+)?",
    r"sent from my huawei (cell )?phone",
    r"original message",
    r"customer support team",
    r"(january|february|march|april|may|june|july|august|september|october|november|december)",
    r"\b(mon|tues|wednes|thurs|fri|satur|sun)day\b",
    r"\d{2}[:.]\d{2}",
]


def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate interactions.

    The raw Purchasing.csv contains rows duplicated verbatim (same ticket,
    same text). Duplicates inflate the dataset and, worse, can leak the
    same message into both the training and the test split.
    """
    before = len(df)
    df = df.drop_duplicates(
        subset=[config.TICKET_ID_COL, config.INTERACTION_CONTENT_COL, "y2", "y3", "y4"]
    )
    logger.info("Row-level de-duplication removed %d rows.", before - len(df))
    return df.reset_index(drop=True)


def clean_text(text: str) -> str:
    """Deterministic text cleaning applied to a single string."""
    text = html.unescape(str(text))          # '&amp;' -> '&'
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)     # URLs
    for pattern in _NOISE_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text) # punctuation / non-latin remnants
    text = re.sub(r"\d+", " ", text)          # bare numbers carry no label signal
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Combine the two text columns into one cleaned model input column."""
    df = df.copy()
    combined = (
        df[config.TICKET_SUMMARY_COL].fillna("").astype(str)
        + " "
        + df[config.INTERACTION_CONTENT_COL].fillna("").astype(str)
    )
    df["clean_text"] = combined.apply(clean_text)
    return df


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the three label columns.

    y3/y4 are frequently unlabelled in the raw data (35 and 43 empty values
    respectively). Dropping those rows would discard ~20% of an already
    small dataset, so empty values become an explicit 'Unknown' class:
    'not labelled' is itself information the support team can act on.
    """
    df = df.copy()
    df["y1"] = df["y1"].apply(lambda v: html.unescape(str(v)).strip())
    for col in config.LABEL_COLUMNS:
        df[col] = (
            df[col]
            .fillna(config.UNLABELLED_TOKEN)
            .astype(str)
            .apply(lambda v: html.unescape(v).strip())
            .replace("", config.UNLABELLED_TOKEN)
            .replace("nan", config.UNLABELLED_TOKEN)
        )
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full fixed preprocessing chain used by both training and inference
    (inference skips label preparation when labels are absent)."""
    df = deduplicate_rows(df)
    df = build_model_input(df)
    if "y2" in df.columns:
        df = prepare_labels(df)
    return df
