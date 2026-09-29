"""Batch inference pipeline.

Run from the project root:

    python -m src.predict --input data/new_messages.csv --output outputs/predictions.csv

Training-serving consistency is guaranteed by construction:
- the SAME fixed preprocessing code (src.preprocessing) is imported and
  applied, not re-implemented;
- the TF-IDF vectorizer is LOADED from artifacts and used with
  ``transform`` only - it is never re-fitted on inference data;
- the manifest records which model/version produced each output file.
"""
import argparse
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd

from src import config, features, preprocessing

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("predict")

REQUIRED_ARTIFACTS = [
    config.ARTIFACTS_DIR / "model.joblib",
    config.ARTIFACTS_DIR / "vectorizer.joblib",
    config.ARTIFACTS_DIR / "model_manifest.json",
]


def validate_artifacts_exist() -> None:
    missing = [str(p) for p in REQUIRED_ARTIFACTS if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required artefacts missing (run `python -m src.train` first):\n- "
            + "\n- ".join(missing))
    logger.info("All required artefacts present.")


def validate_inference_input(df: pd.DataFrame) -> None:
    problems = []
    for col in [config.TICKET_SUMMARY_COL, config.INTERACTION_CONTENT_COL]:
        if col not in df.columns:
            problems.append(f"Missing required column: '{col}'")
    if not problems:
        empty = df[config.INTERACTION_CONTENT_COL].fillna("").astype(str).str.strip().eq("")
        if empty.any():
            problems.append(f"{int(empty.sum())} rows have empty interaction content "
                            f"(row indices: {list(df.index[empty])[:10]})")
    if problems:
        raise ValueError("Inference input validation failed:\n- " + "\n- ".join(problems))
    logger.info("Inference input validation passed (%d rows).", len(df))


def validate_prediction_output(input_df: pd.DataFrame, output_df: pd.DataFrame) -> None:
    problems = []
    if len(input_df) != len(output_df):
        problems.append("Output row count does not match input row count.")
    pred_cols = [f"predicted_{c}" for c in config.LABEL_COLUMNS]
    for col in pred_cols + ["model_name", "model_version", "prediction_timestamp"]:
        if col not in output_df.columns:
            problems.append(f"Missing output column: {col}")
    for col in pred_cols:
        if col in output_df.columns and output_df[col].isna().any():
            problems.append(f"Missing predictions in column: {col}")
    if problems:
        raise ValueError("Prediction output validation failed:\n- " + "\n- ".join(problems))
    logger.info("Prediction output validation passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference on new messages.")
    parser.add_argument("--input", required=True, help="CSV with new messages")
    parser.add_argument("--output", required=True, help="Where to write predictions CSV")
    args = parser.parse_args()

    validate_artifacts_exist()
    model = joblib.load(config.ARTIFACTS_DIR / "model.joblib")
    vectorizer = joblib.load(config.ARTIFACTS_DIR / "vectorizer.joblib")
    import json
    with open(config.ARTIFACTS_DIR / "model_manifest.json") as f:
        manifest = json.load(f)

    new_df = pd.read_csv(args.input, skipinitialspace=True)
    new_df.columns = [c.strip() for c in new_df.columns]
    validate_inference_input(new_df)

    # Same fixed preprocessing as training (imported, not re-implemented).
    prepared = preprocessing.build_model_input(new_df)
    X_new = features.transform(vectorizer, prepared["clean_text"])

    preds = model.predict(X_new)
    confs = model.predict_confidence(X_new)

    output_df = pd.DataFrame()
    if "message_id" in new_df.columns:
        output_df["message_id"] = new_df["message_id"]
    else:
        output_df["row_id"] = range(len(new_df))
    output_df["text_reference"] = (
        new_df[config.TICKET_SUMMARY_COL].fillna("").astype(str).str.slice(0, 80))
    for j, col in enumerate(config.LABEL_COLUMNS):
        output_df[f"predicted_{col}"] = preds[:, j]
        output_df[f"confidence_{col}"] = confs[:, j].round(4)
    output_df["needs_human_review"] = (confs < config.REVIEW_THRESHOLD).any(axis=1)
    output_df["model_name"] = manifest["model_name"]
    output_df["model_version"] = manifest["model_version"]
    output_df["prediction_timestamp"] = datetime.now(timezone.utc).isoformat()

    validate_prediction_output(new_df, output_df)
    output_df.to_csv(args.output, index=False)
    logger.info("Wrote %d predictions to %s "
                "(%d flagged for human review).",
                len(output_df), args.output,
                int(output_df["needs_human_review"].sum()))


if __name__ == "__main__":
    main()
