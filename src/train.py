"""Training pipeline entry point.

Run from the project root:

    python -m src.train

Steps: load -> validate -> preprocess -> split -> fit TF-IDF (train only)
-> train two model configurations -> evaluate both -> save artefacts of
the selected model.
"""
import json
import logging
import random
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src import config, data_loader, evaluation, features, preprocessing
from src.models.chained_model import ChainedMultiLabelModel
from src.models.independent_model import IndependentMultiLabelModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.OUTPUTS_DIR / "train_log.txt", mode="w"),
    ],
)
logger = logging.getLogger("train")


def main() -> None:
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # 1. Load + validate ----------------------------------------------------
    df = data_loader.load_and_validate()

    # 2. Fixed preprocessing ------------------------------------------------
    df = preprocessing.preprocess(df)
    logger.info("After preprocessing: %d rows", len(df))

    # 3. Split BEFORE any learned preprocessing ------------------------------
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=df["y2"],  # y2 has 3 well-populated classes; y3/y4 are too
                            # sparse to stratify on (many singleton classes)
    )
    logger.info("Split: %d train / %d test rows", len(train_df), len(test_df))

    # 4. Learned preprocessing: TF-IDF fitted on TRAIN ONLY ------------------
    vectorizer = features.fit_vectorizer(train_df["clean_text"])
    X_train = features.transform(vectorizer, train_df["clean_text"])
    X_test = features.transform(vectorizer, test_df["clean_text"])

    Y_train = train_df[config.LABEL_COLUMNS]
    Y_test = test_df[config.LABEL_COLUMNS]

    # 5. Train and evaluate both configurations ------------------------------
    all_metrics = []
    models = {}
    for model in [IndependentMultiLabelModel(), ChainedMultiLabelModel()]:
        logger.info("Training configuration: %s", model.name)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)

        metrics = evaluation.evaluate_model(model.name, Y_test, y_pred)
        logger.info("Metrics for %s: %s", model.name, metrics)
        all_metrics.append(metrics)
        models[model.name] = model

        evaluation.save_classification_reports(model.name, Y_test, y_pred)
        evaluation.save_confusion_matrix_plot(
            model.name, Y_test["y2"].to_numpy(), y_pred[:, 0], "y2")
        evaluation.save_error_analysis(model.name, test_df, y_pred)

    evaluation.save_metrics(all_metrics)

    # 6. Select and persist the serving model --------------------------------
    # Selection criterion: hierarchical prefix accuracy, because it matches
    # how the support-routing task consumes the labels (a wrong y2 makes the
    # rest useless).
    best = max(all_metrics, key=lambda m: m["hierarchical_prefix_accuracy"])
    best_model = models[best["model"]]
    logger.info("Selected serving model: %s", best["model"])

    joblib.dump(best_model, config.ARTIFACTS_DIR / "model.joblib")
    joblib.dump(vectorizer, config.ARTIFACTS_DIR / "vectorizer.joblib")
    manifest = {
        "model_name": best["model"],
        "model_version": config.MODEL_VERSION,
        "label_columns": config.LABEL_COLUMNS,
        "tfidf_params": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in config.TFIDF_PARAMS.items()},
        "rf_params": {k: v for k, v in config.RF_PARAMS.items()},
        "random_seed": config.RANDOM_SEED,
        "test_size": config.TEST_SIZE,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(train_df),
        "test_rows": len(test_df),
        "metrics": best,
    }
    with open(config.ARTIFACTS_DIR / "model_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Artefacts saved to %s", config.ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
