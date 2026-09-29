# Customer Message Classification — Production-Oriented Refactor

A refactored, modular ML pipeline for hierarchical multi-label classification
of customer support messages (labels `y2`, `y3`, `y4` = Type 2/3/4), built on
top of the provided starter prototype.

## Project structure

```
customer-message-classifier/
├── data/                     # AppGallery.csv, Purchasing.csv, new_messages.csv
├── src/
│   ├── config.py             # all paths, columns, hyperparameters
│   ├── data_loader.py        # loading + input validation
│   ├── preprocessing.py      # fixed (non-learned) cleaning, dedup, labels
│   ├── features.py           # TF-IDF: fit on train only, transform elsewhere
│   ├── evaluation.py         # per-label / exact-match / hierarchical metrics
│   ├── train.py              # training pipeline entry point
│   ├── predict.py            # batch inference entry point
│   └── models/
│       ├── base_model.py     # common interface (predict + confidence)
│       ├── independent_model.py  # Config A: MultiOutputClassifier(RF)
│       └── chained_model.py      # Config B: hierarchical chain y2→y3→y4
├── artifacts/                # model.joblib, vectorizer.joblib, model_manifest.json
├── outputs/                  # metrics, reports, confusion matrices, predictions
└── requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

(Python 3.10+; developed and tested with scikit-learn 1.8, pandas 3.0.)

## Run the training pipeline

From the project root:

```bash
python -m src.train
```

This loads and validates the raw CSVs, applies fixed preprocessing,
splits the data (stratified on `y2`) **before** fitting TF-IDF on the
training split only, trains two configurations (independent multi-output
vs hierarchical chain), evaluates both with per-label, exact-match and
hierarchical prefix accuracy, and saves the better configuration to
`artifacts/` together with a `model_manifest.json`.

## Run the batch inference pipeline

```bash
python -m src.predict --input data/new_messages.csv --output outputs/predictions.csv
```

Loads the saved artefacts (never re-fits anything), applies the same
preprocessing code used in training, and writes a prediction file with:
message id, text reference, predicted y2/y3/y4, per-level confidence,
`needs_human_review` flag (any confidence < 0.50), model name/version
and a UTC timestamp.

## Outputs produced

- `outputs/metrics.json` — comparison of both configurations
- `outputs/classification_report_<model>_<label>.csv`
- `outputs/confusion_matrix_<model>_y2.png`
- `outputs/error_analysis_<model>.csv`
- `outputs/predictions.csv` — batch inference output
- `outputs/train_log.txt` — full training log (evidence of execution)
