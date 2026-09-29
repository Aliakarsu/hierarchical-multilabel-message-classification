"""Configuration A: independent multi-output classifier.

One Random Forest per label level (y2, y3, y4); each classifier sees only
the TF-IDF features and is unaware of the other levels. This ignores the
Type2 -> Type3 -> Type4 dependency but is simple and trivially
parallelisable.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from src import config
from src.models.base_model import BaseMultiLabelModel


class IndependentMultiLabelModel(BaseMultiLabelModel):
    name = "independent_rf"

    def __init__(self):
        base = RandomForestClassifier(**config.RF_PARAMS)
        self.model = MultiOutputClassifier(base, n_jobs=-1)

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model.predict(X))

    def predict_confidence(self, X) -> np.ndarray:
        confs = []
        for estimator in self.model.estimators_:
            proba = estimator.predict_proba(X)
            confs.append(proba.max(axis=1))
        return np.column_stack(confs)
