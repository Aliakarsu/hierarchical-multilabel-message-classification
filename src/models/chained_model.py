"""Configuration B: chained (hierarchical) multi-label classifier.

The label levels form a hierarchy (Type2 -> Type3 -> Type4): the meaning
of y3 depends on y2, and y4 on both. This model makes that dependency
explicit:

    y2 = f2(text)
    y3 = f3(text, y2)
    y4 = f4(text, y2, y3)

During training each stage sees the *true* upstream labels; during
prediction it sees the *predicted* upstream labels (so upstream mistakes
propagate, exactly as the hierarchical accuracy metric assumes).

sklearn's built-in ClassifierChain targets binary indicator matrices, so
a small custom chain is implemented here for the multi-class case.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.models.base_model import BaseMultiLabelModel


class ChainedMultiLabelModel(BaseMultiLabelModel):
    name = "chained_rf"

    def __init__(self):
        self.stages = []      # one RandomForest per label level
        self.encoders = []    # OneHotEncoder for upstream labels per stage
        self.label_columns = list(config.LABEL_COLUMNS)

    @staticmethod
    def _augment(X, upstream_labels, encoder):
        """Append one-hot encoded upstream labels to the feature matrix."""
        if upstream_labels is None:
            return X
        encoded = encoder.transform(upstream_labels)
        return sparse.hstack([X, encoded]).tocsr()

    def fit(self, X, Y):
        Y = pd.DataFrame(Y, columns=self.label_columns)
        upstream = None
        for i, col in enumerate(self.label_columns):
            if i == 0:
                encoder = None
                X_stage = X
            else:
                encoder = OneHotEncoder(handle_unknown="ignore")
                encoder.fit(upstream)
                X_stage = self._augment(X, upstream, encoder)
            clf = RandomForestClassifier(**config.RF_PARAMS)
            clf.fit(X_stage, Y[col])
            self.stages.append(clf)
            self.encoders.append(encoder)
            # Teacher forcing: next stage trains on TRUE upstream labels.
            upstream = Y[self.label_columns[: i + 1]].to_numpy()
        return self

    def _predict_with_conf(self, X):
        n = X.shape[0]
        preds = np.empty((n, len(self.stages)), dtype=object)
        confs = np.zeros((n, len(self.stages)))
        upstream = None
        for i, clf in enumerate(self.stages):
            X_stage = X if i == 0 else self._augment(X, upstream, self.encoders[i])
            proba = clf.predict_proba(X_stage)
            idx = proba.argmax(axis=1)
            preds[:, i] = clf.classes_[idx]
            confs[:, i] = proba[np.arange(n), idx]
            # At inference the chain consumes its own PREDICTED labels.
            upstream = preds[:, : i + 1]
        return preds, confs

    def predict(self, X) -> np.ndarray:
        preds, _ = self._predict_with_conf(X)
        return preds

    def predict_confidence(self, X) -> np.ndarray:
        _, confs = self._predict_with_conf(X)
        return confs
