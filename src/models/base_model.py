"""Abstract interface every multi-label model in this project implements.

Compared with the starter's BaseModel, the interface is explicit about
multi-output prediction and adds confidence scores, which the batch
inference pipeline and the human-review flag depend on.
"""
from abc import ABC, abstractmethod

import numpy as np


class BaseMultiLabelModel(ABC):
    """Predicts all label columns (y2, y3, y4) for each input row."""

    name: str = "base"

    @abstractmethod
    def fit(self, X, Y) -> "BaseMultiLabelModel":
        """Train on feature matrix X and label frame Y (columns = label levels)."""

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return an (n_samples, n_labels) array of predicted labels."""

    @abstractmethod
    def predict_confidence(self, X) -> np.ndarray:
        """Return an (n_samples, n_labels) array with the probability of the
        predicted class for each label level."""
