"""Learned feature extraction (TF-IDF).

The starter prototype called ``TfidfVectorizer.fit_transform`` on the
*entire* dataset before splitting, which leaks test-set vocabulary and
document frequencies into the features used to train the model. Here the
vectorizer is fitted on the training split only and then *reused* (via
``transform``) for the test split and, after being saved to disk, for
batch inference.
"""
import logging

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config

logger = logging.getLogger(__name__)


def fit_vectorizer(train_texts) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(**config.TFIDF_PARAMS)
    vectorizer.fit(train_texts)
    logger.info("TF-IDF fitted on training data only: %d terms.",
                len(vectorizer.vocabulary_))
    return vectorizer


def transform(vectorizer: TfidfVectorizer, texts) -> sparse.csr_matrix:
    return vectorizer.transform(texts)
