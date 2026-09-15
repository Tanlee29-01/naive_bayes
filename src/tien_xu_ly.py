"""Data splitting and standardization implemented without sklearn."""

import numpy as np


def custom_train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True):
    """Split arrays into train and test partitions."""
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be 2-D, y must be 1-D, and lengths must match")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    indices = np.arange(len(X))
    if shuffle:
        np.random.default_rng(random_state).shuffle(indices)
    test_count = max(1, int(np.ceil(len(X) * test_size)))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


class CustomStandardScaler:
    """Standardize columns using statistics learned from the training set."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a non-empty 2-D array")
        self.mean_ = np.mean(X, axis=0)
        standard_deviation = np.std(X, axis=0)
        self.scale_ = np.where(standard_deviation == 0, 1.0, standard_deviation)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has an unexpected number of features")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _check_is_fitted(self):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("CustomStandardScaler must be fitted first")
