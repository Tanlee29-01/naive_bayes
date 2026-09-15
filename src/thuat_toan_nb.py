"""A Gaussian Naive Bayes classifier implemented with NumPy."""

import numpy as np


class CustomGaussianNB:
    def __init__(self, var_smoothing=1e-9):
        if var_smoothing < 0:
            raise ValueError("var_smoothing must be non-negative")
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None
        self.var_ = None
        self.epsilon_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(X) == 0:
            raise ValueError("X and y have incompatible or empty shapes")
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_prior_ = counts.astype(float) / len(y)
        self.theta_ = np.array([X[y == label].mean(axis=0) for label in self.classes_])
        self.var_ = np.array([X[y == label].var(axis=0) for label in self.classes_])
        self.epsilon_ = self.var_smoothing * max(float(np.var(X, axis=0).max()), 1e-12)
        self.var_ = np.maximum(self.var_ + self.epsilon_, np.finfo(float).eps)
        return self

    def _joint_log_likelihood(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.theta_.shape[1]:
            raise ValueError("X has an unexpected number of features")
        log_prior = np.log(self.class_prior_)
        log_density = -0.5 * (
            np.log(2.0 * np.pi * self.var_)[None, :, :]
            + ((X[:, None, :] - self.theta_[None, :, :]) ** 2) / self.var_[None, :, :]
        )
        return log_prior[None, :] + log_density.sum(axis=2)

    def predict_log_proba(self, X):
        joint = self._joint_log_likelihood(X)
        normalizer = np.max(joint, axis=1, keepdims=True)
        log_sum_exp = normalizer + np.log(np.exp(joint - normalizer).sum(axis=1, keepdims=True))
        return joint - log_sum_exp

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def _check_is_fitted(self):
        if self.classes_ is None:
            raise RuntimeError("CustomGaussianNB must be fitted first")
