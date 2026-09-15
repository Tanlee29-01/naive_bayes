"""Leakage-safe preprocessing utilities implemented without sklearn."""

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
        self.epsilon = 1e-9

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a non-empty 2-D array")
        self.mean_ = np.mean(X, axis=0)
        standard_deviation = np.std(X, axis=0)
        self.scale_ = standard_deviation
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has an unexpected number of features")
        return (X - self.mean_) / (self.scale_ + self.epsilon)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _check_is_fitted(self):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("CustomStandardScaler must be fitted first")


class CustomMissingValueImputer:
    """Fill missing values using statistics learned only from the fit data."""

    def __init__(self, strategy="mean"):
        if strategy not in {"mean", "median", "most_frequent"}:
            raise ValueError("strategy must be mean, median, or most_frequent")
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X):
        values = np.asarray(X, dtype=object)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("X must be a non-empty 2-D array")
        statistics = []
        for column in values.T:
            observed = column[~self._missing_mask(column)]
            if len(observed) == 0:
                statistics.append(0.0)
            elif self.strategy == "most_frequent":
                unique, counts = np.unique(observed, return_counts=True)
                statistics.append(unique[np.argmax(counts)])
            else:
                numeric = np.asarray(observed, dtype=float)
                statistics.append(np.mean(numeric) if self.strategy == "mean" else np.median(numeric))
        self.statistics_ = np.asarray(statistics, dtype=object)
        return self

    def transform(self, X):
        self._check_is_fitted()
        values = np.asarray(X, dtype=object).copy()
        if values.ndim != 2 or values.shape[1] != len(self.statistics_):
            raise ValueError("X has an unexpected number of features")
        for column_index, statistic in enumerate(self.statistics_):
            values[self._missing_mask(values[:, column_index]), column_index] = statistic
        try:
            return values.astype(float)
        except (TypeError, ValueError):
            return values

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @staticmethod
    def _missing_mask(values):
        return np.array([value is None or (isinstance(value, float) and np.isnan(value)) for value in values])

    def _check_is_fitted(self):
        if self.statistics_ is None:
            raise RuntimeError("CustomMissingValueImputer must be fitted first")


class CustomCategoricalEncoder:
    """Ordinal-encode categories learned on train; unknown categories become -1."""

    def __init__(self, unknown_value=-1):
        self.unknown_value = unknown_value
        self.categories_ = None

    def fit(self, X):
        values = np.asarray(X, dtype=object)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("X must be a non-empty 2-D array")
        self.categories_ = [list(dict.fromkeys(column.tolist())) for column in values.T]
        return self

    def transform(self, X):
        self._check_is_fitted()
        values = np.asarray(X, dtype=object)
        if values.ndim != 2 or values.shape[1] != len(self.categories_):
            raise ValueError("X has an unexpected number of features")
        encoded = np.full(values.shape, self.unknown_value, dtype=float)
        for column_index, categories in enumerate(self.categories_):
            lookup = {category: index for index, category in enumerate(categories)}
            encoded[:, column_index] = [lookup.get(value, self.unknown_value) for value in values[:, column_index]]
        return encoded

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _check_is_fitted(self):
        if self.categories_ is None:
            raise RuntimeError("CustomCategoricalEncoder must be fitted first")


class CustomIQROutlierRemover:
    """Learn IQR bounds on train and remove outliers only in fit_transform."""

    def __init__(self, multiplier=1.5):
        if multiplier <= 0:
            raise ValueError("multiplier must be positive")
        self.multiplier = multiplier
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("X must be a non-empty 2-D array")
        first_quartile, third_quartile = np.percentile(values, [25, 75], axis=0)
        spread = third_quartile - first_quartile
        self.lower_bounds_ = first_quartile - self.multiplier * spread
        self.upper_bounds_ = third_quartile + self.multiplier * spread
        return self

    def transform(self, X):
        self._check_is_fitted()
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.lower_bounds_):
            raise ValueError("X has an unexpected number of features")
        return values

    def fit_transform(self, X, y=None):
        self.fit(X)
        values = np.asarray(X, dtype=float)
        keep = np.all((values >= self.lower_bounds_) & (values <= self.upper_bounds_), axis=1)
        filtered = values[keep]
        if y is None:
            return filtered
        targets = np.asarray(y)
        if len(targets) != len(values):
            raise ValueError("X and y must have the same length")
        return filtered, targets[keep]

    def _check_is_fitted(self):
        if self.lower_bounds_ is None:
            raise RuntimeError("CustomIQROutlierRemover must be fitted first")


class CustomRandomOverSampler:
    """Randomly oversample minority classes in train data only."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if len(X) != len(y) or len(y) == 0:
            raise ValueError("X and y must have the same non-zero length")
        self.classes_, counts = np.unique(y, return_counts=True)
        self.target_count_ = int(counts.max())
        return self

    def transform(self, X, y):
        if not hasattr(self, "classes_"):
            raise RuntimeError("CustomRandomOverSampler must be fitted first")
        X = np.asarray(X)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        indices = np.concatenate([
            rng.choice(class_indices, self.target_count_, replace=True)
            if len(class_indices) < self.target_count_ else class_indices
            for class_indices in [np.flatnonzero(y == label) for label in self.classes_]
        ])
        rng.shuffle(indices)
        return X[indices], y[indices]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)
