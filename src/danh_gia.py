"""Cross-validation, grid search, and classification metrics without sklearn."""

import numpy as np


class CustomStratifiedKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if self.n_splits > len(y):
            raise ValueError("n_splits cannot exceed the number of samples")
        _, counts = np.unique(y, return_counts=True)
        if np.any(counts < self.n_splits):
            raise ValueError(
                f"Each class must contain at least n_splits={self.n_splits} samples; "
                f"smallest class has {int(counts.min())}"
            )
        rng = np.random.default_rng(self.random_state)
        folds = [[] for _ in range(self.n_splits)]
        for label in np.unique(y):
            indices = np.flatnonzero(y == label)
            if self.shuffle:
                rng.shuffle(indices)
            for position, index in enumerate(indices):
                folds[position % self.n_splits].append(index)
        for fold in folds:
            validation_indices = np.array(sorted(fold), dtype=int)
            training_indices = np.setdiff1d(np.arange(len(y)), validation_indices)
            yield training_indices, validation_indices


class CustomEvaluator:
    @staticmethod
    def confusion_matrix(y_true, y_pred, labels=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        CustomEvaluator._validate_lengths(y_true, y_pred)
        if len(y_true) == 0:
            raise ValueError("y_true and y_pred must not be empty")
        labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else np.asarray(labels)
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        lookup = {label: index for index, label in enumerate(labels)}
        for actual, predicted in zip(y_true, y_pred):
            matrix[lookup[actual], lookup[predicted]] += 1
        return matrix, labels

    @staticmethod
    def metrics(y_true, y_pred, positive_label=1):
        matrix, labels = CustomEvaluator.confusion_matrix(y_true, y_pred)
        if positive_label not in labels:
            raise ValueError("positive_label is absent from y_true and y_pred")
        positive_index = int(np.flatnonzero(labels == positive_label)[0])
        true_positive = matrix[positive_index, positive_index]
        false_positive = matrix[:, positive_index].sum() - true_positive
        false_negative = matrix[positive_index, :].sum() - true_positive
        accuracy = float(np.trace(matrix) / matrix.sum())
        precision = float(true_positive / (true_positive + false_positive)) if true_positive + false_positive else 0.0
        recall = float(true_positive / (true_positive + false_negative)) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

    @staticmethod
    def roc_curve(y_true, positive_scores, positive_label=1):
        y_true = np.asarray(y_true)
        scores = np.asarray(positive_scores, dtype=float)
        if len(y_true) != len(scores):
            raise ValueError("y_true and positive_scores must have the same length")
        if len(y_true) == 0:
            raise ValueError("y_true and positive_scores must not be empty")
        positives = y_true == positive_label
        negatives = ~positives
        positive_count, negative_count = positives.sum(), negatives.sum()
        if positive_count == 0 or negative_count == 0:
            raise ValueError("ROC requires both positive and negative samples")
        order = np.argsort(-scores, kind="mergesort")
        sorted_positive = positives[order]
        thresholds = scores[order]
        true_positive = np.cumsum(sorted_positive)
        false_positive = np.cumsum(~sorted_positive)
        distinct = np.r_[np.flatnonzero(np.diff(thresholds)), len(thresholds) - 1]
        tpr = np.r_[0.0, true_positive[distinct] / positive_count, 1.0]
        fpr = np.r_[0.0, false_positive[distinct] / negative_count, 1.0]
        return fpr, tpr

    @staticmethod
    def auc(fpr, tpr):
        fpr = np.asarray(fpr, dtype=float)
        tpr = np.asarray(tpr, dtype=float)
        if len(fpr) != len(tpr):
            raise ValueError("fpr and tpr must have the same length")
        if len(fpr) == 0:
            raise ValueError("fpr and tpr must not be empty")
        return float(np.trapezoid(tpr, fpr))

    @staticmethod
    def _validate_lengths(y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")


class CustomGridSearchCV:
    """Grid search for var_smoothing using stratified CV and all required metrics."""

    def __init__(self, estimator_class, param_grid, n_splits=5, random_state=42):
        self.estimator_class = estimator_class
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if not self.param_grid:
            raise ValueError("param_grid must not be empty")
        results = []
        splitter = CustomStratifiedKFold(self.n_splits, True, self.random_state)
        for parameter in self.param_grid:
            fold_scores = []
            for train_indices, valid_indices in splitter.split(X, y):
                model = self.estimator_class(var_smoothing=parameter).fit(X[train_indices], y[train_indices])
                predictions = model.predict(X[valid_indices])
                probabilities = model.predict_proba(X[valid_indices])
                metrics = CustomEvaluator.metrics(y[valid_indices], predictions)
                positive_index = np.flatnonzero(model.classes_ == 1)
                if len(positive_index) == 0:
                    raise ValueError("ROC-AUC requires positive_label=1 in the model classes")
                fpr, tpr = CustomEvaluator.roc_curve(
                    y[valid_indices], probabilities[:, int(positive_index[0])]
                )
                metrics["roc_auc"] = CustomEvaluator.auc(fpr, tpr)
                fold_scores.append(metrics)
            summary = {"var_smoothing": parameter, "fold_scores": fold_scores}
            for name in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
                values = np.array([score[name] for score in fold_scores])
                summary[f"mean_{name}"] = float(values.mean())
                summary[f"std_{name}"] = float(values.std())
            results.append(summary)
        self.cv_results_ = results
        self.best_result_ = max(results, key=lambda result: (result["mean_f1_score"], result["mean_roc_auc"]))
        self.best_params_ = {"var_smoothing": self.best_result_["var_smoothing"]}
        return self
