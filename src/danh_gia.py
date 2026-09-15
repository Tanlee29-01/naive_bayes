"""Cross-validation and classification metrics without sklearn."""

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
        return float(np.trapezoid(tpr, fpr))
