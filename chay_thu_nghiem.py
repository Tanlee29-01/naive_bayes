"""Run the complete leakage-safe custom Gaussian Naive Bayes pipeline."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.danh_gia import CustomEvaluator, CustomGridSearchCV
from src.thuat_toan_nb import CustomGaussianNB
from src.tien_xu_ly import (
    CustomIQROutlierRemover,
    CustomMissingValueImputer,
    CustomStandardScaler,
    custom_train_test_split,
)

DATA_PATH = PROJECT_DIR / "data" / "customer_purchase_data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TARGET = "PurchaseStatus"


def run_pipeline():
    # 1. Raw ingestion: no statistics or transformations are applied here.
    data = pd.read_csv(DATA_PATH)
    feature_names = [column for column in data.columns if column != TARGET]
    X_raw = data[feature_names].to_numpy(dtype=float)
    y_raw = data[TARGET].to_numpy()

    # 2. Freeze the test set before EDA, fitting, or any learned transformation.
    X_train_raw, X_test_raw, y_train, y_test = custom_train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )

    # 3. EDA is intentionally restricted to the training partition.
    train_frame = pd.DataFrame(X_train_raw, columns=feature_names)
    print("Training shape:", X_train_raw.shape)
    print("Test shape:", X_test_raw.shape)
    print("Training label distribution:\n", pd.Series(y_train).value_counts().sort_index())
    print("Training summary:\n", train_frame.describe().round(3))
    print("Training correlation:\n", train_frame.corr(numeric_only=True).round(3))

    # 4. Every preprocessing statistic is learned from train only.
    imputer = CustomMissingValueImputer(strategy="mean").fit(X_train_raw)
    X_train_imputed = imputer.transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)
    outlier_remover = CustomIQROutlierRemover().fit(X_train_imputed)
    X_train_clean, y_train_clean = outlier_remover.fit_transform(X_train_imputed, y_train)
    X_test_clean = outlier_remover.transform(X_test_imputed)
    scaler = CustomStandardScaler().fit(X_train_clean)
    X_train = scaler.transform(X_train_clean)
    X_test = scaler.transform(X_test_clean)

    # 5. Tune only inside the training partition.
    grid_search = CustomGridSearchCV(
        CustomGaussianNB,
        param_grid=[1e-9, 1e-8, 1e-7, 1e-5, 1e-3, 1e-1],
        n_splits=5,
        random_state=42,
    ).fit(X_train, y_train_clean)
    print("\nCross-validation results:")
    for result in grid_search.cv_results_:
        print(f"var_smoothing={result['var_smoothing']:.0e}")
        for name in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
            print(f"  {name}: mean={result['mean_' + name]:.4f}, std={result['std_' + name]:.4f}")
    print("Best parameters:", grid_search.best_params_)

    # 6. Fit the selected model on all cleaned training data.
    model = CustomGaussianNB(**grid_search.best_params_).fit(X_train, y_train_clean)

    # 6. Final evaluation is performed once on the untouched test partition.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    metrics = CustomEvaluator.metrics(y_test, predictions)
    matrix, labels = CustomEvaluator.confusion_matrix(y_test, predictions)
    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
    fpr, tpr = CustomEvaluator.roc_curve(y_test, probabilities[:, positive_index])
    roc_auc = CustomEvaluator.auc(fpr, tpr)

    print("\nFinal test metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("Confusion matrix (rows=true, columns=predicted), labels", labels.tolist())
    print(matrix)
    metrics["roc_auc"] = roc_auc
    print(f"roc_auc: {roc_auc:.4f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    _save_plots(train_frame, fpr, tpr, roc_auc, matrix, labels)
    return metrics


def _save_plots(train_frame, fpr, tpr, roc_auc, matrix, labels):
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(train_frame[TARGET] if TARGET in train_frame else train_frame.iloc[:, 0], bins=20)
    axis.set_title("Training feature distribution")
    axis.set_xlabel("First training feature")
    axis.set_ylabel("Count")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "eda_feature_distribution.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"Custom Gaussian NB (AUC={roc_auc:.3f})")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set_title("ROC curve - frozen test set")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_title("Confusion matrix - frozen test set")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    run_pipeline()
