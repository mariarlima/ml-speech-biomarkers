import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
)
from sklearn.utils import resample
from scipy.stats import sem, t

from .metrics import rmse_score


def evaluate_on_test(model, X_test, y_test):
    """Evaluate classification model on test data with standard metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    roc = roc_auc_score(y_test, y_proba)

    return rec, spec, roc, acc, y_proba


def evaluate_on_test_regression(model, X_test, y_test):
    """Evaluate regression model and return MAE, RMSE, and predictions."""
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, 30)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = rmse_score(y_test, y_pred)

    return mae, rmse, y_pred


def evaluate_on_test_with_calibration(calibrated_model, X_test, y_test, n_repeats=10):
    """Evaluate calibrated classifier on test set using bootstrap resampling."""
    all_y_true = []
    all_y_probs = []

    for i in range(n_repeats):
        X_resampled, y_resampled = resample(
            X_test, y_test, n_samples=len(X_test), random_state=i
        )
        y_proba = calibrated_model.predict_proba(X_resampled)[:, 1]
        all_y_true.append(y_resampled)
        all_y_probs.append(y_proba)

    return all_y_true, all_y_probs


def splits_performance(y_test, y_proba_bootstrap, idx, n_repeats=10, confidence=0.95):
    """Compute performance and confidence intervals for specific splits."""
    accuracies = []
    positive_rates = []

    for i in range(n_repeats):
        y_pred = (y_proba_bootstrap[i] > 0.5).astype(int)
        acc = (y_pred[idx] == y_test[idx]).mean()
        pos_rate = y_pred[idx].mean()

        accuracies.append(acc)
        positive_rates.append(pos_rate)

    acc_mean = np.mean(accuracies)
    acc_ci = sem(accuracies) * t.ppf((1 + confidence) / 2.0, len(accuracies) - 1)

    pos_mean = np.mean(positive_rates)
    pos_ci = sem(positive_rates) * t.ppf((1 + confidence) / 2.0, len(positive_rates) - 1)

    return (
        f"{100 * acc_mean:.1f}",
        f"{100 * (acc_mean - acc_ci):.1f}",
        f"{100 * (acc_mean + acc_ci):.1f}",
        f"{100 * pos_mean:.1f}",
        f"{100 * (pos_mean - pos_ci):.1f}",
        f"{100 * (pos_mean + pos_ci):.1f}",
    )


def extract_results_classif_test(df):
    """Format classification results from bootstrap confidence intervals."""
    r = df.iloc[0]

    recall = f"{r['Recall Mean']:.1f} ({r['Recall Lower CI']:.1f} - {r['Recall Upper CI']:.1f})"
    spec = f"{r['Specificity Mean']:.1f} ({r['Specificity Lower CI']:.1f} - {r['Specificity Upper CI']:.1f})"
    auc = f"{r['ROC-AUC Mean']:.1f} ({r['ROC-AUC Lower CI']:.1f} - {r['ROC-AUC Upper CI']:.1f})"
    acc = f"{r['Accuracy Mean']:.1f} ({r['Accuracy Lower CI']:.1f} - {r['Accuracy Upper CI']:.1f})"

    return recall, spec, auc, acc


def evaluate_on_test_with_calibration(calibrated_model, X_test, y_test, n_repeats=10):
    """
    Evaluate the calibrated model on the test set with bootstrap sampling.
    """

    # Initialize a list to store bootstrap probabilities for class 1
    all_y_true, all_y_probs = [], []

    for i in range(n_repeats):
        # Create a bootstrap sample of the test set
        X_test_r, y_test_r = resample(X_test, y_test, n_samples=len(X_test), random_state=i)

        # Predict class probabilities with the calibrated model
        y_proba = calibrated_model.predict_proba(X_test_r)[:, 1]  # Class 1 probabilities

        # Store the true labels and predicted probabilities
        all_y_true.append(y_test_r)
        all_y_probs.append(y_proba)
    
    return all_y_true, all_y_probs


def splits_performace(y_test, y_proba_bootstrap, idx, n_repeats=10, confidence=0.95):
    """
    Compute subgroup accuracy and positive-prediction rate with bootstrap CIs.
    """

    # Normalize indexer
    idx = np.asarray(idx)

    accuracies = []
    positive_rates = []

    # Use up to n_repeats available samples
    for i, proba in enumerate(y_proba_bootstrap[:n_repeats]):
        y_pred = (proba > 0.5).astype(int)

        y_pred_sub = y_pred[idx]
        y_true_sub = np.asarray(y_test)[idx]

        acc = accuracy_score(y_true_sub, y_pred_sub)
        accuracies.append(acc)
        positive_rates.append(y_pred_sub.mean())

    # Accuracy CI
    acc_mean = float(np.mean(accuracies))
    acc_se = sem(accuracies)
    acc_ci = acc_se * t.ppf((1 + confidence) / 2.0, len(accuracies) - 1)

    # Positive-rate CI
    pos_mean = float(np.mean(positive_rates))
    pos_se = sem(positive_rates)
    pos_ci = pos_se * t.ppf((1 + confidence) / 2.0, len(positive_rates) - 1)

    return (
        f"{100 * acc_mean:.1f}",
        f"{100 * (acc_mean - acc_ci):.1f}",
        f"{100 * (acc_mean + acc_ci):.1f}",
        f"{100 * pos_mean:.1f}",
        f"{100 * (pos_mean - pos_ci):.1f}",
        f"{100 * (pos_mean + pos_ci):.1f}",
    )
