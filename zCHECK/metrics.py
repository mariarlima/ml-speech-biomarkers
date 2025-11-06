import numpy as np
import pandas as pd
from scipy.stats import loguniform, sem, t
from sklearn.metrics import (
    confusion_matrix,
    mean_squared_error,
    recall_score,
    accuracy_score,
    roc_auc_score,
)


def specificity_score(y_true, y_pred):
    """Compute specificity (true negative rate)."""
    return recall_score(y_true, y_pred, pos_label=0)


def rmse_score(y_true, y_pred):
    """Compute root mean squared error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_metrics(y_true, y_pred):
    """Compute sensitivity, specificity, ROC AUC, and accuracy."""
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1)
    spec = recall_score(y_true, y_pred, pos_label=0)
    rocauc = roc_auc_score(y_true, y_pred)

    return sens, spec, rocauc, acc


def calculate_confusion_items(y_true, y_pred):
    """Return confusion matrix components: TP, TN, FP, FN."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp, tn, fp, fn


def calc_spec(y_true, y_pred):
    """Alias for computing specificity."""
    return specificity_score(y_true, y_pred)


def calculate_metrics_with_ci(scores, confidence=0.95):
    # Calculate the mean and 95% confidence interval for the scores
    mean_score = np.mean(scores)
    se = sem(scores)  # Standard error of the mean
    ci = se * t.ppf((1 + confidence) / 2., len(scores) - 1)
    
    # Convert to percentages
    mean_score_percent = mean_score * 100
    lower_ci_percent = (mean_score - ci) * 100
    upper_ci_percent = (mean_score + ci) * 100
    
    # Format results as "mean (lower CI - upper CI)%"
    formatted_result = f"{mean_score_percent:.1f} ({lower_ci_percent:.1f} - {upper_ci_percent:.1f})"
    return formatted_result


def calc_metrics_folds_bootstrap_probabilities(proba, y_true, green_thresh, amber_thresh):
    accuracies, recalls, specs, aucs = [], [], [], []
    
    for a in proba:
        a = pd.DataFrame(a, columns=['prob'])
        
        # Selective classification by removing amber values
        green_idx = a[a.prob < green_thresh].index.tolist()
        red_idx = a[a.prob >= amber_thresh].index.tolist()
        
        # Filter only green and red (remove amber)
        valid_idx = green_idx + red_idx
        y_true_filtered = y_true[valid_idx]
        proba_filtered = a.prob[valid_idx].values
        
        # AUC ROC on probabilities
        auc = roc_auc_score(y_true_filtered, proba_filtered)
        
        # Binary predictions for accuracy, recall, etc.
        y_pred_filtered = (proba_filtered > 0.5).astype(int)
        
        # Calculate the metrics
        sens, spec, _, acc = calculate_metrics(y_true_filtered, y_pred_filtered)
        
        # Append the results
        recalls.append(sens)
        specs.append(spec)
        aucs.append(auc)
        accuracies.append(acc)
    
    return recalls, specs, aucs, accuracies