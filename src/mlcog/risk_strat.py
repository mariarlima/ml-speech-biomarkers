import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics

from .metrics import calc_spec


def calc_strat_stats(y_true, y_proba, resolution=0.05, ignore_amber=False):
    """
    Compute stratified metrics over (green, amber) threshold grids and return
    a table sorted by Youden's J (sens + spec - 1).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels {0,1}.
    y_proba : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    resolution : float, default=0.05
        Step size for threshold search.
    ignore_amber : bool, default=False
        If True, evaluate only Green vs Red (drop Amber) and compute ROC AUC
        using probabilities. Otherwise, compute ROC AUC on predicted labels.

    Returns
    -------
    pandas.DataFrame
        Columns: ['green','amber','sens','spec','J','roc_auc','proportion_amber'],
        sorted by 'J' descending.
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()

    # Threshold grids
    green_thresholds = np.arange(0.05, 0.55, resolution)
    amber_thresholds = 1 - green_thresholds  # symmetric sweep

    results = []
    n = len(y_proba)

    for green in green_thresholds:
        for amber in amber_thresholds:
            if green >= amber:
                continue  # enforce Green < Amber

            # Stratify by thresholds: Green <= g, Amber (g, a], Red > a
            stratified = np.select(
                [y_proba <= green, y_proba <= amber],
                ['Green', 'Amber'],
                default='Red'
            )

            proportion_amber = np.mean(stratified == 'Amber')

            if ignore_amber:
                mask = (stratified == 'Green') | (stratified == 'Red')
                if not np.any(mask):
                    continue

                y_true_i = y_true[mask]
                y_proba_i = y_proba[mask]
                # Need at least two classes to compute metrics
                if np.unique(y_true_i).size < 2:
                    continue

                y_pred_i = (stratified[mask] == 'Red').astype(int)  # Red as positive
                roc_auc = skmetrics.roc_auc_score(y_true_i, y_proba_i)
            else:
                y_true_i = y_true
                y_pred_i = (stratified == 'Red').astype(int)  # Red as positive
                roc_auc = skmetrics.roc_auc_score(y_true_i, y_pred_i)

            sens = skmetrics.recall_score(y_true_i, y_pred_i)
            spec = calc_spec(y_true_i, y_pred_i)
            j_stat = sens + spec - 1

            results.append({
                'green': float(green),
                'amber': float(amber),
                'sens': sens,
                'spec': spec,
                'J': j_stat,
                'roc_auc': roc_auc,
                'proportion_amber': proportion_amber,
            })

    df = pd.DataFrame(results)
    return df.sort_values(by='J', ascending=False).reset_index(drop=True)
