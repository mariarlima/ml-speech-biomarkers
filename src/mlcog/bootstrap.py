import numpy as np
from scipy.stats import sem, t
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from .evaluation import evaluate_on_test, evaluate_on_test_regression


def fit_and_evaluate_bootstrap_classification(
    best_hyperparams, X_train, y_train, X_test, y_test, n_repeats=10, confidence=0.95
):
    """Perform bootstrap evaluation for classification models."""
    results = []
    bootstrap_probs = {name: [] for name in best_hyperparams}
    metric_names = ['Recall', 'Specificity', 'ROC-AUC', 'Accuracy']

    # Create pipelines with scaling
    model_pipelines = {
        name: Pipeline([('scaler', StandardScaler()), ('clf', model)])
        for name, model in best_hyperparams.items()
    }

    for model_name, pipeline in model_pipelines.items():
        bootstrap_metrics = {metric: [] for metric in metric_names}

        for i in range(n_repeats):
            X_resampled, y_resampled = resample(
                X_train, y_train, n_samples=len(X_train), random_state=i
            )
            pipeline[-1].random_state = i
            pipeline.fit(X_resampled, y_resampled)

            scores, probs = evaluate_on_test(pipeline, X_test, y_test)[:-1], evaluate_on_test(pipeline, X_test, y_test)[-1]
            for metric, score in zip(metric_names, scores):
                bootstrap_metrics[metric].append(score)
            bootstrap_probs[model_name].append(probs)

        summary = {'Model': model_name}
        for metric in metric_names:
            values = bootstrap_metrics[metric]
            mean = np.mean(values)
            ci = sem(values) * t.ppf((1 + confidence) / 2, len(values) - 1)
            summary[f'{metric} Mean'] = mean
            summary[f'{metric} Lower CI'] = mean - ci
            summary[f'{metric} Upper CI'] = mean + ci

        results.append(summary)

    return results, bootstrap_probs


def fit_and_evaluate_bootstrap_regression(
    best_hyperparams, X_train, y_train, X_test, y_test, n_repeats=10, confidence=0.95
):
    """Perform bootstrap evaluation for regression models."""
    results = []
    predictions = {name: [] for name in best_hyperparams}
    metric_names = ['MAE', 'RMSE']

    model_pipelines = {
        name: Pipeline([('scaler', StandardScaler()), ('regressor', model)])
        for name, model in best_hyperparams.items()
    }

    for model_name, pipeline in model_pipelines.items():
        bootstrap_metrics = {metric: [] for metric in metric_names}

        for i in range(n_repeats):
            X_resampled, y_resampled = resample(
                X_train, y_train, n_samples=len(X_train), random_state=i
            )
            pipeline[-1].random_state = i
            pipeline.fit(X_resampled, y_resampled)

            mae, rmse, y_pred = evaluate_on_test_regression(pipeline, X_test, y_test)
            bootstrap_metrics['MAE'].append(mae)
            bootstrap_metrics['RMSE'].append(rmse)
            predictions[model_name].append(y_pred)

        summary = {'Model': model_name}
        for metric in metric_names:
            values = bootstrap_metrics[metric]
            mean = np.mean(values)
            ci = sem(values) * t.ppf((1 + confidence) / 2, len(values) - 1)
            summary[f'{metric} Mean'] = mean
            summary[f'{metric} Lower CI'] = mean - ci
            summary[f'{metric} Upper CI'] = mean + ci

        results.append(summary)

    return results, predictions
