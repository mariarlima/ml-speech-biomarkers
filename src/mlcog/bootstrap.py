import numpy as np
from scipy.stats import sem, t
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.base import clone

from .evaluation import evaluate_on_test, evaluate_on_test_regression


def fit_and_evaluate_bootstrap_classification(best_hyperparams, X_train, y_train, X_test, y_test, n_repeats=10, confidence=0.95):
    """
    Output final DataFrame with evaluation metrics from unseen dataset
    """
    evaluation_results = []
    i = 0

    # turn the models into pipelines
    model_dict = {
        model_name: Pipeline([
            ('scaler', StandardScaler()),
            ('clf', clf)
        ])
        for model_name, clf in best_hyperparams.items()
    }

    bootstrap_probabilities = {model_name: [] for model_name in model_dict.keys()}

    for model_name, clf in model_dict.items():
        metric_names = ['Recall', 'Specificity', 'ROC-AUC', 'Accuracy']
        bootstrap_metrics = {'Recall': [], 'Specificity': [], 'ROC-AUC': [], 'Accuracy': []}

        for _ in range(n_repeats):
            # Create a bootstrap sample
            X_train_r, y_train_r = resample(X_train, y_train, n_samples=len(X_train), random_state=i)

            clf.set_params(clf__random_state=i)
            clf.fit(X_train_r, y_train_r)

            # Evaluate the model on the original test set
            r = evaluate_on_test(clf, X_test, y_test)
            scores, probabilities = r[:-1], r[-1]
            i += 1

            scores_dict = dict(zip(metric_names, scores))
            for metric, score in scores_dict.items():
                bootstrap_metrics[metric].append(score)

            bootstrap_probabilities[model_name].append(probabilities)

        # Calculate the 95% CI for the mean of the metrics for this model
        metrics_ci = {}
        for metric, scores in bootstrap_metrics.items():
            mean_score = np.mean(scores)
            se = sem(scores)
            ci = se * t.ppf((1 + confidence) / 2, len(scores) - 1)
            metrics_ci[metric] = (mean_score, mean_score - ci, mean_score + ci)

        # Result format
        result = {'Model': model_name}
        for metric, (mean, lower_ci, upper_ci) in metrics_ci.items():
            result[f'{metric} Mean'] = mean
            result[f'{metric} Lower CI'] = lower_ci
            result[f'{metric} Upper CI'] = upper_ci
        evaluation_results.append(result)

    return evaluation_results, bootstrap_probabilities


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
