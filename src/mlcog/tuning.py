import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (
    RandomizedSearchCV, 
    StratifiedKFold, 
    train_test_split,
    cross_val_predict
)
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics as skmetrics
from sklearn.base import clone

from .metrics import specificity_score
from .utils.io import get_cv_model_path


def hyperparameter_tuning(model, params, X, y):
    """Run RandomizedSearchCV over given parameters with 10-fold cross-validation."""
    search = RandomizedSearchCV(
        model, params, n_iter=50, cv=10, verbose=1,
        random_state=42, n_jobs=-1
    )
    search.fit(X, y)
    return search


def crossval(model_name, model, params, X, y, feature_set):
    """Run cross-validation with multiple metrics and save best estimator."""
    model_map = {
        'Logistic Regression': 'lr',
        'SVM': 'svm',
        'Random Forest': 'rf',
        'Neural Network': 'nn',
        'XGBoost': 'xgboost',
    }

    scoring = {
        'sensitivity': make_scorer(recall_score),
        'specificity': make_scorer(specificity_score),
        'roc_auc': 'roc_auc',
        'accuracy': make_scorer(accuracy_score),
    }

    search = RandomizedSearchCV(
        model, params, n_iter=50, cv=10, verbose=1,
        random_state=42, n_jobs=-1, scoring=scoring, refit='accuracy'
    )
    search.fit(X, y)

    abbr = model_map.get(model_name, model_name.lower())
    path = get_cv_model_path(feature_set, abbr)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, path)

    metrics = ['sensitivity', 'specificity', 'roc_auc', 'accuracy']
    best = search.best_index_
    scores = {'Model': model_name}

    for m in metrics:
        mean = 100 * search.cv_results_[f'mean_test_{m}'][best]
        std = 100 * search.cv_results_[f'std_test_{m}'][best]
        scores[m.capitalize()] = f"{mean:.1f} ({std:.1f})"

    return scores


def crossvalidation_with_predicted_values(model_name, model, params, X, y, feature_set):
    """Run cross-validation and return prediction probabilities."""
    model_map = {'Random Forest': 'rf'}

    scoring = {
        'sensitivity': make_scorer(recall_score),
        'specificity': make_scorer(specificity_score),
        'roc_auc': 'roc_auc',
        'accuracy': make_scorer(accuracy_score),
    }

    search = RandomizedSearchCV(
        model, params, n_iter=50, cv=10, verbose=1,
        random_state=42, n_jobs=-1, scoring=scoring, refit='accuracy'
    )
    search.fit(X, y)
    best_model = search.best_estimator_

    abbr = model_map.get(model_name, model_name.lower())
    path = get_cv_model_path(feature_set, abbr)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, path)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    all_probs = np.zeros(len(X))

    for train_idx, val_idx in skf.split(X, y):
        fold_model = clone(best_model)
        fold_model.fit(X[train_idx], y[train_idx])
        all_probs[val_idx] = fold_model.predict_proba(X[val_idx])[:, 1]

    best = search.best_index_
    scores = {'Model': model_name}

    for m in ['sensitivity', 'specificity', 'roc_auc', 'accuracy']:
        mean = 100 * search.cv_results_[f'mean_test_{m}'][best]
        std = 100 * search.cv_results_[f'std_test_{m}'][best]
        scores[m.capitalize()] = f"{mean:.1f} ({std:.1f})"

    return {'scores': scores, 'y_proba_matrix': all_probs}


def crossvalidation_regression(model_name, model, params, X, y, feature_set):
    """Run cross-validation for regression tasks and save best estimator."""
    model_map = {
        'Ridge': 'rr',
        'SVR': 'svr',
        'RFR': 'rfr',
        'MLP': 'mlp',
        'XGBoost': 'xgb',
    }

    scoring = {
        'RMSE': 'neg_root_mean_squared_error',
        'MAE': 'neg_mean_absolute_error',
    }

    search = RandomizedSearchCV(
        model, params, n_iter=50, cv=10, verbose=1,
        random_state=42, n_jobs=-1, scoring=scoring, refit='RMSE'
    )
    search.fit(X, y)

    abbr = model_map.get(model_name, model_name.lower())
    path = get_cv_model_path(feature_set, abbr, reg=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, path)

    best = search.best_index_
    scores = {'Model': model_name}

    for m in scoring:
        mean = abs(search.cv_results_[f'mean_test_{m}'][best])
        std = search.cv_results_[f'std_test_{m}'][best]
        scores[m] = f"{mean:.1f} ({std:.1f})"

    return scores


def crossval_calibration(model, params, X, y):
    """Run calibration with sigmoid method on best model."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    search = RandomizedSearchCV(
        model, params, n_iter=50, cv=10, verbose=1,
        random_state=42, n_jobs=-1, scoring='f1'
    )
    search.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(search.best_estimator_, method='sigmoid', cv='prefit')
    calibrated.fit(X_val, y_val)

    return calibrated


def crossval_predict_with_folds(model, X, y, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Generate predictions using cross_val_predict
    predictions = cross_val_predict(model, X, y, cv=skf, method='predict_proba')
    
    # For probabilities, choose the column corresponding to the positive class
    probs = predictions[:, 1]  # Assuming binary classification for simplicity
    
    # Create an array to store fold numbers
    fold_numbers = np.empty(len(y), dtype=int)
    
    # Generate fold numbers for each sample
    for fold, (_, test_index) in enumerate(skf.split(X, y)):
        fold_numbers[test_index] = fold
    
    # Create the results DataFrame
    results_df = pd.DataFrame({
        'prob': probs,
        'fold': fold_numbers,
        'y_true': y
    })
    
    return results_df
