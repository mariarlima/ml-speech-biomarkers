import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import xgboost as xgb

_RNG = 42  # Reproducible defaults


def create_models():
    """Create baseline classifiers with sensible defaults."""
    lr = LogisticRegression(max_iter=10_000, random_state=_RNG)

    svm = SVC(probability=True, random_state=_RNG)

    rf = RandomForestClassifier(criterion="gini", random_state=_RNG)

    mlp = MLPClassifier(
        hidden_layer_sizes=(400,),
        activation="logistic",
        solver="sgd",
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        batch_size="auto",
        max_iter=10_000,
        random_state=_RNG,
    )

    xgb_model = xgb.XGBClassifier(
        random_state=_RNG,
        eval_metric="logloss",
        # tree_method="hist",  # Uncomment if supported; speeds up training.
    )

    return {"lr": lr, "svm": svm, "rf": rf, "nn": mlp, "xgboost": xgb_model}


def create_param_grids():
    """Create hyperparameter distributions for randomized search."""
    lr_params = {
        "C": loguniform(1e-5, 1e2),
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
    }

    svm_params = {
        "C": loguniform(1e-4, 1e3),
        "gamma": ["scale", "auto"] + list(np.geomspace(1e-6, 1.0, 10)),
        "kernel": ["linear", "rbf"],
    }

    rf_params = {
        "n_estimators": list(range(50, 501, 50)),
        "max_depth": [None] + list(range(3, 21)),
        "min_samples_split": [2, 3, 4, 5],
        "min_samples_leaf": [1, 2, 3],
    }

    nn_params = {
        "learning_rate_init": loguniform(1e-3, 1e-2),
        "batch_size": [16, 32, 64, 128, 166],  # 166 = total training recordings
        "alpha": loguniform(1e-4, 1e-3),  # L2 regularization
    }

    xgb_params = {
        "learning_rate": uniform(0.01, 0.49),
        "n_estimators": randint(50, 501),
        "max_depth": randint(1, 11),
        "subsample": uniform(0.01, 0.99),
        "reg_alpha": uniform(0.0, 0.001),
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0.0, 0.4),
    }

    return {
        "lr": lr_params,
        "svm": svm_params,
        "rf": rf_params,
        "nn": nn_params,
        "xgboost": xgb_params,
    }


def create_regression_models():
    """Create baseline regressors with sensible defaults."""
    rr = Ridge()  # Ridge has no random_state
    svr = SVR()
    rf = RandomForestRegressor(random_state=_RNG)
    nn = MLPRegressor(
        hidden_layer_sizes=(400,),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        batch_size="auto",
        max_iter=10_000,
        random_state=_RNG,
    )
    xgb_model = xgb.XGBRegressor(
        random_state=_RNG,
        objective="reg:squarederror",
    )

    return {"Ridge": rr, "SVR": svr, "RFR": rf, "MLP": nn, "XGBoost": xgb_model}


def create_param_grids_regression():
    """Create hyperparameter distributions for regression randomized search."""
    rr_params = {
        "alpha": loguniform(1e-3, 10),
    }

    svr_params = {
        "C": loguniform(1e-2, 1e2),
        "gamma": ["scale", "auto"],
        "kernel": ["linear", "rbf"],
    }

    rf_params = {
        "n_estimators": list(range(50, 201, 50)),
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    nn_params = {
        "learning_rate_init": loguniform(1e-3, 1e-1),
        "batch_size": [16, 32],
        "alpha": loguniform(1e-3, 1e-2),
    }

    xgb_params = {
        "learning_rate": uniform(0.01, 0.3),
        "n_estimators": randint(50, 201),
        "max_depth": randint(2, 6),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_alpha": uniform(0, 1),
        "gamma": uniform(0, 0.4),
    }

    return {
        "Ridge": rr_params,
        "SVR": svr_params,
        "RFR": rf_params,
        "MLP": nn_params,
        "XGBoost": xgb_params,
    }
