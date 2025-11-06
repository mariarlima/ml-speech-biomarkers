from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, make_scorer, roc_auc_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform, sem, t
import joblib
import pandas as pd
import numpy as np
import os 
import torch
import xgboost as xgb
from scipy.stats import uniform, randint
import pickle
import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plotting
import nltk
import spacy
from nltk import word_tokenize, pos_tag
from sklearn import metrics as skmetrics
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import transforms
import typing


def create_models():
    """
    Function to create models
    """
    # Define models with default settings (hyperparameters will be tuned later)
    lr = LogisticRegression(max_iter=10000, random_state=42)
    svm = SVC(probability=True, random_state=42)
    rf = RandomForestClassifier(criterion='gini', random_state=42)
    nn = MLPClassifier(
        hidden_layer_sizes=(400,),
        activation='logistic',
        solver='sgd',  # stochastic gradient descent
        learning_rate='adaptive',  # adaptive learning rate
        learning_rate_init=0.001,  # initial learning rate, can be adjusted based on needs
        batch_size='auto',  # auto means min(200, n_samples)
        max_iter=10000,
        random_state=42
    )
    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
    return {'lr': lr, 'svm': svm, 'rf': rf, 'nn': nn, 'xgboost': xgb_model}

def create_param_grids():
    """
    Function to create hyperparameter grids
    """
    lr_params = {
        'C': loguniform(1e-5, 1e2),  # Sampling over a wider range
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']  # 'saga' supports both 'l1' and 'l2', but 'liblinear' is more efficient for small datasets
    }
    svm_params = {
        'C': loguniform(1e-4, 1e3),  # Sampling over a wide range
        'gamma': ['scale', 'auto'] + list(loguniform(1e-6, 1).rvs(size=10)),  # Including 'scale' and 'auto', and some random values between 1e-6 and 1
        'kernel': ['linear', 'rbf']  # Commonly used kernels
    }
    rf_params = {
        'n_estimators': list(range(50, 501, 50)),  # From 50 to 500 trees in steps of 50
        'max_depth': [None] + list(range(3, 21)),  # No max depth or from 3 to 20
        # note: Overfitting: Deeper trees can capture more complex relationships in the data but also run a higher risk of overfitting, 
        # especially when data is limited. With only 166 samples, a very deep tree (e.g., depth of 100) might overfit the training data, 
        # leading to poor generalization on unseen data. providing the option for a tree to grow very deep can still be risky in terms 
        # of overfitting when dealing with small datasets.
        'min_samples_split': [2, 3, 4, 5],
        'min_samples_leaf': [1, 2, 3]
        # CAREFUL WITH SMALL TRAINING SET!
        # This gives the model enough flexibility to learn from the data but still has some constraints to prevent overfitting.]
    }
    nn_params = {
        # 'learning_rate_init': loguniform(1e-5, 1e-1),  # Sampling learning rates over a range
        'learning_rate_init': loguniform(1e-3, 1e-2),  # Learning rate between [0.001, 0.01]
        'batch_size': [16, 32, 64, 128, 166], # 166 is the total recordings for the training data
        # 'alpha': [0, 1e-3, 1e-4, 1e-5]  # Regularization strength for weights
        'alpha': loguniform(1e-4, 1e-3),  # L2 regularization between [0, 0.001]
    }

    xgb_params = {
        'learning_rate': uniform(0.01, 0.49),  # uniform distribution starting at 0.01 with range of 0.49
        'n_estimators': randint(50, 501),  # random integers between 50 (inclusive) and 501 (exclusive)
        'max_depth': randint(1, 11),  # random integers between 1 (inclusive) and 11 (exclusive)
        'subsample': uniform(0.01, 0.99),  # uniform distribution starting at 0.01 with range of 0.99
        'reg_alpha': uniform(0, 0.001),  # L1 regularization term. uniform distribution starting at 0 with range of 0.001
        'colsample_bytree': uniform(0.7, 0.3).rvs(size=10),  # uniform distribution starting at 0.7 with range of 0.3
        'gamma': uniform(0, 0.4)  # uniform distribution starting at 0 with range of 0.4
    }

    return {'lr': lr_params, 'svm': svm_params, 'rf': rf_params, 'nn': nn_params, 'xgboost': xgb_params}


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def hyperparameter_tuning(model, params, X, y):
    """
    Function to perform hyperparameter tuning
    """
    random_search = RandomizedSearchCV(model, params, n_iter=50, cv=10, verbose=1, random_state=42, n_jobs=-1)
    random_search.fit(X, y)
    return random_search


def create_df_embeddings(path_label_1, path_label_0):
    """
    Load embeddings from .pt files and assign labels and groups
    """
    # Initialize lists to hold data, labels, and groups
    data_list = []
    label_list = []
    group_list = []

    # Define folder-label mapping
    folder_label_mapping = {
        path_label_1: 1,
        path_label_0: 0
    }
    # Load embeddings from .pt files and assign labels and groups
    for folder, label in folder_label_mapping.items():
        for filename in os.listdir(folder):
            if filename.endswith('.pt'):
                # Load the embedding
                embedding = torch.load(os.path.join(folder, filename))
                
                # Append to data and label lists
                data_list.append(embedding.numpy())
                label_list.append(label)
                
                # Use filename without extension as the group identifier
                group_id = os.path.splitext(filename)[0]
                group_list.append(group_id)

    # Convert lists to NumPy arrays for compatibility with scikit-learn
    X = np.array(data_list)
    y = np.array(label_list)
    groups = np.array(group_list)

    # Create a DataFrame for better visualization and potential use in pandas-based pipelines
    df = pd.DataFrame({
        'data': list(X),
        'label': y,
        'pid': groups
    })
    return df


def create_df_from_pt_files(csv_path, folder_path):
    """
    Load data from .pt files in the specified folder_path, match labels from the provided csv_path,
    and return a DataFrame with columns [data], [label], and [pid].
    """

    # Read csv file
    df_task = pd.read_csv(csv_path)
    # Create a mapping dictionary for 'Dx' to labels (0 or 1)
    dx_label_mapping = df_task.set_index('ID')['Dx'].apply(lambda x: 0 if x == 'Control' else 1).to_dict()

    data_list = []
    label_list = []
    pid_list = []

    # Load embeddings from .pt files, assign labels, and extract pids
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            # Load the embedding
            embedding = torch.load(os.path.join(folder_path, filename))
            data_list.append(embedding.numpy())

            # Extract pid from filename and modify it
            pid = filename.split('_')[-1].split('.pt')[0]
            modified_pid = 'adrsd' + pid
            pid_list.append(modified_pid)

            # Fetch label using the modified pid
            label = dx_label_mapping.get(modified_pid, None)  # Default to None if pid not found
            label_list.append(label)

    # Convert lists to NumPy arrays
    X = np.array(data_list)
    y = np.array(label_list)
    pids = np.array(pid_list)

    df = pd.DataFrame({
        'data': list(X),
        'label': y,
        'pid': pids
    })

    return df


def create_df_from_dict(embeddings_dict, csv_path):
    """
    Load data from the embeddings_dict, match labels from the provided csv_path,
    and return a DataFrame with columns [data], [label], and [pid].
    """

    # Read csv file
    df_task = pd.read_csv(csv_path)
    # Create a mapping dictionary for 'Dx' to labels (0 or 1)
    dx_label_mapping = df_task.set_index('ID')['Dx'].apply(lambda x: 0 if x == 'Control' else 1).to_dict()

    data_list = []
    label_list = []
    pid_list = []

    # Load embeddings from embeddings_dict, assign labels, and extract pids
    for pid, embedding in embeddings_dict.items():
        data_list.append(embedding)
        pid_modified = 'adrsd' + pid
        # print(pid_modified)
        pid_list.append(pid_modified)

        # Fetch label using the pid
        label = dx_label_mapping.get(pid_modified, None)  # Default to None if pid not found
        label_list.append(label)

    # Convert lists to a list of arrays for data and list for labels and pids
    df = pd.DataFrame({
        'data': data_list,
        'label': label_list,
        'pid': pid_list
    })

    return df


def concatenate_feature_vectors(df1, df2, feature1_name, feature2_name):
    """
    Merges two dataframes on 'pid', checks if labels match, and concatenates their feature vectors.

    :param df1: First DataFrame
    :param df2: Second DataFrame
    :param feature1_name: Suffix for features of df1
    :param feature2_name: Suffix for features of df2
    :return: DataFrame with concatenated feature vectors
    """
    # Merge the dataframes on 'pid'
    merged_df = pd.merge(df1, df2, on='pid', suffixes=(f'_{feature1_name}', f'_{feature2_name}'))
    
    # Assert that labels match
    label_cols = [col for col in merged_df.columns if 'label' in col]
    assert all(merged_df[label_cols[0]] == merged_df[label_cols[1]]), "Labels do not match for some pids"
    
    # Concatenate the vectors
    def concatenate_vectors(row):
        vector1 = row[f'data_{feature1_name}']
        vector2 = row[f'data_{feature2_name}']
        return np.concatenate([vector1, vector2])

    merged_df['fusion'] = merged_df.apply(concatenate_vectors, axis=1)

    # Rename label column and keep only necessary columns
    merged_df = merged_df.rename(columns={label_cols[0]: 'label'})
    merged_df = merged_df[['pid', 'label', 'fusion']]

    return merged_df


def load_best_params_UPD(feature_set):
    model_name_mapping = {
    'Logistic Regression': 'lr',
    'SVM': 'svm',
    'Random Forest': 'rf',
    'Neural Network': 'nn',
    'XGBoost': 'xgboost'
    }

    best_hyperparams = {}

    # Load each saved model into the best_models dictionary
    for model_name, abbreviation in model_name_mapping.items():
        filename = f"10fcv_{abbreviation}.pkl"
        file_path = os.path.join("./hyperparam-tuning-UPD", feature_set, filename)
        # file_path = os.path.join(f"./hyperparam-tuning/{feature_set}/new-liwc07/{filename}")
        
        # Load the model if the file exists
        if os.path.exists(file_path):
            print(file_path)
            best_model = joblib.load(file_path)
            best_hyperparams[model_name] = best_model
    
    return best_hyperparams


def specificity_score(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp)


def crossvalidation_UPD(model_name, model, params, X, y, feature_set):
    """
    Function to perform hyperparameter tuning and evaluation on training set (mean, standard deviation) using single 10-fold CV
    """

    model_name_mapping = {
        'Logistic Regression': 'lr',
        'SVM': 'svm',
        'Random Forest': 'rf',
        'Neural Network': 'nn',
        'XGBoost': 'xgboost'
    }

    scoring = {
        'sensitivity': make_scorer(recall_score),  # Sensitivity is the same as recall
        'specificity': make_scorer(specificity_score),
        'roc_auc': 'roc_auc',
        'accuracy': make_scorer(accuracy_score)
    }

    random_search = RandomizedSearchCV(
        model, params, 
        n_iter=50, cv=10, verbose=1, 
        random_state=42, n_jobs=-1, 
        scoring=scoring, refit='accuracy'  # Use accuracy or any metric of choice for refitting
    )
    
    random_search.fit(X, y)

    # Save the best model
    abbreviation = model_name_mapping.get(model_name, model_name.lower())
    filename = f"10fcv_{abbreviation}.pkl"
    joblib.dump(random_search.best_estimator_, f"./hyperparam-tuning-UPD/{feature_set}/{filename}")

    # Extracting scores
    metrics = ['sensitivity', 'specificity', 'roc_auc', 'accuracy']
    scores = {'Model': model_name}
    
    for metric in metrics:
        mean_score_key = f'mean_test_{metric}'
        std_score_key = f'std_test_{metric}'
        best_index = random_search.best_index_
        mean_score = random_search.cv_results_[mean_score_key][best_index] * 100
        std_score = random_search.cv_results_[std_score_key][best_index] * 100
        scores[metric.capitalize()] = f"{mean_score:.1f} ({std_score:.1f})"
    
    return scores


def crossvalidation_with_predicted_values(model_name, model, params, X, y, feature_set):
    """
    Function to perform hyperparameter tuning and evaluation on training set (mean, standard deviation) using single 10-fold CV
    """

    model_name_mapping = {
        'Random Forest': 'rf',
    }

    scoring = {
        'sensitivity': make_scorer(recall_score),  # Sensitivity is the same as recall
        'specificity': make_scorer(specificity_score),
        'roc_auc': 'roc_auc',
        'accuracy': make_scorer(accuracy_score)
    }

    random_search = RandomizedSearchCV(
        model, params, 
        n_iter=50, cv=10, verbose=1, 
        random_state=42, n_jobs=-1, 
        scoring=scoring, refit='accuracy'  # Use accuracy or any metric of choice for refitting
    )
    
    random_search.fit(X, y)
    best_model = random_search.best_estimator_

    # Save the best model
    abbreviation = model_name_mapping.get(model_name, model_name.lower())
    filename = f"10fcv_{abbreviation}.pkl"
    joblib.dump(best_model, f"./hyperparam-tuning-UPD/{feature_set}/{filename}")

    # Step 2: Predict using each fold model on full dataset
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    n_samples = len(X)
    all_probs = np.zeros((n_samples, 10))

    from sklearn.base import clone 

    for fold_idx, (train_idx, _) in enumerate(skf.split(X, y)):
        model_fold = clone(best_model)
        model_fold.fit(X[train_idx], y[train_idx])
        probs = model_fold.predict_proba(X)[:, 1]
        all_probs[:, fold_idx] = probs  # Each column is prediction from one fold's model


    # Extracting scores
    metrics = ['sensitivity', 'specificity', 'roc_auc', 'accuracy']
    scores = {'Model': model_name}
    best_index = random_search.best_index_
    
    for metric in metrics:
        mean_score = random_search.cv_results_[f'mean_test_{metric}'][best_index] * 100
        std_score = random_search.cv_results_[f'std_test_{metric}'][best_index] * 100
        scores[metric.capitalize()] = f"{mean_score:.1f} ({std_score:.1f})"

    return {
        'scores': scores,
        'y_proba_matrix': all_probs
    }


def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


### INCLUDE BOOTSTRAP SAMPLING ###
def evaluate_on_test_new(model, X_test, y_test):
    """
    Evaluate the model on the given test set.
    """
    y_pred = model.predict(X_test)
    # Predict class probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)
    roc_auc = roc_auc_score(y_test, y_proba)
    return recall, specificity, roc_auc, accuracy, y_proba


def evaluate_on_test_regression(model, X_test, y_test):
    """
    Evaluate the model on the given test set using regression metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_clipped = np.clip(y_pred, 0, 30)  # Clip predictions to [0, 30]

    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred_clipped)
    rmse = rmse_score(y_test, y_pred_clipped)  # Optional: Include if R² is considered useful

    # Return the metrics
    return mae, rmse, y_pred_clipped


def fit_and_evaluate_bootstrap_(best_hyperparams, X_train, y_train, X_test, y_test, n_repeats=10, confidence=0.95):
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

    # bootstrap_probabilities = []
    bootstrap_probabilities = {model_name: [] for model_name in best_hyperparams.keys()}

    for model_name, clf in model_dict.items():
        
        metric_names = ['Recall', 'Specificity', 'ROC-AUC', 'Accuracy']
        bootstrap_metrics = {'Recall': [], 'Specificity': [], 'ROC-AUC': [], 'Accuracy': []}

        for _ in range(n_repeats):
            # Create a bootstrap sample
            X_train_r, y_train_r = resample(X_train, y_train, n_samples=len(X_train), random_state=i)

            clf[-1].random_state = i
 
            clf.fit(X_train_r, y_train_r)

            # Evaluate the model on the original test set
            r = evaluate_on_test_new(clf, X_test, y_test)
            scores, probabilities = r[:-1], r[-1]
            i += 1

            # Updated to ensure order 
            scores_dict = dict(zip(metric_names, scores))
            for metric, score in scores_dict.items():
                bootstrap_metrics[metric].append(score)

            # Store the probabilities for this iteration
            bootstrap_probabilities[model_name].append(probabilities)

        # Calculate the 95% CI for the mean of the metrics for this model
        metrics_ci = {}
        for metric, scores in bootstrap_metrics.items():
            mean_score = np.mean(scores)
            se = sem(scores)
            ci = se * t.ppf((1 + confidence) / 2, len(scores) - 1)
            metrics_ci[metric] = (mean_score, mean_score - ci, mean_score + ci)

        # Prepare the result dictionary
        result = {'Model': model_name}
        for metric, (mean, lower_ci, upper_ci) in metrics_ci.items():
            result[f'{metric} Mean'] = mean
            result[f'{metric} Lower CI'] = lower_ci
            result[f'{metric} Upper CI'] = upper_ci
        evaluation_results.append(result)

    return evaluation_results, bootstrap_probabilities


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
    # Store the accuracies from each bootstrap sample
    accuracies = []
    likelihood_pos = []
    
    for i in range(n_repeats):
        # Extract predictions for the current bootstrap
        # y_pred_test = y_proba_bootstrap[i][:, 1] > 0.5  # Convert probabilities to 0/1 predictions
        y_pred_test = y_proba_bootstrap[i]> 0.5  # Convert probabilities to 0/1 predictions
        y_pred_test_subgroup = y_pred_test[idx]  # Filter predictions for the specific subgroup
        y_true_subgroup = y_test[idx]  # Filter true labels for the specific subgroup
        
        # Calculate and store the accuracy for this bootstrap sample
        acc = accuracy_score(y_true_subgroup, y_pred_test_subgroup)
        accuracies.append(acc)

        # Calculate and store the percentage of positive predictions for this bootstrap sample
        # Applying .mean() to this boolean array calculates the fraction of True values, since in this context, 
        # True is treated as 1 and False as 0. This gives the proportion (between 0 and 1) of positive predictions.
        positive_predictions = (y_pred_test_subgroup).mean()
        likelihood_pos.append(positive_predictions)
    
    # Calculate the mean and 95% confidence interval for the accuracies
    mean_acc, se = np.mean(accuracies), sem(accuracies)
    ci = se * t.ppf((1 + confidence) / 2., len(accuracies) - 1)

    # Convert to percentages
    mean_acc_percent = mean_acc * 100
    lower_ci_percent = (mean_acc - ci) * 100
    upper_ci_percent = (mean_acc + ci) * 100

    # Calculate the mean and 95% confidence interval for the positive prediction percentages
    mean_lik, se = np.mean(likelihood_pos), sem(likelihood_pos)
    perc_ci = se * t.ppf((1 + confidence) / 2., len(likelihood_pos) - 1)

    # Convert to percentages - likelihood of positive prediction
    mean_likelihood = mean_lik * 100
    lower_ci_likelihood = (mean_lik - perc_ci) * 100
    upper_ci_likelihood = (mean_lik + perc_ci) * 100

    return f"{mean_acc_percent:.1f}", f"{lower_ci_percent:.1f}", f"{upper_ci_percent:.1f}", f"{mean_likelihood:.1f}", f"{lower_ci_likelihood:.1f}", f"{upper_ci_likelihood:.1f}"


def get_data(feature_name, model_abbr, feature_abbr, probs_name):
    with open(f'./FEATURES/dfs/{feature_name}.pkl', 'rb') as f:
        df_train = pickle.load(f)
    with open(f'./FEATURES/test-features/dfs/{feature_name}.pkl', 'rb') as f:
        df_test = pickle.load(f)
    X_train = df_train['data']
    y_train = df_train['label']
    X_train_2d = np.stack(X_train.values)
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train_2d)
    X_test = df_test['data']
    y_test = df_test['label']
    X_test_2d = np.stack(X_test.values)
    X_scaled_test = scaler.transform(X_test_2d)
    filename = f"10fcv_{model_abbr}.pkl"
    file_path = os.path.join("./hyperparam-tuning", feature_abbr, filename)
    # load best model
    if os.path.exists(file_path):
        best_model = joblib.load(file_path)
    # load probs
    with open(f'./evaluation-test/predict_proba/{probs_name}.pkl', 'rb') as f:
        proba = pickle.load(f)
        
    return X_scaled_train, y_train, X_scaled_test, y_test, best_model, proba


def probs_mmse(probs, groups):
    
    probability_values = groups['prob'].values
    mmse_values = groups.mmse.values
    pid_values = groups.astype({'pid': int}).pid.values
    
    results = []
    for run, probs in enumerate(probability_values):
        results.append(
            pd.DataFrame(
                {   
                    'pid': pid_values,
                    'mmse': mmse_values,
                    # 'probs': probs[:, 1],
                    'probs': probs,
                    'run': run
                }
            )
        )
    results = pd.concat(results)
    results = results.dropna(subset=['mmse'])
    results.mmse = results.mmse.astype(int)
    return results


def plot_hist(results):
    with plotting.paper_theme():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(
            data=results.groupby('pid').mean().reset_index(),
            y='prob',
            x='mmse',
            bins=(15, 10),
            ax=ax,
        )

        ax.axvline(x=26, linestyle='--', color='darkblue')
        ax.axvline(x=20, linestyle='--', color='darkblue')
        ax.axvline(x=10, linestyle='--', color='darkblue')

        # Calculate y position as a percentage of y-axis range
        y_min, y_max = ax.get_ylim()
        y_pos = y_max - 0.02 * (y_max - y_min)  # X% above the bottom of the y-axis

        ax.text(4, y_pos, 'Severe', fontsize=8, color='black')
        ax.text(10.5, y_pos, 'Moderate', fontsize=8, color='black')
        ax.text(20.5, y_pos, 'Mild', fontsize=8, color='black')
        ax.text(26.5, y_pos, 'CN', fontsize=8, color='black')

        ax.set_xlabel('MMSE Score')
        ax.set_ylabel('Positive Probability')

        fig.colorbar(ax.collections[0], ax=ax, label='Number of Participants', shrink=0.8)
        plt.show()


def extract_results_classif_test(df):
    recall_result = f"{round(df['Recall Mean'].values[0], 2)} ({round(df['Recall Lower CI'].values[0], 2)} - {round(df['Recall Upper CI'].values[0], 2)})"
    spec_result = f"{round(df['Specificity Mean'].values[0], 2)} ({round(df['Specificity Lower CI'].values[0], 2)} - {round(df['Specificity Upper CI'].values[0], 2)})"
    auc_result = f"{round(df['ROC-AUC Mean'].values[0], 2)} ({round(df['ROC-AUC Lower CI'].values[0], 2)} - {round(df['ROC-AUC Upper CI'].values[0], 2)})"
    acc_result = f"{round(df['Accuracy Mean'].values[0], 2)} ({round(df['Accuracy Lower CI'].values[0], 2)} - {round(df['Accuracy Upper CI'].values[0], 2)})"
    return recall_result, spec_result, auc_result, acc_result


def calculate_metrics(true, pred):
    rec = recall_score(true, pred)
    acc = accuracy_score(true, pred)
    cm = confusion_matrix(true, pred)
    rocauc =roc_auc_score(true, pred)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = rec
    specificity = TN / (TN + FP) if TN + FP != 0 else 0
    j = sensitivity + specificity - 1
    return sensitivity, specificity, rocauc, acc


def calc_metrics_folds_bootstrap(proba, y_true, green_thresh, amber_thresh):
    accuracies = []
    recalls = []
    specs = []
    aucs = []
    for a in proba:
        a= pd.DataFrame(a, columns=['prob'])
        green_idx = a[a.prob < green_thresh].index.tolist()
        red_idx = a[a.prob >= amber_thresh].index.tolist()
        a_values= a.prob

        y_pred_after_risk_one_repeat = np.concatenate((a_values[green_idx], a_values[red_idx]))
        y_true_after_risk_one_repeat = np.concatenate((y_true[green_idx], y_true[red_idx]))

        y_pred_after_risk_one_repeat = (y_pred_after_risk_one_repeat > 0.5).astype(int)

        sens, spec, auc, acc = calculate_metrics(y_true_after_risk_one_repeat, y_pred_after_risk_one_repeat)
        recalls.append(sens)
        specs.append(spec)
        aucs.append(auc)
        accuracies.append(acc)
    return recalls, specs, aucs, accuracies


def calculate_metrics_with_ci(scores, confidence=0.95):
    from scipy.stats import loguniform, sem, t
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


class ShapDisplay:
    def from_shap_values(
        shap_values_df,
        n_features_plot=15,
        ax=None,
    ):
        point_alpha = 0.75
        cmap = "winter"
        # taking the data in the middle 90% of the distribution
        feature_value_quatiles = [0.025, 0.975]
        # shap_quantiles = [0.025, 0.975]

        shap_quantiles = [0, 1]

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))


        f_order = (
            shap_values_df.groupby("Feature")
            .agg({"SHAP Value": lambda x: np.abs(x).mean()})
            .sort_values("SHAP Value", ascending=False)
            .index
        )

        plot_df = (
            shap_values_df.reset_index(drop=True)
            .set_index("Feature")
            .query(
                "Value < Value.quantile(@feature_value_quatiles[1])"
                "& Value > Value.quantile(@feature_value_quatiles[0])"
            )
            # .query(
            #     "`SHAP Value` < `SHAP Value`.quantile(@shap_quantiles[1])"
            #     "& `SHAP Value` > `SHAP Value`.quantile(@shap_quantiles[0])"
            # )
            .reset_index()
        )


        plot_df = plot_df.loc[lambda df: df["Feature"].isin(f_order[:n_features_plot])]
        plot_df = plot_df.sort_values("Feature", key=lambda x: x.map(f_order.get_loc))

        ax = sns.scatterplot(
            data=plot_df,
            x="SHAP Value",
            y="Feature",
            hue="Value",
            palette=cmap,
            legend=False,
            s=5,
            ax=ax,
            alpha=point_alpha,
            edgecolor=None,
        )
        pts = ax.collections[0]
        pts.set_offsets(
            pts.get_offsets()
            + np.c_[np.zeros(len(plot_df)), np.random.normal(0, 0.1, len(plot_df))]
        )
        ax.margins(y=0.1)
        ax.autoscale_view()

        # ax.set_title("SHAP and Feature Values", y=0.98)
        ax.yaxis.grid(True)
        ax.set_ylabel("")
        ax.set_xlabel("SHAP Value (%)")
        # ax.set_xlim(-1.1*plot_df['SHAP Value'].abs().max(), 1.1*plot_df['SHAP Value'].abs().max())
        # ax.vlines(0, ax.get_ylim()[0]*1.1, 0, color='k', linestyle='--', linewidth=2, zorder=0)
        ax.axvline(0, color="black", linestyle="--", linewidth=1, zorder=0)

        norm = plt.Normalize(plot_df["Value"].min(), plot_df["Value"].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Remove the legend and add a colorbar
        # ax.get_legend().remove()
        cbar = ax.figure.colorbar(sm, ax=ax, pad=0.01, shrink=0.66)
        cbar.outline.set_linewidth(0)
        cbar.set_alpha(point_alpha)
        cbar.set_label("Normalised Feature Value")
        cbar.ax.tick_params(
            axis="both", which="major", pad=1, length=2, width=0.5, right=True
        )

        # Add custom labels for the colorbar
        cbar.set_ticks([plot_df["Value"].min(), plot_df["Value"].max()])
        cbar.set_ticklabels(["Low", "High"])

        ax.tick_params(
            axis="both",
            which="major",
            pad=1,
            length=2,
            width=0.5,
            bottom=True,
            left=True,
        )
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)

        return ax
    

def update_with_defaults(kwargs_dict, default_dict):
    kwargs_dict = kwargs_dict.copy()
    for key, value in default_dict.items():
        if key not in kwargs_dict:
            kwargs_dict[key] = value
    return kwargs_dict

colors = [
    # "#104E8B", # dark blue
    'xkcd:cerulean',
    "#00B2EE", 
    "#88CCEE",
    "mediumaquamarine", 
]  
import matplotlib.colors as mcolors
colors = [mcolors.to_rgba(c, alpha=0.97) for c in colors]

def waterfallplot(
    data: pd.DataFrame = None,
    x: str = None,
    y: str = None,
    order: typing.List[str] = None,
    base: float = 0,
    orient: str = "h",
    estimator: typing.Union[str, typing.Callable] = "sum",
    cmap: str = None,
    alpha: bool = 0.75,
    # positive_colour: str = "#648fff",
    # negative_colour: str = "#fe6100",
    positive_colour: str = colors[0],
    negative_colour: str = colors[3],
    width: float = 0.8,
    bar_label: bool = True,
    ax: plt.Axes = None,
    arrow_kwargs: typing.Dict[str, typing.Any] = {},
    bar_kwargs: typing.Dict[str, typing.Any] = {},
    bar_label_kwargs: typing.Dict[str, typing.Any] = {},
):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    horizontal = orient == "h"

    categories = y if horizontal else x
    values = x if horizontal else y
    unique_categories = data[categories].unique()
    order = unique_categories if order is None else order
    bar_labels, bar_values = (
        data.groupby(categories)[values]
        .agg(estimator)
        .loc[order]
        .reset_index()
        .values.T
    )

    # if not all of the categories are in the order
    categories_not_in_order = unique_categories[~np.isin(unique_categories, order)]
    if len(categories_not_in_order) > 0:
        bar_labels_other, bar_values_other = (
            data.groupby(categories)[values]
            .agg(estimator)
            .loc[categories_not_in_order]
            .reset_index()
            .values.T
        )
        bar_labels_other = ["Other"]
        bar_values_other = [bar_values_other.sum()]
        bar_labels = np.concatenate([bar_labels, bar_labels_other])
        bar_values = np.concatenate([bar_values, bar_values_other])

    pos_bar = np.arange(len(bar_labels))
    height = bar_values
    bottom = np.concatenate([np.array([base]), base + np.cumsum(bar_values)])[:-1]
    ends = bottom + bar_values

    if horizontal:
        height = height[::-1]
        bottom = bottom[::-1]
        ends = ends[::-1]
        bar_labels = bar_labels[::-1]

    if horizontal:
        bars = ax.barh(
            y=pos_bar, height=width, left=bottom, alpha=0, width=height, **bar_kwargs
        )
    else:
        bars = ax.bar(
            pos_bar, height=height, bottom=bottom, alpha=0, width=width, **bar_kwargs
        )

    min_value = np.min([base, ends.min()])
    max_value = np.max([base, ends.max()])

    whole_range = max_value - min_value

    if horizontal:
        ax.set_xlim(
            min_value - 0.1 * whole_range,
            max_value + 0.1 * whole_range,
        )
    else:
        ax.set_ylim(
            min_value - 0.1 * whole_range,
            max_value + 0.1 * whole_range,
        )

    bar_boxes = [rect.get_bbox().get_points() for rect in bars.patches]

    if cmap is not None:
        cmap = sns.color_palette(cmap, len(pos_bar))
    else:
        cmap = []
        for bar_box in bar_boxes:
            if horizontal:
                if (bar_box[1, 0] - bar_box[0, 0]) > 0:
                    cmap.append(positive_colour)
                else:
                    cmap.append(negative_colour)
            else:
                if (bar_box[1, 1] - bar_box[0, 1]) > 0:
                    cmap.append(positive_colour)
                else:
                    cmap.append(negative_colour)

    arrow_kwargs = update_with_defaults(
        arrow_kwargs,
        {
            "head_width": width,
            "length_includes_head": True,
            "alpha": alpha,
            "linewidth": 0,
            "width": width,
            "head_length": whole_range * 0.025,
        },
    )

    for nb, bar_box in enumerate(bar_boxes):
        if horizontal:
            y_bar = pos_bar[nb]
            x_bar = bar_box[0, 0]
            dx_bar = bar_box[1, 0] - bar_box[0, 0]
            dy_bar = 0
        else:
            x_bar = pos_bar[nb]
            y_bar = bar_box[0, 1]
            dy_bar = bar_box[1, 1] - bar_box[0, 1]
            dx_bar = 0

        ax.arrow(
            x=x_bar,
            y=y_bar,
            dy=dy_bar,
            dx=dx_bar,
            color=cmap[nb] if cmap is not None else cmap,
            **arrow_kwargs,
        )

    # if horizontal:
    #    ax.axvline(base, color='black', linestyle='--', linewidth=2, zorder=0)
    # else:
    #    ax.axhline(base, color='black', linestyle='--', linewidth=2, zorder=0)

    bar_label_kwargs = update_with_defaults(
        bar_label_kwargs,
        {
            "fmt": "%.2f",
            "label_type": "center",
            "padding": 0,
        },
    )

    if bar_label:
        ax.bar_label(
            bars,
            **bar_label_kwargs,
        )

    if horizontal:
        ax.set_yticks(pos_bar)
        ax.set_yticklabels(bar_labels)
    else:
        ax.set_xticks(pos_bar)
        ax.set_xticklabels(bar_labels)

    ax.set_ylabel(y)
    ax.set_xlabel(x)

    ax.grid(False, axis="y" if horizontal else "x")

    return ax

colors = [
    # "#104E8B", # dark blue
    'xkcd:cerulean',
    'xkcd:powder blue', #"#00B2EE", 
    "#88CCEE",
    "mediumaquamarine", 
]  


def risk_feature_plot(shap_values_point_df, base, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    order = (
        shap_values_point_df.groupby("Feature")
        .sum()
        .reset_index()
        .set_index("Feature")
        .sort_values("Risk", key=abs, ascending=False)
        .index[:5]
    )
    # bar_labels = (
    #     shap_values_point_df.set_index("Feature").loc[order]["Value"].values[::-1]
    # )
    bar_labels = (
        shap_values_point_df.set_index("Feature").loc[order]["Value Norm"].values[::-1]
    )

    ax = waterfallplot(
        data=shap_values_point_df,
        y="Feature",
        x="Risk",
        order=order,
        base=base,
        orient="h",
        cmap=None,
        estimator="sum",
        # positive_colour="#648fff",
        # negative_colour="#fe6100",
        positive_colour=colors[0],
        negative_colour=colors[1],
        width=0.8,
        bar_label=True,
        ax=ax,
        arrow_kwargs={},
        bar_kwargs={},
        bar_label_kwargs={
            "fmt": "%.0f %%",
            "labels": [""]
            + [f"+{x:.1f}" if x >= 0 else f"{x:.1f}" for x in bar_labels],
        },
    )

    proba_value = shap_values_point_df["Risk"].sum() + base

    ax.axvline(proba_value, color="black", linestyle="--", linewidth=0.5, zorder=0)

    text_trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    ax.text(
        proba_value * 0.99,
        0.0,
        f"Final Risk: {proba_value:.0f}%",
        ha="right",
        va="center",
        transform=text_trans,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
        ),
        fontsize=12,
    )

    ax.tick_params(
        axis="both",
        which="major",
        pad=1,
        length=2,
        width=0.5,
        bottom=True,
        left=True,
    )

    ax.grid(True, axis="y")
    ax.xaxis.label.set_size(13.5)
    ax.tick_params(axis='y', labelsize=13.5)
    ax.tick_params(axis='x', labelsize=11)
    ax.set_ylabel('')

    return ax


def list_of_array_to_df_with_melt(x, feature_names, value_name):
    return pd.concat([
        (
            pd.DataFrame(
                x_r,
                columns=feature_names
            )
            .assign(
                Point=lambda df: np.arange(df.shape[0])
            )
            .melt(
                id_vars=['Point'],
                var_name='Feature',
                value_name=value_name
            )
            .assign(run=r)
        )
        for r, x_r in enumerate(x)
    ])


def list_of_array_to_df(x, feature_name):
    return pd.concat([
        (
            pd.DataFrame(
                x_r,
                columns=[feature_name]
            )
            .assign(
                Point=lambda df: np.arange(df.shape[0])
            )
            .assign(run=r)
        )
        for r, x_r in enumerate(x)
    ])

# Function to clean and tokenize using spaCy
def clean_and_tokenize_spacy(transcript, lang='en'):
    # Load the spaCy English and Spanish models
    nlp_en = spacy.load("en_core_web_sm")
    nlp_es = spacy.load("es_core_news_sm")
    # Select the appropriate spaCy model based on language
    if lang == 'es':
        doc = nlp_es(transcript)
    else:
        doc = nlp_en(transcript)
        # print(doc)

    # Extract tokens (words) excluding punctuation and whitespace
    words = [token.text.lower() for token in doc if token.is_alpha]
    return words, doc


# Function to count propositions (for Propositional Idea Density)
def calculate_pid(text):
    # Tokenize the text into words and tag with part-of-speech (POS) tags
    if isinstance(text, spacy.tokens.doc.Doc):
        text = text.text  # Extract text from spacy Doc object
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    
    # Set of POS tags that match the definition (verbs, adjectives, adverbs, prepositions, conjunctions)
    proposition_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', # Verbs
                        'JJ', 'JJR', 'JJS',                       # Adjectives
                        'RB', 'RBR', 'RBS',                       # Adverbs
                        'IN',                                     # Prepositions
                        'CC'}                                     # Conjunctions
    
    # Count number of propositions (verbs, adjectives, adverbs, prepositions, conjunctions)
    propositions = [word for word, tag in pos_tags if tag in proposition_tags]
    
    # Calculate PID (propositions / total words)
    pid = len(propositions) / len(words) if len(words) > 0 else 0
    
    return pid


# Function to calculate proportion of consecutive duplicate words using spaCy
def calculate_consecutive_duplicates_spacy(words):
    duplicate_count = 0
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            duplicate_count += 1
    return duplicate_count / len(words) if len(words) > 0 else 0


# Function to calculate linguistic features including POS rates and other metrics
def calculate_ling_nlp(transcript, lang, alpha=0.165):
    from collections import Counter
    import math
    # Clean and tokenize the transcript using spaCy
    words, doc = clean_and_tokenize_spacy(transcript, lang)

    # Total number of words (tokens)
    N = len(words)
    
    # Frequency of each unique word
    word_freq = Counter(words)
    
    # Total number of unique words (types)
    V = len(word_freq)
    
    # Number of hapax legomena (words that occur only once)
    V1 = sum(1 for count in word_freq.values() if count == 1)

    # Calculate Brunet's Index
    brunet_index = N ** (V ** (-alpha)) if N > 0 and V > 0 else 0
    brunet_index = round(brunet_index, 2)
    
    # Calculate Honoré's Statistic
    honors_statistic = (100 * math.log(N)) / (1 - (V1 / V)) if V > V1 and N > 0 else None
    honors_statistic = round(honors_statistic, 2) if honors_statistic else None
    
    # Calculate Type-Token Ratio (TTR) corrected for text length (CTTR)
    cttr = V / math.sqrt(2 * N) if N > 0 else 0
    cttr = round(cttr, 2)

    # Calculate Propositional Idea Density (PID)
    pid = calculate_pid(doc)
    pid = round(pid, 2)
    
    # Calculate proportion of consecutive duplicate words
    duplicate_proportion = calculate_consecutive_duplicates_spacy(words)
    duplicate_proportion = round(duplicate_proportion, 2)

    return brunet_index, honors_statistic, cttr, pid, duplicate_proportion


def text_analysis_features(p, lang='en'):
    import glob
    all_files = sorted(glob.glob(os.path.join(p, '*.txt')))
    total_files = len(all_files)

    # Initialize an empty list to store results
    results = []

    print(f'Processing {total_files} files')
    # Loop over all files in the directory.
    for i, f in enumerate(all_files, 1):
        # print(f'Processing {i}/{total_files}')
        if f.endswith(".txt"): 
            with open(f, 'r', encoding='ISO-8859-1') as file:
                content = file.read()
            # Extract the file id from the filename.
            file_id = f.rsplit('.', 1)[0][-3:]
            brunet_index, honors_statistic, cttr, pid, duplicate_proportion = calculate_ling_nlp(content, lang=lang)
            # Append results to the list
            results.append({
                'PID': file_id,
                'Brunet': brunet_index,
                'Honore': honors_statistic,
                'CTTR': cttr,
                'PIDensity': pid,
                'Duplic': duplicate_proportion  
            })

    # Convert the results into a DataFrame
    df = pd.DataFrame(results)
    return df

def text_analysis_features_UPDATED_PID(p, lang='en'):
    import glob
    all_files = sorted(glob.glob(os.path.join(p, '*.txt')))
    total_files = len(all_files)

    # Initialize an empty list to store results
    results = []

    print(f'Processing {total_files} files')
    # Loop over all files in the directory.
    for i, f in enumerate(all_files, 1):
        # print(f'Processing {i}/{total_files}')
        if f.endswith(".txt"): 
            with open(f, 'r', encoding='ISO-8859-1') as file:
                content = file.read()
            # Extract the file id from the filename.
            file_id = f.rsplit('.', 1)[0][-3:]
            brunet_index, honors_statistic, cttr, pid, duplicate_proportion = calculate_ling_nlp(content, lang=lang)
            # Append results to the list
            results.append({
                'PID': file_id,
                'Brunet': brunet_index,
                'Honore': honors_statistic,
                'CTTR': cttr,
                'PIDensity': depid,
                'Duplic': duplicate_proportion  
            })

    # Convert the results into a DataFrame
    df = pd.DataFrame(results)
    return df


def calc_spec(y_true, y_pred):
        """
        Calculate specificity.
        """
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        return tn / (tn + fp) if (tn + fp) != 0 else 0

def calc_strat_stats(y_true, y_proba, resolution=0.05, ignore_amber=False):
    """
    Calculate statistics based on stratified risk levels and find optimal thresholds.

    Args:
    - y_true: true labels (0 or 1).
    - y_proba: Predicted probabilities for the positive class from 10fCV
    - resolution: Step size for threshold search.
    - ignore_amber: If True, only consider Green and Red categories. In this case, the probability will be used for ROC AUC instead of the predicted labels.

    Returns:
    - DataFrame with columns: green, amber, sens, spec, weighted_score.
    """

    # Define thresholds
    green_thresholds = np.arange(0.05, 0.55, resolution)
    amber_thresholds = 1 - green_thresholds  # Inverse relation for simplification

    results = []
    total_samples = len(y_proba)
    for green in green_thresholds:
        for amber in amber_thresholds:
            # Ensure proper threshold logic
            if green < amber:
                # Stratify based on thresholds
                stratified = np.select(
                    [y_proba <= green, y_proba <= amber],
                    ['Green', 'Amber'],
                    default='Red'
                )

                # Calculate the proportion of Amber in the dataset
                proportion_amber = np.sum(stratified == 'Amber') / total_samples

                if ignore_amber:
                    # only consider Green and Red
                    green_and_red_idx = (stratified == 'Green') | (stratified == 'Red')
                    stratified, y_proba_i, y_true_i = map(
                        lambda x: x[green_and_red_idx],
                        [stratified, y_proba, y_true]
                    )

                    if len(y_true_i) == 0:
                        continue
                    
                    if len(np.unique(y_true_i)) < 2:
                        continue

                    y_pred_i = np.where(stratified == 'Red', 1, 0)  # Consider Red as positive
                    roc_auc = skmetrics.roc_auc_score(y_true_i, y_proba_i)

                else:
                    y_true_i = y_true
                    y_pred_i = np.where(stratified == 'Red', 1, 0)  # Consider Red as positive
                    roc_auc = skmetrics.roc_auc_score(y_true_i, y_pred_i)


                # Calculate metrics
                sensitivity = skmetrics.recall_score(y_true_i, y_pred_i)
                specificity = calc_spec(y_true_i, y_pred_i)
                youdens = sensitivity + specificity - 1
                # Calculate ROC AUC for this stratification

                
                # Compile results
                results.append({
                    'green': green,
                    'amber': amber,
                    'sens': sensitivity,
                    'spec': specificity,
                    'J': youdens,
                    'roc_auc': roc_auc,
                    'proportion_amber': proportion_amber
                })

    # Convert to DataFrame and return
    results_df = pd.DataFrame(results)

    return results_df.sort_values(by='J', ascending=False)


def calculate_metrics_probabilities(true, pred):
    rec = recall_score(true, pred)
    acc = accuracy_score(true, pred)
    cm = confusion_matrix(true, pred)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if TP + FN != 0 else 0
    specificity = TN / (TN + FP) if TN + FP != 0 else 0
    j = sensitivity + specificity - 1
    return sensitivity, specificity, acc


def calc_metrics_folds_bootstrap_probabilities(proba, y_true, green_thresh, amber_thresh):
    accuracies = []
    recalls = []
    specs = []
    aucs = []
    
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
        sens, spec, acc = calculate_metrics_probabilities(y_true_filtered, y_pred_filtered)
        
        # Append the results
        recalls.append(sens)
        specs.append(spec)
        aucs.append(auc)
        accuracies.append(acc)
    
    return recalls, specs, aucs, accuracies


def probs_split(probs_bestmodel, bestmodel, groups, split):
    
    # probs_bestmodel = probs[f'{bestmodel}']

    probability_values = [
        np.array(
            probs[np.arange(len(probs))] 
        ) for probs in probs_bestmodel
        ]
    
    groups['mapping'] = groups['mapping'].str.replace('adrsdt', '')
    groups.rename(columns={'mapping': 'pid'}, inplace=True)
    groups = groups.sort_values('pid').reset_index(drop=True)
    # print('Columns:', groups.columns)

    split_values = groups[split].values
    pid_values = groups.astype({'pid': int}).pid.values
    
    results = []
    for run, probs in enumerate(probability_values):
        results.append(
            pd.DataFrame(
                {   
                    'pid': pid_values,
                    split: split_values,
                    'probs': probs[:, 1],
                    'run': run
                }
            )
        )
    results = pd.concat(results)
    results = results.dropna(subset=[split])
    results[split] = results[split]
    results = results.reset_index(drop=True)
    return results

# Define a function to calculate TP, TN, FP, FN
def calculate_confusion_items(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp, tn, fp, fn


def calc_items(threshold, probs, y_test, idx):
    # Predicted labels based on threshold
    y_pred = (np.array(probs) > threshold).astype(int)
    # Calculate confusion matrix items for Females
    tp, tn, fp, fn = calculate_confusion_items(y_test[idx], y_pred[idx])
    return tp, tn, fp, fn