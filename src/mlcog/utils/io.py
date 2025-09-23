import os
import pandas as pd
import pickle
import joblib
import numpy as np
from pathlib import Path
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from ..evaluation import splits_performace


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_cv_model_path(feature_set, abbr, reg=False):
    """Return the path to the saved cross-validation model."""
    prefix = "10fcv_reg_" if reg else "10fcv_"
    return PROJECT_ROOT / "data" / "cv_eval" / feature_set / f"{prefix}{abbr}.pkl"


def _load_best(feature_set, mapping, reg=False):
    """Load best models from standardized cv_eval directory."""
    best_models = {}

    for model_name, abbr in mapping.items():
        path = get_cv_model_path(feature_set, abbr, reg=reg)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            best_models[model_name] = joblib.load(path)

    return best_models


def load_best_params(model_map, feature_set, reg=False):
    """
    Load best models for classification using standardized cv model path.
    model_map_class = {
        'Logistic Regression': 'lr',
        'SVM': 'svm',
        'Random Forest': 'rf',
        'Neural Network': 'nn',
        'XGBoost': 'xgboost',
    }
    model_map_reg = {
        'Ridge': 'rr',
        'SVR': 'svr',
        'RFR': 'rfr',
        'MLP': 'mlp',
        'XGBoost': 'xgb',
    }
    """
    return _load_best(feature_set, mapping=model_map, reg=reg)


def get_data(feature_name, model_abbr, feature_abbr, probs_name):
    """Load features, labels, model, and predicted probabilities."""
    with open(f"./FEATURES/dfs/{feature_name}.pkl", "rb") as f:
        df_train = pickle.load(f)
    with open(f"./FEATURES/test-features/dfs/{feature_name}.pkl", "rb") as f:
        df_test = pickle.load(f)

    X_train = np.stack(df_train["data"].values)
    y_train = df_train["label"]
    X_test = np.stack(df_test["data"].values)
    y_test = df_test["label"]

    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)

    model_path = os.path.join("./hyperparam-tuning", feature_abbr, f"10fcv_{model_abbr}.pkl")
    best_model = joblib.load(model_path) if os.path.exists(model_path) else None

    with open(f"./evaluation-test/predict_proba/{probs_name}.pkl", "rb") as f:
        proba = pickle.load(f)

    return X_scaled_train, y_train, X_scaled_test, y_test, best_model, proba


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


def groups(y_test, proba, idx):
    mean_acc, lower_ci, upper_ci, mean_lik, lower_lik, upper_lik = splits_performace(y_test, proba, idx)
    acc_str = f"Mean Accuracy: {mean_acc} ({lower_ci} - {upper_ci})"
    lik_str = f"Mean Likelihood of positive: {mean_lik} ({lower_lik} - {upper_lik})"
    return acc_str, lik_str
