from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, LeaveOneGroupOut, cross_val_score, KFold, train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import loguniform, sem, t
import joblib
import pandas as pd
import numpy as np
import os 
import torch
import xgboost as xgb
from scipy.stats import uniform, randint
import pickle
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns
from ..src.mlcog.utils import plotting
import nltk
import spacy
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from sklearn import metrics as skmetrics
from ideadensity import depid


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
# functions for plots


def extract_results_classif_test(df):
    recall_result = f"{round(df['Recall Mean'].values[0], 2)} ({round(df['Recall Lower CI'].values[0], 2)} - {round(df['Recall Upper CI'].values[0], 2)})"
    spec_result = f"{round(df['Specificity Mean'].values[0], 2)} ({round(df['Specificity Lower CI'].values[0], 2)} - {round(df['Specificity Upper CI'].values[0], 2)})"
    auc_result = f"{round(df['ROC-AUC Mean'].values[0], 2)} ({round(df['ROC-AUC Lower CI'].values[0], 2)} - {round(df['ROC-AUC Upper CI'].values[0], 2)})"
    acc_result = f"{round(df['Accuracy Mean'].values[0], 2)} ({round(df['Accuracy Lower CI'].values[0], 2)} - {round(df['Accuracy Upper CI'].values[0], 2)})"
    return recall_result, spec_result, auc_result, acc_result

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer, roc_auc_score, mean_squared_error, mean_absolute_error
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


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import transforms
import typing

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

def calculate_depid(text):
    text = "This is a test of DEPID-R. This is a test of DEPID-R"
    density, word_count, dependencies = depid(text, is_depid_r=True)
    return density

# Function to calculate proportion of consecutive duplicate words using spaCy
def calculate_consecutive_duplicates_spacy(words):
    duplicate_count = 0
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            duplicate_count += 1
    return duplicate_count / len(words) if len(words) > 0 else 0

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

    depid = calculate_depid(doc)
    depid = round(depid, 5)
    
    # Calculate proportion of consecutive duplicate words
    duplicate_proportion = calculate_consecutive_duplicates_spacy(words)
    duplicate_proportion = round(duplicate_proportion, 2)

    return brunet_index, honors_statistic, cttr, depid, duplicate_proportion


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
            brunet_index, honors_statistic, cttr, depid, duplicate_proportion = calculate_ling_nlp(content, lang=lang)
            # Append results to the list
            results.append({
                'PID': file_id,
                'Brunet': brunet_index,
                'Honore': honors_statistic,
                'CTTR': cttr,
                'DEPID': depid,
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


def calculate_metrics_probabilities(true, pred):
    rec = recall_score(true, pred)
    acc = accuracy_score(true, pred)
    cm = confusion_matrix(true, pred)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if TP + FN != 0 else 0
    specificity = TN / (TN + FP) if TN + FP != 0 else 0
    j = sensitivity + specificity - 1
    return sensitivity, specificity, acc





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