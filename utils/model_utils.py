"""
Utility functions for loading data, training models, and making predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42


def map_disease_to_category(disease: str) -> str:
    """
    Map a specific disease name to a broader disease category.
    """
    respiratory = {
        "Asthma", "Bronchitis", "Common Cold", "Influenza", "Pneumonia",
        "Allergic Rhinitis"
    }
    cardiovascular = {
        "Hypertension", "Stroke", "Coronary Artery Disease", "Heart Failure",
        "Myocardial Infarction"
    }
    endocrine_metabolic = {
        "Diabetes", "Hyperthyroidism", "Hypothyroidism", "Obesity",
        "Metabolic Syndrome"
    }
    gi_renal = {
        "Gastroenteritis", "Crohn's Disease", "Ulcerative Colitis",
        "Pancreatitis", "Kidney Disease", "Urinary Tract Infection",
        "Irritable Bowel Syndrome", "Kidney Stones"
    }
    musc_autoimmune = {
        "Rheumatoid Arthritis", "Osteoarthritis", "Osteoporosis", "Lupus",
        "Psoriasis", "Multiple Sclerosis"
    }
    neurological = {
        "Migraine", "Epilepsy", "Parkinson's Disease", "Alzheimer's Disease",
        "Multiple Sclerosis"
    }
    psychiatric = {
        "Depression", "Anxiety Disorders", "Bipolar Disorder",
        "Schizophrenia", "PTSD"
    }
    cancer = {
        "Liver Cancer", "Kidney Cancer", "Lung Cancer", "Breast Cancer",
        "Colon Cancer"
    }

    if disease in respiratory:
        return "Respiratory"
    if disease in cardiovascular:
        return "Cardiovascular"
    if disease in endocrine_metabolic:
        return "Endocrine/Metabolic"
    if disease in gi_renal:
        return "GI/Renal"
    if disease in musc_autoimmune:
        return "Musculoskeletal/Autoimmune"
    if disease in neurological:
        return "Neurological"
    if disease in psychiatric:
        return "Psychiatric"
    if disease in cancer:
        return "Cancer"

    return "Other"


def load_and_prepare_data():
    """
    Load and prepare the dataset for modeling.
    Returns a dictionary with all necessary data components.
    """
    # Find the CSV file
    csv_path = Path("/home/user/Health-Multiclass/Disease_symptom_and_patient_profile_dataset 2.csv")

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    # Load data
    df = pd.read_csv(csv_path)

    # Create disease categories
    df["DiseaseCategory"] = df["Disease"].apply(map_disease_to_category)

    # Merge small categories into 'Other'
    MIN_CAT_COUNT = 20
    cat_counts = df["DiseaseCategory"].value_counts()
    small_cats = cat_counts[cat_counts < MIN_CAT_COUNT].index.tolist()
    df["DiseaseCategory"] = df["DiseaseCategory"].apply(
        lambda c: "Other" if c in small_cats else c
    )

    # Define features
    feature_cols = [
        "Fever",
        "Cough",
        "Fatigue",
        "Difficulty Breathing",
        "Age",
        "Gender",
        "Blood Pressure",
        "Cholesterol Level"
    ]

    X = df[feature_cols].copy()
    y = df["DiseaseCategory"].copy()

    # Feature engineering
    X["SymptomCount"] = (
        (X["Fever"] == "Yes").astype(int)
        + (X["Cough"] == "Yes").astype(int)
        + (X["Fatigue"] == "Yes").astype(int)
        + (X["Difficulty Breathing"] == "Yes").astype(int)
    )
    X["IsElderly"] = (X["Age"] >= 60).astype(int)
    X["MetabolicRisk"] = (
        ((X["Blood Pressure"] == "High") | (X["Cholesterol Level"] == "High"))
    ).astype(int)

    numeric_features = ["Age", "SymptomCount", "IsElderly", "MetabolicRisk"]
    categorical_features = [
        "Fever",
        "Cough",
        "Fatigue",
        "Difficulty Breathing",
        "Gender",
        "Blood Pressure",
        "Cholesterol Level"
    ]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    # Fit preprocessor
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Apply SMOTE
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

    return {
        'df': df,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_proc': X_train_proc,
        'X_test_proc': X_test_proc,
        'X_train_bal': X_train_bal,
        'y_train_bal': y_train_bal,
        'preprocessor': preprocessor,
        'classes': sorted(y.unique()),
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }


def train_all_models(X_train, y_train, preprocessor):
    """
    Train all models and return them in a dictionary.
    """
    # Apply preprocessing and SMOTE
    X_train_proc = preprocessor.transform(X_train)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

    models = {}

    # 1. Logistic Regression
    log_reg = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    log_reg.fit(X_train_bal, y_train_bal)
    models['logistic_regression'] = log_reg

    # 2. Decision Tree with GridSearch
    dt_param_grid = {
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"]
    }

    dt_grid = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=dt_param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    dt_grid.fit(X_train_bal, y_train_bal)
    models['decision_tree'] = dt_grid.best_estimator_

    # 3. Neural Network
    mlp = MLPClassifier(
        hidden_layer_sizes=(25, 10),
        activation='tanh',
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=RANDOM_STATE
    )
    mlp.fit(X_train_bal, y_train_bal)
    models['neural_network'] = mlp

    # 4. K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=16)
    knn.fit(X_train_bal, y_train_bal)
    models['knn'] = knn

    return models


def get_model_predictions(patient_data, session_state):
    """
    Get predictions from all models for given patient data.
    Returns a dictionary with predictions from each model.
    """
    # Add engineered features
    patient_data["SymptomCount"] = (
        (patient_data["Fever"] == "Yes").astype(int)
        + (patient_data["Cough"] == "Yes").astype(int)
        + (patient_data["Fatigue"] == "Yes").astype(int)
        + (patient_data["Difficulty Breathing"] == "Yes").astype(int)
    )
    patient_data["IsElderly"] = (patient_data["Age"] >= 60).astype(int)
    patient_data["MetabolicRisk"] = (
        ((patient_data["Blood Pressure"] == "High") |
         (patient_data["Cholesterol Level"] == "High"))
    ).astype(int)

    # Preprocess
    patient_proc = session_state['preprocessor'].transform(patient_data)

    predictions = {}

    # Get predictions from each model
    models = {
        'Logistic Regression': session_state.get('logistic_regression'),
        'Decision Tree': session_state.get('decision_tree'),
        'Neural Network': session_state.get('neural_network'),
        'K-Nearest Neighbors': session_state.get('knn')
    }

    for name, model in models.items():
        if model is not None:
            probs = model.predict_proba(patient_proc)[0]
            classes = model.classes_

            # Get top 3
            top3_idx = np.argsort(probs)[::-1][:3]

            predictions[name] = {
                'top1': classes[top3_idx[0]],
                'top3_labels': [classes[i] for i in top3_idx],
                'top3_probs': [probs[i] for i in top3_idx]
            }

    return predictions


def top_k_accuracy(probs, y_true, classes, k=3):
    """Calculate top-k accuracy."""
    correct = 0
    for i, true_label in enumerate(y_true):
        topk_idx = np.argsort(probs[i])[::-1][:k]
        topk_labels = classes[topk_idx]
        if true_label in topk_labels:
            correct += 1
    return correct / len(y_true)


def calculate_metrics(session_state):
    """
    Calculate comprehensive metrics for all models.
    """
    X_test_proc = session_state['X_test_proc']
    y_test = session_state['y_test']

    models = {
        'Logistic Regression': session_state.get('logistic_regression'),
        'Decision Tree': session_state.get('decision_tree'),
        'Neural Network': session_state.get('neural_network'),
        'K-Nearest Neighbors': session_state.get('knn')
    }

    metrics_dict = {}

    for name, model in models.items():
        if model is not None:
            # Predictions
            y_pred = model.predict(X_test_proc)
            probs = model.predict_proba(X_test_proc)
            classes = model.classes_

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            top3_acc = top_k_accuracy(probs, y_test.to_numpy(), classes, k=3)

            # F1 scores
            weighted_f1 = f1_score(y_test, y_pred, average='weighted')
            macro_f1 = f1_score(y_test, y_pred, average='macro')

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=classes)

            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            metrics_dict[name] = {
                'accuracy': accuracy,
                'top3_accuracy': top3_acc,
                'weighted_f1': weighted_f1,
                'macro_f1': macro_f1,
                'confusion_matrix': cm,
                'classification_report': {k: v for k, v in report.items() if isinstance(v, dict)},
                'y_pred': y_pred
            }

    return metrics_dict
