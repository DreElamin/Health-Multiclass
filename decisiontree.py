import kagglehub
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score

from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# --------------------------
# 0. Helper: map Disease -> Category
# --------------------------

def map_disease_to_category(disease: str) -> str:
    """
    Map a specific disease name to a broader disease category.
    Anything not explicitly listed falls into 'Other'.
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


# --------------------------
# 1. Load dataset
# --------------------------

print("Downloading dataset...")
path = kagglehub.dataset_download(
    "uom190346a/disease-symptoms-and-patient-profile-dataset"
)
print("Dataset downloaded to:", path)

csv_path = path + "/Disease_symptom_and_patient_profile_dataset.csv"
df = pd.read_csv(csv_path)

print("\n=== Raw Data Loaded ===")
print(df.head())
print("\nOriginal disease counts (top 20):")
print(df["Disease"].value_counts().head(20))

# --------------------------
# 2. Create DiseaseCategory and merge small categories
# --------------------------

df["DiseaseCategory"] = df["Disease"].apply(map_disease_to_category)

print("\n=== Disease Categories (raw) ===")
print(df["DiseaseCategory"].value_counts())

# Merge very small categories into 'Other'
MIN_CAT_COUNT = 20  # you can adjust this for more or fewer categories
cat_counts = df["DiseaseCategory"].value_counts()

small_cats = cat_counts[cat_counts < MIN_CAT_COUNT].index.tolist()
df["DiseaseCategory"] = df["DiseaseCategory"].apply(
    lambda c: "Other" if c in small_cats else c
)

print(f"\nAfter merging categories with < {MIN_CAT_COUNT} samples into 'Other':")
print(df["DiseaseCategory"].value_counts())

# --------------------------
# 3. Define features and target
# --------------------------

target = "DiseaseCategory"

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
y = df[target].copy()

# Simple feature engineering
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

# --------------------------
# 4. Train-test split
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTrain size: {len(X_train)} Test size: {len(X_test)}")
print("Category distribution in train:")
print(Counter(y_train))
print("\nCategory distribution in test:")
print(Counter(y_test))

# --------------------------
# 5. Preprocess + SMOTE + GridSearch
# --------------------------

# Preprocessing for features
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

# Fit preprocessor on train and transform
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Apply SMOTE to balance classes in the training set
print("\nApplying SMOTE to balance classes...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

print("Balanced class counts (train after SMOTE):")
print(Counter(y_train_bal))

# Base Decision Tree
base_dt = DecisionTreeClassifier(random_state=RANDOM_STATE)

# Hyperparameter grid for Decision Tree
param_grid = {
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "criterion": ["gini", "entropy"]
}

print("\nRunning GridSearchCV for Decision Tree...")
grid = GridSearchCV(
    estimator=base_dt,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X_train_bal, y_train_bal)

print("\nBest Decision Tree params:")
print(grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

best_dt = grid.best_estimator_

# --------------------------
# 6. Evaluate on test set
# --------------------------

y_pred = best_dt.predict(X_test_proc)
acc = accuracy_score(y_test, y_pred)

print("\n=== Decision Tree Performance on Disease Categories (with SMOTE + tuning) ===")
print("Accuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# --------------------------
# 7. Top-1 and Top-3 accuracy
# --------------------------

probs = best_dt.predict_proba(X_test_proc)
classes = best_dt.classes_

def top_k_accuracy(probs, y_true, classes, k=3):
    correct = 0
    for i, true_label in enumerate(y_true):
        topk_idx = np.argsort(probs[i])[::-1][:k]
        topk_labels = classes[topk_idx]
        if true_label in topk_labels:
            correct += 1
    return correct / len(y_true)

top1 = top_k_accuracy(probs, y_test.to_numpy(), classes, k=1)
top3 = top_k_accuracy(probs, y_test.to_numpy(), classes, k=3)

print(f"\nTop-1 accuracy: {top1:.3f}")
print(f"Top-3 accuracy: {top3:.3f}")

# --------------------------
# 8. Example WebMD-style prediction (Top 3 categories)
# --------------------------

example_input = pd.DataFrame([{
    "Fever": "Yes",
    "Cough": "No",
    "Fatigue": "Yes",
    "Difficulty Breathing": "No",
    "Age": 35,
    "Gender": "Male",
    "Blood Pressure": "High",
    "Cholesterol Level": "High"
}])

# Add engineered features to example
example_input["SymptomCount"] = (
    (example_input["Fever"] == "Yes").astype(int)
    + (example_input["Cough"] == "Yes").astype(int)
    + (example_input["Fatigue"] == "Yes").astype(int)
    + (example_input["Difficulty Breathing"] == "Yes").astype(int)
)
example_input["IsElderly"] = (example_input["Age"] >= 60).astype(int)
example_input["MetabolicRisk"] = (
    ((example_input["Blood Pressure"] == "High") |
     (example_input["Cholesterol Level"] == "High"))
).astype(int)

example_proc = preprocessor.transform(example_input)
example_probs = best_dt.predict_proba(example_proc)[0]

print("\nDisease categories learned by model:", classes)

# sort probabilities and show top 3
top3_idx = np.argsort(example_probs)[::-1][:3]

print("\n=== Example Prediction: Top 3 Disease Categories ===")
for rank, idx in enumerate(top3_idx, start=1):
    print(f"{rank}. {classes[idx]}  (probability = {example_probs[idx]:.3f})")

print("\nMost likely category:", classes[top3_idx[0]])

import matplotlib.pyplot as plt
from sklearn import tree

plt.figure(figsize=(14, 8))
tree.plot_tree(
    best_dt,
    feature_names=preprocessor.get_feature_names_out(),
    class_names=best_dt.classes_,
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=3        # only show first 3 levels
)
plt.tight_layout()
plt.show()