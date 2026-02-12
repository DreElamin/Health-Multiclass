import kagglehub
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ============== 1. Download and load data ==============

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


# ============== 2. Map Disease -> DiseaseCategory ==============

def map_disease_to_category(name: str) -> str:
    """
    Map a specific disease string to a broader disease category.
    This is a simple rule based mapping using substrings.
    """
    n = name.lower()

    # Respiratory diseases
    if any(s in n for s in [
        "asthma", "bronchitis", "pneumonia", "cold", "influenza", "flu",
        "rhinitis", "respiratory"
    ]):
        return "Respiratory"

    # Cardiovascular diseases
    if any(s in n for s in [
        "hypertension", "coronary artery", "stroke", "heart failure",
        "myocardial", "cardio"
    ]):
        return "Cardiovascular"

    # Endocrine / metabolic
    if any(s in n for s in [
        "diabetes", "hypothyroidism", "hyperthyroidism", "thyroid",
        "hypoglycemia", "metabolic", "cholesterol"
    ]):
        return "Endocrine/Metabolic"

    # Neurological and neurodevelopmental
    if any(s in n for s in [
        "parkinson", "alzheimer", "multiple sclerosis", "migraine",
        "epilepsy", "seizure", "autism", "neuropathy", "sclerosis"
    ]):
        return "Neurological"

    # Musculoskeletal / autoimmune / rheumatologic / skin
    if any(s in n for s in [
        "arthritis", "rheumatoid", "osteoporosis", "osteoarthritis",
        "fibromyalgia", "psoriasis", "lupus", "spondylitis"
    ]):
        return "Musculoskeletal/Autoimmune"

    # Gastrointestinal / liver / pancreas / kidney / urinary
    if any(s in n for s in [
        "crohn", "ulcerative colitis", "colitis", "gastro", "pancreatitis",
        "liver", "hepatitis", "kidney", "renal", "urinary tract", "uti",
        "bowel", "stomach"
    ]):
        return "GI/Renal"

    # Psychiatric
    if any(s in n for s in [
        "depression", "anxiety", "bipolar", "schizophrenia", "eating disorder"
    ]):
        return "Psychiatric"

    # Cancer (catch any other cancers we did not catch above)
    if "cancer" in n or "carcinoma" in n or "leukemia" in n or "lymphoma" in n:
        return "Cancer"

    # Fallback category
    return "Other"


df["DiseaseCategory"] = df["Disease"].apply(map_disease_to_category)

print("\n=== Disease Categories ===")
print(df["DiseaseCategory"].value_counts())

# Optionally, filter categories with very few samples (for example < 10)
min_count = 10
category_counts = df["DiseaseCategory"].value_counts()
keep_categories = category_counts[category_counts >= min_count].index

df = df[df["DiseaseCategory"].isin(keep_categories)].copy()

print("\nAfter filtering rare categories (min_count = 10):")
print(df["DiseaseCategory"].value_counts())


# ============== 3. Define features and target ==============

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

X = df[feature_cols]
y = df[target]


# ============== 4. Train test split ==============

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y  # keep category distribution similar in train and test
)

print("\nTrain size:", X_train.shape[0], "Test size:", X_test.shape[0])
print("Category distribution in train:")
print(y_train.value_counts())
print("\nCategory distribution in test:")
print(y_test.value_counts())


# ============== 5. Preprocessing ==============

numeric_features = ["Age"]
categorical_features = [
    "Fever",
    "Cough",
    "Fatigue",
    "Difficulty Breathing",
    "Gender",
    "Blood Pressure",
    "Cholesterol Level"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)


# ============== 6. Logistic regression on categories ==============

log_reg = LogisticRegression(
    multi_class="multinomial",  # multiclass over categories
    solver="lbfgs",
    max_iter=1000
)

clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", log_reg)
])

print("\nTraining logistic regression model to predict DiseaseCategory...")
clf.fit(X_train, y_train)


# ============== 7. Evaluation ==============

y_pred = clf.predict(X_test)

print("\n=== Model Performance on Disease Categories ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---- Top-k accuracy (k=1 and k=3) ----

def top_k_accuracy(probs, y_true, classes, k=3):
    correct = 0
    for i, true_label in enumerate(y_true):
        topk_idx = np.argsort(probs[i])[::-1][:k]
        topk_labels = classes[topk_idx]
        if true_label in topk_labels:
            correct += 1
    return correct / len(y_true)

probs_test = clf.predict_proba(X_test)
classes = clf.named_steps["model"].classes_

top1 = top_k_accuracy(probs_test, y_test.to_numpy(), classes, k=1)
top3 = top_k_accuracy(probs_test, y_test.to_numpy(), classes, k=3)

print(f"Top-1 Accuracy (standard): {top1:.3f}")
print(f"Top-3 Accuracy: {top3:.3f}")


# ============== 8. Example “WebMD style” prediction ==============

print("\nDisease categories learned by model:", classes)

example_input = pd.DataFrame([{
    "Fever": "Yes",
    "Cough": "No",
    "Fatigue": "Yes",
    "Difficulty Breathing": "No",
    "Age": 35,
    "Gender": "Male",
    "Blood Pressure": "Normal",
    "Cholesterol Level": "High"
}])

example_probs = clf.predict_proba(example_input)[0]

# Show top-3 predicted categories
top3_idx = np.argsort(example_probs)[::-1][:3]

print("\n=== Example: Top 3 Predicted Categories ===")
for rank, idx in enumerate(top3_idx, start=1):
    print(f"{rank}. {classes[idx]}  (probability = {example_probs[idx]:.4f})")

print("\nMost likely category:", classes[top3_idx[0]])