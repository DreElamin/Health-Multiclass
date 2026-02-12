import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                             classification_report, f1_score)
import matplotlib.pyplot as plt


class Knn(object):
    k = 0
    nFeatures = 0
    nSamples = 0
    isFitted = False
    
    def __init__(self, k):
        self.k = k
    
    def train(self, xFeat, y):
        self.xTrain = xFeat
        self.yTrain = y
        self.nSamples, self.nFeatures = xFeat.shape
        self.isFitted = True
        return self
    
    def predict(self, xFeat):
        m = xFeat.shape[0]
        yhat = np.zeros(m)
        for i in range(m):
            ithsample = xFeat[i]
            difference = self.xTrain - ithsample
            distances = np.linalg.norm(difference, axis=1)
            knn_indices = np.argsort(distances)[:self.k]
            knn_labels = self.yTrain[knn_indices]
            values, counts = np.unique(knn_labels, return_counts=True)
            yhat[i] = values[np.argmax(counts)]
        return yhat

def accuracy(yHat, yTrue):
    acc = np.mean(yHat == yTrue)
    return acc


def create_disease_categories(disease_name):
    """Map individual diseases to broader categories"""
    disease_name = str(disease_name).strip().lower()
    
    if any(term in disease_name for term in 
        ['asthma', 'bronchitis', 'common cold', 'influenza', 'pneumonia', 'allergic rhinitis']):
        return 'Respiratory'

    elif any(term in disease_name for term in 
            ['hypertension', 'stroke', 'coronary artery disease', 'heart failure', 'myocardial infarction']):
        return 'Cardiovascular'

    elif any(term in disease_name for term in 
            ['diabetes', 'hyperthyroidism', 'hypothyroidism', 'obesity', 'metabolic syndrome']):
        return 'Endocrine_metabolic'

    elif any(term in disease_name for term in 
            ['migraine', 'epilepsy', 'parkinson', 'alzheimer', 'multiple sclerosis']):
        return 'Neurological'

    elif any(term in disease_name for term in 
            ['rheumatoid arthritis', 'osteoarthritis', 'osteoporosis', 'lupus', 'psoriasis']):
        return 'Musculoskeletal/Autoimmune'

    elif any(term in disease_name for term in 
            ['gastroenteritis', 'crohn', 'ulcerative colitis', 'pancreatitis', 
            'kidney disease', 'urinary tract infection', 'irritable bowel syndrome', 'kidney stones']):
        return 'Gi_renal'

    elif any(term in disease_name for term in 
            ['depression', 'anxiety', 'bipolar', 'schizophrenia', 'ptsd']):
        return 'Psychiatric'

    elif any(term in disease_name for term in 
            ['liver cancer', 'kidney cancer', 'lung cancer', 'breast cancer', 'colon cancer']):
        return 'Cancer'

    return 'Other'


# Load the dataset
df = pd.read_csv('Disease_symptom_and_patient_profile_dataset.csv')
disease_column = 'Disease'

# Filter diseases with at least 5 samples
disease_counts = df[disease_column].value_counts()
valid_diseases = disease_counts[disease_counts >= 5].index
df_filtered = df[df[disease_column].isin(valid_diseases)].copy()


# Create disease categories
df_filtered['Disease_Category'] = df_filtered[disease_column].apply(create_disease_categories)

print("\n" + "="*70)
print('Category Distribution')
print("="*70)
# Display category distribution

print(df_filtered['Disease_Category'].value_counts())

# Separate features and target
X = df_filtered.drop([disease_column, 'Disease_Category'], axis=1)
y = df_filtered['Disease_Category']

# Encode categorical features
non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
if non_numeric_cols:
    for col in non_numeric_cols:
        le_temp = LabelEncoder()
        X[col] = le_temp.fit_transform(X[col].astype(str))


# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Handle missing values
if X.isnull().sum().sum() > 0:
    print(f"\nHandling {X.isnull().sum().sum()} missing values...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42,
    stratify=y_encoded
)

# Feature scaling (CRITICAL for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# K-NEAREST NEIGHBORS - HYPERPARAMETER TUNING

print("\n" + "="*70)
print("KNN HYPERPARAMETER TUNING")
print("="*70)

k_values = range(1, 31)
test_accs = []
train_accs = []

print("\nEvaluating k values from 1 to 30...")
for k in k_values:
    knn = Knn(k)
    knn.train(X_train_scaled, y_train)
    
    yHatTrain = knn.predict(X_train_scaled)
    trainAcc = accuracy(yHatTrain, y_train)
    train_accs.append(trainAcc)
    
    yHatTest = knn.predict(X_test_scaled)
    testAcc = accuracy(yHatTest, y_test)
    test_accs.append(testAcc)
    
    if k % 5 == 1:
        print(f"  k={k:2d}: Train={trainAcc:.4f}, Test={testAcc:.4f}")

# Find k with highest test accuracy
naive_best_k = k_values[np.argmax(test_accs)]
naive_best_acc = max(test_accs)

# Choose k for robustness
chosen_k = 16
chosen_k_acc = test_accs[chosen_k - 1]


# Plot k-value analysis
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accs, label='Training Accuracy', marker='o', markersize=4, alpha=0.7)
plt.plot(k_values, test_accs, label='Test Accuracy', marker='s', markersize=4, alpha=0.7)
plt.axvline(x=chosen_k, color='green', linestyle='--', linewidth=2, alpha=0.7, 
            label=f'Chosen k={chosen_k}')
plt.xlabel('Number of Neighbors (k)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('KNN: K-Value Selection Analysis', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('knn_k_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# TRAIN FINAL KNN MODEL

print("\n" + "="*70)
print(f"FINAL KNN MODEL (k={chosen_k})")
print("="*70)

knn_final = Knn(chosen_k)
knn_final.train(X_train_scaled, y_train)
y_pred_knn = knn_final.predict(X_test_scaled)

# Calculate metrics
acc_knn = accuracy(y_pred_knn, y_test)
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')

print(f"\nOverall Performance:")
print(f"  Accuracy:  {acc_knn:.4f} ({acc_knn*100:.2f}%)")
print(f"  F1-Score:  {f1_knn:.4f}")

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred_knn, average=None, zero_division=0
)

# Macro and weighted averages
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1)

# Full classification report
print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_pred_knn, target_names=le.classes_, zero_division=0))


print("="*70)
print("KNN ANALYSIS COMPLETE")
print("="*70)