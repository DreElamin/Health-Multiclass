# Multi-Class Health Condition Classification

A machine learning project that classifies patient health conditions into disease categories based on self-reported symptoms and demographic data — inspired by real-world symptom checkers like WebMD.

**Authors:** James, Elamin, Jonathan — CS334 Final Project

---

## Overview

This project investigates whether classical supervised learning models can approximate a simplified symptom-based disease screening system. Using a publicly available dataset of patient symptoms, demographics, and diagnoses, we frame the task as a **multi-class classification problem** and evaluate models at both the fine-grained disease level and the broader disease category level.

Key challenges addressed:
- Limited sample size (~350 records)
- Severe class imbalance (100+ original disease labels)
- Overlapping symptom profiles across conditions
- Noisy, binary/categorical symptom features

---

## Models Implemented

| Model | File | Top-1 Accuracy | Top-3 Accuracy |
|---|---|---|---|
| Logistic Regression | `logistic_regression.py` | 32.9% | 62.9% |
| Decision Tree | `decision_tree.py` | 42.9% | 70.0% |
| Neural Network (MLP) | `neural_network.py` | — | — |
| K-Nearest Neighbors | `knn.py` | 33.3% | — |

---

## Dataset

- **Source:** Kaggle (synthetic patient health records)
- **Size:** ~350 patient records
- **Features:** Age, Gender, Fever, Cough, Fatigue, Difficulty Breathing, Blood Pressure, Cholesterol Level
- **Original Labels:** 100+ disease diagnoses (highly imbalanced)
- **Grouped Labels:** Diseases were mapped into 9 broader categories for practical classification:
  - Respiratory
  - Cardiovascular
  - Gastrointestinal/Renal
  - Neurological
  - Endocrine/Metabolic
  - Musculoskeletal/Autoimmune
  - Psychiatric
  - Cancer
  - Other

---

## Methods

### Preprocessing
- Binary/categorical encoding of symptom features
- Stratified train/test split to preserve class proportions
- **SMOTE** (Synthetic Minority Oversampling Technique) to address class imbalance
- **StandardScaler** normalization (used for KNN and Neural Network)

### Logistic Regression
Multinomial logistic regression with softmax outputs, providing class probability estimates well-suited for ranked, top-k diagnostic suggestions.

### Decision Tree
Decision tree classifier with impurity-based splitting. Features `Age`, `Difficulty Breathing`, and `Cholesterol Level` emerged as the most influential early splits. Tuned via hyperparameter selection and SMOTE resampling.

### Neural Network (MLP)
Multi-layer perceptron tuned via grid search. Optimal architecture: **two hidden layers (25, 10 units)**, activation `tanh`, alpha `1e-4`, learning rate `1e-3`. Performance was limited by dataset size.

### K-Nearest Neighbors
Non-parametric instance-based classifier using Euclidean distance with majority voting. **k = 16** selected via grid search over k = 1–30. Feature scaling applied via StandardScaler.

---

## Key Results

- **Decision Tree** achieved the best overall performance (Top-1: 42.9%, Top-3: 70.0%)
- **Respiratory** was the most consistently well-predicted category across all models, owing to distinctive features (e.g., Difficulty Breathing) and higher support in the dataset
- **Neurological** and **Psychiatric** categories proved hardest to classify — their symptoms are poorly captured by the available features
- Top-3 accuracy substantially outperformed Top-1 across all models, mirroring how real symptom checkers present ranked differential diagnoses rather than a single answer
- KNN achieved excellent Respiratory recall (0.92) but failed on 6 out of 9 categories due to class imbalance and feature overlap

---

## Repo Structure

```
├── knn.py
├── decision_tree.py
├── logistic_regression.py
├── neural_network.py
├── data/
│   └── disease_dataset.csv
└── README.md
```

---

## Requirements

```bash
pip install scikit-learn pandas numpy imbalanced-learn matplotlib
```

---

## Usage

Each model script can be run independently. Ensure the dataset CSV is available at the expected path (see each file for configuration).

```bash
python logistic_regression.py
python decision_tree.py
python neural_network.py
python knn.py
```

---

## Discussion

While top-1 accuracy is modest, the models demonstrate meaningful directional value — the correct disease category is frequently included among the top-3 predictions. This aligns with how clinical symptom checkers operate in practice. With access to larger, richer real-world datasets (including lab results, longitudinal data, and more granular symptom descriptions), these modeling approaches could achieve substantially stronger performance.

> This project is a proof-of-concept and is not intended for clinical use.

---

## References

- Munsch et al., *Frontiers in Medicine*, 2022 — Evaluating diagnostic accuracy of symptom checkers
- Ramanath et al., *Journal of Biomedical Informatics*, 2021 — ML models for symptom-to-diagnosis mapping
