# 🏥 Health ML Dashboard - User Guide

An interactive, polished dashboard for exploring multi-class health condition classification using machine learning.

## 🚀 Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`.

## 📱 Dashboard Features

### 🏠 Home Page
- **Project Overview**: Introduction to the ML health classification project
- **Quick Stats**: Key metrics and performance indicators
- **Model Performance**: Visual comparison of all 4 models
- **Getting Started Guide**: Step-by-step instructions

### 🔍 Symptom Checker (Interactive Prediction)
The most exciting feature! Input patient symptoms and get real-time predictions:

1. **Patient Information**:
   - Age (1-100)
   - Gender (Male/Female)

2. **Symptoms**:
   - Fever (Yes/No)
   - Cough (Yes/No)
   - Fatigue (Yes/No)
   - Difficulty Breathing (Yes/No)

3. **Vital Signs**:
   - Blood Pressure (Normal/High/Low)
   - Cholesterol Level (Normal/High/Low)

4. **Get Predictions**:
   - Click the "Get Predictions" button
   - See predictions from all 4 models
   - Each model shows:
     - Most likely diagnosis
     - Top 3 predictions with probabilities
     - Interactive probability charts

### 📊 Model Comparison
Comprehensive performance analysis:

- **Overall Performance Metrics**:
  - Accuracy (Top-1 and Top-3)
  - F1 Scores (Weighted and Macro)
  - Best model identification

- **Visualizations** (4 tabs):
  1. **Accuracy Comparison**: Bar charts comparing all models
  2. **Per-Class Performance**: Precision, Recall, F1 for each disease category
  3. **Confusion Matrices**: Visual representation of prediction patterns
  4. **Feature Importance**: Decision Tree feature importance analysis

### 📈 Data Insights
Explore the dataset characteristics:

- **Dataset Overview**:
  - Total records, features, and categories
  - Sample data preview
  - Statistical summaries

- **Feature Analysis**:
  - Age distribution histogram
  - Gender distribution pie chart
  - Symptom frequency analysis
  - Blood pressure & cholesterol distributions

- **Class Distribution**:
  - Disease category breakdown
  - Top 20 original diseases
  - Class imbalance visualization

### ℹ️ About
Detailed project information:
- Methodology and approach
- Model descriptions and architectures
- Key findings and insights
- Technical stack
- References and citations

## 🎯 How to Use

### First Time Setup

1. **Launch the Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

2. **Load Data** (Required):
   - Click "Load/Reload Data" button in the sidebar
   - Wait for confirmation message
   - Dataset statistics will appear in sidebar

3. **Train Models** (Required):
   - Click "Train Models" button in the sidebar
   - This will train all 4 models (takes ~1 minute)
   - Wait for confirmation message

4. **Start Exploring**:
   - Navigate to "Symptom Checker" to make predictions
   - Check "Model Comparison" for detailed analytics
   - Explore "Data Insights" to understand the dataset

### Making Predictions

1. Go to "🔍 Symptom Checker" page
2. Fill in patient information:
   - Adjust age slider
   - Select gender
   - Choose symptoms (Yes/No)
   - Set vital signs
3. Click "🔮 Get Predictions"
4. Review predictions from all models
5. Compare probabilities across models

### Comparing Models

1. Go to "📊 Model Comparison" page
2. View overall metrics table
3. Explore different tabs:
   - Compare accuracy metrics
   - Analyze per-class performance
   - View confusion matrices
   - Check feature importance

## 📊 Models Included

### 1. Logistic Regression
- **Type**: Linear classifier with multinomial regression
- **Strengths**: Fast, interpretable, probability estimates
- **Performance**: Top-1: 32.9% | Top-3: 62.9%

### 2. Decision Tree ⭐
- **Type**: Tree-based classifier with SMOTE
- **Strengths**: Non-linear, interpretable, handles categorical data
- **Performance**: Top-1: 42.9% | Top-3: 70.0%
- **Best Overall Model**

### 3. Neural Network
- **Type**: Multi-layer perceptron (MLP)
- **Architecture**: Two hidden layers (25, 10 units)
- **Strengths**: Captures complex patterns
- **Performance**: Top-1: ~35% | Top-3: ~60%

### 4. K-Nearest Neighbors
- **Type**: Instance-based classifier
- **K-value**: 16 (optimized)
- **Strengths**: Simple, no training phase
- **Performance**: Top-1: 33.3%

## 🎨 Dashboard Features

### Interactive Visualizations
- **Plotly Charts**: Interactive hover, zoom, pan
- **Color-Coded Metrics**: Easy visual comparison
- **Responsive Design**: Works on different screen sizes

### User Experience
- **Clean Interface**: Modern, professional design
- **Clear Navigation**: Sidebar with page selection
- **Real-Time Feedback**: Loading indicators and status messages
- **Helpful Tooltips**: Guidance throughout the app

### Performance Metrics
- **Accuracy**: Overall and top-3
- **F1 Scores**: Weighted and macro averages
- **Confusion Matrices**: Detailed prediction patterns
- **Per-Class Metrics**: Precision, recall, F1 for each category

## 🔧 Technical Details

### Technology Stack
- **Framework**: Streamlit 1.28+
- **ML Libraries**: scikit-learn, imbalanced-learn
- **Data Processing**: pandas, numpy
- **Visualization**: plotly, matplotlib, seaborn

### Data Processing Pipeline
1. Load CSV dataset
2. Map diseases to categories
3. Feature engineering (SymptomCount, IsElderly, MetabolicRisk)
4. Train-test split (80/20, stratified)
5. One-hot encoding for categorical features
6. SMOTE for class balancing
7. Model training and evaluation

### Disease Categories (9 total)
1. **Respiratory** - Asthma, Bronchitis, Pneumonia, etc.
2. **Cardiovascular** - Hypertension, Stroke, Heart Disease
3. **GI/Renal** - Gastroenteritis, Kidney Disease, UTI
4. **Neurological** - Migraine, Epilepsy, Parkinson's
5. **Endocrine/Metabolic** - Diabetes, Thyroid conditions
6. **Musculoskeletal/Autoimmune** - Arthritis, Lupus
7. **Psychiatric** - Depression, Anxiety, Bipolar
8. **Cancer** - Various cancer types
9. **Other** - Miscellaneous conditions

## ⚠️ Important Notes

### Limitations
- **Dataset Size**: Only ~350 records
- **Class Imbalance**: Some categories have limited samples
- **Feature Set**: Limited to basic symptoms and demographics
- **Proof of Concept**: Not for clinical use

### Disclaimer
> **This dashboard is for educational purposes only and is NOT intended for clinical use.**
> Always consult with qualified healthcare professionals for medical advice and diagnosis.

## 🐛 Troubleshooting

### Dashboard won't start
```bash
# Make sure streamlit is installed
pip install streamlit

# Try running with full path
python -m streamlit run dashboard.py
```

### Data won't load
- Ensure `Disease_symptom_and_patient_profile_dataset 2.csv` exists in the project directory
- Check file permissions
- Verify CSV format is correct

### Models won't train
- Ensure data is loaded first
- Check that all required libraries are installed
- Verify sufficient memory is available

### Visualizations not appearing
- Update plotly: `pip install --upgrade plotly`
- Clear browser cache
- Try a different browser

## 📚 Additional Resources

- **Original Models**: See `decisiontree.py`, `logistic regression.py`, `Knearest.py`
- **Project README**: See main `README.md` for project details
- **Requirements**: See `requirements.txt` for dependencies

## 🎓 Academic Context

This dashboard was created as part of the CS334 Final Project by James, Elamin, and Jonathan.
It demonstrates the application of classical machine learning techniques to healthcare symptom
classification, with a focus on practical usability and interpretability.

## 📧 Feedback

For questions, issues, or suggestions, please contact the project authors.

---

**Made with ❤️ using Streamlit and scikit-learn**
