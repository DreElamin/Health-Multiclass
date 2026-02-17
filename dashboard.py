"""
Multi-Class Health Condition Classification Dashboard
A polished interactive dashboard for exploring and using ML models for disease prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Health ML Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #1976d2;
    }
    .stMetric label {
        color: #1565c0 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #0d47a1 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #424242 !important;
        font-size: 0.85rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f5f5f5;
        border-radius: 10px 10px 0px 0px;
        color: #424242;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1976d2;
        color: white !important;
    }
    h1 {
        color: #1565c0;
        font-weight: 700;
    }
    h2 {
        color: #1976d2;
        font-weight: 600;
    }
    h3 {
        color: #424242;
        font-weight: 600;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Import model utilities
from utils.model_utils import (
    load_and_prepare_data,
    train_all_models,
    get_model_predictions,
    calculate_metrics
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Sidebar
with st.sidebar:
    st.markdown("# 🏥")
    st.title("Navigation")

    page = st.radio(
        "Go to",
        ["🏠 Home", "🔍 Symptom Checker", "📊 Model Comparison", "📈 Data Insights", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Dataset Info")
    if st.session_state.data_loaded:
        st.success("✅ Data Loaded")
        st.info(f"**Records:** {st.session_state.get('n_samples', 0)}")
        st.info(f"**Features:** {st.session_state.get('n_features', 0)}")
    else:
        st.warning("⏳ Data Not Loaded")

    st.divider()

    if st.button("🔄 Load/Reload Data", width="stretch"):
        with st.spinner("Loading data..."):
            try:
                data_dict = load_and_prepare_data()
                st.session_state.update(data_dict)
                st.session_state.data_loaded = True
                st.success("Data loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    if st.session_state.data_loaded and st.button("🚀 Train Models", width="stretch"):
        with st.spinner("Training models... This may take a minute."):
            try:
                models_dict = train_all_models(
                    st.session_state.X_train,
                    st.session_state.y_train,
                    st.session_state.preprocessor
                )
                st.session_state.update(models_dict)
                st.session_state.models_trained = True
                st.success("Models trained successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error training models: {str(e)}")

# Main content based on selected page
if page == "🏠 Home":
    st.title("🏥 Multi-Class Health Condition Classification")
    st.markdown("### Machine Learning for Symptom-Based Disease Screening")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Models Implemented",
            value="4",
            delta="Logistic Reg, Decision Tree, Neural Net, KNN"
        )

    with col2:
        st.metric(
            label="Disease Categories",
            value="9",
            delta="From 100+ original diseases"
        )

    with col3:
        st.metric(
            label="Best Top-3 Accuracy",
            value="70.0%",
            delta="Decision Tree"
        )

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## 📋 Overview

        This project investigates whether classical supervised learning models can approximate
        a simplified symptom-based disease screening system, similar to WebMD or other health
        symptom checkers.

        ### 🎯 Key Features

        - **Multi-Model Comparison**: Compare performance across 4 different ML models
        - **Interactive Predictions**: Input symptoms and get real-time disease predictions
        - **Top-K Accuracy**: Models provide ranked differential diagnoses (Top-1 and Top-3)
        - **Class Imbalance Handling**: Uses SMOTE to address severe class imbalance
        - **Comprehensive Metrics**: Detailed performance analysis with visualizations

        ### 🔬 Dataset

        - **Source**: Kaggle (synthetic patient health records)
        - **Size**: ~350 patient records
        - **Features**: Age, Gender, Fever, Cough, Fatigue, Difficulty Breathing,
          Blood Pressure, Cholesterol Level
        - **Labels**: 9 disease categories

        ### 📊 Disease Categories

        1. **Respiratory** - Asthma, Bronchitis, Pneumonia, Common Cold, Influenza
        2. **Cardiovascular** - Hypertension, Stroke, Heart Disease
        3. **Gastrointestinal/Renal** - Gastroenteritis, Kidney Disease, UTI
        4. **Neurological** - Migraine, Epilepsy, Parkinson's, Alzheimer's
        5. **Endocrine/Metabolic** - Diabetes, Thyroid conditions
        6. **Musculoskeletal/Autoimmune** - Arthritis, Lupus, Psoriasis
        7. **Psychiatric** - Depression, Anxiety, Bipolar Disorder
        8. **Cancer** - Various cancer types
        9. **Other** - Miscellaneous conditions
        """)

    with col2:
        st.markdown("### 🚀 Quick Start")
        st.info("""
        1. Click **Load/Reload Data** in the sidebar
        2. Click **Train Models** to train all models
        3. Navigate to **Symptom Checker** to make predictions
        4. Explore **Model Comparison** for detailed metrics
        """)

        st.markdown("### 📈 Model Performance")

        performance_data = pd.DataFrame({
            'Model': ['Decision Tree', 'KNN', 'Logistic Reg', 'Neural Net'],
            'Top-1': [42.9, 33.3, 32.9, 35.0],
            'Top-3': [70.0, 55.0, 62.9, 60.0]
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Top-1 Accuracy',
            x=performance_data['Model'],
            y=performance_data['Top-1'],
            marker_color='#1f77b4'
        ))
        fig.add_trace(go.Bar(
            name='Top-3 Accuracy',
            x=performance_data['Model'],
            y=performance_data['Top-3'],
            marker_color='#ff7f0e'
        ))

        fig.update_layout(
            barmode='group',
            title='Model Performance Comparison',
            yaxis_title='Accuracy (%)',
            height=300,
            showlegend=True
        )

        st.plotly_chart(fig, width="stretch")

    st.divider()

    st.markdown("""
    ### ⚠️ Important Disclaimer

    > This project is a **proof-of-concept** and is **NOT intended for clinical use**.
    > Always consult with qualified healthcare professionals for medical advice and diagnosis.
    """)

elif page == "🔍 Symptom Checker":
    st.title("🔍 Interactive Symptom Checker")
    st.markdown("### Enter patient symptoms to get disease predictions from all models")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the data first using the sidebar button.")
    elif not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first using the sidebar button.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 👤 Patient Information")
            age = st.slider("Age", 1, 100, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])

            st.markdown("#### 🩺 Symptoms")
            fever = st.selectbox("Fever", ["No", "Yes"])
            cough = st.selectbox("Cough", ["No", "Yes"])
            fatigue = st.selectbox("Fatigue", ["No", "Yes"])
            difficulty_breathing = st.selectbox("Difficulty Breathing", ["No", "Yes"])

            st.markdown("#### 🔬 Vital Signs")
            blood_pressure = st.selectbox("Blood Pressure", ["Normal", "High", "Low"])
            cholesterol = st.selectbox("Cholesterol Level", ["Normal", "High", "Low"])

        with col2:
            st.markdown("#### 🎯 Predictions")

            if st.button("🔮 Get Predictions", width="stretch", type="primary"):
                # Create input dataframe
                patient_data = pd.DataFrame([{
                    "Fever": fever,
                    "Cough": cough,
                    "Fatigue": fatigue,
                    "Difficulty Breathing": difficulty_breathing,
                    "Age": age,
                    "Gender": gender,
                    "Blood Pressure": blood_pressure,
                    "Cholesterol Level": cholesterol
                }])

                # Get predictions from all models
                predictions = get_model_predictions(
                    patient_data,
                    st.session_state
                )

                # Display predictions for each model
                for model_name, pred_info in predictions.items():
                    with st.expander(f"**{model_name}**", expanded=True):
                        st.markdown(f"**Most Likely Diagnosis:** `{pred_info['top1']}`")

                        st.markdown("**Top 3 Predictions:**")

                        # Create a nice visualization for top 3
                        top3_df = pd.DataFrame({
                            'Category': pred_info['top3_labels'],
                            'Probability': pred_info['top3_probs']
                        })

                        fig = go.Figure(go.Bar(
                            x=top3_df['Probability'],
                            y=top3_df['Category'],
                            orientation='h',
                            marker=dict(
                                color=top3_df['Probability'],
                                colorscale='Blues',
                                showscale=False
                            ),
                            text=[f"{p:.1%}" for p in top3_df['Probability']],
                            textposition='auto',
                        ))

                        fig.update_layout(
                            height=200,
                            margin=dict(l=0, r=0, t=20, b=0),
                            xaxis_title="Probability",
                            yaxis_title="",
                            xaxis=dict(range=[0, 1])
                        )

                        st.plotly_chart(fig, width="stretch")

                # Show symptom summary
                st.markdown("---")
                st.markdown("#### 📋 Input Summary")
                symptoms_present = []
                if fever == "Yes":
                    symptoms_present.append("Fever")
                if cough == "Yes":
                    symptoms_present.append("Cough")
                if fatigue == "Yes":
                    symptoms_present.append("Fatigue")
                if difficulty_breathing == "Yes":
                    symptoms_present.append("Difficulty Breathing")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.info(f"**Symptoms Present:** {', '.join(symptoms_present) if symptoms_present else 'None'}")
                with col_b:
                    st.info(f"**Risk Factors:** {blood_pressure} BP, {cholesterol} Cholesterol")

elif page == "📊 Model Comparison":
    st.title("📊 Model Performance Comparison")
    st.markdown("### Detailed metrics and visualizations for all models")

    if not st.session_state.data_loaded or not st.session_state.models_trained:
        st.warning("⚠️ Please load data and train models first using the sidebar buttons.")
    else:
        # Calculate metrics for all models
        metrics_dict = calculate_metrics(st.session_state)

        # Overall metrics comparison
        st.markdown("### 🎯 Overall Performance")

        metrics_df = pd.DataFrame({
            'Model': list(metrics_dict.keys()),
            'Accuracy': [m['accuracy'] for m in metrics_dict.values()],
            'Top-3 Accuracy': [m['top3_accuracy'] for m in metrics_dict.values()],
            'Weighted F1': [m['weighted_f1'] for m in metrics_dict.values()],
            'Macro F1': [m['macro_f1'] for m in metrics_dict.values()]
        })

        col1, col2, col3, col4 = st.columns(4)

        best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Model']
        best_acc = metrics_df['Accuracy'].max()

        with col1:
            st.metric("Best Model", best_model, f"{best_acc:.1%} accuracy")
        with col2:
            st.metric("Avg Top-1 Accuracy", f"{metrics_df['Accuracy'].mean():.1%}")
        with col3:
            st.metric("Avg Top-3 Accuracy", f"{metrics_df['Top-3 Accuracy'].mean():.1%}")
        with col4:
            st.metric("Avg F1 Score", f"{metrics_df['Weighted F1'].mean():.3f}")

        # Metrics table
        st.markdown("### 📋 Detailed Metrics Table")
        st.dataframe(
            metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Top-3 Accuracy', 'Weighted F1', 'Macro F1']),
            width="stretch"
        )

        # Visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Accuracy Comparison",
            "🎯 Per-Class Performance",
            "📈 Confusion Matrices",
            "🔍 Feature Importance"
        ])

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                # Top-1 vs Top-3 comparison
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Top-1 Accuracy',
                    x=metrics_df['Model'],
                    y=metrics_df['Accuracy'] * 100,
                    marker_color='#1f77b4'
                ))
                fig.add_trace(go.Bar(
                    name='Top-3 Accuracy',
                    x=metrics_df['Model'],
                    y=metrics_df['Top-3 Accuracy'] * 100,
                    marker_color='#ff7f0e'
                ))
                fig.update_layout(
                    title='Top-1 vs Top-3 Accuracy',
                    yaxis_title='Accuracy (%)',
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, width="stretch")

            with col2:
                # F1 scores comparison
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Weighted F1',
                    x=metrics_df['Model'],
                    y=metrics_df['Weighted F1'],
                    marker_color='#2ca02c'
                ))
                fig.add_trace(go.Bar(
                    name='Macro F1',
                    x=metrics_df['Model'],
                    y=metrics_df['Macro F1'],
                    marker_color='#d62728'
                ))
                fig.update_layout(
                    title='F1 Score Comparison',
                    yaxis_title='F1 Score',
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, width="stretch")

        with tab2:
            # Per-class performance
            st.markdown("#### Per-Class Precision, Recall, and F1-Score")

            selected_model = st.selectbox(
                "Select Model",
                list(metrics_dict.keys())
            )

            class_metrics = metrics_dict[selected_model]['classification_report']

            # Create DataFrame for per-class metrics
            class_df = pd.DataFrame({
                'Class': list(class_metrics.keys()),
                'Precision': [class_metrics[c]['precision'] for c in class_metrics.keys()],
                'Recall': [class_metrics[c]['recall'] for c in class_metrics.keys()],
                'F1-Score': [class_metrics[c]['f1-score'] for c in class_metrics.keys()],
                'Support': [class_metrics[c]['support'] for c in class_metrics.keys()]
            })

            # Horizontal bar chart for per-class metrics
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Precision',
                y=class_df['Class'],
                x=class_df['Precision'],
                orientation='h',
                marker_color='#1f77b4'
            ))
            fig.add_trace(go.Bar(
                name='Recall',
                y=class_df['Class'],
                x=class_df['Recall'],
                orientation='h',
                marker_color='#ff7f0e'
            ))
            fig.add_trace(go.Bar(
                name='F1-Score',
                y=class_df['Class'],
                x=class_df['F1-Score'],
                orientation='h',
                marker_color='#2ca02c'
            ))
            fig.update_layout(
                title=f'{selected_model} - Per-Class Metrics',
                xaxis_title='Score',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, width="stretch")

            st.dataframe(class_df, width="stretch")

        with tab3:
            # Confusion matrices
            st.markdown("#### Confusion Matrices")

            selected_models = st.multiselect(
                "Select Models to Compare",
                list(metrics_dict.keys()),
                default=list(metrics_dict.keys())[:2]
            )

            cols = st.columns(len(selected_models))

            for i, model_name in enumerate(selected_models):
                with cols[i]:
                    cm = metrics_dict[model_name]['confusion_matrix']

                    fig = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=st.session_state.classes,
                        y=st.session_state.classes,
                        color_continuous_scale='Blues',
                        title=model_name
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, width="stretch")

        with tab4:
            st.markdown("#### Feature Importance (Decision Tree)")

            if 'decision_tree' in st.session_state:
                # Get feature importances
                feature_names = st.session_state.preprocessor.get_feature_names_out()
                importances = st.session_state.decision_tree.feature_importances_

                # Create DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(15)

                # Plot
                fig = go.Figure(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker=dict(
                        color=importance_df['Importance'],
                        colorscale='Viridis',
                        showscale=True
                    )
                ))
                fig.update_layout(
                    title='Top 15 Most Important Features',
                    xaxis_title='Importance',
                    yaxis_title='Feature',
                    height=600
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Train models to see feature importance.")

elif page == "📈 Data Insights":
    st.title("📈 Data Insights & Exploration")
    st.markdown("### Explore the dataset characteristics and distributions")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the data first using the sidebar button.")
    else:
        df = st.session_state.df

        tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "🔍 Feature Analysis", "📈 Class Distribution"])

        with tab1:
            st.markdown("### 📋 Dataset Statistics")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Total Features", len(df.columns) - 1)
            with col3:
                st.metric("Disease Categories", df['DiseaseCategory'].nunique())
            with col4:
                st.metric("Original Diseases", df['Disease'].nunique())

            st.markdown("### 🔍 Sample Data")
            st.dataframe(df.head(20), width="stretch")

            st.markdown("### 📊 Feature Summary")
            st.dataframe(df.describe(), width="stretch")

        with tab2:
            st.markdown("### 🔬 Feature Distributions")

            col1, col2 = st.columns(2)

            with col1:
                # Age distribution
                fig = px.histogram(
                    df,
                    x='Age',
                    nbins=30,
                    title='Age Distribution',
                    color_discrete_sequence=['#1f77b4']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, width="stretch")

                # Gender distribution
                gender_counts = df['Gender'].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=gender_counts.index,
                    values=gender_counts.values,
                    hole=.3
                )])
                fig.update_layout(title='Gender Distribution', height=300)
                st.plotly_chart(fig, width="stretch")

            with col2:
                # Symptom frequencies
                symptoms = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
                symptom_counts = {s: (df[s] == 'Yes').sum() for s in symptoms}

                fig = go.Figure(go.Bar(
                    x=list(symptom_counts.keys()),
                    y=list(symptom_counts.values()),
                    marker_color='#ff7f0e'
                ))
                fig.update_layout(
                    title='Symptom Frequencies',
                    xaxis_title='Symptom',
                    yaxis_title='Count',
                    height=300
                )
                st.plotly_chart(fig, width="stretch")

                # Blood Pressure & Cholesterol
                bp_counts = df['Blood Pressure'].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=bp_counts.index,
                    values=bp_counts.values,
                    hole=.3
                )])
                fig.update_layout(title='Blood Pressure Distribution', height=300)
                st.plotly_chart(fig, width="stretch")

        with tab3:
            st.markdown("### 🎯 Disease Category Distribution")

            category_counts = df['DiseaseCategory'].value_counts()

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = go.Figure(go.Bar(
                    x=category_counts.index,
                    y=category_counts.values,
                    marker=dict(
                        color=category_counts.values,
                        colorscale='Viridis',
                        showscale=True
                    )
                ))
                fig.update_layout(
                    title='Disease Category Distribution',
                    xaxis_title='Category',
                    yaxis_title='Count',
                    height=400,
                    xaxis={'categoryorder': 'total descending'}
                )
                st.plotly_chart(fig, width="stretch")

            with col2:
                st.markdown("#### Category Breakdown")
                for cat, count in category_counts.items():
                    percentage = (count / len(df)) * 100
                    st.metric(cat, count, f"{percentage:.1f}%")

            st.markdown("### 🔬 Original Disease Distribution (Top 20)")
            disease_counts = df['Disease'].value_counts().head(20)

            fig = go.Figure(go.Bar(
                y=disease_counts.index,
                x=disease_counts.values,
                orientation='h',
                marker_color='#2ca02c'
            ))
            fig.update_layout(
                title='Top 20 Original Diseases',
                xaxis_title='Count',
                yaxis_title='Disease',
                height=600
            )
            st.plotly_chart(fig, width="stretch")

else:  # About page
    st.title("ℹ️ About This Project")

    st.markdown("""
    ## 🎓 CS334 Final Project

    **Authors:** James, Elamin, Jonathan

    ### 🎯 Project Goals

    This project investigates whether classical supervised learning models can approximate a
    simplified symptom-based disease screening system. Using publicly available patient data,
    we frame the task as a multi-class classification problem and evaluate multiple ML approaches.

    ### 🔬 Methodology

    #### Data Preprocessing
    - Binary/categorical encoding of symptom features
    - Stratified train/test split (80/20) to preserve class proportions
    - **SMOTE** (Synthetic Minority Oversampling Technique) to address severe class imbalance
    - Feature engineering (SymptomCount, IsElderly, MetabolicRisk)
    - StandardScaler normalization for distance-based models

    #### Models Implemented

    **1. Logistic Regression**
    - Multinomial logistic regression with softmax outputs
    - Provides class probability estimates for ranked predictions
    - Top-1: 32.9% | Top-3: 62.9%

    **2. Decision Tree**
    - Impurity-based splitting with hyperparameter tuning via GridSearchCV
    - Key features: Age, Difficulty Breathing, Cholesterol Level
    - Tuned with SMOTE resampling
    - Top-1: 42.9% | Top-3: 70.0% ⭐ **Best Overall**

    **3. Neural Network (MLP)**
    - Multi-layer perceptron with grid search optimization
    - Optimal architecture: Two hidden layers (25, 10 units)
    - Activation: tanh, Alpha: 1e-4, Learning rate: 1e-3
    - Performance limited by small dataset size

    **4. K-Nearest Neighbors**
    - Instance-based classifier with Euclidean distance
    - k=16 selected via grid search (k=1 to 30)
    - Excellent Respiratory recall (0.92) but struggles with rare classes
    - Top-1: 33.3%

    ### 📊 Key Findings

    - **Decision Tree** achieved best overall performance (70% Top-3 accuracy)
    - **Respiratory** conditions most consistently predicted across all models due to
      distinctive features and higher support
    - **Neurological** and **Psychiatric** categories hardest to classify—symptoms poorly
      captured by available features
    - **Top-3 accuracy** substantially outperformed Top-1 across all models, mirroring how
      real symptom checkers present ranked differential diagnoses
    - **Class imbalance** significantly impacted performance, partially mitigated by SMOTE

    ### 🚀 Technical Stack

    - **Framework**: Streamlit
    - **ML Libraries**: scikit-learn, imbalanced-learn
    - **Data Processing**: pandas, numpy
    - **Visualization**: plotly, matplotlib, seaborn

    ### 📝 Challenges Addressed

    - Limited sample size (~350 records)
    - Severe class imbalance (100+ original disease labels)
    - Overlapping symptom profiles across conditions
    - Noisy, binary/categorical symptom features

    ### 🔮 Future Improvements

    With access to larger, richer datasets including:
    - Lab results and biomarkers
    - Longitudinal patient data
    - More granular symptom descriptions
    - Medical history and comorbidities

    These modeling approaches could achieve substantially stronger performance.

    ### ⚠️ Disclaimer

    > **This project is a proof-of-concept and is NOT intended for clinical use.**
    > Always consult with qualified healthcare professionals for medical advice and diagnosis.

    ### 📚 References

    - Munsch et al., *Frontiers in Medicine*, 2022 — Evaluating diagnostic accuracy of symptom checkers
    - Ramanath et al., *Journal of Biomedical Informatics*, 2021 — ML models for symptom-to-diagnosis mapping
    - Chawla et al., 2002 — SMOTE: Synthetic Minority Over-sampling Technique

    ### 📧 Contact

    For questions or feedback about this project, please reach out to the authors.

    ---

    Made with ❤️ using Streamlit and scikit-learn
    """)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Multi-Class Health Condition Classification Dashboard | CS334 Final Project</p>
        <p>⚠️ For educational purposes only - Not for clinical use</p>
    </div>
""", unsafe_allow_html=True)
