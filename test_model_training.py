"""
Test to verify model training works after the fix
"""
import pandas as pd
import numpy as np
from utils.model_utils import load_and_prepare_data, train_all_models

try:
    print("Loading data...")
    data = load_and_prepare_data()
    print("✓ Data loaded successfully!")
    
    print("\nTraining models...")
    models = train_all_models(
        data['X_train'],
        data['y_train'],
        data['preprocessor']
    )
    
    print("✓ All models trained successfully!")
    print(f"\nTrained models:")
    for model_name in models.keys():
        print(f"  - {model_name}")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
