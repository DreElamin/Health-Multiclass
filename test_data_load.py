"""
Quick test to verify data loading works
"""
from utils.model_utils import load_and_prepare_data

try:
    print("Attempting to load data...")
    data = load_and_prepare_data()
    print(f"✓ Data loaded successfully!")
    print(f"  - Total samples: {data['n_samples']}")
    print(f"  - Total features: {data['n_features']}")
    print(f"  - Disease categories: {', '.join(data['classes'])}")
except Exception as e:
    print(f"✗ Error loading data: {e}")
