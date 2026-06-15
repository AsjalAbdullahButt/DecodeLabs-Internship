"""
PHASE 2 — Data Loader Module
Loads Iris dataset from sklearn and exports to CSV
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import os


def load_iris_data():
    """
    Load Iris dataset from sklearn and convert to pandas DataFrame.
    
    Returns:
        X (pd.DataFrame): Features with columns [sepal_length, sepal_width, petal_length, petal_width]
        y (pd.Series): Target class names [setosa, versicolor, virginica]
    """
    # Load raw Iris dataset
    iris = load_iris()
    
    # Create DataFrame with feature names
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = pd.DataFrame(iris.data, columns=feature_names)
    
    # Map numeric labels to string names
    label_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    y = pd.Series([label_map[label] for label in iris.target], name='species')
    
    # Export raw CSV
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    data_df = X.copy()
    data_df['species'] = y
    csv_path = os.path.join(output_dir, 'iris_raw.csv')
    data_df.to_csv(csv_path, index=False)
    
    # Print dataset info
    print("\n" + "=" * 50)
    print("DATA LOADER — Iris Dataset")
    print("=" * 50)
    print(f"Dataset shape: {X.shape}")
    print(f"\nClass distribution:\n{y.value_counts()}")
    print(f"\nFirst 5 rows:\n{data_df.head()}")
    print(f"\nDataset exported to: {csv_path}")
    
    return X, y
