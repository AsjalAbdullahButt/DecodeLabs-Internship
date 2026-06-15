"""
PHASE 3 — Preprocessor Module
Handles train-test split and standardization (StandardScaler)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(X, y):
    """
    Split and standardize data to prevent data leakage.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target labels
    
    Returns:
        X_train_scaled (np.ndarray): Scaled training features
        X_test_scaled (np.ndarray): Scaled test features
        y_train (pd.Series): Training labels
        y_test (pd.Series): Test labels
    """
    # Split data: 80% train, 20% test with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        shuffle=True, 
        stratify=y
    )
    
    # Fit StandardScaler ONLY on training data (prevent data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test data using train scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Print info
    print("\n" + "=" * 50)
    print("PREPROCESSOR — Train-Test Split & Scaling")
    print("=" * 50)
    print(f"Training set size: {X_train_scaled.shape[0]}")
    print(f"Test set size: {X_test_scaled.shape[0]}")
    print(f"Train/Test split: {X_train_scaled.shape[0]}/{X_test_scaled.shape[0]}")
    print(f"\nX_train_scaled mean: {X_train_scaled.mean(axis=0)}")
    print(f"X_train_scaled std: {X_train_scaled.std(axis=0)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test
