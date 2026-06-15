"""
PHASE 6 — Main Entry Point
Orchestrates the complete KNN classification pipeline
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_iris_data
from preprocessor import preprocess
from model import KNNClassifierModel
from evaluator import evaluate


def main():
    """Run the complete classification pipeline."""
    
    print("\n" + "=" * 60)
    print("KNN DATA CLASSIFICATION PIPELINE — IRIS DATASET")
    print("DecodeLabs Internship — Project 2")
    print("=" * 60)
    
    # Phase 1: Load Data
    print("\n" + "=" * 50)
    print("PHASE 1 — DATA LOADING")
    print("=" * 50)
    X, y = load_iris_data()
    
    # Phase 2: Preprocess Data
    print("\n" + "=" * 50)
    print("PHASE 2 — PREPROCESSING")
    print("=" * 50)
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess(X, y)
    
    # Phase 3: Initial Model Training
    print("\n" + "=" * 50)
    print("PHASE 3 — INITIAL MODEL TRAINING")
    print("=" * 50)
    model = KNNClassifierModel(k=5)
    model.train(X_train_scaled, y_train)
    
    # Phase 4: Hyperparameter Tuning
    print("\n" + "=" * 50)
    print("PHASE 4 — HYPERPARAMETER TUNING")
    print("=" * 50)
    optimal_k = model.tune_k(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Phase 5: Retrain with Optimal K
    print("\n" + "=" * 50)
    print("PHASE 5 — RETRAINING WITH OPTIMAL K")
    print("=" * 50)
    model = KNNClassifierModel(k=optimal_k)
    model.train(X_train_scaled, y_train)
    
    # Phase 6: Make Predictions
    print("\n" + "=" * 50)
    print("PHASE 6 — MAKING PREDICTIONS")
    print("=" * 50)
    predictions = model.predict(X_test_scaled)
    print(f"Predictions made for {len(predictions)} test samples")
    
    # Phase 7: Evaluate Model
    print("\n" + "=" * 50)
    print("PHASE 7 — MODEL EVALUATION")
    print("=" * 50)
    class_names = ['setosa', 'versicolor', 'virginica']
    accuracy, f1_weighted = evaluate(y_test, predictions, class_names, k_value=optimal_k)
    
    # Final Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nFinal Results:")
    print(f"  Optimal K: {optimal_k}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Weighted F1 Score: {f1_weighted:.4f}")
    print(f"\nAll outputs saved to: outputs/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
