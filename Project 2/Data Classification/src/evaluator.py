"""
PHASE 5 — Evaluator Module
Model evaluation with confusion matrix, F1 score, and classification report
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import os


def evaluate(y_test, predictions, class_names, k_value=5):
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        y_test (pd.Series): True test labels
        predictions (np.ndarray): Predicted labels
        class_names (list): List of class names
        k_value (int): K value used for KNN (for reporting)
    """
    print("\n" + "=" * 50)
    print("EVALUATOR — Model Performance Metrics")
    print("=" * 50)
    
    # Compute accuracy
    accuracy = np.mean(predictions == y_test.values)
    print(f"\nAccuracy Score: {accuracy:.4f}")
    
    # Compute classification report
    report = classification_report(y_test, predictions, target_names=class_names)
    print(f"\nClassification Report:\n{report}")
    
    # Compute weighted F1 score
    f1_weighted = f1_score(y_test, predictions, average='weighted')
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    
    # Save classification report to file
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("KNN CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy Score: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nWeighted F1 Score: {f1_weighted:.4f}\n")
    
    print(f"\nClassification report saved to: {report_path}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=class_names)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'KNN Confusion Matrix (k={k_value})', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {cm_path}")
    
    return accuracy, f1_weighted
