"""
PHASE 4 — Model Module
KNN classifier with training, prediction, and hyperparameter tuning
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os


class KNNClassifierModel:
    """KNN Classifier with hyperparameter tuning capabilities."""
    
    def __init__(self, k=5):
        """
        Initialize KNN model.
        
        Args:
            k (int): Number of neighbors (default: 5)
        """
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """
        Train KNN model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (pd.Series): Training labels
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"\nModel trained with k={self.k}")
    
    def predict(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test (np.ndarray): Test features
        
        Returns:
            np.ndarray: Predicted labels
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X_test)
    
    def tune_k(self, X_train, y_train, X_test, y_test, k_range=range(1, 21)):
        """
        Tune hyperparameter k by testing error rates.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (pd.Series): Training labels
            X_test (np.ndarray): Test features
            y_test (pd.Series): Test labels
            k_range (range): Range of k values to test (default: 1-20)
        
        Returns:
            int: Optimal k value
        """
        print("\n" + "=" * 50)
        print("MODEL TUNING — K Hyperparameter Optimization")
        print("=" * 50)
        
        error_rates = []
        
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            predictions = knn.predict(X_test)
            error_rate = 1 - accuracy_score(y_test, predictions)
            error_rates.append(error_rate)
        
        # Find optimal k
        optimal_idx = np.argmin(error_rates)
        optimal_k = list(k_range)[optimal_idx]
        min_error_rate = error_rates[optimal_idx]
        
        print(f"Optimal k: {optimal_k} (error rate: {min_error_rate:.4f})")
        
        # Plot error rate vs K
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, error_rates, marker='o', linestyle='-', linewidth=2, markersize=6)
        plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        plt.xlabel('K Value', fontsize=12)
        plt.ylabel('Error Rate', fontsize=12)
        plt.title('KNN Error Rate vs K Value', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'k_tuning_curve.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"K tuning curve saved to: {plot_path}")
        
        return optimal_k
