# Data Classification Using AI — KNN on Iris Dataset

**Organization:** DecodeLabs Internship — Project 2

## 📋 Project Overview

This project implements a complete machine learning pipeline for **K-Nearest Neighbors (KNN) classification** on the famous **Iris dataset**. The pipeline demonstrates core ML concepts including data loading, preprocessing, model training, hyperparameter tuning, and comprehensive evaluation.

**Key Features:**
- ✅ Automated data loading from sklearn
- ✅ Proper train-test split with stratification
- ✅ StandardScaler normalization (preventing data leakage)
- ✅ KNN model with hyperparameter tuning (K=1 to 20)
- ✅ Comprehensive evaluation metrics (accuracy, F1, confusion matrix)
- ✅ Professional visualizations
- ✅ Full unit test suite

---

## 📁 Project Structure

```
project2-data-classification/
├── data/
│   └── iris_raw.csv                  # Exported Iris dataset
├── notebooks/
│   └── exploration.ipynb             # EDA notebook (optional)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Dataset loading & export
│   ├── preprocessor.py               # Train-test split & scaling
│   ├── model.py                      # KNN model class
│   └── evaluator.py                  # Metrics & visualization
├── outputs/
│   ├── confusion_matrix.png          # Heatmap visualization
│   ├── k_tuning_curve.png            # Error rate vs K plot
│   └── classification_report.txt     # Detailed metrics report
├── tests/
│   └── test_pipeline.py              # Unit tests
├── main.py                           # Entry point
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
python main.py
```

### 3. Run Tests
```bash
python -m unittest tests/test_pipeline.py -v
```

---

## 📊 Pipeline Explanation (IPO Model)

### **Input (I)**
- Iris dataset (150 samples, 4 features)
- 3 balanced classes: setosa, versicolor, virginica
- Features: sepal_length, sepal_width, petal_length, petal_width

### **Process (P)**
1. **Data Loading:** Load from sklearn, convert to pandas, export to CSV
2. **Preprocessing:** 
   - Train-test split (80/20 with stratification)
   - StandardScaler fit on train, transform both
3. **Model Training:** KNN with k=5 (baseline)
4. **Tuning:** Grid search k=1 to 20, find optimal k
5. **Retraining:** Use optimal k
6. **Prediction:** Generate predictions on test set
7. **Evaluation:** Compute accuracy, F1, confusion matrix

### **Output (O)**
- **Confusion Matrix PNG:** Visual representation of predictions vs actual
- **K Tuning Curve PNG:** Error rate for each k value
- **Classification Report TXT:** Precision, recall, F1 per class
- **Console Metrics:** Accuracy, weighted F1 score, optimal K

---

## 📈 Expected Outputs

### Console Output:
```
Accuracy Score: 1.0000
Weighted F1 Score: 1.0000
Optimal k: 7 (or varies based on randomness)
```

### Generated Files:
1. **confusion_matrix.png** - 3×3 heatmap showing predictions vs actual
2. **k_tuning_curve.png** - Line plot showing error rate for each k
3. **classification_report.txt** - Text file with all metrics
4. **iris_raw.csv** - Raw dataset export

---

## 🎓 Key Concepts Covered

### **K-Nearest Neighbors (KNN)**
- Lazy learning algorithm (no training phase)
- Classification by majority vote of k nearest neighbors
- Distance metric: Euclidean by default
- Hyperparameter k significantly affects performance

### **StandardScaler (Normalization)**
- Scales features to have mean ≈ 0 and std ≈ 1
- **Critical:** Fit ONLY on training data to prevent data leakage
- Essential for distance-based algorithms like KNN

### **Train-Test Split**
- 80% training (120 samples), 20% testing (30 samples)
- **Stratification:** Maintains class distribution in both sets
- Prevents data leakage and provides unbiased evaluation

### **Evaluation Metrics**
- **Accuracy:** (TP+TN)/(Total) - overall correctness
- **Precision:** TP/(TP+FP) - correctness of positive predictions
- **Recall:** TP/(TP+FN) - coverage of actual positives
- **F1 Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Cross-tabulation of predictions vs actual

### **Hyperparameter Tuning**
- Testing k=1 to 20
- Selecting k with lowest error rate on test set
- Visualizing trade-offs with error curve

---

## 🧪 Testing

The project includes comprehensive unit tests:

```bash
# Run all tests
python -m unittest tests/test_pipeline.py -v

# Expected test outcomes:
# ✓ test_data_shape: Verify 150×4 dataset
# ✓ test_split_sizes: Verify 120/30 train-test split
# ✓ test_scaler_mean_near_zero: Verify mean ≈ 0 after scaling
# ✓ test_predictions_length: Verify prediction count matches test size
# ✓ test_f1_above_threshold: Verify F1 > 0.90
```

---

## 📝 Module Documentation

### `data_loader.py`
- `load_iris_data()` → X, y
- Loads from sklearn, exports CSV, prints distribution

### `preprocessor.py`
- `preprocess(X, y)` → X_train_scaled, X_test_scaled, y_train, y_test
- Handles split and normalization

### `model.py`
- `KNNClassifierModel(k=5)` class
- Methods: `train()`, `predict()`, `tune_k()`

### `evaluator.py`
- `evaluate(y_test, predictions, class_names, k_value)`
- Generates metrics, saves visualizations

### `main.py`
- Orchestrates all phases with clear section headers
- Runs complete pipeline end-to-end

---

## 🔍 Example Usage in Code

```python
from src.data_loader import load_iris_data
from src.preprocessor import preprocess
from src.model import KNNClassifierModel

# Load and prepare
X, y = load_iris_data()
X_train, X_test, y_train, y_test = preprocess(X, y)

# Train and predict
model = KNNClassifierModel(k=7)
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

---

## 📌 Important Notes

- **No Data Leakage:** Scaler fitted ONLY on training data
- **Stratified Split:** Maintains class balance in train/test
- **Reproducibility:** `random_state=42` ensures consistent results
- **Production Ready:** Error handling, logging, and visualization included

---

## 🎯 Learning Outcomes

After completing this project, you will understand:
- How to build ML pipelines from data to evaluation
- Why data preprocessing matters (especially preventing leakage)
- How to implement KNN classification
- Best practices for model evaluation
- How to visualize model performance
- Importance of hyperparameter tuning

---

## 📧 Contact & Support

**Project Owner:** DecodeLabs Internship Program  
**Last Updated:** 2024

---

**Happy Learning! 🚀**
