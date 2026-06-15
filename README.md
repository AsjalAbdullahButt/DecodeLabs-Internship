# 🧠 DecodeLabs Internship — AI/ML Projects

**Intern:** Asjal Abdullah Butt  
**Program:** DecodeLabs Internship 2026  
**Track:** Artificial Intelligence & Machine Learning  
**Institution:** FAST-NUCES, Lahore — BS Computer Science (2026)

---

## 📌 Overview

This repository contains two projects completed as part of the DecodeLabs AI/ML internship program. Both projects are written in Python and demonstrate applied skills across NLP rule systems, supervised learning, data preprocessing, model evaluation, and software design principles.

---

## 🗂️ Repository Structure

```
DecodeLabs-Internship/
├── Project 1/
│   └── Rule-Based-Chatbot/
│       ├── chatbot/
│       │   ├── __init__.py
│       │   └── engine.py          # Core intent matching logic
│       ├── main.py                # REPL entry point
│       └── README.md
│
└── Project 2/
    └── Data Classification/
        ├── src/
        │   ├── data_loader.py     # Dataset loading & CSV export
        │   ├── preprocessor.py    # Train-test split & scaling
        │   ├── model.py           # KNN classifier + tuning
        │   └── evaluator.py       # Metrics & visualizations
        ├── data/
        │   └── iris_raw.csv
        ├── outputs/
        │   ├── confusion_matrix.png
        │   ├── k_tuning_curve.png
        │   └── classification_report.txt
        ├── main.py                # Pipeline orchestrator
        ├── requirements.txt
        └── README.md
```

---

## 🤖 Project 1 — Rule-Based Chatbot

### What It Does

A terminal-based conversational chatbot that matches user input to pre-defined responses using a Python dictionary. No ML, no external APIs — pure logic and data structures.

### Architecture

The chatbot is split into two modules:

- **`engine.py`** — owns the knowledge base (a `dict`) and exposes two functions: `sanitize()` and `get_response()`
- **`main.py`** — owns the REPL loop, handles I/O, and coordinates the execution flow

This separation keeps concerns clean: the engine does not know about the terminal, and `main.py` does not know how responses are retrieved.

### Key Design Decisions

**Dictionary over if-elif chains**

The knowledge base is a flat Python `dict`. Lookup via `.get()` runs in O(1) average time regardless of how many intents exist. An if-elif chain would degrade to O(n) as intents grow.

| Approach | 10 Intents | 100 Intents | 1,000 Intents |
|---|---|---|---|
| if-elif | ~5 comparisons | ~50 comparisons | ~500 comparisons |
| dict `.get()` | 1 lookup | 1 lookup | 1 lookup |

**Exit check before knowledge base lookup**

`bye` exists in the knowledge base as a farewell response. If the exit check ran after the lookup, the user could never exit using `bye` — they'd just get a response back. The code deliberately places the exit check first to avoid this collision.

**Sanitization before matching**

`sanitize()` applies `.lower().strip()` before any lookup. This ensures `"  HELLO  "`, `"Hello"`, and `"hello"` all resolve to the same key. The function is O(n) on string length but runs once per turn.

**Graceful Ctrl+C handling**

`KeyboardInterrupt` is caught in `main()` and exits with code 0. Without this, an interrupt would print a raw stack trace — bad UX for a terminal app.

### Supported Intents

| Category | Triggers |
|---|---|
| Greetings | `hello`, `hi`, `hey` |
| Farewell / Exit | `bye`, `goodbye`, `exit`, `quit` |
| Identity | `who are you`, `what are you` |
| Capability | `what can you do`, `help` |
| Status | `how are you` |
| Gratitude | `thanks`, `thank you` |
| Time | `what time is it` |
| Fun | `tell me a joke` |

### How to Run

```bash
cd "Project 1/Rule-Based-Chatbot"
python main.py
```

No dependencies. Requires Python 3.8+.

### Limitations (Honest Assessment)

- **Exact-match only.** `"hi there"` does not match `"hi"`. Any deviation from a known key hits the fallback.
- **No context or memory.** Each input is treated independently. There is no conversation state.
- **Static knowledge base.** Adding new intents requires editing `engine.py` directly.

These are expected constraints for a rule-based system. The value of this project is in its clean architecture and correct use of data structures — not breadth of coverage.

---

## 📊 Project 2 — KNN Data Classification Pipeline

### What It Does

A complete, modular machine learning pipeline that trains a K-Nearest Neighbors classifier on the Iris dataset, tunes the hyperparameter K, evaluates performance across multiple metrics, and saves visualizations and reports to disk.

### Pipeline Phases

```
Phase 1 → Load Data       (data_loader.py)
Phase 2 → Preprocess      (preprocessor.py)
Phase 3 → Train (k=5)     (model.py)
Phase 4 → Tune K          (model.py → tune_k())
Phase 5 → Retrain         (model.py)
Phase 6 → Predict         (model.py)
Phase 7 → Evaluate        (evaluator.py)
```

`main.py` orchestrates all seven phases in sequence with clear console section headers. Each phase is handled by a dedicated module — no phase knows about another's internals.

### Key Design Decisions

**Modular src/ structure**

Each concern (loading, preprocessing, modeling, evaluation) lives in its own file. This makes each component independently testable and replaceable. Swapping KNN for Random Forest only requires changes to `model.py`.

**Data leakage prevention**

`StandardScaler` is fit exclusively on training data. The test set is then transformed using the training scaler's parameters — never fit again on test data. Fitting on the full dataset before splitting is one of the most common preprocessing mistakes; this project correctly avoids it.

**Stratified train-test split**

`train_test_split` is called with `stratify=y`. This guarantees that each class (setosa, versicolor, virginica) is proportionally represented in both train and test sets. Without stratification, random splits can create imbalanced subsets, especially on small datasets.

**K tuning via error rate curve**

The model tests k values from 1 to 20, computes the error rate for each, and selects the k with the minimum error. Results are saved as a PNG plot (`k_tuning_curve.png`) so the tradeoff is visible, not just a printed number.

**OOP model class**

`KNNClassifierModel` wraps `sklearn`'s `KNeighborsClassifier` in a class with a `is_trained` guard. Calling `.predict()` before `.train()` raises a `RuntimeError` explicitly — better than letting sklearn raise a cryptic internal error.

### Model Results

These are the actual outputs from the saved classification report:

```
Accuracy Score:    0.9667  (96.67%)
Weighted F1 Score: 0.9666
```

**Per-class breakdown:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| setosa | 1.00 | 1.00 | 1.00 | 10 |
| versicolor | 0.91 | 1.00 | 0.95 | 10 |
| virginica | 1.00 | 0.90 | 0.95 | 10 |
| **Weighted Avg** | **0.97** | **0.97** | **0.97** | **30** |

Setosa is perfectly classified (linearly separable in feature space). Versicolor and virginica share overlapping feature distributions, accounting for the single misclassification.

### Generated Outputs

| File | Description |
|---|---|
| `outputs/confusion_matrix.png` | 3×3 heatmap — predicted vs actual per class |
| `outputs/k_tuning_curve.png` | Error rate plotted against k=1 to 20 |
| `outputs/classification_report.txt` | Full precision/recall/F1 report |
| `data/iris_raw.csv` | Exported raw dataset (150 rows, 5 columns) |

### How to Run

```bash
cd "Project 2/Data Classification"
pip install -r requirements.txt
python main.py
```

**Dependencies:**

```
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2
matplotlib==3.8.2
seaborn==0.13.0
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| ML Library | scikit-learn |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Architecture | Modular OOP (src/ pattern) |
| Version Control | Git / GitHub |

---

## 💡 Skills Demonstrated

- Clean Python module architecture with separation of concerns
- O(1) intent matching using hash-based dictionary lookups
- Supervised classification with KNN
- Correct preprocessing pipeline (no data leakage)
- Hyperparameter tuning with visual analysis
- Multi-metric model evaluation (accuracy, F1, confusion matrix)
- Writing maintainable, well-documented code

---

## 📬 Contact

**Asjal Abdullah Butt**  
BS Computer Science — FAST-NUCES Lahore (2026)  
[GitHub](https://github.com/AsjalAbdullahButt) • [LinkedIn](https://www.linkedin.com/in/asjalabdullahbutt)
