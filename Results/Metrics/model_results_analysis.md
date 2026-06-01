# Model Results Analysis

## Model Overview

- Algorithm: Random Forest Classifier
- Training set: 400 transactions (80%)
- Test set: 100 transactions (20%)
- Random state: 42

---

## Results

### Accuracy
Overall accuracy on test set: **97%**

---

### Classification Report

| | Precision | Recall | F1-score | Support |
|--|-----------|--------|----------|---------|
| 0 (planned) | 0.97 | 0.94 | 0.95 | 31 |
| 1 (impulsive) | 0.97 | 0.99 | 0.98 | 69 |
| macro avg | 0.97 | 0.96 | 0.96 | 100 |
| weighted avg | 0.97 | 0.97 | 0.97 | 100 |

---

## Interpretation

### What the numbers mean

- **Precision 0.97** — When the model predicts a purchase is impulsive, it is correct 97% of the time
- **Recall 0.99 (class 1)** — The model correctly identifies 99% of all actual impulsive purchases
- **Recall 0.94 (class 0)** — The model correctly identifies 94% of all actual planned purchases
- **F1-score** — Balanced score between precision and recall. 0.98 for impulsive class is very strong

### Class imbalance
The test set contains 69 impulsive and 31 planned purchases — reflecting the dataset's imbalance. The model performs slightly better on the impulsive class (larger group).

---

## Limitations

- Dataset is synthetic — results may not reflect real-world performance
- Rules used to generate labels are the same patterns the model learned, which inflates accuracy
- 500 samples is small for a production ML model
- Model file (.pkl) not saved — cannot be reused without retraining

---

## Next Steps

- Test model on real transaction data
- Save trained model as .pkl for deployment
- Integrate model with FastAPI for real-time predictions
- Experiment with XGBoost for comparison
