# Model Results Analysis

## Model Overview

- Algorithm: Neural network (MLPClassifier) wrapped in a scikit-learn Pipeline
- Hidden layers: (16, 8), ReLU activation, Adam optimizer
- Training: epoch-by-epoch with early stopping on the validation set
- Target: **graded impulse level** (0=none, 1=mild, 2=moderate, 3=strong)
- Dataset: 10,000 synthetic transactions with **label noise**
- Split: 8000 train (80%) / 1000 validation (10%) / 1000 test (10%), stratified
- Class balancing: minority levels oversampled in the training set only
- Random state: 42

---

## Why macro-F1 (not accuracy)?

The classes are imbalanced and this is a 4-level (multiclass) problem, so the
**headline metric is macro-F1** — the average of the per-level F1 scores,
weighting all four levels equally so the smaller "strongly impulsive" class
still counts. Accuracy would be dominated by the easy majority class.

---

## Why the score is ~0.67 (and why that's honest)

Earlier versions scored ~0.90 — but those labels were a **deterministic rule**,
so the model could essentially memorize a lookup table, which inflated the
score. The dataset now adds **Gaussian noise** to the impulse score before
labeling, so identical-looking purchases don't always get the same level. The
task is now genuinely hard, and ~0.67 macro-F1 reflects **real learning**, not
memorization.

---

## Results (held-out test set)

| Metric | Value |
|--------|-------|
| Macro-F1 | **0.67** |
| Weighted-F1 | 0.68 |
| Accuracy | 0.68 |

### Per-level report

| Level | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Not impulsive | 0.89 | 0.73 | 0.80 | 328 |
| Mildly impulsive | 0.56 | 0.66 | 0.60 | 259 |
| Moderately impulsive | 0.55 | 0.53 | 0.54 | 225 |
| Strongly impulsive | 0.72 | 0.78 | 0.75 | 188 |

The two middle levels (mild/moderate) are the hardest — their boundary is
genuinely fuzzy, which is expected. The clear ends (none / strong) score best.

---

## Validation against baselines (5-fold cross-validation)

`compare_models.py` runs 5-fold cross-validation so the score is an average
across folds (mean ± std), not one lucky split:

| Model | Macro-F1 (mean ± std) |
|-------|----------------------|
| Majority baseline (always predict the most common level) | 0.12 ± 0.00 |
| Logistic regression | 0.66 ± 0.01 |
| **Neural network** | **0.67 ± 0.01** |

Two takeaways:
- The neural network **far outperforms the majority baseline** (0.67 vs 0.12),
  confirming it is genuinely learning.
- It only **ties logistic regression** — on data this size, the neural net's
  extra complexity adds little. A useful, honest finding to report.

---

## Limitations

- Dataset is synthetic — labels still come from a (now noisy) generator.
- The neural network does not clearly beat a simple linear model here.
- No personalization yet (impulse is relative to each individual user).

---

## Next Steps

- Collect real labelled data through the feedback loop and retrain.
- Add per-user baselines (deviation from a user's normal spending).
- Tune hyperparameters via search on the validation set.
- Integrate the pipeline with the FastAPI backend for real-time predictions.
