# System Design

## Overview

The system detects impulsive purchases before a transaction is completed.
The AI model analyzes purchase features — including how much the user
deliberated and whether the item is a genuine need — and returns a **graded
impulse level** (none / mild / moderate / strong) plus a confidence, so the app
can respond proportionally instead of with a blunt yes/no.

---

## System Architecture

User attempts purchase
→ Data Input Layer (hour, price, category, frequency, is_essential,
  deliberation_minutes, on_wishlist)
→ ML Pipeline (one artifact: OneHotEncoder + log/StandardScaler + neural network)
→ Predicted impulse level (0–3) + per-level probabilities
    → 0 Not impulsive  : ✅ proceeds
    → 1 Mildly         : 🟡 stay aware
    → 2 Moderately     : ⚠️ second thought
    → 3 Strongly       : 🚨 reality check
→ Feedback Layer (user can correct the level → stored for retraining)
→ Backend API (FastAPI)
→ Database (PostgreSQL)

---

## Layer Details

| Layer | Technology | Status |
|-------|------------|--------|
| Data input | Python, Pandas | Done |
| Preprocessing | OneHotEncoder + log1p + StandardScaler (in Pipeline) | Done |
| ML prediction model | Neural network / MLPClassifier (Scikit-learn) | Done |
| Graded 4-level output | predict_proba over 4 classes | Done |
| Class balancing | Oversampling on the training set | Done |
| Model validation | 5-fold cross-validation vs baselines | Done |
| Feedback loop | CSV capture + auto-merge on retrain | Done |
| Backend API | FastAPI | In progress |
| Frontend alert system | TBD | In progress |
| Database | PostgreSQL | In progress |

---

## Model Details

- Algorithm: MLPClassifier (neural network), hidden layers (16, 8), ReLU + Adam
- Packaging: scikit-learn Pipeline (preprocessing + model in one `.pkl`) so
  inference always matches training
- Training: epoch-by-epoch with early stopping on a validation set
- Split: 80% train / 10% validation / 10% test (stratified)
- Class balancing: minority levels oversampled in the training set only
- Input features: hour, price, category, frequency, is_essential,
  deliberation_minutes, on_wishlist
- Output: graded impulse level 0–3 with per-level probabilities
- Headline metric: **macro-F1 ≈ 0.67** on the held-out test set (synthetic data)

---

## Design Principle

Impulse buying is modelled as **low deliberation + emotional trigger**, not as
"expensive" or "late at night". This is why an essential, planned purchase
(e.g. a fridge bought at night after moving) is correctly treated as planned.
