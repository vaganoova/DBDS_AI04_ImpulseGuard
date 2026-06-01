# System Design

## Overview

The system consists of five layers that work together to detect impulsive
purchases before a transaction is completed.

The AI model analyzes transaction features and predicts impulsive behavior
in real time.

---

## System Architecture

User attempts purchase
→ Data Input Layer (hour, price, category, frequency)
→ Preprocessing Layer (LabelEncoder for category)
→ ML Prediction Model (Random Forest Classifier)
→ If impulsive: Alert sent to user
→ If planned: Purchase proceeds
→ Backend API (FastAPI)
→ Database (PostgreSQL)

---

## Layer Details

| Layer | Technology | Status |
|-------|------------|--------|
| Data input | Python, Pandas | Done |
| Preprocessing | LabelEncoder (Scikit-learn) | Done |
| ML prediction model | Random Forest (Scikit-learn) | Done |
| Backend API | FastAPI | In progress |
| Frontend alert system | TBD | In progress |
| Database | PostgreSQL | In progress |

---

## Model Details

- Algorithm: Random Forest Classifier
- Estimators: 100 trees
- Train/test split: 80/20
- Input features: hour, price, category, frequency
- Output: 0 (planned) or 1 (impulsive)
- Accuracy: ~97% on synthetic data
