# DBDS_AI04_ImpulseGuard

## Project Overview

ImpulseGuard is an AI-powered system designed to detect impulsive purchases before a transaction is completed. The system analyzes user spending behavior and predicts whether a purchase is impulsive or planned, with the goal of helping users make more conscious financial decisions.

---

## Objectives

- Generate synthetic transaction datasets
- Train a machine learning classification model
- Detect impulsive purchase behavior
- Build a prototype alert system
- Visualize spending insights

---

## Technologies

- Python
- Pandas
- Scikit-learn
- XGBoost
- FastAPI
- PostgreSQL
- GitHub

---

## Repository Structure

```text
Data/
  Synthetic/
    transactions.csv        ← 500 synthetic transactions
Source/
  data_generator.py         ← generates synthetic dataset
  model_trainer.py          ← initial model training script
  model_evaluator.py        ← model training with evaluation metrics
Documentation/
  research_impulse_buying.md
  feature_selection_rationale.md
Results/
  Visualizations/
    feature_importance.png  ← feature importance chart
  Metrics/
    model_metrics.txt       ← accuracy and classification report
Logs/
```

---

## How It Works

### 1. Data Generation
Since real transaction data was unavailable, a synthetic dataset of 500 transactions was generated using rule-based logic. Each transaction includes:

| Feature | Description |
|---------|-------------|
| `hour` | Hour of purchase (0–23) |
| `price` | Purchase amount |
| `category` | Product category (clothing, food, electronics, entertainment, home, beauty) |
| `frequency` | How often the user buys in that category (1–20) |
| `is_impulsive` | Label: 1 = impulsive, 0 = planned |

### 2. Model Training
A Random Forest classifier was trained on the dataset using an 80/20 train/test split.

```
Training set: 400 transactions
Test set:     100 transactions
Model:        Random Forest (100 estimators)
```

### 3. Evaluation
The model was evaluated using accuracy score and a full classification report (precision, recall, F1-score), saved to `Results/Metrics/model_metrics.txt`.

---

## Current Results

- Generated synthetic dataset (500 transactions)
- Trained Random Forest classifier (80/20 split)
- Achieved ~97% accuracy on synthetic data
- Generated feature importance analysis

---

## System Architecture (Planned)

```
User attempts purchase
        ↓
FastAPI receives transaction data
        ↓
AI model predicts: impulsive or planned
        ↓
Impulsive → Alert sent to user
Planned   → Transaction proceeds
        ↓
Result stored in PostgreSQL
```

---

## Limitations

- Dataset is synthetic — real-world accuracy may differ
- Rule-based data generation means the model learns predefined patterns
- Model file (.pkl) not yet saved for deployment
- FastAPI and PostgreSQL integration not yet implemented

---

## Team

| Name | Role |
|------|------|
| Tina | Team Lead / AI & Backend Developer |
| Danna | Frontend / UX & Prototype Designer |
| Yewon | Research & Documentation |
| Zoe | Research & Documentation |

---

## Daily Logs

### 22 May 2026
- Created GitHub repository
- Defined repository structure
- Selected project technologies
- Planned AI workflow
- Designed initial system architecture
- Conducted research on impulse buying behavior
- Selected dataset features based on research findings

### 30 May 2026
- Synthetic dataset generation
- First ML model
- Model evaluation
