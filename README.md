# DBDS_AI04_ImpulseGuard

## Project Overview

ImpulseGuard is an AI-powered system designed to detect impulsive purchases before a transaction is completed. The system analyzes user spending behavior and predicts how impulsive a purchase is, with the goal of helping users make more conscious financial decisions.

---

## Objectives

- Generate synthetic transaction datasets
- Train a machine learning classification model
- Detect impulsive purchase behavior on a graded scale
- Build usable prototype interfaces (Telegram bot, Chrome extension)
- Close the loop with human feedback
- Visualize spending insights

---

## Technologies

- Python
- Pandas
- Scikit-learn
- FastAPI (backend API)
- JavaScript / HTML (Chrome extension)
- python-telegram-bot
- GitHub

---

## Repository Structure

```text
Data/
  Synthetic/
    transactions.csv          ← 10,000 synthetic transactions
  Real_User_Data/
    feedback.csv              ← real user corrections (created at runtime)
Source/
  Data_Generation/
    generate_dataset.py       ← generates synthetic dataset (with label noise)
  ML_Model/
    config.py                 ← paths, feature list, hyperparameters
    preprocess.py             ← shared loading, preprocessing, oversampling
    train_model.py            ← epoch-based training with early stopping
    predict_purchase.py       ← graded 4-level prediction + feedback capture
    compare_models.py         ← cross-validation vs baselines
    feature_importance.py     ← permutation importance chart
  Backend/
    api.py                    ← FastAPI server (/predict, /feedback)
  Bot/
    telegram_bot.py           ← Telegram chatbot interface
  Extension/
    manifest.json             ← Chrome extension config
    popup.html / popup.js     ← extension popup UI and logic
Results/
  Models/
    impulse_pipeline.pkl      ← preprocessing + model in one artifact
  Visualizations/
    feature_importance.png    ← feature importance chart
  Metrics/
    model_metrics.txt         ← macro-F1 + per-level report
Logs/
  predicted_transactions.csv  ← log of every prediction (not used for training)
```

---

## How It Works

### 1. Data Generation
Since real transaction data was unavailable, a synthetic dataset of 10,000 transactions was generated using rule-based logic. Each transaction includes:

| Feature | Description |
|---------|-------------|
| `hour` | Hour of purchase (0–23) |
| `price` | Purchase amount |
| `category` | Product category (clothing, food, electronics, entertainment, home, beauty) |
| `frequency` | How often the user buys in that category (1–20) |
| `is_essential` | Whether the item is a genuine need (1) or a want (0) |
| `deliberation_minutes` | How long the user thought before buying |
| `on_wishlist` | Whether it was planned ahead (1) or not (0) |
| `impulse_level` | Label: 0=none, 1=mild, 2=moderate, 3=strong |

Crucially, the label treats impulse as **low deliberation + emotional
trigger**, not as "expensive" — so a needed, planned purchase (e.g. a fridge
bought late at night after moving) is correctly labeled as planned.
**Gaussian noise** is added to the score before labeling, so the model has to
*learn* the pattern instead of memorizing a fixed rule.

### 2. Model Training
A neural network (MLPClassifier) is trained **epoch by epoch** on an 80/10/10
split. After each epoch its macro-F1 is measured on the validation set, and the
best epoch is kept (early stopping). Minority levels are oversampled in the
training set. Preprocessing and the model are bundled into a single
scikit-learn Pipeline.

```
Train:       8000 transactions (80%)  ← minority levels oversampled
Validation:  1000 transactions (10%)  ← checked after every epoch
Test:        1000 transactions (10%)  ← used once, at the end
Model:       MLPClassifier, hidden layers (16, 8)
```

### 3. Evaluation
The headline metric is **macro-F1** (not accuracy, because the classes are
imbalanced and there are 4 of them), saved in
`Results/Metrics/model_metrics.txt`. `compare_models.py` runs 5-fold
cross-validation against a majority baseline and logistic regression.

### 4. Prediction & Feedback
`predict_purchase.py` returns a **graded impulse level** (none / mild / moderate
/ strong) with per-level probabilities. The user can then correct the level;
the correction is logged to `Data/Real_User_Data/feedback.csv` and folded into
the next training run.

### 5. Interfaces
Two interfaces send the same 7 features to the same model:
- **Telegram bot** — a guided conversation with tappable menus, ending in a
  graded result and a feedback step.
- **Chrome extension** — a popup form that calls the FastAPI backend over
  HTTP and shows a color-coded result with confidence.

---

## Current Results

- Generated synthetic dataset (10,000 transactions, 7 features, with label noise)
- Trained epoch-based neural network (80/10/10 split, early stopping, balanced)
- Achieved **macro-F1 ≈ 0.67** on the held-out test set (synthetic data)
- 5-fold cross-validation: neural net (0.67) far beats the majority baseline
  (0.12) and ties logistic regression (0.66)
- Permutation importance shows deliberation, planning and necessity dominate —
  price is the weakest predictor
- Graded 4-level output and a human-in-the-loop feedback loop
- Working Telegram bot and Chrome extension, both connected to a FastAPI backend
- Shared prediction logging across channels (`telegram` vs `extension`)

---

## System Architecture

```
User checks a purchase (Chrome extension or Telegram bot)
        ↓
FastAPI backend receives the purchase features
        ↓
Neural-network pipeline predicts the impulse level (0-3)
        ↓
Result + confidence shown to the user
        ↓
Optional feedback saved to CSV for the next training run
```

---

## Limitations

- Dataset is synthetic — labels come from a (now noisy) rule-based generator
- The neural network does not clearly beat a simple logistic regression here
- No personalization yet (impulse is relative to each individual user)
- Runs locally only — the backend is not yet hosted/deployed
- Feedback volume is still small, so its effect on retraining is limited so far

---

## Team

| Name | Role |
|------|------|
| Tina | Team Lead / AI & Backend Developer |
| Danna | Frontend / UX & Chatbot Developer |
| Yewon | Data Analyst & Product Integration |
| Zoe | Documentation & Code Quality Manager |

---

## Logs

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

### 1 June 2026
- Updated README and all Reports documentation
- Added model results analysis
- Added feature importance analysis
- Developed Random Forest model with Joblib persistence
- Implemented prediction system with explainable AI output
- Developed Telegram bot with ML integration
- Created project presentation slides

### 26 June 2026
- Switched headline metric from accuracy to F1 (imbalanced classes)
- Rebuilt training as an epoch-based neural network with an 80/10/10 split and early stopping
- Fixed the "expensive purchase = impulsive" problem by adding necessity and deliberation features and rewriting the labels
- Added a human-in-the-loop feedback loop
- Refactored into shared config/preprocess modules and a single Pipeline artifact
- Changed the output to a graded 4-level scale (none / mild / moderate / strong)
- Expanded the dataset to 10,000 rows and added class-balancing (oversampling)
- Added label noise so the model learns the pattern instead of memorizing a rule (macro-F1 ~0.90 → ~0.67, an honest number)
- Gave price a mild non-essential role; decoupled deliberation from wishlist; log-transformed deliberation
- Added cross-validation + baseline comparison (neural net ties logistic regression)

### 29 June 2026
- Updated the Telegram bot to use the new pipeline (7 features, graded 4-level result)
- Overhauled Telegram bot UX (persistent menu button, plain text, time-bucket and frequency menus)
- Closed the feedback loop on every channel with a shared `preprocess.save_feedback()` helper
- Built the FastAPI backend (`Source/Backend/api.py`) with `/predict`, `/feedback`, and `/docs`
- Built the Chrome extension MVP (`manifest.json`, `popup.html`, `popup.js`)
- Added shared prediction logging with a `channel` column to distinguish Telegram vs extension usage
- Retrained the model with feedback merged in; re-saved the model artifact for the local scikit-learn version
- Cleaned up `.gitignore` to keep runtime/user data out of the repository
