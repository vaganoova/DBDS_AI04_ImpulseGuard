# Technologies

## Overview

ImpulseGuard is built using Python-based data science and web technologies.
Each technology was selected for a specific purpose in the AI pipeline.

---

## Technology Stack

| Technology | Purpose | Status |
|------------|---------|--------|
| Python | Primary programming language | In use |
| Pandas | Data loading and manipulation | In use |
| NumPy | Numerical operations | In use |
| Scikit-learn | ML model training and evaluation | In use |
| XGBoost | Advanced ML model (planned) | Planned |
| FastAPI | Backend REST API (serves the model) | In use |
| JavaScript / HTML | Chrome extension popup | In use |
| python-telegram-bot | Telegram chatbot interface | In use |
| Matplotlib | Data visualization | In use |
| Seaborn | Statistical charts | In use |
| Jupyter | Exploratory data analysis | In use |
| GitHub | Version control and collaboration | In use |

---

## Why These Technologies?

- **Python** — Industry standard for data science and ML projects
- **Pandas + NumPy** — Essential for data processing and feature engineering
- **Scikit-learn** — Provides the neural network (MLPClassifier), the preprocessing
  steps (OneHotEncoder, log/StandardScaler), the Pipeline that bundles them, and the
  baselines used for comparison (DummyClassifier, LogisticRegression) with cross-validation
- **XGBoost** — Planned as an additional baseline to compare against the neural network
- **FastAPI** — Lightweight, fast framework serving the model as a REST API for the Chrome extension
- **Chrome extension + Telegram bot** — The two user-facing ways to check a purchase
- **Data storage** — Synthetic data, feedback and prediction logs are kept as CSV files (no database needed at this stage)
- **Matplotlib + Seaborn** — Used for feature importance visualization and data exploration
