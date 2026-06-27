"""Shared data loading and preprocessing.

Keeping this in one place means the training, prediction and analysis
scripts cannot drift apart (which is what caused the earlier scaler bug).
"""

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.utils import resample

import config

# deliberation_minutes spans 4 orders of magnitude (1 .. ~20000), so we
# log-transform it before scaling. The remaining numeric features are scaled
# directly.
LOG_FEATURE = "deliberation_minutes"
PLAIN_NUMERIC = [f for f in config.NUMERIC_FEATURES if f != LOG_FEATURE]


def load_dataset():
    """Load the dataset as a DataFrame.

    If real user feedback has been collected (Data/Real_User_Data/feedback.csv),
    it is appended to the synthetic data so the model learns from real,
    user-corrected examples on the next training run.
    """
    df = pd.read_csv(config.DATA_PATH)

    if os.path.exists(config.FEEDBACK_PATH):
        feedback = pd.read_csv(config.FEEDBACK_PATH)
        df = pd.concat([df, feedback], ignore_index=True)
        print(f"Loaded {len(feedback)} real feedback example(s) on top of synthetic data.")

    return df


def split_features_target(df):
    """Return (X, y) using the feature list from config."""
    return df[config.FEATURES], df[config.TARGET]


def build_preprocessor():
    """Build the preprocessing step.

    - category              -> one-hot encoded (no false ordering)
    - deliberation_minutes  -> log1p then standardised (huge dynamic range)
    - other numeric         -> standardised (needed for the neural network)

    Bundling this into the model Pipeline guarantees that prediction uses
    exactly the same preprocessing as training.
    """
    log_pipeline = Pipeline(
        steps=[
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
            ("log", log_pipeline, [LOG_FEATURE]),
            ("num", StandardScaler(), PLAIN_NUMERIC),
        ]
    )


def save_feedback(purchase, level):
    """Append one user-corrected example to the feedback CSV for retraining.

    `purchase` is a dict of the 7 raw features; `level` is the true impulse
    level (0-3) the user reported. Shared by the CLI and the Telegram bot so
    both write feedback the same way.
    """
    row = dict(purchase)
    row[config.TARGET] = int(level)
    os.makedirs(os.path.dirname(config.FEEDBACK_PATH), exist_ok=True)
    write_header = not os.path.exists(config.FEEDBACK_PATH)
    pd.DataFrame([row])[config.FEATURES + [config.TARGET]].to_csv(
        config.FEEDBACK_PATH, mode="a", header=write_header, index=False
    )


def balance_by_oversampling(X_train, y_train, random_state=config.RANDOM_STATE):
    """Oversample minority classes in the TRAINING set up to the largest class.

    MLPClassifier has no `class_weight`, so we balance by resampling the rare
    levels (e.g. "strongly impulsive") with replacement. Apply ONLY to the
    training split — never to validation or test.
    """
    train_df = X_train.copy()
    train_df[config.TARGET] = y_train.values
    max_count = train_df[config.TARGET].value_counts().max()

    balanced_parts = [
        resample(group, replace=True, n_samples=max_count, random_state=random_state)
        for _, group in train_df.groupby(config.TARGET)
    ]
    balanced_df = pd.concat(balanced_parts).sample(frac=1, random_state=random_state)
    return balanced_df[config.FEATURES], balanced_df[config.TARGET]
