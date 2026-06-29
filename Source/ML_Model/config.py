"""Central configuration for the ImpulseGuard ML model.

Paths, the feature list and hyperparameters live here so the training,
prediction and analysis scripts all share one source of truth.
"""

# --- File paths (relative to the project root) ---------------------------
DATA_PATH = "Data/Synthetic/transactions.csv"
MODEL_PATH = "Results/Models/impulse_pipeline.pkl"
METRICS_PATH = "Results/Metrics/model_metrics.txt"
IMPORTANCE_PLOT_PATH = "Results/Visualizations/feature_importance.png"
FEEDBACK_PATH = "Data/Real_User_Data/feedback.csv"
PREDICTIONS_LOG_PATH = "Logs/predicted_transactions.csv"

# --- Features ------------------------------------------------------------
CATEGORICAL_FEATURES = ["category"]
NUMERIC_FEATURES = [
    "hour",
    "price",
    "frequency",
    "is_essential",
    "deliberation_minutes",
    "on_wishlist",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "impulse_level"

# Graded (ordinal) target: impulse buying is a spectrum, not a yes/no.
CLASSES = [0, 1, 2, 3]
LEVEL_NAMES = {
    0: "Not impulsive",
    1: "Mildly impulsive",
    2: "Moderately impulsive",
    3: "Strongly impulsive",
}

# --- Training hyperparameters --------------------------------------------
EPOCHS = 100
RANDOM_STATE = 42
HIDDEN_LAYER_SIZES = (16, 8)
LEARNING_RATE_INIT = 0.01
CV_FOLDS = 5
