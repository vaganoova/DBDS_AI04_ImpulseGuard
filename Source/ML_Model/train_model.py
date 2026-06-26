import copy
import os

import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

import config
import preprocess

# ---------------------------------------------------------------------------
# Load and split: 80% train / 10% validation / 10% test (stratified)
# ---------------------------------------------------------------------------
df = preprocess.load_dataset()
X, y = preprocess.split_features_target(df)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.10, random_state=config.RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=1 / 9,  # 1/9 of 90% -> 10% of the full dataset
    random_state=config.RANDOM_STATE,
    stratify=y_train_val,
)
# Record the natural split sizes BEFORE oversampling, for honest reporting.
n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
print(f"Split sizes -> train: {n_train}, validation: {n_val}, test: {n_test}")

# ---------------------------------------------------------------------------
# Class balancing (only on the TRAINING set) via oversampling.
# Note: this is safe here even though it duplicates rows, because early
# stopping below watches the *natural* (un-oversampled) validation set — if
# duplication caused overfitting, val macro-F1 would drop and we'd keep an
# earlier epoch automatically.
# ---------------------------------------------------------------------------
X_train, y_train = preprocess.balance_by_oversampling(X_train, y_train)
print(
    "After balancing, train level counts:\n"
    + y_train.value_counts().sort_index().to_string()
)

# ---------------------------------------------------------------------------
# Fit preprocessing on TRAIN only, then transform every split.
# ---------------------------------------------------------------------------
preprocessor = preprocess.build_preprocessor()
X_train_t = preprocessor.fit_transform(X_train)
X_val_t = preprocessor.transform(X_val)
X_test_t = preprocessor.transform(X_test)

# ---------------------------------------------------------------------------
# Train the neural network epoch by epoch (partial_fit = one pass per call).
# We keep the model from the epoch with the BEST validation F1 (early stopping),
# instead of the last epoch, to avoid using an overfitted model.
# ---------------------------------------------------------------------------
model = MLPClassifier(
    hidden_layer_sizes=config.HIDDEN_LAYER_SIZES,
    activation="relu",
    solver="adam",
    learning_rate_init=config.LEARNING_RATE_INIT,
    random_state=config.RANDOM_STATE,
)

best_val_f1 = -1.0
best_epoch = -1
best_model = None

# Multiclass target -> use macro-F1 (averages F1 across all four levels
# equally, so the small "strongly impulsive" class still counts).
print("\nEpoch training (validation macro-F1 after each epoch):")
for epoch in range(1, config.EPOCHS + 1):
    model.partial_fit(X_train_t, y_train, classes=config.CLASSES)

    train_f1 = f1_score(y_train, model.predict(X_train_t), average="macro")
    val_f1 = f1_score(y_val, model.predict(X_val_t), average="macro")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        best_model = copy.deepcopy(model)  # snapshot the best model so far

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"  epoch {epoch:3d} | train F1: {train_f1:.3f} | val F1: {val_f1:.3f}"
        )

print(f"\nBest validation F1: {best_val_f1:.3f} (epoch {best_epoch}) -> kept this model")

# ---------------------------------------------------------------------------
# Bundle the fitted preprocessor + best model into ONE pipeline artifact.
# Prediction then loads a single file and cannot mismatch preprocessing.
# ---------------------------------------------------------------------------
pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", best_model)])

os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
joblib.dump(pipeline, config.MODEL_PATH)
print(f"\nPipeline saved to {config.MODEL_PATH}")

# ---------------------------------------------------------------------------
# Final evaluation on the held-out TEST set (used only once).
# Headline metric: macro-F1 (averages all 4 levels equally), with weighted-F1
# and a full per-level precision/recall/F1 report alongside.
# Note: the per-epoch "train F1" printed above is measured on the balanced
# (oversampled) training set, while val/test use the natural class mix — so
# the two are not directly comparable.
# ---------------------------------------------------------------------------
test_predictions = best_model.predict(X_test_t)

macro_f1 = f1_score(y_test, test_predictions, average="macro")
weighted_f1 = f1_score(y_test, test_predictions, average="weighted")
target_names = [config.LEVEL_NAMES[c] for c in config.CLASSES]
report = classification_report(
    y_test, test_predictions, labels=config.CLASSES, target_names=target_names
)

print("\n=== Final TEST results ===")
print(f"Macro F1 (all 4 levels equally): {macro_f1:.3f}")
print(f"Weighted F1 (by class size):     {weighted_f1:.3f}")
print("\nClassification Report (per level):")
print(report)

os.makedirs(os.path.dirname(config.METRICS_PATH), exist_ok=True)
with open(config.METRICS_PATH, "w") as file:
    file.write("Model: MLPClassifier (neural network) in a sklearn Pipeline\n")
    file.write("Target: graded impulse level (0=none, 1=mild, 2=moderate, 3=strong)\n")
    file.write(
        f"Trained up to {config.EPOCHS} epochs | best validation macro-F1 "
        f"{best_val_f1:.3f} at epoch {best_epoch} (early stopping)\n"
    )
    file.write(
        f"Split (before oversampling): train {n_train} / val {n_val} / test {n_test}\n\n"
    )
    file.write("=== Final TEST results (headline metric: macro-F1) ===\n")
    file.write(f"Macro F1 (all 4 levels equally): {macro_f1:.3f}\n")
    file.write(f"Weighted F1 (by class size):     {weighted_f1:.3f}\n\n")
    file.write("Classification Report (per level):\n\n")
    file.write(report)

print(f"\nMetrics saved to {config.METRICS_PATH}")
