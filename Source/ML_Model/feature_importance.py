import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

import config
import preprocess

# ---------------------------------------------------------------------------
# Feature importance for the neural-network pipeline.
#
# A neural network has no built-in `feature_importances_`, so we use
# PERMUTATION IMPORTANCE: shuffle one feature at a time and measure how much
# the F1 score drops. A bigger drop means the model relies more on it.
# Because we pass the whole pipeline, importance is measured on the raw
# input features (the pipeline handles encoding/scaling internally).
# ---------------------------------------------------------------------------
pipeline = joblib.load(config.MODEL_PATH)

df = preprocess.load_dataset()
X, y = preprocess.split_features_target(df)

result = permutation_importance(
    pipeline,
    X,
    y,
    scoring="f1_macro",  # multiclass: average F1 across all four levels
    n_repeats=20,
    random_state=config.RANDOM_STATE,
)
importance = result.importances_mean

# Plot
plt.figure(figsize=(9, 5))
plt.bar(config.FEATURES, importance)
plt.title("Permutation Feature Importance (impact on macro-F1)")
plt.xlabel("Features")
plt.ylabel("Mean drop in macro-F1 when shuffled")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(config.IMPORTANCE_PLOT_PATH)

print("Feature importance chart saved.")
for name, score in sorted(
    zip(config.FEATURES, importance), key=lambda t: t[1], reverse=True
):
    print(f"  {name:22s}: {score:.4f}")
