"""Cross-validated comparison of the model against simple baselines.

Why this exists: a single train/val/test split gives ONE macro-F1 number from
one lucky/unlucky shuffle, and on its own that number means little. This script:

  1. Runs k-fold cross-validation (so we get mean +/- std, not one number).
  2. Compares the neural network against two baselines:
       - "majority"  : always predict the most common level (a sanity floor)
       - "logistic"  : logistic regression (a simple, strong tabular model)
  3. Oversamples INSIDE each fold's training data only (no leakage).

If the neural network can't beat the majority baseline, it has learned nothing;
if it can't beat logistic regression, the extra complexity isn't paying off.
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import config
import preprocess

# The baseline MLP is capped at EPOCHS iterations for a fair comparison; it may
# not fully converge in that budget, which is fine here.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def make_model(name):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logistic":
        return LogisticRegression(max_iter=1000)
    if name == "neural_net":
        return MLPClassifier(
            hidden_layer_sizes=config.HIDDEN_LAYER_SIZES,
            activation="relu",
            solver="adam",
            learning_rate_init=config.LEARNING_RATE_INIT,
            max_iter=config.EPOCHS,
            random_state=config.RANDOM_STATE,
        )
    raise ValueError(name)


def cross_validate(name, X, y):
    """Return per-fold macro-F1 scores for one model."""
    skf = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # Balance + fit preprocessing on the TRAINING fold only.
        X_tr, y_tr = preprocess.balance_by_oversampling(X_tr, y_tr)
        pre = preprocess.build_preprocessor()
        X_tr_t = pre.fit_transform(X_tr)
        X_te_t = pre.transform(X_te)

        model = make_model(name)
        model.fit(X_tr_t, y_tr)
        preds = model.predict(X_te_t)
        scores.append(f1_score(y_te, preds, average="macro"))
    return np.array(scores)


def main():
    df = preprocess.load_dataset()
    X, y = preprocess.split_features_target(df)

    print(f"{config.CV_FOLDS}-fold cross-validation (macro-F1):\n")
    print(f"{'Model':<14}{'mean':>8}{'std':>8}   per-fold")
    print("-" * 60)
    for name in ["majority", "logistic", "neural_net"]:
        scores = cross_validate(name, X, y)
        per_fold = " ".join(f"{s:.3f}" for s in scores)
        print(f"{name:<14}{scores.mean():>8.3f}{scores.std():>8.3f}   {per_fold}")

    print(
        "\nReading this: the neural net should clearly beat 'majority'. "
        "If it only ties 'logistic', the simpler model may be enough."
    )


if __name__ == "__main__":
    main()
