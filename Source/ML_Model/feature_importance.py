import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

#load dataset
df = pd.read_csv("Data/Synthetic/transactions.csv")

#encode category
encoder = LabelEncoder()
df["category"] = encoder.fit_transform(df["category"])

# Features and target
X = df[["hour", "price", "category", "frequency"]]
y = df["is_impulsive"]

#train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

#feature importance
importance = model.feature_importances_

features = ["hour", "price", "category", "frequency"]

plt.figure(figsize=(8,5))
plt.bar(features, importance)
plt.title("Feature Importance for Impulse Purchase Prediction")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.savefig("Results/Visualizations/feature_importance.png")

print("Feature importance chart saved.")
