import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("Data/Synthetic/transactions.csv")

# Convert category text to numbers
encoder = LabelEncoder()
df["category"] = encoder.fit_transform(df["category"])

# Features and target
X = df[["hour", "price", "category", "frequency"]]
y = df["is_impulsive"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

os.makedirs("Results/Models", exist_ok=True)

joblib.dump(model, "Results/Models/impulse_model.pkl")
joblib.dump(encoder, "Results/Models/category_encoder.pkl")

print("Model saved successfully.")

# Predictions
predictions = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(report)

# Save metrics
with open("Results/Metrics/model_metrics.txt", "w") as file:
    file.write(f"Accuracy: {accuracy:.2%}\n\n")
    file.write("Classification Report:\n\n")
    file.write(report)

print("\nMetrics saved to Results/Metrics/model_metrics.txt")
