import joblib
import pandas as pd

#load saved model and encoder
model = joblib.load("Results/Models/impulse_model.pkl")
encoder = joblib.load("Results/Models/category_encoder.pkl")

hour = int(input("Purchase hour (0-23): "))
price = float(input("Price (€): "))
category = input("Category: ").lower()
frequency = int(input("Frequency of similar purchases: "))

purchase = {
    "hour": hour,
    "price": price,
    "category": category,
    "frequency": frequency
}

#convert category text into encoded number
purchase["category"] = encoder.transform([purchase["category"]])[0]

#convert input into dataframe
purchase_df = pd.DataFrame([purchase])

#make prediction
prediction = model.predict(purchase_df)[0]

risk_factors = []

if hour >= 22 or hour <= 5:
    risk_factors.append("Late-night purchase")

if price > 200:
    risk_factors.append("High purchase amount")

if frequency > 10:
    risk_factors.append("High frequency of similar purchases")

if category in ["clothing", "beauty", "entertainment"]:
    risk_factors.append("Non-essential / emotional spending category")

#output result
if prediction == 1:
    print("\n🚨 ImpulseGuard Alert!")
    print("This purchase may be impulsive.")

    if risk_factors:
        print("\nRisk factors detected:")
        for factor in risk_factors:
            print(f"- {factor}")

    print("\nReality check: Do you really need this right now?")
else:
    print("\n✅ ImpulseGuard Assessment")
    print("This purchase seems planned or low-risk.")
