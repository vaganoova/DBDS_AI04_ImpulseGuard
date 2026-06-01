import joblib
import pandas as pd

#load saved model and encoder
model = joblib.load("Results/Models/impulse_model.pkl")
encoder = joblib.load("Results/Models/category_encoder.pkl")

hour = int(input("Purchase hour (0-23): "))
price = float(input("Price: (€): "))
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

#output result
if prediction == 1:
    print("ImpulseGuard Alert!")
    print("This purchase may be impulsive.")
    print("Reality check: Do you really need this right now?")
else:
    print("This purchase seems planned or low-risk.")
