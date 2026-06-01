import joblib
import pandas as pd

#load saved model and encoder
model = joblib.load("Results/Models/impulse_model.pkl")
encoder = joblib.load("Results/Models/category_encoder.pkl")

#example purchase input
purchase = {
    "hour": 23,
    "price": 250,
    "category": "clothing",
    "frequency": 15
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
