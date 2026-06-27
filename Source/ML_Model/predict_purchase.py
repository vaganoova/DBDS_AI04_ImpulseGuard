import joblib
import pandas as pd

import config
import preprocess

# Load the trained pipeline (preprocessing + model in one artifact)
pipeline = joblib.load(config.MODEL_PATH)

# ---------------------------------------------------------------------------
# Collect the purchase details
# ---------------------------------------------------------------------------
hour = int(input("Purchase hour (0-23): "))
price = float(input("Price (€): "))
category = input("Category: ").lower()
frequency = int(input("Frequency of similar purchases: "))
is_essential = int(input("Is this an essential need? (1 = yes, 0 = no): "))
deliberation_minutes = int(input("Minutes spent deciding before buying: "))
on_wishlist = int(input("Was it planned / on a wishlist? (1 = yes, 0 = no): "))

purchase = {
    "hour": hour,
    "price": price,
    "category": category,
    "frequency": frequency,
    "is_essential": is_essential,
    "deliberation_minutes": deliberation_minutes,
    "on_wishlist": on_wishlist,
}

# Build a one-row DataFrame in the expected column order. The pipeline handles
# encoding and scaling, so we pass the raw values straight in.
purchase_df = pd.DataFrame([purchase])[config.FEATURES]

# Predict the impulse LEVEL (0-3), plus how confident the model is in it.
probabilities = pipeline.predict_proba(purchase_df)[0]
predicted_level = int(pipeline.predict(purchase_df)[0])
confidence = probabilities[list(pipeline.classes_).index(predicted_level)]

# ---------------------------------------------------------------------------
# Explain the assessment using the REAL impulse signals.
# ---------------------------------------------------------------------------
risk_factors = []
if deliberation_minutes < 10:
    risk_factors.append("Bought almost instantly (little deliberation)")
if on_wishlist == 0:
    risk_factors.append("Not planned ahead / not on a wishlist")
if hour >= 22 or hour <= 5:
    risk_factors.append("Late-night purchase")
if category in ["clothing", "beauty", "entertainment"]:
    risk_factors.append("Non-essential / emotional spending category")
if frequency > 10:
    risk_factors.append("High frequency of similar purchases")

protective_factors = []
if is_essential == 1:
    protective_factors.append("This is an essential need")
if on_wishlist == 1:
    protective_factors.append("Planned ahead / on a wishlist")
if deliberation_minutes >= 1440:
    protective_factors.append("Considered for a long time before buying")

# ---------------------------------------------------------------------------
# Respond by DEGREE of impulsiveness (0-3), not a blunt yes/no.
# ---------------------------------------------------------------------------
level_response = {
    0: ("✅", "This purchase seems planned or low-risk."),
    1: ("🟡", "Slightly impulsive — probably fine, but stay aware."),
    2: ("⚠️", "Moderately impulsive — worth a second thought."),
    3: ("🚨", "Strongly impulsive — reality check: do you really need this now?"),
}
icon, message = level_response[predicted_level]

print(f"\n{icon} ImpulseGuard: {config.LEVEL_NAMES[predicted_level]} "
      f"({confidence:.0%} confidence)")
print(message)

# Show the full spread across levels so the "in between" is visible.
print("\nLevel probabilities:")
for cls, prob in zip(pipeline.classes_, probabilities):
    print(f"  {config.LEVEL_NAMES[cls]:22s}: {prob:.0%}")

if predicted_level >= 1 and risk_factors:
    print("\nThings to consider:")
    for factor in risk_factors:
        print(f"- {factor}")
elif predicted_level == 0 and protective_factors:
    print("\nWhy this looks fine:")
    for factor in protective_factors:
        print(f"- {factor}")

# ---------------------------------------------------------------------------
# Feedback loop (human-in-the-loop).
# Let the user tell us what really happened. We store the purchase together
# with the TRUE label so train_model.py can learn from real examples next
# time. This is how the model gradually adapts to the user's real habits.
# ---------------------------------------------------------------------------
answer = input(
    "\nWhat was the true level? "
    "(0=none, 1=mild, 2=moderate, 3=strong, Enter = skip): "
).strip()

if answer in ("0", "1", "2", "3"):
    preprocess.save_feedback(purchase, int(answer))
    print("Thanks! Your feedback was saved and will improve the next training run.")
else:
    print("No feedback recorded.")
