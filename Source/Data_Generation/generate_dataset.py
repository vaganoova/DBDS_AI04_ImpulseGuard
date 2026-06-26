import pandas as pd
import numpy as np
import random

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
np.random.seed(42)
random.seed(42)

num_samples = 10000

categories = ["clothing", "food", "electronics", "entertainment", "home", "beauty"]

# How likely a purchase in this category is an ESSENTIAL need (vs a want).
# This is what lets a fridge (home appliance) avoid being called impulsive.
essential_probability = {
    "food": 0.90,
    "home": 0.60,         # appliances are essential, decor is not
    "electronics": 0.40,  # a work laptop vs a gadget
    "clothing": 0.30,     # basics vs fashion
    "beauty": 0.10,
    "entertainment": 0.05,
}

# Categories that tend to be emotional / "want" driven purchases.
emotional_categories = ["clothing", "beauty", "entertainment"]

# ---------------------------------------------------------------------------
# Generate data
# ---------------------------------------------------------------------------
data = []

for i in range(num_samples):

    hour = random.randint(0, 23)
    price = round(random.uniform(5, 500), 2)
    category = random.choice(categories)
    frequency = random.randint(1, 20)  # how often user buys similar items

    # Is this purchase an essential need? Mostly driven by category, plus noise
    # so the model can't perfectly infer it from the category alone.
    is_essential = 1 if random.random() < essential_probability[category] else 0

    # Was the item planned ahead of time (on a wishlist / shopping list)?
    on_wishlist = 1 if random.random() < 0.40 else 0

    # How long the user deliberated before buying, in minutes.
    # Drawn from OVERLAPPING lognormal distributions: wishlist items skew
    # longer, but the ranges overlap heavily, so deliberation is NOT just a
    # deterministic function of on_wishlist (more realistic, harder task).
    if on_wishlist:
        deliberation_minutes = np.random.lognormal(mean=6.0, sigma=1.5)  # median ~400 min
    else:
        deliberation_minutes = np.random.lognormal(mean=3.0, sigma=2.0)  # median ~20 min
    deliberation_minutes = int(max(1, min(deliberation_minutes, 20160)))  # clamp 1 min .. 2 weeks

    # -----------------------------------------------------------------------
    # Label logic (synthetic).
    #
    # KEY IDEA: impulse = lack of deliberation + emotional trigger,
    # NOT price and NOT just the clock. Essential and planned purchases are
    # pulled away from "impulsive" so a needed fridge bought at night is fine.
    # -----------------------------------------------------------------------
    impulse_score = 0

    # Deliberation: the strongest real signal of impulse.
    if deliberation_minutes < 10:
        impulse_score += 2      # bought almost instantly
    elif deliberation_minutes < 60:
        impulse_score += 1

    # Planning.
    if on_wishlist == 1:
        impulse_score -= 2      # clearly planned
    else:
        impulse_score += 1

    # Necessity — this is the fix for the "fridge at night" case.
    if is_essential == 1:
        impulse_score -= 2      # needs are not impulse, even if pricey/late

    # Emotional context (weaker signals).
    if hour >= 22 or hour <= 5:
        impulse_score += 1      # late-night / emotional state
    if category in emotional_categories:
        impulse_score += 1      # want-driven category
    if frequency > 10:
        impulse_score += 1      # repeated unplanned buying

    # Price plays a MILD role, but only for non-essential ("want") purchases:
    # an expensive discretionary buy leans more impulsive. Essentials are
    # exempt, so a needed fridge is never pushed up by its price.
    if is_essential == 0 and price > 300:
        impulse_score += 1

    # Label noise: real behaviour is not a clean rule. Adding Gaussian noise to
    # the score before bucketing means identical-looking purchases do not always
    # get the same label, so the model must LEARN the pattern (and cannot simply
    # memorise a lookup table). This makes the metrics meaningful.
    noisy_score = impulse_score + np.random.normal(0, 1.0)

    # Graded label: instead of a binary 0/1, map the score to a DEGREE of
    # impulsiveness. This reflects that impulse buying is a spectrum.
    #   <= 0 : not impulsive
    #   1-2  : mildly impulsive
    #   3-4  : moderately impulsive
    #   >= 5 : strongly impulsive
    if noisy_score <= 0:
        impulse_level = 0
    elif noisy_score <= 2:
        impulse_level = 1
    elif noisy_score <= 4:
        impulse_level = 2
    else:
        impulse_level = 3

    data.append([
        hour,
        price,
        category,
        frequency,
        is_essential,
        deliberation_minutes,
        on_wishlist,
        impulse_level,
    ])

# ---------------------------------------------------------------------------
# Create dataframe
# ---------------------------------------------------------------------------
df = pd.DataFrame(data, columns=[
    "hour",
    "price",
    "category",
    "frequency",
    "is_essential",
    "deliberation_minutes",
    "on_wishlist",
    "impulse_level",
])

# Save dataset
df.to_csv("Data/Synthetic/transactions.csv", index=False)

print("Dataset created successfully!")
print(f"Samples: {len(df)}")
print("\nLevel balance (0=none, 1=mild, 2=moderate, 3=strong):")
print(df["impulse_level"].value_counts().sort_index())
print("\nFirst rows:")
print(df.head())
