import pandas as pd
import numpy as np
import random

#config 
np.random.seed(42)
random.seed(42)

num_samples = 500

categories = ["clothing", "food", "electronics", "entertainment", "home", "beauty"]

#generate data
data = []

for i in range(num_samples):

    hour = random.randint(0, 23)
    price = round(random.uniform(5, 500), 2)
    category = random.choice(categories)
    frequency = random.randint(1, 20)  # how often user buys similar items

    #rule based label (synthetic)
    impulsive_score = 0

    if hour >= 22 or hour <= 5:
        impulsive_score += 1  # late night buying

    if price > 200:
        impulsive_score += 1  # expensive item

    if frequency > 10:
        impulsive_score += 1  # repeated buying

    if category in ["clothing", "beauty", "entertainment"]:
        impulsive_score += 1  # emotional categories

    label = 1 if impulsive_score >= 2 else 0  # 1 = impulsive

    data.append([
        hour,
        price,
        category,
        frequency,
        label
    ])

#create dataframe
df = pd.DataFrame(data, columns=[
    "hour",
    "price",
    "category",
    "frequency",
    "is_impulsive"
])

#save dataset
df.to_csv("Data/Synthetic/transactions.csv", index=False)

print("Dataset created successfully!")
print(df.head())
