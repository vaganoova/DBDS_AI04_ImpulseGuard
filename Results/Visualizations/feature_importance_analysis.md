# Feature Importance Analysis

## Overview

Feature importance shows how much each feature contributed to the model's
prediction. The higher the value, the more the model relied on that feature
to decide whether a purchase is impulsive or planned.

---

## Results

| Rank | Feature | Importance Score |
|------|---------|-----------------|
| 1 | price | 0.29 |
| 2 | frequency | 0.26 |
| 3 | category | 0.26 |
| 4 | hour | 0.19 |

Total: 1.00 (all features combined)

---

## Interpretation

### 1. price (0.29) — Most important
Price is the strongest predictor of impulsive behavior in our model.
This aligns with research showing that 60.7% of impulse buyers have
spent $100 or more on a single unplanned purchase. Higher-priced
purchases are more likely to trigger post-purchase regret, which is
a key indicator of impulsive behavior.

### 2. frequency (0.26) — Second most important
How often a user buys in the same category is nearly as important as
price. Repeated buying in the same category indicates habitual or
compulsive behavior, which correlates strongly with impulse buying
tendencies.

### 3. category (0.26) — Equal to frequency
Product category plays an equally significant role. Emotional
categories such as clothing, beauty, and entertainment are
consistently linked to higher impulse purchase rates across
research studies.

### 4. hour (0.19) — Least important but still significant
Time of day is the least influential feature, but still contributes
19% to the model's decisions. Research shows 74% of impulse purchases
happen at night, which supports its inclusion as a feature.

---

## Key Insight

All four features contribute relatively equally to the model, with no
single feature dominating. This suggests that impulse buying is a
multi-factor behavior — no single signal is enough on its own, but
the combination of price, frequency, category, and time of day
creates a strong predictive pattern.

---

## Connection to Research

The feature importance results are consistent with impulse buying
research findings:

| Feature | Research Backing | Model Importance |
|---------|-----------------|-----------------|
| price | High-price purchases linked to regret | 0.29 (highest) |
| frequency | Repeated buying = impulsive tendency | 0.26 |
| category | Clothing/beauty/entertainment = emotional | 0.26 |
| hour | 74% of impulse buys happen at night | 0.19 |
