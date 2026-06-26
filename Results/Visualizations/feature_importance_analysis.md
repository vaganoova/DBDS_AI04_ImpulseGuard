# Feature Importance Analysis

## Overview

Because the model is a neural network (which has no built-in
`feature_importances_`), we measure importance with **permutation importance**:
each feature is shuffled in turn and we record how much the macro-F1 score
drops. A larger drop means the model relies on that feature more.

---

## Results

| Rank | Feature | Importance (macro-F1 drop when shuffled) |
|------|---------|------------------------------------------|
| 1 | on_wishlist | 0.235 |
| 2 | is_essential | 0.181 |
| 3 | deliberation_minutes | 0.140 |
| 4 | category | 0.076 |
| 5 | frequency | 0.052 |
| 6 | hour | 0.045 |
| 7 | price | 0.040 |

---

## Key Insight: impulse is about deliberation and planning, not price

The behavioural/context features dominate:

- **on_wishlist** — whether the purchase was planned ahead (strongest signal).
- **is_essential** — whether the item is a genuine need.
- **deliberation_minutes** — how long the user thought before buying.

**price** sits at the bottom (0.040). It now has a *mild* role in the data
(an expensive non-essential purchase leans slightly more impulsive), which is
why it is no longer near-zero — but it remains the weakest feature. This is the
core lesson of the project: *impulse buying is defined by lack of deliberation
and necessity, not by how expensive an item is or what time of day it is.*

### Why this matters — the "fridge at night" problem

An earlier version flagged an expensive late-night purchase (e.g. buying a
fridge after moving) as impulsive, because price and hour were the dominant
features. By adding necessity and deliberation signals — and rewriting the
labels so that essential, planned purchases are not impulsive — the model now
correctly treats a needed fridge as a planned purchase, even when it is
expensive and bought at night.

---

## Connection to Research

| Feature | Research backing | Model importance |
|---------|-----------------|------------------|
| planning (wishlist) | Pre-commitment / lists reduce impulse spending | 0.235 |
| necessity | Needs vs wants is central to impulse definitions | 0.181 |
| deliberation | Impulse = unplanned, low-deliberation buying | 0.140 |
| category | Clothing/beauty/entertainment skew emotional | 0.076 |
| frequency | Repeated buying can be habitual | 0.052 |
| hour | Some impulse buys cluster at night | 0.045 |
| price | Weak on its own — expensive ≠ impulsive | 0.040 |
