# Datasets

## Overview

The project uses:
- Synthetic transaction datasets (500 transactions)
- Manually labeled user purchase data

---

## Features

| Feature | Column Name | Type | Description |
|---------|-------------|------|-------------|
| Purchase time | `hour` | 0–23 | Hour of the day when purchase was made |
| Price | `price` | 5.00–500.00 | Purchase amount |
| Product category | `category` | text | clothing, food, electronics, entertainment, home, beauty |
| Purchase frequency | `frequency` | 1–20 | How often the user buys in that category |
| Label | `is_impulsive` | 0 or 1 | 1 = impulsive purchase, 0 = planned purchase |

---

## Why These Features?

Each feature was selected based on research into impulse buying behavior:

- **`hour`** — 74% of impulse purchases happen at night (22:00–05:00). Time of day is a strong behavioral signal.
- **`price`** — Higher-priced unplanned purchases are strongly linked to post-purchase regret.
- **`category`** — Clothing, beauty, and entertainment are the top impulse buying categories across multiple studies.
- **`frequency`** — Repeated buying in the same category correlates with impulsive and habitual behavior.

---

## Dataset Generation

Since real transaction data was unavailable, the dataset was synthetically generated using `data_generator.py`. Labels were assigned using a rule-based scoring system:

| Condition | Score |
|-----------|-------|
| Hour between 22:00–05:00 | +1 |
| Price > 200 | +1 |
| Frequency > 10 | +1 |
| Category is clothing, beauty, or entertainment | +1 |

Transactions scoring 2 or more points were labeled as impulsive (`is_impulsive = 1`).

---

## Limitations

- Dataset is synthetic — patterns are rule-based, not from real user behavior
- Real-world data would include more noise and edge cases
- 500 samples is relatively small for ML training
