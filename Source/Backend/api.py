"""FastAPI backend for ImpulseGuard.

Serves the trained model over HTTP so the Chrome extension (or any client)
can get predictions. Reuses the exact same pipeline, feature list and feedback
mechanism as the rest of the project.

Run from the project root:
    .venv/bin/python -m uvicorn Source.Backend.api:app --reload
Then open http://127.0.0.1:8000/docs to try it interactively.
"""

import os
import sys

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Reuse the model's config / helpers (single source of truth).
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ML_Model"))
import config  # noqa: E402
import preprocess  # noqa: E402

app = FastAPI(title="ImpulseGuard API", version="1.0")

# Allow the browser extension (a different origin) to call this API.
# For local development we allow all origins; tighten this for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained pipeline once at startup.
pipeline = joblib.load(config.MODEL_PATH)


# --- Request/response schemas -------------------------------------------
class Purchase(BaseModel):
    hour: int = Field(ge=0, le=23)
    price: float = Field(ge=0)
    category: str
    frequency: int = Field(ge=1, le=20)
    is_essential: int = Field(ge=0, le=1)
    deliberation_minutes: int = Field(ge=0)
    on_wishlist: int = Field(ge=0, le=1)


class Feedback(Purchase):
    true_level: int = Field(ge=0, le=3)


def _to_dataframe(purchase: Purchase) -> pd.DataFrame:
    """Turn a Purchase into a one-row DataFrame in the model's column order."""
    return pd.DataFrame([purchase.model_dump()])[config.FEATURES]


# --- Endpoints ----------------------------------------------------------
@app.get("/")
def health():
    """Simple health check."""
    return {"status": "ok", "model": os.path.basename(config.MODEL_PATH)}


@app.post("/predict")
def predict(purchase: Purchase):
    """Return the graded impulse level and per-level probabilities."""
    df = _to_dataframe(purchase)
    level = int(pipeline.predict(df)[0])
    probs = pipeline.predict_proba(df)[0]
    classes = list(pipeline.classes_)

    probabilities = {
        config.LEVEL_NAMES[c]: round(float(probs[i]), 4)
        for i, c in enumerate(classes)
    }
    confidence = round(float(probs[classes.index(level)]), 4)

    # Record the prediction in the shared log (same as the Telegram bot).
    preprocess.log_prediction(purchase.model_dump(), level, channel="extension")

    return {
        "level": level,
        "level_name": config.LEVEL_NAMES[level],
        "confidence": confidence,
        "probabilities": probabilities,
    }


@app.post("/feedback")
def feedback(item: Feedback):
    """Store a user-corrected example for the next training run."""
    purchase = {f: getattr(item, f) for f in config.FEATURES}
    preprocess.save_feedback(purchase, item.true_level)
    return {"status": "saved", "true_level": item.true_level}
